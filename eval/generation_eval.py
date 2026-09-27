"""End-to-end answer quality on CUAD, scored by an LLM judge.

For a stratified sample of answerable and unanswerable questions, runs the real
pipeline and measures:
- Faithfulness, per claim: is each claim supported by the chunk it cites?
  (fabrication rate = share of answers with at least one unsupported claim)
- Completeness vs the lawyer-highlighted gold answer (1-5 anchored scale)
- Refusal behaviour, split by stage: retrieval gate (no LLM call) vs the LLM
  itself saying the excerpts don't answer
- Latency per stage, p50/p95
- Length bias check: does the judge's completeness score track answer length?
It also exports every judged claim for blind human labelling (eval/label_app.py),
so the judge itself can be validated (eval/judge_agreement.py).

Usage: python -m eval.generation_eval [--answerable 30] [--unanswerable 20]
"""
import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from scipy.stats import spearmanr

from eval.common import EvalRig, save_report
from eval.judge import judge_completeness, judge_support, make_judge_client
from levi.timing import percentiles

LABEL_DIR = Path(__file__).parent / "labeling"


def sample(questions: list[dict], n_ans: int, n_unans: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ans = [q for q in questions if q["answerable"]]
    unans = [q for q in questions if not q["answerable"]]
    rng.shuffle(ans)
    rng.shuffle(unans)
    return ans[:n_ans] + unans[:n_unans]


async def run(n_ans: int, n_unans: int, seed: int) -> dict:
    rig = EvalRig()
    judge = make_judge_client()
    pipe = rig.pipeline()
    rows, claims_for_labels = [], []

    for i, q in enumerate(sample(rig.questions, n_ans, n_unans, seed)):
        r = (await pipe.ask(q["question"], [q["contract_id"]])).model_dump()
        for retry in range(3):
            if r["status"] != "error":
                break
            # Provider outage or exhausted quota is infrastructure, not answer quality: cool down and retry.
            last = r["llm_attempts"][-1]["outcome"] if r["llm_attempts"] else "?"
            print(f"    error ({last}), retry {retry + 1} in 70s", flush=True)
            await asyncio.sleep(70)
            for p in rig.llm.providers:
                p.breaker.record_success()
            r = (await pipe.ask(q["question"], [q["contract_id"]])).model_dump()
        row = {"qid": q["qid"], "answerable": q["answerable"], "category": q["category"], "question": q["question"],
               "status": r["status"], "answer": r["answer"], "provider": r["provider"], "llm_attempts": r["llm_attempts"],
               "timings_ms": r["timings_ms"], "total_ms": r["total_ms"], "claims": []}

        if r["status"] == "answered":
            # Judge each claim against the exact text the model was shown (incl. neighbour context).
            cites = {c["ref"]: c["text"] for c in r["citations"]}
            for j, claim in enumerate(r["claims"]):
                sources = [cites[ref] for ref in claim["citations"] if ref in cites]
                verdict = await judge_support(judge, claim["text"], sources)
                row["claims"].append({"text": claim["text"], "verdict": verdict.get("verdict"),
                                      "reasoning": verdict.get("reasoning")})
                claims_for_labels.append({"id": f"{q['qid']}#{j}", "question": q["question"], "claim": claim["text"],
                                          "sources": sources, "judge_verdict": verdict.get("verdict")})
            if q["answerable"]:
                comp = await judge_completeness(judge, q["question"], [g["text"] for g in q["gold"]], r["answer"])
                row["completeness"] = comp.get("score")
                row["completeness_reasoning"] = comp.get("reasoning")
        rows.append(row)
        verdicts = "".join((c["verdict"] or "?")[0] for c in row["claims"])
        print(f"{i + 1:>3} {q['qid'][:48]:<48} {'ANS' if q['answerable'] else 'UNA'} {r['status']:<24} "
              f"claims={verdicts or '-':<6} comp={row.get('completeness', '-')}", flush=True)

    return {"rows": rows, "metrics": metrics(rows), "judge_model": "qwen/qwen3.8-27b",
            "answer_model": rig.llm.providers[0].label, "seed": seed}, claims_for_labels


def metrics(rows: list[dict]) -> dict:
    def rate(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    errors = [r for r in rows if r["status"] == "error"]
    rows = [r for r in rows if r["status"] != "error"]  # infrastructure failures are reported, not scored
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["status"] == "answered"]
    claims = [c for r in answered for c in r["claims"]]
    verdicts = Counter(c["verdict"] for c in claims)
    comp = [r for r in ans if r.get("completeness") is not None]
    m = {
        "n": {"answerable": len(ans), "unanswerable": len(unans), "claims_judged": len(claims),
              "excluded_errors": len(errors)},
        "answerable": {
            "answered_rate": rate([r["status"] == "answered" for r in ans]),
            "refused_by_retrieval_gate": rate([r["status"] == "refused_low_confidence" for r in ans]),
            "refused_by_llm": rate([r["status"] == "refused_not_in_document" for r in ans]),
        },
        "unanswerable": {
            "refusal_rate_total": rate([r["status"] in ("refused_low_confidence", "refused_not_in_document")
                                        for r in unans]),
            "refused_by_retrieval_gate": rate([r["status"] == "refused_low_confidence" for r in unans]),
            "refused_by_llm": rate([r["status"] == "refused_not_in_document" for r in unans]),
        },
        "faithfulness": {
            "claims_supported": rate([c["verdict"] == "supported" for c in claims]),
            "claims_partial": rate([c["verdict"] == "partial" for c in claims]),
            "claims_unsupported": rate([c["verdict"] == "unsupported" for c in claims]),
            "answers_with_unsupported_claim": rate([any(c["verdict"] == "unsupported" for c in r["claims"])
                                                    for r in answered]),
            "verdict_counts": dict(verdicts),
        },
        "completeness": {
            "mean_score_1to5": round(mean(r["completeness"] for r in comp), 2) if comp else None,
            "score_ge_4_rate": rate([r["completeness"] >= 4 for r in comp]),
            "score_distribution": dict(sorted(Counter(r["completeness"] for r in comp).items())),
        },
        "latency_ms": {
            "total_answered": percentiles([r["total_ms"] for r in answered]),
            "total_answered_excl_rate_limit_wait": percentiles(
                [r["total_ms"] - r["timings_ms"].get("llm_wait", 0) for r in answered]),
            "llm_model_time": percentiles([r["timings_ms"].get("llm", 0) for r in answered]),
            "llm_rate_limit_wait": percentiles([r["timings_ms"].get("llm_wait", 0) for r in answered]),
            "rerank": percentiles([r["timings_ms"].get("rerank", 0) for r in rows if "rerank" in r["timings_ms"]]),
            "refused_by_gate_total": percentiles([r["total_ms"] for r in rows
                                                  if r["status"] == "refused_low_confidence"]),
        },
    }
    if len(comp) >= 5:
        rho, p = spearmanr([len(r["answer"]) for r in comp], [r["completeness"] for r in comp])
        m["length_bias_check"] = {"spearman_answer_length_vs_score": round(float(rho), 3), "p_value": round(float(p), 3),
                                  "note": "a strong positive rho would suggest the judge rewards longer answers"}
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answerable", type=int, default=30)
    parser.add_argument("--unanswerable", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    report, claims = asyncio.run(run(args.answerable, args.unanswerable, args.seed))
    out = save_report("generation", report)
    LABEL_DIR.mkdir(exist_ok=True)
    with open(LABEL_DIR / "claims_to_label.jsonl", "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(json.dumps(report["metrics"], indent=2))
    print(f"\nReport: {out}\nClaims for human labelling: {LABEL_DIR / 'claims_to_label.jsonl'} ({len(claims)})")


if __name__ == "__main__":
    main()
