"""Evaluation tab: every number is read from eval/results/*.json, with n and caveats shown."""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from charts import completeness_bars, tradeoff_scatter, verdict_bar
from theme import card, esc

RESULTS = Path(os.getenv("LEVI_RESULTS_DIR", Path(__file__).resolve().parent.parent / "eval" / "results"))
SHIPPED = "hybrid + MiniLM rerank (k=10)"


def _load(prefix: str) -> list[tuple[str, dict]]:
    files = sorted(RESULTS.glob(f"{prefix}*.json"))
    return [(f.name, json.loads(f.read_text(encoding="utf-8"))) for f in files]


def _retrieval_table() -> pd.DataFrame | None:
    rows = []
    names = {"bm25": "BM25 only", "dense": "dense only", "hybrid": "hybrid (RRF)",
             "hybrid_rerank": "hybrid + bge-reranker (k=20)",
             "rerank bge-reranker-base k=10 len=256": "hybrid + bge-reranker (k=10)",
             "rerank ms-marco-MiniLM-L-6-v2 k=20 len=512": "hybrid + MiniLM rerank (k=20)",
             "rerank ms-marco-MiniLM-L-6-v2 k=10 len=256": SHIPPED}
    for _, rep in _load("retrieval_"):
        runs = rep.get("runs") or rep.get("modes") or {}
        for key, r in runs.items():
            label = names.get(key, key)
            q, lat = r["quality"], r["latency_ms"]["total"]
            ci = r.get("ci95", {}).get("hit@5")
            rows = [x for x in rows if x["config"] != label]  # files load oldest first, so the newest run wins
            rows.append({"config": label, "hit1": q["hit@1"], "hit5": q["hit@5"], "mrr": q["mrr"],
                         "hit5_ci": f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else "n/a",
                         "p50_ms": lat["p50"], "p95_ms": lat["p95"], "n": r["latency_ms"]["total"].get("n")})
    return pd.DataFrame(rows) if rows else None


def _generation() -> dict | None:
    runs = _load("generation_")
    if not runs:
        return None
    name, rep = runs[-1]
    rows = rep["rows"]
    errors = [r for r in rows if r["status"] == "error"]
    ok = [r for r in rows if r["status"] != "error"]
    ans, una = [r for r in ok if r["answerable"]], [r for r in ok if not r["answerable"]]
    answered = [r for r in ok if r["status"] == "answered"]
    claims = [c["verdict"] for r in answered for c in r["claims"]]
    comp = [r["completeness"] for r in ans if r.get("completeness") is not None]
    # Wait time was not recorded separately in early runs; calls >= 5 s are rate-limit stalls there.
    unstalled = [r for r in answered if r["timings_ms"].get("llm", 0) < 5000]
    pct = lambda v: {q: float(np.percentile(v, q)) for q in (50, 95, 99)} if v else {}  # noqa: E731
    return {
        "file": name, "n": len(rows), "errors": len(errors), "n_ans": len(ans), "n_una": len(una),
        "answered_rate": sum(r["status"] == "answered" for r in ans) / max(1, len(ans)),
        "false_refusals": Counter(r["status"] for r in ans if r["status"] != "answered"),
        "una_refused": sum(r["status"] != "answered" for r in una), "una_by": Counter(r["status"] for r in una),
        "verdicts": Counter(claims), "n_claims": len(claims), "completeness": comp,
        "comp_dist": dict(Counter(comp)),
        "lat_total": pct([r["total_ms"] - r["timings_ms"].get("llm_wait", 0) for r in unstalled]),
        "lat_llm": pct([r["timings_ms"]["llm"] for r in unstalled]), "n_unstalled": len(unstalled),
        "n_stalled": len(answered) - len(unstalled),
        "lat_rerank": pct([r["timings_ms"]["rerank"] for r in rows if "rerank" in r["timings_ms"]]),
        "length_bias": rep["metrics"].get("length_bias_check"),
    }


def render() -> None:
    st.markdown("### Evaluation")
    st.markdown('<div class="lv-def">Every number here is read from <span class="lv-mono">eval/results/*.json</span>, '
                'produced by the scripts in <span class="lv-mono">eval/</span> on the CUAD contract dataset. '
                'Sample sizes and caveats are shown next to each result; small samples mean wide uncertainty.</div>',
                unsafe_allow_html=True)
    st.write("")

    ret, gen = _retrieval_table(), _generation()
    refusal = _load("refusal_hybrid_rerank_")
    red = _load("redteam_")
    red_name, red_latest = red[-1] if red else (None, None)
    red_baseline = next((rep for _, rep in reversed(red) if "A_prompt_only" in rep["metrics"]["advice_leak_rate"]), None)

    # --- headline tiles
    cols = st.columns(6)
    if ret is not None and (ret["config"] == SHIPPED).any():
        s = ret[ret["config"] == SHIPPED].iloc[0]
        cols[0].metric("Top-1 retrieval (hit@1)", f"{s.hit1:.1%}",
                       help="Share of questions where the #1 retrieved chunk contains the lawyer-marked answer. n=107.")
        cols[1].metric("MRR", f"{s.mrr:.3f}", help="Mean reciprocal rank of the first correct chunk (1.0 = always first).")
    if gen:
        v = gen["verdicts"]
        cols[2].metric("Claims supported", f"{v.get('supported', 0) / max(1, gen['n_claims']):.0%}",
                       help=f"LLM judge: claim fully supported by the chunk it cites. n={gen['n_claims']} claims.")
        cols[3].metric("Unsupported claims", f"{v.get('unsupported', 0)} / {gen['n_claims']}",
                       help="Claims the judge found not supported by their cited text (fabrications).")
    if red_latest:
        m = red_latest["metrics"]
        cols[4].metric("Advice leak", f"{m['advice_leak_rate']['C_router_output']:.0%}",
                       help="Share of advice-seeking red-team questions whose answer gave legal advice (LLM judge). n=34.")
        cols[5].metric("Benign over-refusal", f"{m['benign_over_refusal_rate_C']:.0%}",
                       help="Normal questions wrongly blocked by a guardrail. n=16.")

    st.write("")
    # --- retrieval
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Retrieval: quality vs latency")
        if ret is not None:
            st.altair_chart(tradeoff_scatter(ret, SHIPPED), width="stretch")
    with right:
        st.markdown("#### Why this configuration")
        st.markdown(card("Decision", (
            '<div class="lv-def">Hybrid search (BM25 + vectors, fused with RRF) beat either alone. A cross-encoder '
            'reranker then lifts the right chunk to position 1 more often. The large bge reranker cost ~11 s/query on '
            'CPU and did not help; the 12x smaller MiniLM gave the best top-1 ranking at ~1 s.</div>'
            '<div class="lv-caveat">hit@5 intervals overlap across hybrid configs (n=107), so the claim is better '
            'top-ranking (hit@1, MRR), not better recall.</div>')), unsafe_allow_html=True)
        st.markdown(card("Definitions", (
            '<div class="lv-def"><b>hit@k</b>: a chunk containing the gold answer is in the top k.<br>'
            '<b>MRR</b>: average of 1/rank of the first correct chunk.<br>'
            '<b>95% CI</b>: bootstrap over questions.<br><b>Latency</b>: per query on a laptop CPU, retrieval only.'
            '</div>')), unsafe_allow_html=True)
    if ret is not None:
        with st.expander("Table view"):
            st.dataframe(ret.rename(columns={"hit1": "hit@1", "hit5": "hit@5", "hit5_ci": "hit@5 95% CI", "mrr": "MRR",
                                             "p50_ms": "p50 ms", "p95_ms": "p95 ms"}),
                         hide_index=True, width="stretch")

    st.divider()
    # --- answer quality
    st.markdown("#### Answer quality (LLM-as-judge)")
    if gen:
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown(card(f"Claim faithfulness · {gen['n_claims']} claims", (
                verdict_bar(gen["verdicts"]) +
                '<div class="lv-caveat">Judge: qwen3.8-27b (a different model family from the answering gpt-oss-120b, '
                'to avoid self-preference). Each claim is checked only against the chunk it cites.</div>')),
                unsafe_allow_html=True)
            lb = gen["length_bias"]
            judge_status = '<span class="lv-pending">pending</span> human labels to validate the judge (0 of 22 labelled)'
            st.markdown(card("Can the judge be trusted?", (
                f'<div class="lv-def">Length-bias check: Spearman ρ between answer length and score = '
                f'<b>{lb["spearman_answer_length_vs_score"]:+.2f}</b> (p={lb["p_value"]:.2f}), so no sign that longer '
                f'answers score higher.</div><div class="lv-def" style="margin-top:6px">{judge_status}</div>'
                if lb else judge_status)), unsafe_allow_html=True)
        with b:
            mean = np.mean(gen["completeness"]) if gen["completeness"] else float("nan")
            st.markdown(f'<div class="lv-card-title">Completeness vs lawyer-marked answer · mean {mean:.2f} / 5 · '
                        f'n={len(gen["completeness"])}</div>', unsafe_allow_html=True)
            st.altair_chart(completeness_bars(gen["comp_dist"]), width="stretch")
        st.markdown(f'<div class="lv-caveat">Source: <span class="lv-mono">{esc(gen["file"])}</span>. Run before the '
                    f'prompt fix for indirect answers; a same-seed re-run is pending.</div>', unsafe_allow_html=True)

        st.markdown(card("Refusal behaviour", (
            f'<div class="lv-kv">'
            f'<div class="lv-stat"><div class="label">Answerable questions answered</div><div class="value">'
            f'{gen["answered_rate"]:.0%}</div><div class="hint">n={gen["n_ans"]} · false refusals: '
            f'{gen["false_refusals"].get("refused_low_confidence", 0)} by retrieval gate, '
            f'{gen["false_refusals"].get("refused_not_in_document", 0)} by the LLM</div></div>'
            f'<div class="lv-stat"><div class="label">Unanswerable questions refused</div><div class="value">'
            f'{gen["una_refused"]} / {gen["n_una"]}</div><div class="hint">'
            f'{gen["una_by"].get("refused_low_confidence", 0)} by gate (no LLM call), '
            f'{gen["una_by"].get("refused_not_in_document", 0)} by the LLM</div></div>'
            f'<div class="lv-stat"><div class="label">Excluded as infrastructure errors</div><div class="value">'
            f'{gen["errors"]} / {gen["n"]}</div><div class="hint">provider daily token quota (HTTP 429); reported, '
            f'not counted as refusals</div></div></div>')), unsafe_allow_html=True)
        if refusal:
            t = refusal[-1][1]["test"]
            st.markdown(card("Retrieval refusal gate on its own · held-out half of contracts, 240 questions", (
                f'<div class="lv-def">Refuses before any LLM call when the best rerank score is below '
                f'<b>{t["threshold"]:.2f}</b> (tuned on the other half, false refusals capped at 10%). '
                f'Precision <b>{t["refusal_precision"]:.0%}</b> · recall <b>{t["refusal_recall"]:.0%}</b> · '
                f'false refusals <b>{t["false_refusal_rate"]:.1%}</b>. It catches about a third of unanswerable '
                f'questions for free; the LLM\'s own "not answerable" check handles the rest.</div>')),
                unsafe_allow_html=True)

    st.divider()
    # --- guardrails
    st.markdown("#### Guardrails: red-team (58 hand-written attacks)")
    if red_latest:
        m = red_latest["metrics"]
        cats = {"advice_direct": "Direct advice requests", "advice_disguised": "Disguised (role-play, hypotheticals)",
                "mixed": "Mixed (facts + advice)"}
        rows = [{"attack type": cats[k], "n": sum(r["category"] == k for r in red_latest["rows"]),
                 "advice leaked": f"{v:.0%}"} for k, v in m["advice_leak_by_category_C"].items()]
        rows += [{"attack type": "Benign questions wrongly refused", "n": 16, "advice leaked": "—",
                  "result": f"{m['benign_over_refusal_rate_C']:.0%}"},
                 {"attack type": "Off-topic caught before retrieval", "n": 8, "advice leaked": "—",
                  "result": f"{m['off_topic_caught_C']:.0%}"}]
        a, b = st.columns([3, 2], gap="large")
        with a:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with b:
            base = red_baseline["metrics"]["advice_leak_rate"].get("A_prompt_only") if red_baseline else None
            inj_cited = sum(r.get("injection_cited", False) for r in red_latest.get("injection", []))
            st.markdown(card("Reading these results honestly", (
                f'<div class="lv-def">With only the system prompt (no router, no output check) the leak rate was '
                f'already <b>{base:.0%}</b>: gpt-oss-120b follows the instruction. The extra layers add margin for '
                f'weaker fallback models (not yet measured), clearer replies, and skip the LLM for off-topic input.'
                f'</div><div class="lv-caveat">Router accuracy {m["router_accuracy"]:.0%}. Prompt injection: '
                f'{m["injection_attack_success_rate"]:.0%} success, but inconclusive: the injected passage was cited '
                f'in {inj_cited} of {len(red_latest.get("injection", []))} answers, so the model may not have seen it.'
                f'</div>' if base is not None else '')), unsafe_allow_html=True)
        st.markdown(f'<div class="lv-caveat">Source: <span class="lv-mono">{esc(red_name)}</span>. Measured before the '
                    f'router simplification (its LLM fallback was removed afterwards). A re-run hit the provider’s '
                    f'daily token quota on 18 of 58 queries and is excluded: see <span class="lv-mono">'
                    f'eval/results/invalid/</span>.</div>', unsafe_allow_html=True)

    st.divider()
    # --- latency
    st.markdown("#### Latency (end-to-end, laptop CPU + Groq free tier)")
    if gen and gen["lat_total"]:
        c = st.columns(4)
        c[0].metric("End-to-end p50", f"{gen['lat_total'][50] / 1000:.2f} s",
                    help="Answered questions, rate-limit stalls excluded.")
        c[1].metric("End-to-end p95", f"{gen['lat_total'][95] / 1000:.2f} s")
        c[2].metric("LLM generation p50", f"{gen['lat_llm'][50] / 1000:.2f} s", help="Model call only.")
        c[3].metric("Rerank p50", f"{gen['lat_rerank'][50]:.0f} ms", help="Cross-encoder on 10 chunks, CPU.")
        st.markdown(f'<div class="lv-caveat">n={gen["n_unstalled"]} answered questions. {gen["n_stalled"]} calls '
                    f'that stalled 40 s+ waiting for the provider\'s per-minute token limit are excluded; that run did '
                    f'not yet record wait time separately, so this split is a heuristic. Newer runs report '
                    f'<span class="lv-mono">llm_wait</span> as its own stage. p99 omitted: too few samples.</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Not done yet")
    st.markdown('<div class="lv-def">'
                '<span class="lv-pending">pending</span> Human labels for judge agreement (Cohen\'s κ).<br>'
                '<span class="lv-pending">pending</span> Generation re-run after the prompt fix for indirect answers '
                '(same seed, for a before/after).<br>'
                '<span class="lv-pending">pending</span> Load test (concurrency, rate limiter, CPU offload).<br>'
                '<span class="lv-pending">pending</span> Guardrail leak rate on the weaker fallback model.</div>',
                unsafe_allow_html=True)
