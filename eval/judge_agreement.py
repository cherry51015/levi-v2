"""How much should we trust the judge? Compare its verdicts to human labels.

Reports raw agreement and Cohen's kappa (agreement corrected for chance: if 90%
of claims are "supported", two raters who always say "supported" agree 90% of
the time while knowing nothing - kappa would be 0 for them).

Usage: python -m eval.judge_agreement
"""
import json
from collections import Counter

from sklearn.metrics import cohen_kappa_score

from eval.common import save_report
from eval.generation_eval import LABEL_DIR


def main() -> None:
    claims = {json.loads(l)["id"]: json.loads(l) for l in
              (LABEL_DIR / "claims_to_label.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    human = {}
    for line in (LABEL_DIR / "human_labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            human[row["id"]] = row["label"]  # last label wins if you relabelled

    pairs = [(human[i], claims[i]["judge_verdict"]) for i in human if i in claims and claims[i]["judge_verdict"]]
    h, j = [p[0] for p in pairs], [p[1] for p in pairs]
    to_bin = lambda xs: ["supported" if x == "supported" else "not_supported" for x in xs]  # noqa: E731

    report = {
        "n": len(pairs),
        "three_way": {"agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3),
                      "cohen_kappa": round(cohen_kappa_score(h, j), 3)},
        "binary_supported_vs_not": {"agreement": round(sum(a == b for a, b in zip(to_bin(h), to_bin(j))) / len(pairs), 3),
                                    "cohen_kappa": round(cohen_kappa_score(to_bin(h), to_bin(j)), 3)},
        "confusion_human_to_judge": {f"{a} -> {b}": n for (a, b), n in sorted(Counter(pairs).items())},
        "disagreements": [{"id": i, "claim": claims[i]["claim"], "human": human[i], "judge": claims[i]["judge_verdict"]}
                          for i in human if i in claims and human[i] != claims[i]["judge_verdict"]],
    }
    out = save_report("judge_agreement", report)
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=2))
    print(f"\n{len(report['disagreements'])} disagreements - read them, they show where the judge is weak.\nReport: {out}")


if __name__ == "__main__":
    main()
