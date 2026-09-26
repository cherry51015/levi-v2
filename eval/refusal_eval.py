"""Tune the retrieval-confidence refusal threshold on CUAD.

CUAD asks every clause question of every contract; when a contract has no
such clause, the question is unanswerable. The system should refuse those
and answer the rest. We sweep thresholds on half the contracts (tune split)
and report precision/recall of refusal on the other half (test split), so the
reported number isn't overfit to the data that chose the threshold.

Usage: python -m eval.refusal_eval [--mode hybrid|hybrid_rerank] [--max-false-refusal 0.10]
"""
import argparse
import json
import time

import numpy as np

from eval.cuad import DATA_DIR
from eval.retrieval_eval import RESULTS_DIR, build_store, load_jsonl
from levi.pipeline import retrieval_confidence
from levi.retrieval import Embedder, Reranker, Retriever


def refusal_stats(conf: np.ndarray, answerable: np.ndarray, threshold: float) -> dict:
    refuse = conf < threshold
    tp = int(np.sum(refuse & ~answerable))  # correctly refused
    fp = int(np.sum(refuse & answerable))  # wrongly refused an answerable question
    fn = int(np.sum(~refuse & ~answerable))  # answered (well, attempted) an unanswerable one
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": round(float(threshold), 4), "refusal_precision": round(precision, 3),
            "refusal_recall": round(recall, 3), "false_refusal_rate": round(fp / max(1, int(answerable.sum())), 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hybrid", choices=("hybrid", "hybrid_rerank"))
    parser.add_argument("--max-false-refusal", type=float, default=0.10,
                        help="refusing answerable questions is costly; cap it and maximise recall under the cap")
    args = parser.parse_args()

    contracts = {c["contract_id"]: c for c in load_jsonl(DATA_DIR / "cuad_contracts.jsonl")}
    questions = load_jsonl(DATA_DIR / "cuad_questions.jsonl")
    embedder = Embedder()
    store, _, _ = build_store(contracts, embedder, "memory")
    retriever = Retriever(store, embedder, Reranker() if args.mode == "hybrid_rerank" else None)

    ids = sorted(contracts)
    tune_ids, test_ids = set(ids[::2]), set(ids[1::2])
    conf, answerable, split = [], [], []
    for q in questions:
        chunks = retriever.retrieve(q["question"], [q["contract_id"]], mode=args.mode)
        conf.append(retrieval_confidence(chunks))
        answerable.append(q["answerable"])
        split.append("tune" if q["contract_id"] in tune_ids else "test")
    conf, answerable, split = np.array(conf), np.array(answerable), np.array(split)

    tune = split == "tune"
    candidates = np.unique(conf[tune])
    sweep = [refusal_stats(conf[tune], answerable[tune], t) for t in candidates]
    allowed = [s for s in sweep if s["false_refusal_rate"] <= args.max_false_refusal]
    best = max(allowed, key=lambda s: (s["refusal_recall"], s["refusal_precision"])) if allowed else sweep[0]
    test = refusal_stats(conf[~tune], answerable[~tune], best["threshold"])

    separation = {
        "answerable_conf_mean": round(float(conf[answerable].mean()), 3),
        "unanswerable_conf_mean": round(float(conf[~answerable].mean()), 3),
    }
    report = {"mode": args.mode, "max_false_refusal": args.max_false_refusal, "chosen_on_tune": best,
              "test": test, "separation": separation, "n_questions": len(questions)}
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"refusal_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nSet LEVI_REFUSAL_THRESHOLD={best['threshold']} (report: {out})")


if __name__ == "__main__":
    main()
