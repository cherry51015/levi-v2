"""Retrieval ablation: BM25 vs dense vs hybrid (RRF) vs hybrid + rerank variants.

For each configuration, reports quality (hit@k, MRR, nDCG@5, with bootstrap
95% CIs) and latency (p50/p95 per stage) on the CUAD answerable questions.
The point is the tradeoff table: what each stage buys in quality and costs
in milliseconds.

Usage:
  python -m eval.retrieval_eval
  python -m eval.retrieval_eval --rerankers BAAI/bge-reranker-base:20:512 \
      cross-encoder/ms-marco-MiniLM-L-6-v2:10:256
  python -m eval.retrieval_eval --store qdrant --rerankers none   # backend parity check
  (reranker spec = model:rerank_k:max_length; pass --rerankers none to skip)
"""
import argparse
import json
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

import numpy as np

from eval.cuad import DATA_DIR
from eval.metrics import bootstrap_ci, hit_at_k, ndcg_at_k, overlaps, reciprocal_rank
from levi.config import settings
from levi.ingest import chunk_document, from_text
from levi.retrieval import Embedder, Reranker, Retriever
from levi.store import ChunkStore, InMemoryStore, QdrantStore, new_meta
from levi.timing import StageTimer, percentiles

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = DATA_DIR / "cache"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_store(contracts: dict, embedder: Embedder, backend: str) -> tuple[ChunkStore, dict, list[float]]:
    """Put every contract in the store. Vectors are cached on disk, keyed by model + chunking config."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{settings.embed_model.replace('/', '_')}_{settings.chunk_words}_{settings.chunk_overlap}"
    if backend == "qdrant":
        store: ChunkStore = QdrantStore(tempfile.mkdtemp(prefix="levi_eval_qdrant_"), dim=embedder.dim)
    else:
        store = InMemoryStore()
    chunks_by_contract, index_ms = {}, []
    for cid, c in contracts.items():
        doc = from_text(c["text"], doc_id=cid)
        chunks = chunk_document(doc, settings.chunk_words, settings.chunk_overlap)
        cache = CACHE_DIR / f"{cid}_{tag}.npy"
        if cache.exists():
            vecs = np.load(cache)
        else:
            t = time.perf_counter()
            vecs = embedder.embed_passages([ch.text for ch in chunks])
            index_ms.append((time.perf_counter() - t) * 1000)
            np.save(cache, vecs)
        store.add_document(new_meta(cid, c["title"], chunks, doc.text), chunks, vecs)
        chunks_by_contract[cid] = chunks
    return store, chunks_by_contract, index_ms


def evaluate(retriever: Retriever, mode: str, chunks_by_contract: dict, questions: list[dict]) -> dict:
    per_q = defaultdict(list)
    stage_ms = defaultdict(list)
    # Warm-up call so the first query doesn't pay one-time costs.
    retriever.retrieve(questions[0]["question"], [questions[0]["contract_id"]], mode=mode)

    for q in questions:
        spans = [(g["start"], g["end"]) for g in q["gold"]]
        n_relevant = sum(overlaps(c.char_start, c.char_end, spans) for c in chunks_by_contract[q["contract_id"]])

        timer = StageTimer()
        results = retriever.retrieve(q["question"], [q["contract_id"]], mode=mode, timer=timer)
        rels = [overlaps(r.chunk.char_start, r.chunk.char_end, spans) for r in results]

        for k in (1, 3, 5):
            per_q[f"hit@{k}"].append(hit_at_k(rels, k))
        per_q["mrr"].append(reciprocal_rank(rels))
        per_q["ndcg@5"].append(ndcg_at_k(rels, 5, n_relevant))
        for stage, ms in timer.stages.items():
            stage_ms[stage].append(ms)
        stage_ms["total"].append(timer.total_ms)

    return {
        "quality": {m: round(mean(v), 3) for m, v in per_q.items()},
        "ci95": {m: bootstrap_ci(per_q[m]) for m in ("hit@5", "mrr")},
        "latency_ms": {s: percentiles(v) for s, v in stage_ms.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerankers", nargs="+",
        default=[f"{settings.rerank_model}:{settings.rerank_k}:{settings.rerank_max_length}"],
    )
    parser.add_argument("--store", choices=("memory", "qdrant"), default="memory")
    parser.add_argument("--skip-base", action="store_true", help="only run the reranker variants")
    args = parser.parse_args()

    contracts = {c["contract_id"]: c for c in load_jsonl(DATA_DIR / "cuad_contracts.jsonl")}
    questions = [q for q in load_jsonl(DATA_DIR / "cuad_questions.jsonl") if q["answerable"]]

    embedder = Embedder()
    store, chunks_by_contract, index_ms = build_store(contracts, embedder, args.store)
    n_chunks = [len(ch) for ch in chunks_by_contract.values()]

    report = {
        "config": settings.as_dict(),
        "store": args.store,
        "n_contracts": len(contracts),
        "n_questions": len(questions),
        "indexing_ms_uncached": percentiles(index_ms),
        "chunks_per_contract": {"mean": round(mean(n_chunks), 1), "max": max(n_chunks)},
        "runs": {},
    }

    if not args.skip_base:
        base = Retriever(store, embedder)
        for mode in ("bm25", "dense", "hybrid"):
            report["runs"][mode] = evaluate(base, mode, chunks_by_contract, questions)

    for spec in args.rerankers:
        if spec == "none":
            continue
        model, k, max_len = spec.rsplit(":", 2)
        t = time.perf_counter()
        reranker = Reranker(model, max_length=int(max_len))
        load_s = time.perf_counter() - t
        retriever = Retriever(store, embedder, reranker, cfg=replace(settings, rerank_k=int(k)))
        name = f"rerank {model.split('/')[-1]} k={k} len={max_len}"
        report["runs"][name] = evaluate(retriever, "hybrid_rerank", chunks_by_contract, questions)
        report["runs"][name]["model_load_s"] = round(load_s, 1)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{len(questions)} answerable questions over {len(contracts)} contracts "
          f"(avg {report['chunks_per_contract']['mean']} chunks each)\n")
    print(f"{'config':<48}{'hit@1':>7}{'hit@5':>7}{'hit@5 95% CI':>16}{'MRR':>7}{'p50 ms':>9}{'p95 ms':>9}")
    for name, r in report["runs"].items():
        q, lat, ci = r["quality"], r["latency_ms"]["total"], r["ci95"]["hit@5"]
        print(f"{name:<48}{q['hit@1']:>7.3f}{q['hit@5']:>7.3f}{str(ci):>16}{q['mrr']:>7.3f}"
              f"{lat['p50']:>9.1f}{lat['p95']:>9.1f}")
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
