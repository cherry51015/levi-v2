"""Hybrid retrieval: BM25 + dense, fused with RRF, then cross-encoder rerank.

Why each piece exists:
- BM25 catches exact legal terms ("indemnification", "Section 12.3") that
  embeddings blur together.
- Dense (bi-encoder) catches paraphrases ("end the contract early" vs
  "termination for convenience") that BM25 misses.
- RRF merges the two rankings using only rank positions, so we never have
  to calibrate BM25 scores (unbounded) against cosine scores ([-1, 1]).
- The cross-encoder reads query and chunk *together*, which is far more
  accurate but too slow to run on every chunk - so it only rescores the
  top candidates (recall first, then precision).

Storage is behind the ChunkStore interface (levi/store.py): numpy in memory
for tests/eval, embedded Qdrant for the persistent document library.
"""
import re
from collections import OrderedDict, defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from levi.config import Settings, settings
from levi.ingest import Document, chunk_document
from levi.store import ChunkStore, DocumentMeta, new_meta
from levi.schemas import Chunk, ScoredChunk
from levi.timing import StageTimer
from levi.tracing import observe, update_span

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Embedder:
    """Pretrained bge-base-en-v1.5 from Hugging Face, loaded with sentence-transformers."""

    # bge models are trained with this instruction on the query side only.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = settings.embed_model):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([self.QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)
        return vec[0].astype(np.float32)


class Reranker:
    """Pretrained cross-encoder from Hugging Face: reads (query, chunk) together and scores relevance."""

    def __init__(self, model_name: str = settings.rerank_model, max_length: int = settings.rerank_max_length):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device="cpu", max_length=max_length)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        # Raw relevance logits (ms-marco MiniLM: roughly -11..+10). Only the order matters for
        # ranking; the refusal threshold is tuned on this same scale.
        return np.asarray(self.model.predict([(query, t) for t in texts], batch_size=16))


class BM25Index:
    """Keyword index over the chunks of whichever documents are in scope."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.chunks[i], float(scores[i])) for i in top]


def index_document(store: ChunkStore, embedder: Embedder, doc: Document, filename: str,
                   cfg: Settings = settings) -> tuple[DocumentMeta, bool]:
    """Chunk, embed and store a document. Returns (meta, was_cached).

    doc_id is a hash of the file bytes, so re-uploading the same file skips
    re-embedding entirely - the most expensive step of ingestion.
    """
    if store.has_document(doc.doc_id):
        meta = next((m for m in store.list_documents() if m.doc_id == doc.doc_id), None)
        if meta is not None:
            return meta, True
    chunks = chunk_document(doc, cfg.chunk_words, cfg.chunk_overlap)
    vectors = embedder.embed_passages([c.text for c in chunks])
    meta = new_meta(doc.doc_id, filename, chunks, doc.text)
    store.add_document(meta, chunks, vectors)
    return meta, False


def merge_window(chunks: list[Chunk], overlap: int) -> str:
    """Join consecutive chunks into one passage, dropping the words each chunk repeats from the previous one."""
    words = chunks[0].text.split()
    for nxt in chunks[1:]:
        words += nxt.text.split()[overlap:]
    return " ".join(words)


class Retriever:
    BM25_CACHE_SIZE = 32

    def __init__(self, store: ChunkStore, embedder: Embedder, reranker: Reranker | None = None,
                 cfg: Settings = settings):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg
        # BM25 statistics (IDF) depend on which documents are searched together, so
        # the index is built per scope and kept in a small LRU cache.
        self._bm25_cache: OrderedDict[tuple[str, ...], BM25Index] = OrderedDict()

    def _bm25_for(self, doc_ids: list[str]) -> BM25Index | None:
        key = tuple(sorted(doc_ids))
        if key in self._bm25_cache:
            self._bm25_cache.move_to_end(key)
            return self._bm25_cache[key]
        chunks = self.store.get_chunks(list(key))
        if not chunks:
            return None
        index = BM25Index(chunks)
        self._bm25_cache[key] = index
        if len(self._bm25_cache) > self.BM25_CACHE_SIZE:
            self._bm25_cache.popitem(last=False)
        return index

    def invalidate(self, doc_id: str) -> None:
        for key in [k for k in self._bm25_cache if doc_id in k]:
            del self._bm25_cache[key]

    def expand_context(self, results: list[ScoredChunk], doc_ids: list[str]) -> list[ScoredChunk]:
        """Small-to-big retrieval: search with small chunks (precise matching), but give the LLM
        each top chunk together with its neighbours (legal clauses depend on surrounding text)."""
        window, top_n = self.cfg.context_window, self.cfg.context_expand_top
        index = self._bm25_for(doc_ids) if window > 0 and results else None
        if index is None:
            return results
        by_position = {(c.doc_id, c.index): c for c in index.chunks}  # the scope's chunks are already cached
        expanded = []
        for rank, sc in enumerate(results):
            if rank < top_n:
                c = sc.chunk
                parts = [by_position.get((c.doc_id, c.index + d)) for d in range(-window, window + 1)]
                sc = sc.model_copy(update={"context": merge_window([p for p in parts if p], self.cfg.chunk_overlap)})
            expanded.append(sc)
        return expanded

    @observe(name="retrieve", as_type="retriever", capture_input=False, capture_output=False)
    def retrieve(self, query: str, doc_ids: list[str], mode: str = "hybrid_rerank",
                 timer: StageTimer | None = None, query_vec: np.ndarray | None = None) -> list[ScoredChunk]:
        """query_vec lets the caller reuse an embedding it already computed (the intent router does)."""
        results = self._retrieve(query, doc_ids, mode, timer, query_vec)
        update_span(input={"query": query, "doc_ids": doc_ids, "mode": mode},
                    output=[{"chunk_id": r.chunk.id, "page": r.chunk.page, "score": round(r.score, 4)}
                            for r in results])
        return results

    def _retrieve(self, query: str, doc_ids: list[str], mode: str, timer: StageTimer | None,
                  query_vec: np.ndarray | None) -> list[ScoredChunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("hybrid_rerank needs a Reranker")
        timer = timer or StageTimer()
        cfg = self.cfg

        dense_by_id: dict[str, float] = {}

        def scored(hits: list[tuple[Chunk, float]], stage: str) -> list[ScoredChunk]:
            return [ScoredChunk(chunk=c, score=s, stage=stage, dense_score=dense_by_id.get(c.id)) for c, s in hits]

        bm25_hits: list[tuple[Chunk, float]] = []
        if mode != "dense":
            with timer.stage("bm25"):
                bm25 = self._bm25_for(doc_ids)
                bm25_hits = bm25.search(query, cfg.candidates_k) if bm25 else []
            if mode == "bm25":
                return scored(bm25_hits[: cfg.final_k], "bm25")

        if query_vec is None:
            with timer.stage("embed_query"):
                query_vec = self.embedder.embed_query(query)
        with timer.stage("dense"):
            dense_hits = self.store.dense_search(query_vec, doc_ids, cfg.candidates_k)
            dense_by_id = {c.id: s for c, s in dense_hits}
        if mode == "dense":
            return scored(dense_hits[: cfg.final_k], "dense")

        with timer.stage("fusion"):
            by_id = {c.id: c for c, _ in dense_hits + bm25_hits}
            fused = rrf_fuse([[c.id for c, _ in dense_hits], [c.id for c, _ in bm25_hits]], k=cfg.rrf_k)
            fused_hits = [(by_id[cid], s) for cid, s in fused]
        if mode == "hybrid":
            return scored(fused_hits[: cfg.final_k], "rrf")

        candidates = fused_hits[: cfg.rerank_k]
        with timer.stage("rerank"):
            rerank_scores = self.reranker.score(query, [c.text for c, _ in candidates])
        order = np.argsort(-rerank_scores, kind="stable")[: cfg.final_k]
        return [ScoredChunk(chunk=candidates[j][0], score=float(rerank_scores[j]), stage="rerank",
                            dense_score=dense_by_id.get(candidates[j][0].id)) for j in order]