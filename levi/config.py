"""Every tunable knob lives here, so an eval run is reproducible from its config."""
import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    embed_model: str = os.getenv("LEVI_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    rerank_model: str = os.getenv("LEVI_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    chunk_words: int = int(os.getenv("LEVI_CHUNK_WORDS", "200"))
    chunk_overlap: int = int(os.getenv("LEVI_CHUNK_OVERLAP", "40"))
    candidates_k: int = 20  # results pulled from each first-stage retriever
    # Reranker choice comes from the ablation in eval/retrieval_eval.py: MiniLM-L6 with k=10, len=256
    # beat bge-reranker-base on hit@1/MRR at ~1/10 the latency on CPU.
    rerank_k: int = 10  # fused candidates handed to the cross-encoder
    rerank_max_length: int = 256  # tokens per (query, chunk) pair; cross-encoder cost grows with this
    final_k: int = 5  # chunks that reach the LLM
    # Small-to-big: the top `context_expand_top` chunks are widened by `context_window` neighbouring chunks
    # on each side before going to the LLM, so a clause isn't cut mid-sentence by the 200-word window.
    # Limited to the top 3 to keep prompts inside the free tier's 8k tokens/minute.
    context_window: int = int(os.getenv("LEVI_CONTEXT_WINDOW", "1"))
    context_expand_top: int = int(os.getenv("LEVI_CONTEXT_EXPAND_TOP", "3"))
    rrf_k: int = 60  # RRF damping constant (value from the original RRF paper)
    retrieval_mode: str = os.getenv("LEVI_RETRIEVAL_MODE", "hybrid_rerank")
    # Below this top rerank score (a raw logit) we refuse without calling the LLM. The gate only filters
    # clearly irrelevant questions; close calls go to the LLM, which refused every unanswerable question in
    # the generation eval. Tuned on half the CUAD contracts with false refusals capped at 2%; on the other
    # half: 90% precision, 4.3% false refusals, 23% of unanswerables caught for free (eval/refusal_eval.py).
    # History: -7.49 with a 10% cap gave 85% precision but 8.5% false refusals. Re-tune if the reranker changes.
    refusal_threshold: float = float(os.getenv("LEVI_REFUSAL_THRESHOLD", "-8.65"))
    # Run CPU-bound work (embedding, BM25, rerank) in a thread pool so it doesn't block the event loop.
    # A switch, not a constant, so the load test can measure the difference.
    offload_cpu: bool = os.getenv("LEVI_OFFLOAD_CPU", "1") == "1"
    qdrant_path: str = os.getenv("LEVI_QDRANT_PATH", "data/qdrant")
    max_upload_mb: int = int(os.getenv("LEVI_MAX_UPLOAD_MB", "10"))
    # /ask budget for the public demo: burst of 10, then 6 per minute (about what the free LLM tier sustains).
    ask_burst: int = int(os.getenv("LEVI_ASK_BURST", "10"))
    ask_per_minute: float = float(os.getenv("LEVI_ASK_PER_MINUTE", "6"))

    def as_dict(self) -> dict:
        return asdict(self)


settings = Settings()
