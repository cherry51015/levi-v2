"""Shared setup: the real pipeline, in-process, over the CUAD contracts."""
import json
from pathlib import Path

from eval.cuad import DATA_DIR
from eval.retrieval_eval import build_store, load_jsonl
from levi.config import settings
from levi.guardrails import IntentRouter
from levi.llm import LLMClient, default_providers
from levi.pipeline import Pipeline
from levi.retrieval import Embedder, Reranker, Retriever

RESULTS_DIR = Path(__file__).parent / "results"
SETS_DIR = Path(__file__).parent / "sets"


def answer_client() -> LLMClient:
    """Primary model only, and wait out rate limits instead of falling back:
    an eval must measure one model, not whichever one happened to be available."""
    return LLMClient(default_providers()[:1], max_retry_after_s=65, deadline_s=180)


class EvalRig:
    def __init__(self, **pipeline_kwargs):
        self.contracts = {c["contract_id"]: c for c in load_jsonl(DATA_DIR / "cuad_contracts.jsonl")}
        self.questions = load_jsonl(DATA_DIR / "cuad_questions.jsonl")
        self.embedder = Embedder()
        self.store, self.chunks_by_contract, _ = build_store(self.contracts, self.embedder, "memory")
        self.retriever = Retriever(self.store, self.embedder, Reranker())
        self.router = IntentRouter.train_default(self.embedder)
        self.llm = answer_client()

    def pipeline(self, **kwargs) -> Pipeline:
        return Pipeline(self.retriever, self.router, self.llm, settings.refusal_threshold,
                        retrieval_mode="hybrid_rerank", offload_cpu=False, **kwargs)


def load_set(name: str) -> list[dict]:
    return [json.loads(line) for line in (SETS_DIR / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_report(prefix: str, report: dict) -> Path:
    import time

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
