"""The /ask request path, with every guardrail layer visible in order:

  embed query (once) -> [L1] intent router (classifier on that embedding)
  -> confidently off_topic? reply without retrieval or LLM
  -> hybrid retrieval (reuses the query embedding)
  -> [L3] refusal if retrieval confidence is below the tuned threshold (no LLM call)
  -> [L2] generate: stricter "facts only" mode for advice-seeking questions
  -> [L3] citation enforcement  -> [L4] output advice check
  -> response with per-stage timings
"""
import asyncio
from typing import Literal

from pydantic import BaseModel

from levi.answer import REFUSAL_LOW_CONFIDENCE, GroundedAnswer, generate_answer
from levi.guardrails import ADVICE_NOTE, OFF_TOPIC_REPLY, IntentRouter, contains_advice
from levi.llm import LLMClient
from levi.retrieval import Retriever
from levi.schemas import ScoredChunk
from levi.timing import StageTimer
from levi.tracing import current_trace_id, observe, update_span


class AskResponse(GroundedAnswer):
    status: Literal["answered", "refused_low_confidence", "refused_not_in_document",
                    "refused_advice", "off_topic", "error"]
    intent: str
    intent_confidence: float
    intent_source: Literal["classifier", "uncertain", "disabled"]
    notice: str | None = None
    retrieval_confidence: float | None = None
    timings_ms: dict[str, float] = {}
    total_ms: float = 0.0
    trace_id: str | None = None  # Langfuse trace, for debugging a specific answer


def retrieval_confidence(chunks: list[ScoredChunk]) -> float:
    """Best evidence that *something* relevant was found.

    Uses the top rerank score when available: it separates answerable from unanswerable
    questions far better than cosine on CUAD (eval/refusal_eval.py). Otherwise fall back to
    the best cosine similarity (RRF scores only encode rank, not how good a match is).
    """
    if not chunks:
        return 0.0
    if chunks[0].stage == "rerank":
        return max(c.score for c in chunks)
    return max((c.dense_score for c in chunks if c.dense_score is not None), default=0.0)


class Pipeline:
    def __init__(self, retriever: Retriever, router: IntentRouter, llm: LLMClient,
                 refusal_threshold: float, retrieval_mode: str = "hybrid", offload_cpu: bool = True,
                 use_router: bool = True, use_output_check: bool = True):
        """use_router / use_output_check exist so the red-team eval can measure each layer's contribution."""
        self.retriever = retriever
        self.router = router
        self.llm = llm
        self.refusal_threshold = refusal_threshold
        self.retrieval_mode = retrieval_mode
        self.offload_cpu = offload_cpu
        self.use_router = use_router
        self.use_output_check = use_output_check

    async def _cpu(self, fn, *args, **kwargs):
        # CPU-bound calls inside an async handler freeze every other request on the
        # event loop; to_thread moves them to a worker thread.
        if self.offload_cpu:
            return await asyncio.to_thread(fn, *args, **kwargs)
        return fn(*args, **kwargs)

    @observe(name="intent_router", as_type="guardrail", capture_input=False, capture_output=False)
    def _route(self, question: str, qvec, timer: StageTimer) -> tuple[str, float, str]:
        """Layer 1. The classifier's label is trusted only when it is confident.

        When unsure, fail open to retrieval: the refusal gate and the LLM's own
        "not answerable" check decide whether the document covers the question.
        Exception: an uncertain "advice" guess still gets the safer facts-only mode,
        since that mode costs nothing if the question was actually informational.
        (An earlier version asked an LLM to classify uncertain queries; it added
        ~2.4s and misrouted document questions like "is a trip mentioned?" as off_topic.)
        """
        with timer.stage("route"):
            intent, conf = self.router.predict(qvec)
        source = "classifier"
        if not self.router.is_confident(conf):
            source = "uncertain"
            if intent != "advice":
                intent = "informational"
        update_span(input=question, output={"intent": intent, "confidence": round(conf, 3), "source": source})
        return intent, conf, source

    @observe(name="ask")
    async def ask(self, question: str, doc_ids: list[str]) -> AskResponse:
        timer = StageTimer()

        with timer.stage("embed_query"):
            qvec = await self._cpu(self.retriever.embedder.embed_query, question)
        if self.use_router:
            intent, conf, source = self._route(question, qvec, timer)
        else:
            intent, conf, source = "informational", 1.0, "disabled"
        base = dict(intent=intent, intent_confidence=round(conf, 3), intent_source=source)

        def finish(**fields) -> AskResponse:
            return AskResponse(**base, **fields, timings_ms={k: round(v, 1) for k, v in timer.stages.items()},
                               total_ms=round(timer.total_ms, 1), trace_id=current_trace_id())

        if intent == "off_topic":
            return finish(status="off_topic", answer=OFF_TOPIC_REPLY)

        chunks = await self._cpu(self.retriever.retrieve, question, doc_ids, mode=self.retrieval_mode,
                                 timer=timer, query_vec=qvec)
        confidence = retrieval_confidence(chunks)
        if confidence < self.refusal_threshold:
            # Nothing relevant enough: refuse without spending an LLM call.
            if intent == "advice":
                # "Should I sign?" rarely matches a clause; "couldn't find anything" would mislead.
                return finish(status="refused_advice", answer=ADVICE_NOTE, retrieval_confidence=round(confidence, 3))
            return finish(status="refused_low_confidence", answer=REFUSAL_LOW_CONFIDENCE,
                          retrieval_confidence=round(confidence, 3))

        with timer.stage("expand_context"):
            chunks = self.retriever.expand_context(chunks, doc_ids)

        advice = intent == "advice"
        try:
            grounded = await generate_answer(self.llm, question, chunks, timer=timer,
                                             advice_mode=advice,
                                             is_advice=contains_advice if self.use_output_check else None)
        except Exception as e:  # LLMUnavailable after the whole fallback chain
            attempts = [a.__dict__ for a in getattr(e, "attempts", [])]
            return finish(status="error", answer=f"The answer service is temporarily unavailable ({type(e).__name__}).",
                          retrieval_confidence=round(confidence, 3), llm_attempts=attempts)

        fields = grounded.model_dump()
        if grounded.status == "refused_advice" or (advice and grounded.status != "answered"):
            fields.update(status="refused_advice", answer=ADVICE_NOTE)
        elif advice:
            fields["notice"] = ADVICE_NOTE
        return finish(**fields, retrieval_confidence=round(confidence, 3))
