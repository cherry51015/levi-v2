"""Pipeline wiring with fakes: no models, no network."""
import asyncio
import json

import numpy as np

from levi.guardrails import contains_advice
from levi.ingest import chunk_document, from_text
from levi.llm import LLMResult
from levi.pipeline import Pipeline
from levi.retrieval import Retriever
from levi.store import InMemoryStore, new_meta

DIM = 4


class FakeEmbedder:
    dim = DIM

    def embed_query(self, q):
        return np.array([1, 0, 0, 0], dtype=np.float32)


class FakeRouter:
    def __init__(self, label, conf=0.95):
        self.label, self.conf = label, conf

    def predict(self, qvec):
        return self.label, self.conf

    def is_confident(self, c):
        return c >= 0.6


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        return LLMResult(text=self.replies.pop(0), provider="fake/m", latency_ms=1, attempts=[], usage={})


def setup(router, llm, doc_similarity=0.9, threshold=0.5):
    store = InMemoryStore()
    doc = from_text("The notice period is thirty days. " * 30, doc_id="d")
    chunks = chunk_document(doc, 40, 5)
    # Every chunk has the same cosine similarity to the fake query vector.
    v = np.array([doc_similarity, np.sqrt(1 - doc_similarity ** 2), 0, 0], dtype=np.float32)
    store.add_document(new_meta("d", "d.txt", chunks, doc.text), chunks, np.tile(v, (len(chunks), 1)))
    return Pipeline(Retriever(store, FakeEmbedder()), router, llm, refusal_threshold=threshold)


def answer(text, cite="c1"):
    return json.dumps({"answerable": True, "claims": [{"text": text, "citations": [cite]}]})


def ask(p, q="What is the notice period?"):
    return asyncio.run(p.ask(q, ["d"]))


def test_informational_answer():
    r = ask(setup(FakeRouter("informational"), FakeLLM(answer("Notice is 30 days."))))
    assert r.status == "answered" and r.notice is None and "embed_query" in r.timings_ms


def test_off_topic_skips_retrieval_and_llm():
    llm = FakeLLM()
    r = ask(setup(FakeRouter("off_topic"), llm))
    assert r.status == "off_topic" and llm.calls == 0 and "dense" not in r.timings_ms


def test_low_retrieval_confidence_refuses_without_llm():
    llm = FakeLLM()
    r = ask(setup(FakeRouter("informational"), llm, doc_similarity=0.2, threshold=0.5))
    assert r.status == "refused_low_confidence" and llm.calls == 0


def test_advice_question_gets_facts_plus_notice():
    r = ask(setup(FakeRouter("advice"), FakeLLM(answer("Notice is 30 days."))), "Should I give notice now?")
    assert r.status == "answered" and r.notice is not None


def test_output_advice_is_removed():
    reply = json.dumps({"answerable": True, "claims": [
        {"text": "Notice is 30 days.", "citations": ["c1"]},
        {"text": "You should terminate before the renewal date.", "citations": ["c1"]},
    ]})
    r = ask(setup(FakeRouter("informational"), FakeLLM(reply)))
    assert r.status == "answered" and r.advice_claims_removed == 1 and "should" not in r.answer


def test_answer_that_is_only_advice_becomes_refusal():
    r = ask(setup(FakeRouter("advice"), FakeLLM(answer("I recommend you sign it."))), "Should I sign?")
    assert r.status == "refused_advice"


def test_uncertain_off_topic_guess_fails_open_to_retrieval():
    # "Is a trip to France mentioned?" looks off-topic but is a question about the document.
    llm = FakeLLM(answer("Notice is 30 days."))
    r = ask(setup(FakeRouter("off_topic", conf=0.45), llm))
    assert r.intent == "informational" and r.intent_source == "uncertain"
    assert r.status == "answered" and llm.calls == 1


def test_uncertain_advice_guess_keeps_safe_mode():
    r = ask(setup(FakeRouter("advice", conf=0.45), FakeLLM(answer("Notice is 30 days."))), "Is this ok?")
    assert r.intent == "advice" and r.notice is not None


def test_confident_off_topic_is_blocked():
    llm = FakeLLM()
    r = ask(setup(FakeRouter("off_topic", conf=0.9), llm))
    assert r.status == "off_topic" and llm.calls == 0


def test_advice_detector_ignores_document_language():
    assert contains_advice("You should consult a lawyer before signing.")
    assert contains_advice("I would recommend negotiating the cap.")
    assert not contains_advice("The Tenant shall give thirty days' notice.")
    assert not contains_advice("The Licensee must pay royalties quarterly.")
    assert not contains_advice("The agreement recommends mediation before arbitration.")
