"""Document-level questions: scoping by reference, overview answers, and the advice + not-found wording."""
import asyncio
import json

import numpy as np
import pytest

from levi.documents import is_document_level, resolve_documents
from levi.guardrails import ADVICE_NOTE, NOT_FOUND_AND_ADVICE
from levi.ingest import chunk_document, from_text
from levi.llm import LLMResult
from levi.overview import build_overview, select_passages
from levi.pipeline import Pipeline
from levi.retrieval import Retriever
from levi.store import InMemoryStore, QdrantStore, new_meta


def meta(doc_id, filename, t, title=None):
    m = new_meta(doc_id, filename, [], "x")
    return m.model_copy(update={"uploaded_at": t, "overview": {"title": title, "claims": []} if title else None})


DOCS = [meta("a", "01_web_hosting_agreement.txt", 1), meta("b", "04_residential_lease.docx", 2),
        meta("c", "contract.txt", 3, title="Cooperation agreement")]


@pytest.mark.parametrize("q,expected", [
    ("who are the participants in the second document?", ["b"]),
    ("summarise the first file", ["a"]),
    ("what is the last contract about?", ["c"]),
    ("what does the lease say about pets?", ["b"]),
    ("termination in the web hosting agreement", ["a"]),
    ("what does contract.txt say about notices?", ["c"]),
    ("what is the notice period?", None),  # names no document
    ("what does the agreement say about fees?", None),  # 'agreement' is too generic to pick one
    ("summarise the fifth document", None),  # out of range
])
def test_resolve_documents(q, expected):
    assert resolve_documents(q, DOCS) == expected


def test_resolve_needs_more_than_one_document():
    assert resolve_documents("the second document", DOCS[:1]) is None


@pytest.mark.parametrize("q", ["give me a summary of the second document", "what is the second document about?",
                               "what is this about", "who are the parties?", "Summarize it", "what kind of agreement is this",
                               "what are the key terms"])
def test_document_level_questions(q):
    assert is_document_level(q)


@pytest.mark.parametrize("q", ["what is the notice period for termination?", "what is the renewal term?",
                               "does the lease allow pets?", "is there a cap on liability?"])
def test_clause_questions_are_not_document_level(q):
    assert not is_document_level(q)


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        return LLMResult(text=self.replies.pop(0), provider="fake/m", latency_ms=1, attempts=[], usage={})


def lease_chunks():
    text = ("This lease is between Meera (Landlord) and Arjun (Tenant). " * 15 +
            "The monthly rent is INR 32,000 payable by the 5th. " * 15 +
            "This lease is governed by the laws of India. " * 15)
    return chunk_document(from_text(text, doc_id="d"), 40, 5)


def test_select_passages_keeps_document_order_and_cap():
    picked = select_passages(lease_chunks())
    assert [c.index for c in picked] == sorted(c.index for c in picked) and len(picked) <= 7


def test_overview_drops_uncited_claims():
    reply = json.dumps({"document_type": "Residential lease", "claims": [
        {"text": "A lease between Meera and Arjun.", "citations": ["c1"]},
        {"text": "Invented fact.", "citations": ["c99"]}]})
    ov = asyncio.run(build_overview(FakeLLM(reply), lease_chunks()))
    assert ov["title"] == "Residential lease"
    assert [c["text"] for c in ov["claims"]] == ["A lease between Meera and Arjun."]
    assert ov["claims"][0]["chunk_ids"] == ["d:0"]


def test_overview_with_nothing_grounded_is_none():
    reply = json.dumps({"document_type": "x", "claims": [{"text": "made up", "citations": []}]})
    assert asyncio.run(build_overview(FakeLLM(reply), lease_chunks())) is None


# --- pipeline integration (fakes, no models)
class FakeEmbedder:
    dim = 4

    def embed_query(self, q):
        return np.array([1, 0, 0, 0], dtype=np.float32)


class FakeRouter:
    def __init__(self, label="informational"):
        self.label = label

    def predict(self, qvec):
        return self.label, 0.9

    def is_confident(self, c):
        return True


def pipeline(llm, similarity=0.9, router=None):
    store = InMemoryStore()
    for doc_id, name, t in (("lease", "lease.docx", 1), ("nda", "nda.pdf", 2)):
        chunks = chunk_document(from_text(f"{name} clause text about rent and parties. " * 40, doc_id=doc_id), 40, 5)
        v = np.array([similarity, np.sqrt(1 - similarity ** 2), 0, 0], dtype=np.float32)
        m = new_meta(doc_id, name, chunks, "x").model_copy(update={"uploaded_at": t})
        store.add_document(m, chunks, np.tile(v, (len(chunks), 1)))
    return Pipeline(Retriever(store, FakeEmbedder()), router or FakeRouter(), llm, refusal_threshold=0.5)


def overview_reply(name):
    return json.dumps({"document_type": name, "claims": [{"text": f"This is the {name}.", "citations": ["c1"]}]})


def test_document_level_question_is_answered_from_a_cached_overview():
    llm = FakeLLM(overview_reply("NDA"))
    p = pipeline(llm)
    r1 = asyncio.run(p.ask("what is the second document about?", ["lease", "nda"]))
    assert r1.answer_mode == "overview" and r1.scoped_to == ["nda"] and r1.status == "answered"
    assert r1.claims[0].text == "This is the NDA." and r1.citations[0].chunk_id.startswith("nda:")
    r2 = asyncio.run(p.ask("summarise the second document", ["lease", "nda"]))
    assert r2.answer_mode == "overview" and llm.calls == 1  # second question reuses the stored overview


def test_overview_failure_falls_back_to_passage_search():
    llm = FakeLLM("not json", json.dumps({"answerable": True, "claims": [{"text": "Rent is due.", "citations": ["c1"]}]}))
    r = asyncio.run(pipeline(llm).ask("what is this about?", ["lease"]))
    assert r.answer_mode == "passages" and r.status == "answered"


def test_advice_question_with_nothing_found_says_both():
    llm = FakeLLM()
    r = asyncio.run(pipeline(llm, similarity=0.1, router=FakeRouter("advice")).ask("should I sign the lease?", ["lease"]))
    assert r.status == "refused_low_confidence" and r.answer == NOT_FOUND_AND_ADVICE and llm.calls == 0


def test_advice_question_model_finds_nothing_says_both():
    llm = FakeLLM(json.dumps({"answerable": False, "claims": []}))
    r = asyncio.run(pipeline(llm, router=FakeRouter("advice")).ask("is the liability cap fair to me?", ["lease"]))
    assert r.answer == NOT_FOUND_AND_ADVICE and r.answer != ADVICE_NOTE


def test_qdrant_update_meta_persists_overview(tmp_path):
    s = QdrantStore(str(tmp_path / "q"), dim=4)
    chunks = chunk_document(from_text("word " * 100, doc_id="d"), 40, 5)
    s.add_document(new_meta("d", "d.txt", chunks, "x"), chunks, np.ones((len(chunks), 4), dtype=np.float32))
    m = s.list_documents()[0]
    s.update_meta(m.model_copy(update={"overview": {"title": "T", "claims": []}}))
    assert s.list_documents()[0].overview == {"title": "T", "claims": []}
    s.client.close()


@pytest.mark.parametrize("q,expected", [
    ("Is there a non-compete clause? Should I worry about it?", True),
    ("Is there a cap on liability? should i sign on it?", True),
    ("Is it worth renewing?", True),
    ("What would you do here?", True),
    ("What should the tenant do if rent is late?", False),  # asks what the document requires, not advice
    ("Does the agreement recommend mediation?", False),
    ("What is the notice period?", False),
])
def test_asks_for_advice(q, expected):
    from levi.guardrails import asks_for_advice
    assert asks_for_advice(q) is expected


def test_mixed_question_router_missed_still_gets_both_messages():
    llm = FakeLLM()
    r = asyncio.run(pipeline(llm, similarity=0.1).ask("Is there a non-compete? Should I worry?", ["lease"]))
    assert r.answer == NOT_FOUND_AND_ADVICE and llm.calls == 0
