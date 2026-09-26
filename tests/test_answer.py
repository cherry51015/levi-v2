import asyncio
import json

from levi.answer import build_messages, enforce_citations, generate_answer, ModelAnswer
from levi.llm import LLMResult
from levi.schemas import Chunk, ScoredChunk


def chunks(n=3):
    return [
        ScoredChunk(chunk=Chunk(id=f"d:{i}", doc_id="d", index=i, text=f"clause {i} text", char_start=0,
                                char_end=10, page=i + 1), score=0.9, stage="rerank")
        for i in range(n)
    ]


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        return LLMResult(text=self.replies.pop(0), provider="fake/m", latency_ms=1.0, attempts=[], usage={})


def run(llm, question="q?"):
    return asyncio.run(generate_answer(llm, question, chunks()))


def test_document_text_is_delimited_not_in_system_prompt():
    msgs, refs = build_messages("what?", chunks())
    assert set(refs) == {"c1", "c2", "c3"}
    assert "clause 0 text" not in msgs[0]["content"]
    assert '<chunk id="c1" page="1">' in msgs[1]["content"]


def test_answer_with_valid_citations():
    reply = json.dumps({"answerable": True, "claims": [{"text": "Notice is 30 days.", "citations": ["c2"]}]})
    ans = run(FakeLLM(reply))
    assert ans.status == "answered"
    assert ans.answer == "Notice is 30 days. [c2]"
    assert ans.citations[0].chunk_id == "d:1" and ans.citations[0].page == 2


def test_claims_citing_unknown_chunks_are_dropped():
    parsed = ModelAnswer(answerable=True, claims=[
        {"text": "grounded", "citations": ["c1", "c9"]},
        {"text": "invented", "citations": ["c9"]},
        {"text": "uncited", "citations": []},
    ])
    kept, dropped = enforce_citations(parsed, {"c1", "c2"})
    assert [c.text for c in kept] == ["grounded"] and kept[0].citations == ["c1"] and dropped == 2


def test_all_claims_ungrounded_becomes_refusal():
    reply = json.dumps({"answerable": True, "claims": [{"text": "made up", "citations": ["c7"]}]})
    ans = run(FakeLLM(reply))
    assert ans.status == "refused_not_in_document" and ans.dropped_claims == 1


def test_model_says_unanswerable():
    ans = run(FakeLLM(json.dumps({"answerable": False, "claims": []})))
    assert ans.status == "refused_not_in_document"


def test_invalid_json_gets_one_repair_attempt():
    good = json.dumps({"answerable": True, "claims": [{"text": "ok", "citations": ["c1"]}]})
    llm = FakeLLM("not json at all", good)
    assert run(llm).status == "answered" and llm.calls == 2


def test_invalid_twice_is_an_error_not_a_crash():
    llm = FakeLLM("nope", "still nope")
    assert run(llm).status == "error"
