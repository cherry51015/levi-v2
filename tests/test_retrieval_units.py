"""Fast unit tests - no model downloads, safe for CI."""
import pytest

from eval.metrics import hit_at_k, ndcg_at_k, overlaps, reciprocal_rank
from levi.ingest import chunk_document, from_text, normalize
from levi.retrieval import BM25Index, rrf_fuse


def make_doc(n_words: int):
    return from_text(" ".join(f"w{i}" for i in range(n_words)), doc_id="d")


def test_chunks_are_exact_slices_and_cover_the_document():
    doc = make_doc(1000)
    chunks = chunk_document(doc, chunk_words=200, overlap=40)
    for c in chunks:
        assert doc.text[c.char_start : c.char_end] == c.text
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(doc.text)


def test_chunk_overlap():
    chunks = chunk_document(make_doc(1000), chunk_words=200, overlap=40)
    assert chunks[0].text.split()[-40:] == chunks[1].text.split()[:40]


def test_short_document_gives_one_chunk():
    assert len(chunk_document(make_doc(10), chunk_words=200, overlap=40)) == 1


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_document(make_doc(10), chunk_words=50, overlap=50)


def test_normalize_folds_ligatures():
    assert normalize("ﬁnal ﬂoor") == "final floor"


def test_rrf_rewards_agreement_between_rankers():
    # "b" is 2nd in both lists; "a" is 1st in one list and absent from the other.
    fused = dict(rrf_fuse([["a", "b", "c"], ["x", "b", "y"]], k=60))
    assert fused["b"] > fused["a"]


def test_rrf_uses_ranks_not_scores():
    assert rrf_fuse([["a", "b"]], k=60)[0] == ("a", 1 / 61)


def test_bm25_finds_exact_term():
    doc = from_text("alpha beta gamma " * 5 + "indemnification clause here " + "delta epsilon " * 50, doc_id="d")
    chunks = chunk_document(doc, chunk_words=10, overlap=0)
    top_chunk, _ = BM25Index(chunks).search("indemnification", k=1)[0]
    assert "indemnification" in top_chunk.text


def test_metrics():
    rels = [False, True, False]
    assert hit_at_k(rels, 1) == 0.0 and hit_at_k(rels, 2) == 1.0
    assert reciprocal_rank(rels) == 0.5
    assert ndcg_at_k([True], 5, n_relevant=1) == 1.0
    assert 0 < ndcg_at_k(rels, 3, n_relevant=1) < 1


def test_overlaps():
    assert overlaps(0, 10, [(5, 15)])
    assert not overlaps(0, 10, [(10, 20)])  # half-open intervals
