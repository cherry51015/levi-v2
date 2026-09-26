"""Both storage backends must behave identically - same tests, parametrized."""
import numpy as np
import pytest

from levi.ingest import chunk_document, from_text
from levi.store import InMemoryStore, QdrantStore, new_meta

DIM = 8


@pytest.fixture(params=["memory", "qdrant"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryStore()
        return
    s = QdrantStore(str(tmp_path / "qdrant"), dim=DIM)
    yield s
    s.client.close()


def add_doc(store, doc_id, n_words, seed):
    doc = from_text(" ".join(f"{doc_id}w{i}" for i in range(n_words)), doc_id=doc_id)
    chunks = chunk_document(doc, chunk_words=20, overlap=5)
    vecs = np.random.default_rng(seed).normal(size=(len(chunks), DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add_document(new_meta(doc_id, f"{doc_id}.txt", chunks, doc.text), chunks, vecs)
    return chunks, vecs


def test_add_list_delete(store):
    add_doc(store, "a", 100, seed=1)
    add_doc(store, "b", 50, seed=2)
    assert {m.doc_id for m in store.list_documents()} == {"a", "b"}
    assert store.has_document("a")
    store.delete_document("a")
    assert not store.has_document("a")
    assert [m.doc_id for m in store.list_documents()] == ["b"]


def test_get_chunks_in_order(store):
    chunks, _ = add_doc(store, "a", 100, seed=1)
    assert [c.id for c in store.get_chunks(["a"])] == [c.id for c in chunks]


def test_dense_search_finds_exact_vector_and_respects_doc_filter(store):
    chunks_a, vecs_a = add_doc(store, "a", 100, seed=1)
    add_doc(store, "b", 100, seed=2)
    top_chunk, score = store.dense_search(vecs_a[3], ["a"], k=3)[0]
    assert top_chunk.id == chunks_a[3].id
    assert score == pytest.approx(1.0, abs=1e-4)
    assert all(c.doc_id == "b" for c, _ in store.dense_search(vecs_a[3], ["b"], k=3))


def test_qdrant_persists_across_restart(tmp_path):
    path = str(tmp_path / "qdrant")
    s1 = QdrantStore(path, dim=DIM)
    add_doc(s1, "a", 60, seed=1)
    s1.client.close()
    s2 = QdrantStore(path, dim=DIM)
    assert [m.doc_id for m in s2.list_documents()] == ["a"]
    s2.client.close()
