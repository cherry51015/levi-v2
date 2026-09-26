"""Chunk storage behind one small interface, with two interchangeable backends.

- InMemoryStore: numpy matrix per document. Zero setup; used by unit tests and
  offline eval. Lost on restart.
- QdrantStore: Qdrant in embedded mode (on-disk, no server process). Survives
  restarts and filters by doc_id, which is what the document library needs.

The retriever only talks to the ChunkStore interface, so swapping backends is
a one-line change - and the retrieval eval can be re-run on both to prove the
swap didn't change results.
"""
import time
import uuid
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from levi.schemas import Chunk


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    n_chunks: int
    n_words: int
    uploaded_at: float


class ChunkStore(Protocol):
    def add_document(self, meta: DocumentMeta, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def has_document(self, doc_id: str) -> bool: ...
    def list_documents(self) -> list[DocumentMeta]: ...
    def delete_document(self, doc_id: str) -> None: ...
    def get_chunks(self, doc_ids: list[str]) -> list[Chunk]: ...
    def dense_search(self, query_vec: np.ndarray, doc_ids: list[str], k: int) -> list[tuple[Chunk, float]]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._docs: dict[str, tuple[DocumentMeta, list[Chunk], np.ndarray]] = {}

    def add_document(self, meta: DocumentMeta, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._docs[meta.doc_id] = (meta, chunks, vectors)

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def list_documents(self) -> list[DocumentMeta]:
        return sorted((m for m, _, _ in self._docs.values()), key=lambda m: m.uploaded_at)

    def delete_document(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)

    def get_chunks(self, doc_ids: list[str]) -> list[Chunk]:
        return [c for d in doc_ids if d in self._docs for c in self._docs[d][1]]

    def dense_search(self, query_vec: np.ndarray, doc_ids: list[str], k: int) -> list[tuple[Chunk, float]]:
        present = [d for d in doc_ids if d in self._docs]
        if not present:
            return []
        chunks = [c for d in present for c in self._docs[d][1]]
        matrix = np.vstack([self._docs[d][2] for d in present])
        sims = matrix @ query_vec  # cosine similarity: vectors are L2-normalized
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(chunks[i], float(sims[i])) for i in top]


class QdrantStore:
    COLLECTION = "chunks"

    def __init__(self, path: str, dim: int):
        from qdrant_client import QdrantClient, models

        self._m = models
        # Embedded mode: Qdrant runs inside this process and persists to `path`.
        # It holds a file lock, so it suits one API worker; several workers would
        # need the Qdrant server instead (same client code, different constructor).
        self.client = QdrantClient(path=path)
        if not self.client.collection_exists(self.COLLECTION):
            self.client.create_collection(
                self.COLLECTION,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            # Embedded mode scans points directly (no HNSW graph, no payload indexes) - fine for a
            # few thousand chunks. On a Qdrant server we would add a keyword payload index on doc_id.

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        # Qdrant ids must be ints or UUIDs; uuid5 makes them deterministic, so re-uploads overwrite.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _doc_filter(self, doc_ids: list[str]):
        m = self._m
        return m.Filter(must=[m.FieldCondition(key="doc_id", match=m.MatchAny(any=doc_ids))])

    def add_document(self, meta: DocumentMeta, chunks: list[Chunk], vectors: np.ndarray) -> None:
        points = []
        for chunk, vec in zip(chunks, vectors):
            payload = chunk.model_dump()
            if chunk.index == 0:  # document metadata rides on the first chunk
                payload["doc_meta"] = meta.model_dump()
            points.append(self._m.PointStruct(id=self._point_id(chunk.id), vector=vec.tolist(), payload=payload))
        self.client.upsert(self.COLLECTION, points=points, wait=True)

    def has_document(self, doc_id: str) -> bool:
        return self.client.count(self.COLLECTION, count_filter=self._doc_filter([doc_id]), exact=True).count > 0

    def list_documents(self) -> list[DocumentMeta]:
        m = self._m
        first_chunks = m.Filter(must=[m.FieldCondition(key="index", match=m.MatchValue(value=0))])
        records = self._scroll_all(first_chunks)
        metas = [DocumentMeta(**r.payload["doc_meta"]) for r in records if "doc_meta" in r.payload]
        return sorted(metas, key=lambda m_: m_.uploaded_at)

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(self.COLLECTION, points_selector=self._m.FilterSelector(filter=self._doc_filter([doc_id])))

    def get_chunks(self, doc_ids: list[str]) -> list[Chunk]:
        chunks = [Chunk(**{k: v for k, v in r.payload.items() if k != "doc_meta"})
                  for r in self._scroll_all(self._doc_filter(doc_ids))]
        return sorted(chunks, key=lambda c: (doc_ids.index(c.doc_id), c.index))

    def dense_search(self, query_vec: np.ndarray, doc_ids: list[str], k: int) -> list[tuple[Chunk, float]]:
        hits = self.client.query_points(
            self.COLLECTION, query=query_vec.tolist(), query_filter=self._doc_filter(doc_ids),
            limit=k, with_payload=True,
        ).points
        return [(Chunk(**{k_: v for k_, v in h.payload.items() if k_ != "doc_meta"}), float(h.score)) for h in hits]

    def _scroll_all(self, flt) -> list:
        records, offset = [], None
        while True:
            batch, offset = self.client.scroll(
                self.COLLECTION, scroll_filter=flt, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            records.extend(batch)
            if offset is None:
                return records


def new_meta(doc_id: str, filename: str, chunks: list[Chunk], text: str) -> DocumentMeta:
    return DocumentMeta(
        doc_id=doc_id, filename=filename, n_chunks=len(chunks), n_words=len(text.split()), uploaded_at=time.time()
    )
