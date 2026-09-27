from pydantic import BaseModel


class Chunk(BaseModel):
    id: str  # "{doc_id}:{index}" - the id the LLM must cite
    doc_id: str
    index: int
    text: str
    char_start: int  # offsets into the normalized document text
    char_end: int
    page: int | None = None  # 1-based; None for formats without pages


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    stage: str  # which stage produced the score: bm25 | dense | rrf | rerank
    # Cosine similarity to the query, kept even after fusion: RRF scores only encode
    # rank, so they can't tell "best of a bad lot" from a genuine match.
    dense_score: float | None = None
    # Wider text around the chunk (itself plus neighbours) shown to the LLM; None = the chunk alone.
    context: str | None = None

    @property
    def text_for_llm(self) -> str:
        return self.context or self.chunk.text
