"""HTTP API. Thin layer: validation, error mapping, request logging. Logic lives in levi/."""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.ratelimit import TokenBucket
from levi.config import settings
from levi.guardrails import IntentRouter
from levi.ingest import EmptyDocument, UnsupportedFileType, load_bytes
from levi.llm import LLMClient
from levi.pipeline import AskResponse, Pipeline
from levi.retrieval import Embedder, Reranker, Retriever, index_document
from levi.store import DocumentMeta, QdrantStore

log = logging.getLogger("levi")
LOG_PATH = Path("logs/requests.jsonl")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    doc_ids: list[str] | None = None  # None = search every document in the library


class UploadResponse(BaseModel):
    document: DocumentMeta
    cached: bool  # True when this exact file was already indexed (no re-embedding)
    ingest_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup, not per request, then warm them up so the
    # first real user doesn't pay one-time initialisation costs.
    t0 = time.perf_counter()
    embedder = Embedder()
    reranker = Reranker() if settings.retrieval_mode == "hybrid_rerank" else None
    Path(settings.qdrant_path).parent.mkdir(parents=True, exist_ok=True)
    store = QdrantStore(settings.qdrant_path, dim=embedder.dim)
    retriever = Retriever(store, embedder, reranker)
    router = IntentRouter.train_default(embedder)
    embedder.embed_query("warm-up")
    llm = LLMClient()
    app.state.store, app.state.retriever, app.state.llm = store, retriever, llm
    app.state.pipeline = Pipeline(retriever, router, llm, settings.refusal_threshold,
                                  settings.retrieval_mode, settings.offload_cpu)
    app.state.startup_s = round(time.perf_counter() - t0, 1)
    log.info("startup complete in %ss; providers=%s", app.state.startup_s, [p.label for p in llm.providers])
    yield
    await llm.aclose()
    store.client.close()


app = FastAPI(title="Levi v2", version="0.1.0", lifespan=lifespan)
ask_bucket = TokenBucket(settings.ask_burst, settings.ask_per_minute / 60)


@app.middleware("http")
async def request_log(request: Request, call_next):
    """One structured log line per request, with an id the client can quote in a bug report."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    response = await call_next(request)
    record = {"ts": time.time(), "request_id": request_id, "method": request.method, "path": request.url.path,
              "status": response.status_code, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    response.headers["x-request-id"] = request_id
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return response


@app.post("/documents", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
    t0 = time.perf_counter()
    try:
        # Parsing and embedding are CPU-bound: keep them off the event loop.
        doc = await asyncio.to_thread(load_bytes, data, file.filename or "upload.txt")
    except (UnsupportedFileType, EmptyDocument) as e:
        raise HTTPException(400, str(e))
    retriever = request.app.state.retriever
    meta, cached = await asyncio.to_thread(index_document, request.app.state.store, retriever.embedder,
                                           doc, file.filename or "upload.txt")
    retriever.invalidate(meta.doc_id)
    return UploadResponse(document=meta, cached=cached, ingest_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.get("/documents", response_model=list[DocumentMeta])
async def list_documents(request: Request):
    return request.app.state.store.list_documents()


@app.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, request: Request):
    store = request.app.state.store
    if not store.has_document(doc_id):
        raise HTTPException(404, "Unknown document")
    store.delete_document(doc_id)
    request.app.state.retriever.invalidate(doc_id)


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request):
    wait = ask_bucket.try_acquire()
    if wait:
        return JSONResponse(status_code=429, headers={"Retry-After": str(int(wait) + 1)},
                            content={"detail": f"Demo is busy - please retry in {int(wait) + 1}s."})
    store = request.app.state.store
    doc_ids = body.doc_ids or [m.doc_id for m in store.list_documents()]
    if not doc_ids:
        raise HTTPException(400, "Upload a document first")
    unknown = [d for d in doc_ids if not store.has_document(d)]
    if unknown:
        raise HTTPException(404, f"Unknown document(s): {unknown}")
    result = await request.app.state.pipeline.ask(body.question, doc_ids)
    if result.status == "error":
        # Upstream (LLM providers) failed: 503 tells clients to retry later, unlike a 500 bug.
        return JSONResponse(status_code=503, content=result.model_dump())
    return result


@app.get("/health")
async def health(request: Request):
    llm = request.app.state.llm
    return {
        "status": "ok",
        "startup_s": request.app.state.startup_s,
        "documents": len(request.app.state.store.list_documents()),
        "retrieval_mode": settings.retrieval_mode,
        "llm_providers": [{"provider": p.label, "breaker": p.breaker.state.value} for p in llm.providers],
    }
