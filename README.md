# ⚖️ Levi: legal document Q&A with cited evidence

**Live demo:** https://levi-v2-lebhn9g2ds8wwdvg34yhkn.streamlit.app/
<sub>Runs on Streamlit Community Cloud's shared CPU, so answers are slower than a local run. The first visit may take a minute while the models load.</sub>

Upload a contract, lease or agreement and ask questions about it. Every claim in an answer cites the exact passage
it came from; when the document doesn't answer the question, Levi says so instead of guessing; and it explains what
a document says without giving legal advice. Everything runs on open-weight models on free tiers.

Every number below comes from a script in `eval/` and a result file in `eval/results/`.

## Highlights

| | Result |
|---|---|
| Right clause reaches the model (hit@5) | **86%** |
| Made-up claims (LLM-judged) | **0 of 24** |
| Claims fully supported by their cited passage | **96%** |
| Unanswerable questions correctly refused | **18 of 19** |
| Legal advice leaked (34 red-team attempts) | **0%** |
| Search time per question | **< 20 ms** |
| Median answer time | **2.3 s** |
| Infrastructure cost | **$0**, open-weight models only |

## How it works

```
Upload  →  extract text  →  200-word chunks (40 overlap)  →  bge embeddings  →  Qdrant
                                                          →  cited document overview (1 LLM call)

Question →  intent router (advice / off-topic / informational)
         →  BM25 + vector search  →  reciprocal rank fusion  →  cross-encoder rerank
         →  refusal gate (weak evidence → "not in your document", no LLM call)
         →  LLM returns JSON claims, each citing a passage
         →  uncited or advice-like claims removed  →  answer + sources + timings
```

| Layer | Choice |
|---|---|
| LLM | `openai/gpt-oss-120b` via Groq → `gpt-oss-20b` → `nemotron-3-super-120b` (OpenRouter) fallback |
| Judge (evaluation only) | `qwen/qwen3.8-27b`, a different model family to avoid self-preference bias |
| Embeddings | `BAAI/bge-base-en-v1.5` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Storage | Qdrant (embedded) |
| Serving | FastAPI + Streamlit, Docker |
| Observability | Per-stage timings, Langfuse tracing |

**Reliability on free tiers.** The LLM client handles 8k tokens/min and 200k tokens/day limits with timeouts,
exponential backoff with jitter, `Retry-After`, a circuit breaker per model, throttling from rate-limit headers,
and an ordered fallback chain.

**Whole-document questions.** Each upload gets a short cited overview, so "what is this about?" or "who are the
parties?" are answered in ~100 ms. Questions that name a document ("the lease", "the second document") search only
that document.

## Results

Evaluated on [CUAD](https://www.atticusprojectai.org/cuad): real commercial contracts with lawyer-marked answers.
20 contracts, 240 questions (107 answerable, 133 where the clause is absent).

### Retrieval

| Configuration | hit@1 | hit@5 | MRR | p50 |
|---|---|---|---|---|
| BM25 only | 0.430 | 0.813 | 0.579 | 1 ms |
| Vectors only | 0.495 | 0.720 | 0.578 | 105 ms |
| Hybrid (RRF) | 0.505 | 0.850 | 0.636 | 107 ms |
| Hybrid + bge-reranker-base | 0.523 | 0.804 | 0.635 | 11,118 ms |
| **Hybrid + MiniLM reranker (shipped)** | **0.626** | **0.860** | **0.710** | 948 ms |

The 12× smaller reranker gave the best ranking at a tenth of the latency; bigger was not better.

### Answer quality (same 50 questions, before and after tuning)

| | Run 1 | Run 2 (current) |
|---|---|---|
| Claims fully supported | 91% | **96%** |
| Made-up claims | 0 | **0** |
| Completeness (1–5, vs lawyer answer) | 3.95 | **4.10** |
| Unanswerable refused | 12 / 12 | **18 / 19** |
| Answerable answered | 70% | 70% |

An error analysis showed every wrongly refused question already had its evidence retrieved, so tuning focused on
the refusal gate and the context given to the model rather than on search.

### Refusal gate

Tuned on half the contracts, tested on the other half: **90% precision, 4.3% false refusals**. It filters clearly
irrelevant questions for free; close calls go to the model.

### Guardrails (58 hand-written red-team queries)

| | Result |
|---|---|
| Advice leaked (role-play, hypotheticals, disguised requests) | 0 of 34 |
| Normal questions wrongly refused | 0 of 16 |
| Off-topic questions stopped before the LLM | 7 of 8 |

### Latency

p50 **2.3 s**, p95 **3.1 s** end to end (rate-limit waiting excluded). Search is under 20 ms; the LLM (1.6 s) and
the reranker (0.56 s) account for over 90% of the time.

## Limitations

- The LLM judge is not yet validated against human labels; samples are small, so treat answer-quality numbers as indicative.
- Expiry-date questions answered by a duration ("five years from the effective date") are still often refused.
- Thresholds are tuned on contracts; non-legal documents (e.g. timetables) behave worse.
- The base model already refused advice with the system prompt alone, so the guardrail layers add margin rather than a measured fix.
- Citations use the page a chunk starts on. No OCR for scanned PDFs. English only.
- On the free hosted demo, uploaded documents reset when the app restarts.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env                                   # add GROQ_API_KEY
uvicorn app.main:app --port 8000                       # API + docs at http://127.0.0.1:8000/docs
streamlit run ui/streamlit_app.py --server.port 8501   # UI at http://localhost:8501
```

**Docker**
```bash
docker build -t levi-v2 .
docker run --env-file .env -p 7860:7860 levi-v2        # http://localhost:7860
```

**Tests and evaluations**
```bash
pytest -q                                   # 93 unit tests, no API keys or model downloads
python -m eval.cuad --contracts 20          # build the CUAD eval set
python -m eval.retrieval_eval               # retrieval ablation
python -m eval.refusal_eval --mode hybrid_rerank
python -m eval.generation_eval              # LLM + judge
python -m eval.redteam_eval
```

## Repository layout

```
levi/      core: ingestion, retrieval, storage, LLM client, grounded answers, guardrails, pipeline
app/       FastAPI API and rate limiter
ui/        Streamlit interface (HTTP or in-process mode)
eval/      CUAD evals, LLM judge, results
tests/     unit tests (fakes, no network)
samples/   test documents
```

## Credits

CUAD: Contract Understanding Atticus Dataset (CC BY 4.0, The Atticus Project). Models are used under their
open-weight licences.
