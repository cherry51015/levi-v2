---
title: Levi
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Levi: legal document Q&A with cited evidence

Upload a contract, lease or agreement and ask questions about it. Every claim in an answer cites the passage it
came from; when the document doesn't answer the question, Levi says so; and it explains what a document says
without giving legal advice. Everything runs on open-weight models on free tiers.

This is a rebuild of an earlier prototype ([v1](https://github.com/cherry51015/Levi-legal_AI_assistant)). The goal of
v2 was to replace "it seems to work" with numbers: every result below comes from a script in `eval/` and a JSON file
in `eval/results/`, and the limitations section lists what failed or is unfinished.

**Status:** runs locally and in Docker (verified). Not yet deployed publicly.

## How it works

```
Upload (PDF / DOCX / TXT)
  → SHA-256 document id (re-uploading the same file skips re-embedding)
  → chunks of 200 words, 40 overlap, keeping page numbers and character offsets
  → bge-base-en-v1.5 embeddings → Qdrant (embedded, on disk)

Question
  → embed once → intent router (logistic regression on that embedding; trusted only when confident)
  → BM25 + vector search → reciprocal rank fusion → MiniLM cross-encoder rerank (top 10 → 5)
  → refusal gate: best rerank score below a tuned threshold → refuse without calling the LLM
  → LLM returns JSON claims, each citing chunk ids (gpt-oss-120b → gpt-oss-20b → nemotron fallback)
  → claims citing anything not retrieved are dropped; advice-style sentences are removed
  → answer + evidence + per-stage timings + Langfuse trace
```

| Component | Choice |
|---|---|
| Answering LLM | `openai/gpt-oss-120b` via Groq (open-weight), fallback `openai/gpt-oss-20b`, then `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter |
| Judge LLM (evaluation only) | `qwen/qwen3.8-27b` via Groq: a different model family from the answering model |
| Embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers, CPU) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU) |
| Storage | Qdrant in embedded mode (tests and offline evals use an in-memory store behind the same interface) |
| API / UI | FastAPI, Streamlit |
| Tracing | Langfuse (optional; off without keys) |

The LLM client handles the free tier's limits (8k tokens/minute and 200k tokens/day per model): per-call timeouts,
exponential backoff with jitter, `Retry-After`, a circuit breaker per model, proactive throttling from the provider's
rate-limit headers, and an ordered fallback chain. These paths are unit-tested with injected failures; the fallback
chain has not yet been exercised under real load.

## Results

Evaluation data: [CUAD](https://www.atticusprojectai.org/cuad) (real commercial contracts with lawyer-marked answer
spans). 20 contracts, 240 clause questions (107 answerable, 133 whose clause the contract doesn't contain).

### Retrieval (107 answerable questions)

| Configuration | hit@1 | hit@5 (95% CI) | MRR | p50 latency |
|---|---|---|---|---|
| BM25 only | 0.430 | 0.813 (0.74–0.89) | 0.579 | 1 ms |
| Dense only | 0.495 | 0.720 (0.64–0.81) | 0.578 | 105 ms |
| Hybrid (RRF) | 0.505 | 0.850 (0.79–0.92) | 0.636 | 107 ms |
| Hybrid + bge-reranker-base (top 20) | 0.523 | 0.804 | 0.635 | 11,118 ms |
| **Hybrid + MiniLM rerank (top 10), shipped** | **0.626** | 0.860 (0.79–0.93) | **0.710** | 948 ms |

The hit@5 intervals overlap, so the defensible claim is better *top-ranking* (hit@1, MRR), not better recall. Latency
is retrieval only, on a laptop CPU. Source: `eval/results/retrieval_*.json`.

### Refusal gate (held-out split)

The threshold was tuned on half the contracts, with false refusals capped at 10%, and tested on the other half:
**85% precision, 32% recall** on unanswerable questions, 8.5% false refusals, with no LLM call. The cheap gate catches
about a third of unanswerable questions; the LLM's own "not answerable" check handles the rest.

### Answer quality (LLM-as-judge, 50 sampled questions)

| Metric | Result | n |
|---|---|---|
| Claims fully supported by their cited chunk | 91% (20) | 22 claims |
| Partially supported | 9% (2) | |
| Unsupported | 0 | |
| Completeness vs the lawyer-marked answer | 3.95 / 5 (76% scored ≥ 4) | 21 answers |
| Answerable questions answered | 70% (9 false refusals: 3 by the gate, 6 by the LLM) | 30 |
| Unanswerable questions refused | 12 of 12 scored | 12 |
| Excluded as infrastructure errors (daily quota, HTTP 429) | 8 of 50 | |

The judge reasons before giving a verdict, checks each claim only against the chunk it cites, and shows no length bias
(Spearman ρ = −0.42 between answer length and score). **The judge has not yet been validated against human labels.**

### Guardrails (58 hand-written red-team queries)

| | Result |
|---|---|
| Advice leaked (34 advice-seeking queries, including role-play and hypotheticals) | 0% |
| Advice leaked with only the system prompt (no router, no output check) | 0% |
| Benign questions wrongly refused | 0 of 16 |
| Off-topic questions stopped before the LLM | 7 of 8 |
| Router accuracy | 86% |

Honest reading: gpt-oss-120b already declined to give advice with the system prompt alone, so the extra layers add
margin rather than fixing a measured leak. A prompt-injection test (0/6 successes) is **inconclusive**, because the
injected passage was never among the retrieved chunks. These results predate a router simplification; a re-run hit
the daily token quota and is excluded (`eval/results/invalid/`).

### Latency

End-to-end for answered questions, excluding rate-limit stalls: **p50 1.57 s, p95 2.54 s** (n=16). LLM call
p50 0.96 s; reranker p50 446 ms. Retrieval itself (BM25 + vector search + fusion) is under 20 ms: the LLM and the
reranker are 90%+ of the time. Five calls that stalled 40 s+ on the per-minute token limit were excluded; that run
predates separate wait-time logging, so the split is a heuristic. Newer requests report `llm_wait` as its own stage.

## Limitations and open items

- Judge not yet validated against human labels (`eval/label_app.py` is ready; 0 of 22 claims labelled).
- The prompt fix for indirect answers ("five years from the effective date" when asked for a date) is unmeasured; a
  same-seed re-run was blocked by the provider's daily quota.
- Small samples: 22 judged claims and 21 answers; treat the generation numbers as indicative.
- The router training data and the red-team set were written by the same author.
- Evals run on the in-memory store; the Qdrant backend passes the same unit tests but the full eval hasn't been re-run on it.
- A citation's page is the page its chunk starts on; a clause at the top of a page can be cited as the previous page.
- Not load-tested; the CPU-offload switch and `/ask` rate limiter are unit-tested only.
- Free Hugging Face Spaces have no persistent disk: uploaded documents are lost when the Space restarts.
- No OCR: scanned PDFs and images are rejected with a clear error.
- English only.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env                                    # add GROQ_API_KEY (+ optional OPENROUTER / LANGFUSE keys)
uvicorn app.main:app --port 8000                        # API docs: http://127.0.0.1:8000/docs
streamlit run ui/streamlit_app.py --server.port 8501    # UI: http://localhost:8501
```

Only one API process can open the embedded Qdrant folder at a time.

**Docker** (API + UI in one container):
```bash
docker build -t levi-v2 .
docker run --env-file .env -p 7860:7860 levi-v2          # http://localhost:7860
```

**Tests and evals:**
```bash
pytest -q                                   # 54 unit tests; no model downloads or API keys needed
python -m eval.cuad --contracts 20          # build the CUAD eval set
python -m eval.retrieval_eval               # retrieval ablation
python -m eval.refusal_eval --mode hybrid_rerank
python -m eval.generation_eval              # uses the LLM + judge (~100k tokens)
python -m eval.redteam_eval
streamlit run eval/label_app.py             # label claims to validate the judge
python -m eval.judge_agreement
```

Sample documents for manual testing are in `samples/`.

## Repository layout

```
levi/        core: ingest, retrieval, store (Qdrant / in-memory), llm client, answer, guardrails, pipeline, tracing
app/         FastAPI app and rate limiter
ui/          Streamlit UI: Ask, Evaluation dashboard, How it works
eval/        CUAD loader, eval scripts, judge, labelling app, results/
tests/       unit tests (fakes for models and HTTP; no network)
samples/     test documents
```

## Data and credits

CUAD: Contract Understanding Atticus Dataset, CC BY 4.0, The Atticus Project. Models are used under their respective
open-weight licences; check each model card.
