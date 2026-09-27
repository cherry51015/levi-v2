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
  → a short overview of the whole document (one LLM call; every claim cites a passage)

Question
  → embed once → intent router (logistic regression on that embedding; trusted only when confident)
  → "the lease" / "the second document" → search only that document
  → "what is it about?" / "summarise" / "who are the parties?" → answer from the stored overview
  → BM25 + vector search → reciprocal rank fusion → MiniLM cross-encoder rerank (top 10 → 5)
  → refusal gate: best rerank score below a tuned threshold → refuse without calling the LLM
  → small-to-big context: the top 3 passages are widened with their neighbouring chunks
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

**Whole-document questions.** Passage search can't answer "what is this document about?": no single passage is
about the whole document. So each upload gets a short overview built from representative passages (the opening, plus
the best matches for parties, term, payment and governing law), with the same citation check as normal answers.
Questions that name a document ("the lease", "the second document", a filename) search only that document, and
questions that mix a factual ask with "should I…?" get the facts plus a note that the decision is a lawyer's.
The overview costs one LLM call per upload; overview answers are served from storage in under 200 ms.

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

The gate only filters clearly irrelevant questions; close calls go to the LLM. Tuned on half the contracts with false
refusals capped at 2%, tested on the other half: **90% precision, 4.3% false refusals**, catching 23% of unanswerable
questions with no LLM call. (The first version, capped at 10%, had 85% precision but 8.5% false refusals.)

### Answer quality (LLM-as-judge, same 50 questions, before and after)

| Metric | Run 1 | Run 2 (current) |
|---|---|---|
| Claims fully supported by their cited passage | 91% (20 / 22) | **96% (23 / 24)** |
| Unsupported (made-up) claims | 0 | **0** |
| Completeness vs the lawyer-marked answer | 3.95 / 5 | **4.10 / 5** |
| Unanswerable questions refused | 12 / 12 | **18 / 19** |
| Answerable questions answered | 70% (21 / 30) | 70% (21 / 30) |
| Excluded as infrastructure errors (daily quota, HTTP 429) | 8 of 50 | 1 of 50 |

Run 2 added a recalibrated refusal gate, small-to-big context, and an instruction to report indirect answers. What
changed, question by question: a licence-grant question the old gate wrongly refused is now answered (4 / 5); an
exclusivity question previously answered from the wrong clause (1 / 5) is now an honest refusal; the one
"unanswerable" question answered in run 2 got a correct, cited answer ("renewal only by mutual written agreement")
that the dataset labels as out of scope. **The indirect-answer instruction did not work:** 5 of 7 expiry-date questions
are still refused, and they remain the main source of wrongly refused questions. An error analysis of run 1 showed that in
every wrongly refused case the evidence *had* been retrieved, so the bottleneck is the answer step, not search.

The judge reasons before giving a verdict, checks each claim against the exact text the model was shown, and shows no
length bias (Spearman ρ = −0.16). **The judge has not yet been validated against human labels.** Small samples: treat
these as indicative.

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

End-to-end for answered questions, with rate-limit waiting logged separately and excluded: **p50 2.31 s, p95 3.11 s**
(n=22, run 2). LLM generation p50 1.62 s; reranker p50 561 ms. Search itself (BM25 + vector search + fusion) takes
under 20 ms in the live app, so the LLM and the reranker are 90%+ of the time. Small-to-big context raised prompts from
about 1.9k to 3.4k tokens: answers got more complete, but generation time rose from ~1.0 s (run 1) and the free tier's
8k tokens/minute now fits about 2 answers per minute instead of 4.

## Limitations and open items

- Judge not yet validated against human labels (`eval/label_app.py` is ready; 0 of 24 claims labelled).
- Expiry-date questions whose answer is a duration ("five years from the effective date") are still usually refused.
- Small samples: 24 judged claims and 22 answers; treat the generation numbers as indicative.
- The safety results predate the router simplification; the re-run hit the daily quota (`eval/results/invalid/`).
- The router training data and the red-team set were written by the same author.
- Evals run on the in-memory store; the Qdrant backend passes the same unit tests but the full eval hasn't been re-run on it.
- A citation's page is the page its chunk starts on; a clause at the top of a page can be cited as the previous page.
- Not load-tested; the CPU-offload switch and `/ask` rate limiter are unit-tested only.
- Free Hugging Face Spaces have no persistent disk: uploaded documents are lost when the Space restarts.
- No OCR: scanned PDFs and images are rejected with a clear error.
- English only.
- Whole-document handling is rule-based (question patterns and filename matching) and has unit tests but no
  labelled evaluation set yet.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env                                    # add GROQ_API_KEY (+ optional OPENROUTER / LANGFUSE keys)
uvicorn app.main:app --port 8000                        # API docs: http://127.0.0.1:8000/docs
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 127.0.0.1   # UI
```

Only one API process can open the embedded Qdrant folder at a time.

**Docker** (API + UI in one container):
```bash
docker build -t levi-v2 .
docker run --env-file .env -p 7860:7860 levi-v2          # http://localhost:7860
```

**Streamlit Community Cloud** (free backup link; small CPU, so answers are slower than locally):
the UI hosts the same FastAPI app in-process (`ui/embedded.py`), so no second server is needed. Create an app from
this repo with main file `ui/streamlit_app.py` (Python 3.12), and in its Secrets set:
```toml
LEVI_UI_MODE = "local"
GROQ_API_KEY = "..."
# optional: OPENROUTER_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
```
Dependencies come from `ui/requirements.txt` (CPU-only PyTorch). Uploaded documents reset when the app restarts.

**Tests and evals:**
```bash
pytest -q                                   # 93 unit tests; no model downloads or API keys needed
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
