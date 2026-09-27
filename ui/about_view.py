"""How it works: a plain five-step story first, the technical architecture on demand."""
import streamlit as st

from theme import card, esc

STEPS = [
    ("Understand the question", "Spots requests for legal advice or off-topic questions.", "intent classifier · <1 ms"),
    ("Find relevant passages", "Searches by exact wording and by meaning, then reranks the best matches.",
     "BM25 + vectors · RRF · cross-encoder"),
    ("Check the evidence", "If nothing relevant enough is found, Levi says so instead of guessing.",
     "tuned score threshold"),
    ("Answer from the passages", "An open-weight LLM writes the answer using only those passages.",
     "gpt-oss-120b · automatic fallback"),
    ("Verify every claim", "Each sentence must cite its source; unsupported or advice-like sentences are removed.",
     "citation + advice checks"),
]

PIPELINE = """
digraph G {
  rankdir=TB; bgcolor="transparent"; pad=0.2; nodesep=0.3; ranksep=0.28;
  node [shape=box, style="rounded,filled", fillcolor="#fcfcfb", color="#c3c2b7", fontname="Segoe UI", fontsize=11,
        fontcolor="#0b0b0b", margin="0.15,0.07"];
  edge [color="#898781", arrowsize=0.6];

  subgraph cluster_ingest { label="Ingest (once per file)"; fontname="Segoe UI"; fontsize=10; fontcolor="#52514e";
    color="#e1e0d9"; style=rounded;
    up [label="Upload\\nPDF · DOCX · TXT"]; hash [label="SHA-256 doc id\\n(repeat upload = cache hit)"];
    chunk [label="Chunk\\n200 words, 40 overlap\\nkeeps page + offsets"]; embed [label="Embed\\nbge-base-en-v1.5"];
    qd [label="Qdrant\\n(embedded, on disk)", fillcolor="#eef4fc", color="#2a78d6"];
    up -> hash -> chunk -> embed -> qd; }

  subgraph cluster_ask { label="Ask (per question)"; fontname="Segoe UI"; fontsize=10; fontcolor="#52514e";
    color="#e1e0d9"; style=rounded;
    q [label="Question"]; qe [label="Embed query"]; router [label="Intent router\\nlogistic regression", fillcolor="#fff8e8", color="#fab219"];
    bm [label="BM25"]; dn [label="Vector search"]; rrf [label="RRF fusion"]; rr [label="MiniLM rerank\\ntop 10 → 5"];
    gate [label="Refusal gate\\nscore < −8.65 → refuse", fillcolor="#fff8e8", color="#fab219"];
    llm [label="LLM · JSON claims\\ngpt-oss-120b → 20b → nemotron", fillcolor="#eef4fc", color="#2a78d6"];
    cite [label="Citation check\\ndrop ungrounded claims", fillcolor="#fff8e8", color="#fab219"];
    adv [label="Advice check", fillcolor="#fff8e8", color="#fab219"]; out [label="Answer + evidence\\n+ timings + trace"];
    q -> qe -> router -> bm; router -> dn; bm -> rrf; dn -> rrf; rrf -> rr -> gate -> llm -> cite -> adv -> out; }
  qd -> dn [style=dashed, color="#c3c2b7"];
}
"""

DECISIONS = [
    ("Hybrid retrieval, not vectors alone",
     "On contracts, keyword search beat vector search (hit@5 0.81 vs 0.72): clause names like 'governing law' are "
     "exact terms. Reciprocal rank fusion merges both rankings without calibrating their scores."),
    ("A small reranker beat a large one", "bge-reranker-base took ~11 s per query on CPU with no gain. MiniLM-L6 "
     "(12x smaller) gave the best ranking (hit@1 0.63, MRR 0.71) at ~1 s."),
    ("Refuse before calling the LLM", "A score threshold filters clearly irrelevant questions at zero LLM cost: "
     "90% precision on held-out contracts. It is tuned to rarely refuse a real question (4.3%), leaving close calls "
     "to the model."),
    ("Claims must cite passages", "The model returns JSON claims with passage ids; anything citing an unretrieved "
     "passage is dropped. 0 unsupported claims out of 22 judged."),
    ("An independent judge", "Answers are graded by Qwen, a different model family from the answering model, "
     "claim by claim, with reasoning before the verdict."),
    ("Whole-document questions", "Passage search can't answer 'what is this about?'. Each document gets a short "
     "overview at upload, one LLM call, every claim citing a passage, and questions that name a document "
     "('the lease', 'the second document') search only that document."),
    ("Built for free-tier limits", "The client reads rate-limit headers and throttles before sending, falls back "
     "across models, and a circuit breaker stops calls to a failing provider."),
]


def render() -> None:
    st.markdown('<div class="lv-hero-eval"><div class="lv-hero-title">How Levi answers a question</div>'
                '<div class="lv-hero-sub">Five steps, each designed so an answer can only come from your document.'
                '</div></div>', unsafe_allow_html=True)
    parts = []
    for i, (title, desc, under) in enumerate(STEPS, start=1):
        if i > 1:
            parts.append('<div class="lv-arrow">›</div>')
        parts.append(f'<div class="lv-step"><div class="n">{i}</div><div class="t">{esc(title)}</div>'
                     f'<div class="d">{esc(desc)}</div><div class="u">{esc(under)}</div></div>')
    st.markdown(f'<div class="lv-steps">{"".join(parts)}</div>', unsafe_allow_html=True)

    with st.expander("Technical architecture"):
        left, right = st.columns([3, 2], gap="large")
        left.graphviz_chart(PIPELINE, width="stretch")
        right.markdown(card("Reading the diagram", (
            '<div class="lv-def">Amber boxes are guardrails or refusal points; blue are storage and the model call. '
            'Every stage is timed per request and traced in Langfuse. Everything runs on open-weight models: '
            'bge-base embeddings and a MiniLM reranker on CPU, gpt-oss via Groq for answers.</div>')),
            unsafe_allow_html=True)

    with st.expander("Design decisions and the evidence behind them"):
        cols = st.columns(2, gap="large")
        for i, (title, body) in enumerate(DECISIONS):
            cols[i % 2].markdown(card(title, f'<div class="lv-def">{esc(body)}</div>'), unsafe_allow_html=True)
