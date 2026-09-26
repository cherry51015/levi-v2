"""How it works: the implemented pipeline and the decisions behind it."""
import streamlit as st

from theme import card

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
    gate [label="Refusal gate\\nscore < −7.49 → refuse", fillcolor="#fff8e8", color="#fab219"];
    llm [label="LLM · JSON claims\\ngpt-oss-120b → 20b → nemotron", fillcolor="#eef4fc", color="#2a78d6"];
    cite [label="Citation check\\ndrop ungrounded claims", fillcolor="#fff8e8", color="#fab219"];
    adv [label="Advice check", fillcolor="#fff8e8", color="#fab219"]; out [label="Answer + evidence\\n+ timings + trace"];
    q -> qe -> router -> bm; router -> dn; bm -> rrf; dn -> rrf; rrf -> rr -> gate -> llm -> cite -> adv -> out; }
  qd -> dn [style=dashed, color="#c3c2b7"];
}
"""

DECISIONS = [
    ("Hybrid retrieval, not vectors alone",
     "On contracts, BM25 beat dense search on hit@5 (0.81 vs 0.72): clause names like 'governing law' are exact "
     "terms. RRF merges the two rankings by rank, so their incompatible score scales never need calibrating."),
    ("Small reranker over large", "bge-reranker-base: ~11 s/query on CPU and no gain. MiniLM-L6 (12x smaller): "
     "best hit@1 (0.63) and MRR (0.71) at ~1 s."),
    ("Refuse before calling the LLM", "The rerank score separates answerable from unanswerable questions; a threshold "
     "tuned on half the contracts refuses 32% of unanswerable ones at 85% precision on the other half, with zero LLM cost."),
    ("Claims must cite chunks", "The model returns JSON claims with chunk ids; a claim citing anything not retrieved is "
     "dropped. The judge found 0 unsupported claims out of 22 (small sample)."),
    ("Different-family judge", "Qwen grades gpt-oss answers, reasoning before verdict, per claim, with a length-bias check."),
    ("Free tier as a design constraint", "8k tokens/min and 200k/day per model. The client reads rate-limit headers "
     "and throttles before sending, falls back across models, and a circuit breaker stops hammering a failing provider."),
    ("No LLM in the router", "An earlier LLM fallback for uncertain queries added ~2.4 s and misrouted document questions. "
     "Now uncertain queries go through retrieval and the refusal gate decides."),
]


def render() -> None:
    st.markdown("### How it works")
    left, _ = st.columns([3, 2])
    left.graphviz_chart(PIPELINE, width="stretch")
    st.markdown('<div class="lv-caveat">Amber = guardrail or refusal point · blue = storage and the model call. '
                'Every stage is timed per request and traced in Langfuse.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("#### Decision log")
    cols = st.columns(2, gap="large")
    for i, (title, body) in enumerate(DECISIONS):
        cols[i % 2].markdown(card(title, f'<div class="lv-def">{body}</div>'), unsafe_allow_html=True)
