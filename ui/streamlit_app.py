"""Levi demo UI. Every decision is made by the API; the UI shows the answer, its evidence,
and how the answer was produced (stages, guardrails, model attempts)."""
import os

import requests
import streamlit as st

import about_view
import eval_view
from charts import confidence_track, stage_timeline
from theme import CSS, GOOD, INK_2, MUTED, WARNING, card, esc, fmt_ms, highlight, pill, stat

API = os.getenv("LEVI_API_URL", "http://127.0.0.1:8000")
REFUSAL_THRESHOLD = float(os.getenv("LEVI_REFUSAL_THRESHOLD", "-7.49"))
EXAMPLES = ["What is the notice period for termination?", "Which law governs this agreement?",
            "Is there a cap on liability?", "Should I sign this agreement?"]

st.set_page_config(page_title="Levi · legal document Q&A", page_icon="⚖️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("history", [])
st.session_state.setdefault("selected", None)


def api(method: str, path: str, **kwargs):
    try:
        return requests.request(method, f"{API}{path}", timeout=180, **kwargs)
    except requests.ConnectionError:
        st.error(f"Can't reach the API at {API}. Start it with: uvicorn app.main:app --port 8000")
        st.stop()


# ---------------------------------------------------------------- sidebar: library + system
with st.sidebar:
    st.markdown("#### Document library")
    upload = st.file_uploader("Add a contract, lease or agreement", type=["pdf", "docx", "txt"],
                              help="Text-based files only; scanned images need OCR, which is out of scope.")
    if upload and st.session_state.get("last_upload") != upload.file_id:
        with st.spinner("Reading, chunking and embedding..."):
            r = api("POST", "/documents", files={"file": (upload.name, upload.getvalue())})
        st.session_state["last_upload"] = upload.file_id
        if r.ok:
            b = r.json()
            note = "already indexed, reused" if b["cached"] else f"indexed in {b['ingest_ms'] / 1000:.1f} s"
            st.success(f"{b['document']['n_chunks']} chunks · {note}")
        else:
            st.error(r.json().get("detail", r.text))

    docs = api("GET", "/documents").json()
    names = {d["doc_id"]: d["filename"] for d in docs}
    selected_docs = []
    for d in docs:
        c1, c2 = st.columns([6, 1])
        if c1.checkbox(d["filename"], value=True, key=d["doc_id"],
                       help=f"{d['n_words']:,} words · {d['n_chunks']} chunks · id {d['doc_id']}"):
            selected_docs.append(d["doc_id"])
        if c2.button(":material/delete:", key=f"del-{d['doc_id']}", help="Remove from library", type="tertiary"):
            api("DELETE", f"/documents/{d['doc_id']}")
            st.rerun()
    if docs:
        st.caption(f"Searching {len(selected_docs)} of {len(docs)} documents")
    else:
        st.caption("No documents yet. Sample files are in the repo's samples/ folder.")

    st.divider()
    st.markdown("#### System")
    health = api("GET", "/health").json()
    rows = "".join(
        f'<li><span class="icon" style="color:{GOOD if p["breaker"] == "closed" else WARNING}">'
        f'{"●" if p["breaker"] == "closed" else "◐"}</span><span class="detail">{esc(p["provider"].rsplit("/", 1)[-1])}'
        f' · breaker {esc(p["breaker"])}</span></li>' for p in health["llm_providers"])
    st.markdown(f'<ul class="lv-checks">{rows}</ul><div class="lv-caveat">Fallback order top to bottom · retrieval: '
                f'{esc(health["retrieval_mode"])} · API startup {health["startup_s"]} s</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- header
st.markdown('<div class="lv-brand"><span class="name">⚖️ Levi</span><span class="tag">legal document Q&A with '
            'cited evidence</span></div><div class="lv-sub">Answers come only from your documents, every claim cites '
            'its source passage, and Levi explains what a document says without giving legal advice.</div>',
            unsafe_allow_html=True)

tab_ask, tab_eval, tab_how = st.tabs(["Ask", "Evaluation", "How it works"])


def render_answer(turn: dict, idx: int) -> None:
    body = turn["response"]
    status = body.get("status", "error")
    st.markdown(pill(status), unsafe_allow_html=True)
    if body.get("notice"):
        st.markdown(f'<div class="lv-notice">{esc(body["notice"])}</div>', unsafe_allow_html=True)
    if status == "answered":
        items = "".join(f'<li>{esc(c["text"])}' + "".join(f'<span class="lv-chip">{esc(ref)}</span>'
                                                          for ref in c["citations"]) + "</li>" for c in body["claims"])
        st.markdown(f'<ul class="lv-claims">{items}</ul>', unsafe_allow_html=True)
        sources = []
        for c in body["citations"]:
            doc = names.get(c["chunk_id"].split(":")[0], "document")
            page = f" · page {c['page']}" if c.get("page") else ""
            sources.append(
                f'<div class="lv-src"><div class="lv-src-head"><span><span class="lv-chip">{esc(c["ref"])}</span> '
                f'<b>{esc(doc)}</b>{esc(page)}</span><span>relevance {c.get("score", 0):.2f}</span></div>'
                f'<div class="lv-src-body">{highlight(c["snippet"], c.get("highlights", []))}</div>'
                f'<details style="margin-top:6px"><summary style="font-size:.78rem;color:{INK_2};cursor:pointer">'
                f'Full passage</summary><div class="lv-src-body" style="margin-top:6px">'
                f'{highlight(c.get("text", ""), c.get("highlights", []))}</div></details></div>')
        st.markdown('<div class="lv-card-title" style="margin-top:10px">Evidence</div>' + "".join(sources),
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="lv-refusal">{esc(body.get("answer") or body.get("detail", ""))}</div>',
                    unsafe_allow_html=True)
    if st.session_state["selected"] != idx and "timings_ms" in body:
        if st.button("Inspect this answer", key=f"inspect-{idx}", type="tertiary"):
            st.session_state["selected"] = idx
            st.rerun()


def guardrail_checks(b: dict) -> str:
    status = b["status"]
    reached_retrieval = status not in ("off_topic",)
    reached_llm = reached_retrieval and status != "refused_low_confidence" and b.get("provider")
    src = {"classifier": "confident", "uncertain": "unsure, so retrieval decides", "disabled": "off"}
    checks = [("✓", GOOD, "Intent router", f'{b["intent"]} · confidence {b["intent_confidence"]:.2f} · '
                                           f'{src.get(b["intent_source"], b["intent_source"])}')]
    if not reached_retrieval:
        checks.append(("↷", MUTED, "Scope", "outside document Q&A: stopped before retrieval, no LLM call"))
    else:
        conf = b.get("retrieval_confidence")
        ok = conf is not None and conf >= REFUSAL_THRESHOLD
        checks.append(("✓" if ok else "∅", GOOD if ok else MUTED, "Refusal gate",
                       f"score {conf:.2f} vs cut-off {REFUSAL_THRESHOLD:g}" + ("" if ok else ": refused, no LLM call")))
    if reached_llm:
        n_claims = len(b.get("claims", []))
        checks.append(("✓", GOOD, "Citation check",
                       f'{n_claims} claims grounded · {b.get("dropped_claims", 0)} dropped for bad citations'))
        removed = b.get("advice_claims_removed", 0)
        checks.append(("!" if removed else "✓", WARNING if removed else GOOD, "Advice check",
                       f"{removed} advice sentences removed" + (" · facts-only mode" if b["intent"] == "advice" else "")))
    else:
        checks += [("–", MUTED, "Citation check", "not reached"), ("–", MUTED, "Advice check", "not reached")]
    return '<ul class="lv-checks">' + "".join(
        f'<li><span class="icon" style="color:{c}">{i}</span><span class="what">{esc(w)}</span>'
        f'<span class="detail">{esc(d)}</span></li>' for i, c, w, d in checks) + "</ul>"


def render_inspector(turn: dict) -> None:
    b = turn["response"]
    st.markdown(f'<div class="lv-card-title">Answer inspector</div><div class="lv-def" style="margin-bottom:8px">'
                f'“{esc(turn["question"])}”</div>', unsafe_allow_html=True)
    if "timings_ms" not in b:
        st.markdown(card("Request failed", f'<div class="lv-def">{esc(b.get("detail", "unknown error"))}</div>'),
                    unsafe_allow_html=True)
        return
    timings = b["timings_ms"]
    model = (b.get("provider") or "none").rsplit("/", 1)[-1]
    attempts = b.get("llm_attempts", [])
    st.markdown(card("Summary", '<div class="lv-kv">' + stat("Total time", fmt_ms(b["total_ms"]),
                                                             f'LLM {fmt_ms(timings.get("llm", 0))}') +
                     stat("Model", model if b.get("provider") else "no LLM call",
                          f"{len(attempts)} attempt(s)" if attempts else "") +
                     stat("Claims", str(len(b.get("claims", []))), f'{len(b.get("citations", []))} sources') + "</div>"),
                unsafe_allow_html=True)
    st.markdown(card("Guardrails", guardrail_checks(b)), unsafe_allow_html=True)
    st.markdown(card("Retrieval confidence (top rerank score)",
                     confidence_track(b.get("retrieval_confidence"), REFUSAL_THRESHOLD)), unsafe_allow_html=True)
    st.markdown('<div class="lv-card-title">Where the time went</div>', unsafe_allow_html=True)
    st.altair_chart(stage_timeline(timings), width="stretch")
    if any(a["outcome"] != "ok" for a in attempts):
        rows = "".join(f'<li><span class="icon" style="color:{GOOD if a["outcome"] == "ok" else WARNING}">'
                       f'{"✓" if a["outcome"] == "ok" else "↻"}</span><span class="what">{esc(a["provider"].rsplit("/", 1)[-1])}'
                       f'</span><span class="detail">{esc(a["outcome"])} · {a["latency_ms"]:.0f} ms</span></li>'
                       for a in attempts)
        st.markdown(card("LLM attempts (fallback chain)", f'<ul class="lv-checks">{rows}</ul>'), unsafe_allow_html=True)
    if b.get("trace_id"):
        st.markdown(f'<div class="lv-caveat">Langfuse trace <span class="lv-mono">{esc(b["trace_id"])}</span></div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------- Ask tab
with tab_ask:
    chat_col, insp_col = st.columns([3, 2], gap="large")
    with chat_col:
        history = st.session_state["history"]
        if not history:
            st.markdown(card("Try asking", '<div class="lv-def">' + "<br>".join(esc(e) for e in EXAMPLES) +
                             '<br><br>Ask something the document does not cover, or ask for advice, to see the '
                             'refusal and advice guardrails.</div>'), unsafe_allow_html=True)
        for i, turn in enumerate(history):
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                render_answer(turn, i)
        question = st.chat_input("Ask about your documents..." if selected_docs else "Add a document to start",
                                 disabled=not selected_docs)
        if question:
            with st.spinner("Retrieving, reranking and answering..."):
                r = api("POST", "/ask", json={"question": question, "doc_ids": selected_docs})
            body = r.json()
            if r.status_code >= 400 and "status" not in body:
                body = {"status": "error", "detail": body.get("detail", r.text)}
            history.append({"question": question, "response": body})
            st.session_state["selected"] = len(history) - 1
            st.rerun()
    with insp_col:
        sel = st.session_state["selected"]
        if sel is not None and sel < len(st.session_state["history"]):
            render_inspector(st.session_state["history"][sel])
        else:
            st.markdown(card("Answer inspector", '<div class="lv-def">Ask a question to see how the answer was '
                             'produced: guardrail decisions, retrieval confidence against the refusal cut-off, and a '
                             'timeline of every pipeline stage.</div>'), unsafe_allow_html=True)

with tab_eval:
    eval_view.render()

with tab_how:
    about_view.render()
