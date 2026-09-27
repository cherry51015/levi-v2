"""Levi demo UI. Every decision is made by the API; the UI shows the answer, its evidence,
and how the answer was produced (stages, guardrails, model attempts)."""
import os

import requests
import streamlit as st

import about_view
import eval_view
from charts import confidence_track, stage_timeline
from theme import CSS, GOOD, INK_2, MUTED, POLISH_CSS, WARNING, card, esc, fmt_ms, hero, highlight, pill, stat

API = os.getenv("LEVI_API_URL", "http://127.0.0.1:8000")
REFUSAL_THRESHOLD = float(os.getenv("LEVI_REFUSAL_THRESHOLD", "-8.65"))
EXAMPLES = ["What is this document about?", "What is the notice period for termination?",
            "Which law governs this agreement?",
            "Is there a cap on liability?", "Should I sign this agreement?"]

st.set_page_config(page_title="Levi · legal document Q&A", page_icon="⚖️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(POLISH_CSS, unsafe_allow_html=True)
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
    st.markdown('<div class="lv-sidebrand"><span>L</span>Levi</div>', unsafe_allow_html=True)
    st.markdown("#### Document library")
    upload = st.file_uploader("Add a contract, lease or agreement", type=["pdf", "docx", "txt"])
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
        if c1.checkbox(d["filename"], value=True, key=d["doc_id"]):
            selected_docs.append(d["doc_id"])
        doc_type = (d.get("overview") or {}).get("title")
        c1.markdown(f'<div class="lv-doctype">{esc(doc_type) + " · " if doc_type else ""}{d["n_words"]:,} words</div>',
                    unsafe_allow_html=True)
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
st.markdown(hero(eval_view.headline_kpis()), unsafe_allow_html=True)

tab_ask, tab_eval, tab_how = st.tabs(["Ask", "Evaluation", "How it works"])


def render_answer(turn: dict, idx: int) -> None:
    body = turn["response"]
    status = body.get("status", "error")
    scope = ""
    if body.get("scoped_to"):
        scope = '<span class="lv-scope">searched only: ' + esc(", ".join(names.get(d, d) for d in body["scoped_to"])) + "</span>"
    st.markdown(pill(status, body.get("answer_mode", "passages")) + scope, unsafe_allow_html=True)
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
                f'<b>{esc(doc)}</b>{esc(page)}</span><span>'
                f'{"relevance " + format(c["score"], ".2f") if c.get("score") is not None else "overview passage"}'
                f'</span></div>'
                f'<div class="lv-src-body">{highlight(c["snippet"], c.get("highlights", []))}</div>'
                f'<details style="margin-top:6px"><summary style="font-size:.78rem;color:{INK_2};cursor:pointer">'
                f'Full passage</summary><div class="lv-src-body" style="margin-top:6px">'
                f'{highlight(c.get("text", ""), c.get("highlights", []))}</div></details></div>')
        st.markdown('<div class="lv-card-title" style="margin-top:10px">Where this comes from</div>' + "".join(sources),
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
    src = {"classifier": "confident", "uncertain": "unsure, so the evidence decides", "disabled": "off"}
    checks = [("✓", GOOD, "Intent router", f'{b["intent"]} · confidence {b["intent_confidence"]:.2f} · '
                                           f'{src.get(b["intent_source"], b["intent_source"])}')]
    if b.get("scoped_to"):
        checks.append(("✓", GOOD, "Document scope",
                       "searched only " + ", ".join(names.get(d, d) for d in b["scoped_to"])))
    if b.get("answer_mode") == "overview":
        checks.append(("✓", GOOD, "Document overview", "answered from the document's cited overview"))
        checks.append(("✓", GOOD, "Citation check", f'{len(b.get("claims", []))} overview claims, each citing a passage'))
        return '<ul class="lv-checks">' + "".join(
            f'<li><span class="icon" style="color:{c}">{i}</span><span class="what">{esc(w)}</span>'
            f'<span class="detail">{esc(d)}</span></li>' for i, c, w, d in checks) + "</ul>"
    if not reached_retrieval:
        checks.append(("↷", MUTED, "Scope", "outside document Q&A: stopped before retrieval, no LLM call"))
    else:
        conf = b.get("retrieval_confidence")
        ok = conf is not None and conf >= REFUSAL_THRESHOLD
        detail = (f"score {conf:.2f} vs cut-off {REFUSAL_THRESHOLD:g}" if conf is not None else "no score") + \
            ("" if ok else ": refused, no LLM call")
        checks.append(("✓" if ok else "∅", GOOD if ok else MUTED, "Refusal gate", detail))
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
    st.markdown(f'<div class="lv-card-title">How this answer was produced</div><div class="lv-def" style="margin-bottom:8px">'
                f'“{esc(turn["question"])}”</div>', unsafe_allow_html=True)
    if "timings_ms" not in b:
        st.markdown(card("Request failed", f'<div class="lv-def">{esc(b.get("detail", "unknown error"))}</div>'),
                    unsafe_allow_html=True)
        return
    timings = b["timings_ms"]
    model = (b.get("provider") or "none").rsplit("/", 1)[-1]
    attempts = b.get("llm_attempts", [])
    llm_ms = timings.get("llm", 0) + timings.get("overview_build", 0)
    if b.get("answer_mode") == "overview":
        time_hint = (f'overview built now: {fmt_ms(timings["overview_build"])}' if "overview_build" in timings
                     else "stored overview, no LLM call")
    else:
        time_hint = f"LLM {fmt_ms(llm_ms)}"
    if timings.get("llm_wait"):
        time_hint += f' · waited {fmt_ms(timings["llm_wait"])} for rate limit'
    st.markdown(card("Summary", '<div class="lv-kv">' + stat("Total time", fmt_ms(b["total_ms"]), time_hint) +
                     stat("Model", model if b.get("provider") else "no LLM call",
                          f"{len(attempts)} attempt(s)" if attempts else "") +
                     stat("Claims", str(len(b.get("claims", []))), f'{len(b.get("citations", []))} sources') + "</div>"),
                unsafe_allow_html=True)
    st.markdown(card("Guardrails", guardrail_checks(b)), unsafe_allow_html=True)
    with st.expander("Timing and retrieval details"):
        st.markdown('<div class="lv-card-title">Where the time went</div>', unsafe_allow_html=True)
        st.altair_chart(stage_timeline(timings), width="stretch")
        st.markdown(card("Retrieval confidence (top rerank score)",
                         confidence_track(b.get("retrieval_confidence"), REFUSAL_THRESHOLD)), unsafe_allow_html=True)
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
            st.markdown('<div class="lv-try"><div class="lv-card-title">Try asking</div><div class="lv-def">Pick a '
                        'question, or ask about something the contract never mentions and watch Levi say so.</div>'
                        '</div>', unsafe_allow_html=True)
            ex_cols = st.columns(2)
            for i, example in enumerate(EXAMPLES):
                if ex_cols[i % 2].button(example, key=f"ex-{i}", disabled=not selected_docs, width="stretch"):
                    st.session_state["pending_question"] = example
                    st.rerun()
        for i, turn in enumerate(history):
            with st.chat_message("user", avatar=":material/person:"):
                st.write(turn["question"])
            with st.chat_message("assistant", avatar="⚖️"):
                render_answer(turn, i)
        question = st.chat_input("Ask about your documents..." if selected_docs else "Add a document to start",
                                 disabled=not selected_docs) or st.session_state.pop("pending_question", None)
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
            steps = [("Finds the passages", "keyword + meaning search in under 20 ms"),
                     ("Checks the evidence", "weak matches are refused, not guessed"),
                     ("Answers with citations", "every sentence points to its source"),
                     ("Verifies the answer", "uncited or advice-like sentences are removed")]
            flow = "".join(f'<div class="lv-flow-step"><div class="dot">{i}</div><div><div class="t">{esc(t)}</div>'
                           f'<div class="d">{esc(d)}</div></div></div>' for i, (t, d) in enumerate(steps, start=1))
            st.markdown(f'<div class="lv-flow"><div class="lv-card-title">What happens when you ask</div>{flow}'
                        f'<div class="lv-def" style="margin-top:10px">After you ask, this panel shows exactly how '
                        f'your answer was produced.</div></div>', unsafe_allow_html=True)

with tab_eval:
    eval_view.render()

with tab_how:
    about_view.render()
