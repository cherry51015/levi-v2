"""Demo UI. Deliberately thin: every decision happens in the API, the UI only displays it."""
import os

import requests
import streamlit as st

API = os.getenv("LEVI_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Levi - legal document Q&A", page_icon="⚖️", layout="wide")
st.title("⚖️ Levi")
st.caption("Ask questions about your legal documents. Answers cite the exact passages they come from. "
           "Levi explains what documents say; it does not give legal advice.")


def api(method: str, path: str, **kwargs):
    try:
        return requests.request(method, f"{API}{path}", timeout=120, **kwargs)
    except requests.ConnectionError:
        st.error(f"Can't reach the API at {API}. Is it running?")
        st.stop()


with st.sidebar:
    st.header("Documents")
    upload = st.file_uploader("Add a document", type=["pdf", "docx", "txt"])
    if upload and st.session_state.get("last_upload") != upload.file_id:
        with st.spinner("Reading and indexing..."):
            r = api("POST", "/documents", files={"file": (upload.name, upload.getvalue())})
        st.session_state["last_upload"] = upload.file_id
        if r.ok:
            body = r.json()
            note = "already indexed, reused" if body["cached"] else f"indexed in {body['ingest_ms'] / 1000:.1f}s"
            st.success(f"{body['document']['filename']}: {body['document']['n_chunks']} chunks ({note})")
        else:
            st.error(r.json().get("detail", r.text))

    docs = api("GET", "/documents").json()
    selected = []
    for d in docs:
        col1, col2 = st.columns([5, 1])
        if col1.checkbox(f"{d['filename']} ({d['n_words']:,} words)", value=True, key=d["doc_id"]):
            selected.append(d["doc_id"])
        if col2.button("🗑", key=f"del-{d['doc_id']}", help="Delete"):
            api("DELETE", f"/documents/{d['doc_id']}")
            st.rerun()
    if not docs:
        st.info("Upload a contract, lease, or agreement to begin.")

question = st.chat_input("e.g. What is the notice period for termination?", disabled=not selected)
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            r = api("POST", "/ask", json={"question": question, "doc_ids": selected})
        body = r.json()
        if r.status_code >= 400 and "detail" in body:
            st.error(body["detail"])
            st.stop()

        if body.get("notice"):
            st.warning(body["notice"])
        if body["status"] == "answered":
            for claim in body["claims"]:
                st.markdown(f"- {claim['text']} " + " ".join(f"`{c}`" for c in claim["citations"]))
            with st.expander("Sources"):
                names = {d["doc_id"]: d["filename"] for d in docs}
                for c in body["citations"]:
                    doc_id = c["chunk_id"].split(":")[0]
                    page = f", page {c['page']}" if c["page"] else ""
                    st.markdown(f"**`{c['ref']}`** {names.get(doc_id, doc_id)}{page}")
                    st.caption(c["snippet"] + "...")
        else:
            st.info(body["answer"])

        with st.expander("How this answer was produced"):
            st.write(f"Intent: **{body['intent']}** ({body['intent_source']}, confidence {body['intent_confidence']})"
                     f" · status: **{body['status']}** · retrieval confidence: {body.get('retrieval_confidence')}"
                     f" · model: {body.get('provider') or '-'}")
            st.bar_chart({k: v for k, v in body["timings_ms"].items()}, horizontal=True)
            st.caption(f"Total {body['total_ms']:.0f} ms · trace id {body.get('trace_id') or 'tracing off'}")
