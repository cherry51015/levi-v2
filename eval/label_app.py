"""Blind human labelling of claims, to validate the LLM judge.

The judge's verdict is hidden so it can't anchor your judgement.
Run: streamlit run eval/label_app.py
"""
import json
from pathlib import Path

import streamlit as st

DIR = Path(__file__).parent / "labeling"
CLAIMS = DIR / "claims_to_label.jsonl"
LABELS = DIR / "human_labels.jsonl"

st.set_page_config(page_title="Label claims", layout="wide")
claims = [json.loads(line) for line in CLAIMS.read_text(encoding="utf-8").splitlines() if line.strip()]
done = {}
if LABELS.exists():
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done[row["id"]] = row["label"]

todo = [c for c in claims if c["id"] not in done]
st.progress(len(done) / max(1, len(claims)), text=f"{len(done)} / {len(claims)} labelled (25+ is enough for agreement)")
if not todo:
    st.success("All claims labelled. Run: python -m eval.judge_agreement")
    st.stop()

item = todo[0]
st.caption(f"Question: {item['question']}")
st.subheader(item["claim"])
st.markdown("**Is this claim supported by the source text below?** Judge only against the text, not your own knowledge.")
for src in item["sources"]:
    st.text_area("Source", src, height=220, disabled=True, key=f"{item['id']}-{hash(src)}")


def save(label: str) -> None:
    with open(LABELS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": item["id"], "label": label}) + "\n")


cols = st.columns(3)
for col, (label, help_text) in zip(cols, [
    ("supported", "Stated in the text, or follows directly"),
    ("partial", "Partly right, but adds or changes a detail"),
    ("unsupported", "Not in the text, or contradicts it"),
]):
    if col.button(label.capitalize(), help=help_text, use_container_width=True):
        save(label)
        st.rerun()
