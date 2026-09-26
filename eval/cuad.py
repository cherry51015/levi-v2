"""Build the eval set from CUAD (Contract Understanding Atticus Dataset, CC BY 4.0).

CUAD gives real commercial contracts, lawyer-written questions per clause
category, and gold answer spans with character offsets. That means retrieval
labels come for free: a retrieved chunk is relevant iff it overlaps a gold span.
Questions with no answer in a given contract (is_impossible) become the
refusal-tuning set for Day 2.

Usage:  python -m eval.cuad --contracts 20 --seed 7
"""
import argparse
import io
import json
import random
import re
import zipfile
from pathlib import Path

import requests

from levi.ingest import normalize

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"

# Clause types a person would actually ask a contract assistant about.
CATEGORIES = [
    "Governing Law",
    "Termination For Convenience",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Non-Compete",
    "Exclusivity",
    "Cap On Liability",
    "Audit Rights",
    "Insurance",
    "Anti-Assignment",
    "License Grant",
]


def download() -> Path:
    test_path = RAW_DIR / "test.json"
    if test_path.exists():
        return test_path
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {CUAD_URL} ...")
    resp = requests.get(CUAD_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith("test.json"))
        test_path.write_bytes(zf.read(name))
    return test_path


def question_text(raw_question: str) -> str:
    # CUAD questions look like: 'Highlight the parts ... related to "X" ... Details: <natural question>'
    m = re.search(r"Details:\s*(.+)", raw_question, flags=re.S)
    return (m.group(1) if m else raw_question).strip()


def build(n_contracts: int, seed: int) -> tuple[list[dict], list[dict]]:
    squad = json.loads(download().read_text(encoding="utf-8"))["data"]
    random.Random(seed).shuffle(squad)

    contracts, questions = [], []
    for entry in squad[:n_contracts]:
        para = entry["paragraphs"][0]
        raw_ctx = para["context"]
        # NFKC can change string length, so gold offsets must be re-found in the normalized text.
        ctx = normalize(raw_ctx)
        contract_id = f"cuad_{len(contracts):03d}"
        contracts.append({"contract_id": contract_id, "title": entry["title"], "text": ctx})

        for qa in para["qas"]:
            category = qa["id"].split("__")[-1]
            if category not in CATEGORIES:
                continue
            spans = sorted({(a["answer_start"], a["text"]) for a in qa["answers"]})
            gold = []
            for start, text in spans:
                norm_text = normalize(text)
                pos = ctx.find(norm_text, max(0, start - 50))
                if pos == -1:
                    pos = ctx.find(norm_text)
                if pos != -1:
                    gold.append({"start": pos, "end": pos + len(norm_text), "text": norm_text})
            answerable = bool(gold)
            questions.append(
                {
                    "qid": f"{contract_id}:{category}",
                    "contract_id": contract_id,
                    "category": category,
                    "question": question_text(qa["question"]),
                    "answerable": answerable,
                    "gold": gold,
                }
            )
    return contracts, questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    contracts, questions = build(args.contracts, args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("cuad_contracts.jsonl", contracts), ("cuad_questions.jsonl", questions)):
        with open(DATA_DIR / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_ans = sum(q["answerable"] for q in questions)
    avg_words = sum(len(c["text"].split()) for c in contracts) / len(contracts)
    print(f"{len(contracts)} contracts (avg {avg_words:,.0f} words), {len(questions)} questions: "
          f"{n_ans} answerable, {len(questions) - n_ans} unanswerable")


if __name__ == "__main__":
    main()
