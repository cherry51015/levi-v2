"""Layered guardrails against legal advice.

Layer 1 - input intent router:
  A logistic-regression classifier on the query embedding we already compute
  for retrieval, so it costs well under a millisecond. Its label is used only
  when it is confident; otherwise the query goes through retrieval as usual.
Layer 2 - safe completion instead of hard refusal:
  "advice" queries still get the factual part of an answer ("clause 9 says
  30 days' notice") in a stricter mode, plus a clear pointer to a lawyer.
  Blocking them outright would refuse many questions a user legitimately needs answered.
Layer 3 - grounding (levi/answer.py): claims must cite retrieved chunks.
Layer 4 - output advice detector: catches recommendation language in the
  drafted answer and removes those claims.
"""
import json
import re
from pathlib import Path

import numpy as np

INTENTS = ("informational", "advice", "off_topic")
TRAIN_PATH = Path(__file__).parent / "data" / "intent_train.jsonl"

ADVICE_NOTE = ("I can explain what your document says, but not what you should do. "
               "For advice on your situation, please consult a qualified lawyer.")
OFF_TOPIC_REPLY = ("I can only answer questions about the documents you've uploaded. "
                   "Try asking about a clause, a date, a party's obligations, or a definition.")


class IntentRouter:
    def __init__(self, confidence_threshold: float = 0.6):
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(max_iter=2000, C=4.0)
        self.threshold = confidence_threshold

    def fit(self, vectors: np.ndarray, labels: list[str]) -> "IntentRouter":
        self.clf.fit(vectors, labels)
        return self

    @classmethod
    def train_default(cls, embedder, path: Path = TRAIN_PATH, **kwargs) -> "IntentRouter":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        vectors = np.vstack([embedder.embed_query(r["text"]) for r in rows])
        return cls(**kwargs).fit(vectors, [r["label"] for r in rows])

    def predict(self, query_vec: np.ndarray) -> tuple[str, float]:
        probs = self.clf.predict_proba(query_vec.reshape(1, -1))[0]
        best = int(np.argmax(probs))
        return str(self.clf.classes_[best]), float(probs[best])

    def is_confident(self, confidence: float) -> bool:
        return confidence >= self.threshold


# Recommendation language aimed at the reader. Deliberately narrow: document text
# like "the Tenant shall" or "the Licensee must" is description, not advice.
_ADVICE_PATTERNS = [
    r"\byou should(?:n't| not)?\b",
    r"\byou (?:may )?want to\b",
    r"\bI (?:would |strongly )?(?:recommend|advise|suggest)\b",
    r"\bwe (?:would )?(?:recommend|advise|suggest)\b",
    r"\bmy (?:advice|recommendation)\b",
    r"\byour best (?:option|move|course)\b",
    r"\b(?:it is|it's|it would be) (?:wise|advisable|a good idea|in your (?:best )?interest)\b",
    r"\byou (?:have|stand) a (?:good|strong|weak) (?:case|chance)\b",
    r"\byou (?:will|would) (?:likely |probably )?(?:win|lose)\b",
    r"\bconsider (?:hiring|suing|filing|negotiating|terminating|signing)\b",
]
_ADVICE_RE = re.compile("|".join(_ADVICE_PATTERNS), flags=re.IGNORECASE)


def contains_advice(text: str) -> bool:
    return bool(_ADVICE_RE.search(text))
