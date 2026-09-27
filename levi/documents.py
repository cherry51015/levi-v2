"""Questions about documents as a whole.

Two small, rule-based helpers (easy to test and to explain; a learned classifier
would need labelled data we don't have):
- resolve_documents: "the second document", "the lease", "contract.txt" narrow the
  search to the documents the user is talking about.
- is_document_level: "summarise", "what is it about", "who are the parties" are
  answered from the document's stored overview instead of passage search.
"""
import re

from levi.store import DocumentMeta

_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
             "fifth": 5, "5th": 5}
_DOC_NOUN = r"(?:document|doc|file|contract|agreement|upload)"
_ORDINAL_RE = re.compile(r"\b(" + "|".join(_ORDINALS) + r"|last)\s+" + _DOC_NOUN + r"s?\b", re.IGNORECASE)
# Words too common in filenames to identify a document on their own.
_GENERIC = {"agreement", "contract", "document", "doc", "docx", "txt", "pdf", "file", "final", "draft", "copy",
            "signed", "version", "the", "and", "for", "with", "sample", "test"}
_WORD = re.compile(r"[a-z]{4,}")

_DOC_LEVEL = [
    r"\bsummar(?:y|ise|ize|ies)\b",
    r"\boverview\b",
    r"\bgist\b",
    r"\bwhat(?:'s| is| are)\s+(?:this|that|it|these|the\s+\w+(?:\s+\w+)?)\s+(?:\w+\s+)?about\b",
    r"\bwhat\s+(?:kind|type|sort)\s+of\s+" + _DOC_NOUN,
    r"\bwho\s+(?:are|is)\s+(?:the\s+)?(?:parties|party|participants|signatories|people involved)\b",
    r"\b(?:main|key)\s+(?:points|terms|provisions)\b",
]
_DOC_LEVEL_RE = re.compile("|".join(_DOC_LEVEL), re.IGNORECASE)


def is_document_level(question: str) -> bool:
    return bool(_DOC_LEVEL_RE.search(question))


def _name_words(meta: DocumentMeta) -> set[str]:
    words = set(_WORD.findall(meta.filename.lower().replace("_", " ")))
    if meta.overview and meta.overview.get("title"):
        words |= set(_WORD.findall(meta.overview["title"].lower()))
    return words - _GENERIC


def resolve_documents(question: str, docs: list[DocumentMeta]) -> list[str] | None:
    """Return the doc_ids the question refers to, or None if it doesn't single any out.

    `docs` must be in the order the user sees them (upload order).
    """
    if len(docs) < 2:
        return None
    q = question.lower()
    m = _ORDINAL_RE.search(q)
    if m:
        word = m.group(1).lower()
        pos = len(docs) if word == "last" else _ORDINALS[word]
        return [docs[pos - 1].doc_id] if pos <= len(docs) else None
    if any(d.filename.lower() in q for d in docs):
        return [d.doc_id for d in docs if d.filename.lower() in q]
    q_words = set(_WORD.findall(q))
    matched = [d.doc_id for d in docs if _name_words(d) & q_words]
    return matched if 0 < len(matched) < len(docs) else None
