"""Ranking metrics over a binary relevance list (rels[i] = is result i relevant)."""
import math


def hit_at_k(rels: list[bool], k: int) -> float:
    """1 if any relevant chunk is in the top k. For single-answer QA this is recall@k."""
    return float(any(rels[:k]))


def reciprocal_rank(rels: list[bool]) -> float:
    for rank, rel in enumerate(rels, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(rels: list[bool], k: int, n_relevant: int) -> float:
    """Rewards relevant chunks more the higher they rank; 1.0 = ideal ordering."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, rel in enumerate(rels[:k], start=1) if rel)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, n_relevant) + 1))
    return dcg / ideal if ideal else 0.0


def bootstrap_ci(values: list[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for a mean, by resampling questions with replacement.

    With ~100 questions, a 0.03 gap between two systems may be noise; the CI shows that.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 3), round(float(hi), 3)


def overlaps(chunk_start: int, chunk_end: int, spans: list[tuple[int, int]]) -> bool:
    """A chunk counts as relevant if it overlaps any gold answer span."""
    return any(s < chunk_end and chunk_start < e for s, e in spans)
