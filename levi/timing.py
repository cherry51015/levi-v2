"""Per-stage latency measurement.

Every request carries a StageTimer; each pipeline stage wraps itself in
`with timer.stage("name"):`. The per-stage breakdown is what lets us say
*which* stage owns the p95, not just that the request was slow.
"""
from contextlib import contextmanager
from time import perf_counter

import numpy as np


class StageTimer:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - start) * 1000
            self.stages[name] = self.stages.get(name, 0.0) + elapsed_ms

    def move(self, src: str, dst: str, ms: float) -> None:
        """Re-attribute part of one stage to another (e.g. rate-limit waiting inside the LLM stage)."""
        if ms <= 0 or src not in self.stages:
            return
        ms = min(ms, self.stages[src])
        self.stages[src] -= ms
        self.stages[dst] = self.stages.get(dst, 0.0) + ms

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())


def percentiles(values_ms: list[float], ps=(50, 95, 99)) -> dict[str, float]:
    if not values_ms:
        return {}
    arr = np.asarray(values_ms)
    out = {f"p{p}": round(float(np.percentile(arr, p)), 2) for p in ps}
    out["mean"] = round(float(arr.mean()), 2)
    out["n"] = len(values_ms)
    return out
