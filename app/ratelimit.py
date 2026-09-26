"""Token-bucket rate limiter for the public demo.

The LLM providers give us ~8k tokens/minute on the free tier; one visitor
hammering /ask would exhaust it for everyone. So the API applies to its
clients the same kind of limit the providers apply to us: a bucket of
`capacity` requests that refills at `refill_per_s`. Empty bucket -> 429 with
Retry-After, which is exactly the signal our own LLM client knows how to handle.

It is global rather than per-IP on purpose: in the deployed Space every request
reaches the API from the Streamlit server on localhost, so all IPs look the same.
"""
import time


class TokenBucket:
    def __init__(self, capacity: float, refill_per_s: float, clock=time.monotonic):
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.clock = clock
        self.tokens = capacity
        self.updated = clock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_s)
        self.updated = now

    def try_acquire(self) -> float:
        """Take one token. Returns 0 on success, else seconds until one is available."""
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0
        return (1 - self.tokens) / self.refill_per_s
