"""LLM client: one OpenAI-compatible interface over several free providers.

Reliability features, each small enough to explain line by line:
- Timeout per call, so a hung provider can't hang the request.
- Retry with exponential backoff + full jitter, only for retryable errors
  (429, 5xx, timeouts). Jitter stops many clients retrying in lockstep.
- Retry-After: if a provider says "wait 30s", we don't wait - we fall back
  to the next provider, because the user is waiting.
- Circuit breaker per provider: after N consecutive failures we stop calling
  it for a cooldown (OPEN), then let one trial request through (HALF_OPEN).
  Retrying a provider that is down only adds latency and load.
- Deadline: the whole call chain has a time budget; we don't start an
  attempt that can't finish inside it.
- Proactive rate limiting: providers report their remaining token budget in
  response headers. Before sending, we estimate the request's tokens; if the
  budget can't cover it, we wait for the reset (if short) or go straight to
  the next provider - instead of spending a round trip to collect a 429.
- Fallback chain: gpt-oss-120b -> gpt-oss-20b (separate rate limit) -> OpenRouter.
"""
import asyncio
import os
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx
from dotenv import load_dotenv

from levi.tracing import observe, update_generation


class LLMUnavailable(RuntimeError):
    """Every provider in the chain failed or was skipped."""

    def __init__(self, message: str, attempts: list | None = None):
        super().__init__(message)
        self.attempts = attempts or []


class _Retryable(Exception):
    def __init__(self, reason: str, retry_after: float | None = None):
        super().__init__(reason)
        self.retry_after = retry_after


class _Fatal(Exception):
    """Won't succeed on retry (bad request, bad key) - move to the next provider."""


class BreakerState(str, Enum):
    CLOSED = "closed"  # normal: calls flow
    OPEN = "open"  # failing: calls are skipped until the cooldown ends
    HALF_OPEN = "half_open"  # cooldown over: one trial call decides


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0, clock=time.monotonic):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> BreakerState:
        if self.opened_at is None:
            return BreakerState.CLOSED
        if self.clock() - self.opened_at >= self.cooldown_s:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def allow(self) -> bool:
        return self.state != BreakerState.OPEN

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        if self.state == BreakerState.HALF_OPEN:
            self.opened_at = self.clock()  # trial failed: back to OPEN for another cooldown
            return
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = self.clock()


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_key: str | None
    timeout_s: float = 15.0
    extra_body: dict = field(default_factory=dict)  # provider-specific knobs, e.g. reasoning effort
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    # Token budget as last reported by the provider's x-ratelimit-* headers.
    tokens_remaining: int | None = None
    tokens_reset_at: float = 0.0  # time.monotonic() when the budget refills

    @property
    def label(self) -> str:
        return f"{self.name}/{self.model}"


@dataclass
class Attempt:
    provider: str
    outcome: str  # ok | retryable: ... | fatal: ... | skipped: ...
    latency_ms: float


@dataclass
class LLMResult:
    text: str
    provider: str
    latency_ms: float
    attempts: list[Attempt]
    usage: dict
    waited_ms: float = 0.0  # time spent waiting for a rate-limit window, not talking to a model


def default_providers() -> list[Provider]:
    load_dotenv()
    groq_key, or_key = os.getenv("GROQ_API_KEY"), os.getenv("OPENROUTER_API_KEY")
    groq = "https://api.groq.com/openai/v1"
    # gpt-oss models reason before answering; "low" keeps that short, since our task is
    # extraction from given text, not multi-step problem solving.
    low_reasoning = {"reasoning_effort": "low"}
    return [
        Provider("groq", groq, os.getenv("LEVI_LLM_PRIMARY", "openai/gpt-oss-120b"), groq_key,
                 extra_body=low_reasoning),
        Provider("groq", groq, os.getenv("LEVI_LLM_SECONDARY", "openai/gpt-oss-20b"), groq_key,
                 extra_body=low_reasoning),
        Provider("openrouter", "https://openrouter.ai/api/v1",
                 os.getenv("LEVI_LLM_FALLBACK", "nvidia/nemotron-3-super-120b-a12b:free"), or_key, timeout_s=30.0),
    ]


class LLMClient:
    def __init__(self, providers: list[Provider] | None = None, max_attempts: int = 2,
                 backoff_base_s: float = 0.5, backoff_cap_s: float = 4.0, max_retry_after_s: float = 5.0,
                 deadline_s: float = 25.0, transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep):
        self.providers = [p for p in (providers or default_providers()) if p.api_key]
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.max_retry_after_s = max_retry_after_s
        self.deadline_s = deadline_s  # default time budget for one complete() call, fallbacks included
        self._http = httpx.AsyncClient(transport=transport)
        self._sleep = sleep

    async def aclose(self) -> None:
        await self._http.aclose()

    def _budget_wait(self, provider: Provider, needed: int) -> float | None:
        """Seconds to wait before this provider can take `needed` tokens; None = don't wait, skip it."""
        if provider.tokens_remaining is None or provider.tokens_remaining >= needed:
            return 0.0
        wait = provider.tokens_reset_at - time.monotonic()
        if wait <= 0:
            return 0.0  # the budget has refilled since we last heard
        return wait if wait <= self.max_retry_after_s else None

    def backoff_s(self, attempt: int) -> float:
        # Full jitter: uniform in [0, min(cap, base * 2^attempt)].
        return random.uniform(0, min(self.backoff_cap_s, self.backoff_base_s * 2 ** attempt))

    async def complete(self, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.0,
                       max_tokens: int = 800, deadline_s: float | None = None) -> LLMResult:
        deadline_s = deadline_s or self.deadline_s
        start = time.monotonic()
        attempts: list[Attempt] = []

        needed = estimate_tokens(messages) + max_tokens
        waited_ms = 0.0
        for provider in self.providers:
            if not provider.breaker.allow():
                attempts.append(Attempt(provider.label, "skipped: circuit open", 0.0))
                continue
            wait = self._budget_wait(provider, needed)
            if wait is None:
                attempts.append(Attempt(provider.label, "skipped: token budget", 0.0))
                continue
            if wait > 0:
                await self._sleep(wait)
                waited_ms += wait * 1000
            for attempt in range(self.max_attempts):
                remaining = deadline_s - (time.monotonic() - start)
                if remaining < 1.0:
                    raise LLMUnavailable(f"deadline of {deadline_s}s exhausted", attempts)
                t0 = time.monotonic()
                try:
                    data = await self._call(provider, messages, json_mode, temperature, max_tokens,
                                            timeout_s=min(provider.timeout_s, remaining))
                except _Fatal as e:
                    provider.breaker.record_failure()
                    attempts.append(Attempt(provider.label, f"fatal: {e}", _ms(t0)))
                    break  # retrying won't help; try the next provider
                except _Retryable as e:
                    provider.breaker.record_failure()
                    attempts.append(Attempt(provider.label, f"retryable: {e}", _ms(t0)))
                    if not provider.breaker.allow():
                        break  # this failure opened the breaker
                    if e.retry_after is not None and e.retry_after > self.max_retry_after_s:
                        break  # rate-limited for longer than we're willing to make the user wait
                    if attempt + 1 < self.max_attempts:
                        pause = e.retry_after if e.retry_after is not None else self.backoff_s(attempt)
                        await self._sleep(pause)
                        waited_ms += pause * 1000
                    continue
                provider.breaker.record_success()
                attempts.append(Attempt(provider.label, "ok", _ms(t0)))
                return LLMResult(
                    text=data["choices"][0]["message"]["content"] or "",
                    provider=provider.label,
                    latency_ms=_ms(start),
                    attempts=attempts,
                    usage=data.get("usage") or {},
                    waited_ms=round(waited_ms, 1),
                )
        raise LLMUnavailable(f"all providers failed: {[(a.provider, a.outcome) for a in attempts]}", attempts)

    @observe(name="llm_attempt", as_type="generation")
    async def _call(self, provider: Provider, messages: list[dict], json_mode: bool, temperature: float,
                    max_tokens: int, timeout_s: float) -> dict:
        update_generation(model=provider.model, metadata={"provider": provider.name, "timeout_s": timeout_s},
                          model_parameters={"temperature": temperature, "max_tokens": max_tokens})
        try:
            data = await self._post(provider, messages, json_mode, temperature, max_tokens, timeout_s)
        except (_Retryable, _Fatal) as e:
            update_generation(level="ERROR", status_message=f"{type(e).__name__.strip('_')}: {e}")
            raise
        usage = data.get("usage") or {}
        update_generation(usage_details={"input": usage.get("prompt_tokens", 0),
                                         "output": usage.get("completion_tokens", 0)})
        return data

    async def _post(self, provider: Provider, messages: list[dict], json_mode: bool, temperature: float,
                    max_tokens: int, timeout_s: float) -> dict:
        body = {"model": provider.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, **provider.extra_body}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            resp = await self._http.post(
                f"{provider.base_url}/chat/completions", json=body, timeout=timeout_s,
                headers={"Authorization": f"Bearer {provider.api_key}"},
            )
        except httpx.TimeoutException:
            raise _Retryable("timeout")
        except httpx.TransportError as e:
            raise _Retryable(f"connection error: {type(e).__name__}")
        _record_budget(provider, resp)

        if resp.status_code == 429 or resp.status_code >= 500:
            raise _Retryable(f"HTTP {resp.status_code}", retry_after=_parse_retry_after(resp))
        if resp.status_code >= 400:
            raise _Fatal(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
            data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError):
            raise _Retryable("malformed response body")
        return data


def estimate_tokens(messages: list[dict]) -> int:
    # ~4 characters per token for English; an estimate is enough for budgeting.
    return sum(len(m.get("content") or "") for m in messages) // 4 + 10 * len(messages)


_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


def parse_duration_s(value: str) -> float | None:
    """Groq-style durations: "577ms", "7.66s", "1m2.5s"."""
    m = _DURATION.fullmatch(value.strip())
    if not m or not any(m.groups()):
        return None
    minutes, seconds, millis = (float(g) if g else 0.0 for g in m.groups())
    return minutes * 60 + seconds + millis / 1000


def _record_budget(provider: Provider, resp: httpx.Response) -> None:
    remaining = resp.headers.get("x-ratelimit-remaining-tokens")
    reset = resp.headers.get("x-ratelimit-reset-tokens")
    if remaining is None:
        return
    try:
        provider.tokens_remaining = int(float(remaining))
    except ValueError:
        return
    reset_s = parse_duration_s(reset) if reset else None
    provider.tokens_reset_at = time.monotonic() + (reset_s or 0.0)


def _parse_retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None  # HTTP-date form; treat as unknown and use our own backoff


def _ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000, 1)
