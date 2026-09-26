"""Failure-injection tests for the LLM client. A fake HTTP transport plays the
providers, so we can force outages, 429s and timeouts deterministically."""
import asyncio

import httpx
import pytest

from levi.llm import BreakerState, CircuitBreaker, LLMClient, LLMUnavailable, Provider, parse_duration_s


def ok_body(text="hello"):
    return {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 5}}


def make_client(behaviours: dict, calls: list, **kwargs):
    """behaviours maps model name -> list of responses (httpx.Response or Exception), consumed in order."""

    def handler(request: httpx.Request):
        model = __import__("json").loads(request.content)["model"]
        calls.append(model)
        outcome = behaviours[model].pop(0) if len(behaviours[model]) > 1 else behaviours[model][0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    providers = [Provider("p", "http://fake", m, api_key="k") for m in behaviours]
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    client = LLMClient(providers, transport=httpx.MockTransport(handler), sleep=fake_sleep, **kwargs)
    return client, sleeps


def run(coro):
    return asyncio.run(coro)


def test_primary_success():
    calls = []
    client, _ = make_client({"a": [httpx.Response(200, json=ok_body("hi"))], "b": [httpx.Response(200, json=ok_body())]}, calls)
    result = run(client.complete([{"role": "user", "content": "x"}]))
    assert result.text == "hi" and result.provider == "p/a" and calls == ["a"]


def test_retry_then_success_on_same_provider():
    calls = []
    client, sleeps = make_client({"a": [httpx.Response(503), httpx.Response(200, json=ok_body())]}, calls)
    result = run(client.complete([{"role": "user", "content": "x"}]))
    assert calls == ["a", "a"] and result.provider == "p/a" and len(sleeps) == 1


def test_fallback_when_primary_down():
    calls = []
    client, _ = make_client({"a": [httpx.Response(500)], "b": [httpx.Response(200, json=ok_body("from b"))]}, calls)
    result = run(client.complete([{"role": "user", "content": "x"}]))
    assert result.text == "from b"
    assert calls == ["a", "a", "b"]
    assert [a.outcome.split(":")[0] for a in result.attempts] == ["retryable", "retryable", "ok"]


def test_timeout_falls_back():
    calls = []
    client, _ = make_client({"a": [httpx.ReadTimeout("slow")], "b": [httpx.Response(200, json=ok_body())]}, calls)
    assert run(client.complete([{"role": "user", "content": "x"}])).provider == "p/b"


def test_long_retry_after_skips_waiting_and_falls_back():
    calls = []
    rate_limited = httpx.Response(429, headers={"retry-after": "30"})
    client, sleeps = make_client({"a": [rate_limited], "b": [httpx.Response(200, json=ok_body())]}, calls)
    result = run(client.complete([{"role": "user", "content": "x"}]))
    assert calls == ["a", "b"] and sleeps == [] and result.provider == "p/b"


def test_short_retry_after_is_honoured():
    calls = []
    client, sleeps = make_client(
        {"a": [httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, json=ok_body())]}, calls)
    run(client.complete([{"role": "user", "content": "x"}]))
    assert sleeps == [1.0]


def test_bad_key_is_not_retried():
    calls = []
    client, _ = make_client({"a": [httpx.Response(401)], "b": [httpx.Response(200, json=ok_body())]}, calls)
    run(client.complete([{"role": "user", "content": "x"}]))
    assert calls == ["a", "b"]


def test_all_down_raises():
    calls = []
    client, _ = make_client({"a": [httpx.Response(500)], "b": [httpx.Response(502)]}, calls)
    with pytest.raises(LLMUnavailable):
        run(client.complete([{"role": "user", "content": "x"}]))


def test_open_breaker_skips_provider_without_calling_it():
    calls = []
    client, _ = make_client({"a": [httpx.Response(500)], "b": [httpx.Response(200, json=ok_body())]}, calls)
    client.providers[0].breaker = CircuitBreaker(failure_threshold=2, cooldown_s=60)
    run(client.complete([{"role": "user", "content": "x"}]))  # 2 failures -> breaker opens
    calls.clear()
    result = run(client.complete([{"role": "user", "content": "x"}]))
    assert calls == ["b"]
    assert result.attempts[0].outcome == "skipped: circuit open"


def test_breaker_state_machine():
    now = [0.0]
    b = CircuitBreaker(failure_threshold=2, cooldown_s=10, clock=lambda: now[0])
    b.record_failure()
    assert b.state == BreakerState.CLOSED
    b.record_failure()
    assert b.state == BreakerState.OPEN and not b.allow()
    now[0] = 10.0
    assert b.state == BreakerState.HALF_OPEN and b.allow()
    b.record_failure()  # trial call fails -> straight back to OPEN
    assert b.state == BreakerState.OPEN
    now[0] = 20.0
    b.record_success()  # trial call succeeds -> CLOSED, counters reset
    assert b.state == BreakerState.CLOSED and b.failures == 0


def test_backoff_has_full_jitter_and_cap():
    client, _ = make_client({"a": [httpx.Response(200, json=ok_body())]}, [])
    samples = [client.backoff_s(attempt=10) for _ in range(200)]
    assert all(0 <= s <= client.backoff_cap_s for s in samples)
    assert max(samples) - min(samples) > 1.0  # actually random, not a fixed delay


def test_parse_duration():
    assert parse_duration_s("577ms") == 0.577
    assert parse_duration_s("7.66s") == 7.66
    assert parse_duration_s("1m2.5s") == 62.5
    assert parse_duration_s("soon") is None


def budget_response(remaining, reset):
    return httpx.Response(200, json=ok_body(), headers={"x-ratelimit-remaining-tokens": str(remaining),
                                                      "x-ratelimit-reset-tokens": reset})


def test_low_budget_with_long_reset_skips_provider_without_calling_it():
    calls = []
    client, sleeps = make_client({"a": [budget_response(50, "30s")], "b": [httpx.Response(200, json=ok_body())]}, calls)
    run(client.complete([{"role": "user", "content": "x"}]))  # a answers, reports only 50 tokens left
    calls.clear()
    result = run(client.complete([{"role": "user", "content": "x" * 400}], max_tokens=100))
    assert calls == ["b"] and sleeps == []
    assert result.attempts[0].outcome == "skipped: token budget"


def test_low_budget_with_short_reset_waits_then_uses_same_provider():
    calls = []
    client, sleeps = make_client({"a": [budget_response(50, "2s")]}, calls)
    run(client.complete([{"role": "user", "content": "x"}]))
    calls.clear()
    run(client.complete([{"role": "user", "content": "x" * 400}], max_tokens=100))
    assert calls == ["a"] and len(sleeps) == 1 and 0 < sleeps[0] <= 2.0


def test_waiting_time_is_reported_separately():
    client, _ = make_client(
        {"a": [httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, json=ok_body())]}, [])
    assert run(client.complete([{"role": "user", "content": "x"}])).waited_ms == 1000.0


def test_unavailable_carries_attempts():
    client, _ = make_client({"a": [httpx.Response(500)]}, [])
    with pytest.raises(LLMUnavailable) as err:
        run(client.complete([{"role": "user", "content": "x"}]))
    assert [a.outcome for a in err.value.attempts] == ["retryable: HTTP 500", "retryable: HTTP 500"]
