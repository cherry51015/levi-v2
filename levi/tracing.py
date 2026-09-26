"""Optional Langfuse tracing (OpenTelemetry under the hood).

A trace is one /ask request; each stage inside it is an observation (a span):
the guardrail router, retrieval, and every individual LLM attempt - so retries
and fallbacks show up as separate generations in the trace view.

Without LANGFUSE_* keys this module is a no-op: the app never depends on it.
"""
import os

import levi.config  # noqa: F401  (loads .env before we read the keys)

ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if ENABLED:
    from langfuse import get_client, observe
else:
    def observe(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func

    def get_client():
        return None


def update_span(**kwargs) -> None:
    if ENABLED:
        get_client().update_current_span(**kwargs)


def update_generation(**kwargs) -> None:
    if ENABLED:
        get_client().update_current_generation(**kwargs)


def current_trace_id() -> str | None:
    return get_client().get_current_trace_id() if ENABLED else None


def flush() -> None:
    if ENABLED:
        get_client().flush()
