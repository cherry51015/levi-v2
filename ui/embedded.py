"""Embedded mode, for single-process hosts such as Streamlit Community Cloud.

Such hosts run only the Streamlit app, with no second server for the API. Instead of
re-implementing anything, the UI hosts the exact same FastAPI app in-process through
FastAPI's TestClient: every request still goes through the same validation, rate
limiting, pipeline and guardrails. Enabled only when LEVI_UI_MODE=local; local and
Docker runs keep talking to the API over HTTP.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # the host runs ui/streamlit_app.py; `app` and `levi` live one level up


class _Response:
    """Gives the in-process response the same `.ok` the HTTP (requests) path uses."""

    def __init__(self, response):
        self._r = response

    @property
    def ok(self) -> bool:
        return self._r.is_success

    def __getattr__(self, name):
        return getattr(self._r, name)


@st.cache_resource(show_spinner="Loading Levi's models (the first start takes about a minute)...")
def _client():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    client.__enter__()  # runs the app's startup once per process: models load, store opens
    return client


def api(method: str, path: str, **kwargs):
    kwargs.pop("timeout", None)
    return _Response(_client().request(method, path, **kwargs))
