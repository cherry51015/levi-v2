#!/usr/bin/env bash
# Start the API, wait until its models are loaded, then start the public UI.
set -euo pipefail

uvicorn app.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

for _ in $(seq 1 180); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "API process exited during startup" >&2
    exit 1
  fi
  sleep 1
done

exec streamlit run ui/streamlit_app.py --server.address 0.0.0.0 --server.port 7860 --server.headless true
