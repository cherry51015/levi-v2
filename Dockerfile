# Single container for a Hugging Face Space: FastAPI (internal :8000) + Streamlit (public :7860).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models_cache \
    LEVI_QDRANT_PATH=/app/data/qdrant

WORKDIR /app

# CPU-only torch first: the default wheel bundles CUDA and is several GB larger.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake model weights into the image so a cold start never waits on a download.
COPY levi/ levi/
RUN python -c "from levi.retrieval import Embedder, Reranker; Embedder(); Reranker()"
ENV HF_HUB_OFFLINE=1

COPY app/ app/
COPY ui/ ui/
COPY .streamlit/ .streamlit/
COPY start.sh .

# Spaces run the container as uid 1000; it needs to write the document store and logs.
RUN useradd -m -u 1000 user && mkdir -p /app/data /app/logs && chown -R user /app
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["bash", "start.sh"]
