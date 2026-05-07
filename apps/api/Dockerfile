# RegRAG API — FastAPI service for Railway deploy.
#
# Build context: this directory (apps/api). Railway builds from the repo root
# but with `dockerfilePath` pointing here; railway.json is configured to do that.

FROM python:3.12-slim

# Install system deps that some Python wheels need
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# uv is a fast pip replacement; install it for the deps step
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install deps first for layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Editable install pulls deps + makes the package importable
RUN uv pip install --system --no-cache -e .

# Railway sets $PORT; default to 8000 for local docker run
ENV PORT=8000
EXPOSE 8000

# Healthcheck: curl /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Use sh -c so $PORT is expanded at runtime
CMD ["sh", "-c", "uvicorn regrag_api.server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
