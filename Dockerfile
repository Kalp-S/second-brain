# Multi-stage Python 3.12 Dockerfile for Second Brain PKM RAG
# ==========================================================
# 1. Builder Stage
# ==========================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy manifest and README for package build
COPY pyproject.toml README.md ./

# Create virtual environment and install project with dependencies
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv -e .

# ==========================================================
# 2. Production Runtime Stage
# ==========================================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Install curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root app user and directory permissions
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy virtual environment and application source
COPY --from=builder /app/.venv /app/.venv
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY sample_vault/ /app/sample_vault/

# Create runtime data directory with proper ownership
RUN mkdir -p /app/data && chown -R appuser:appgroup /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Container Healthcheck verifying FastAPI and Qdrant readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/system/status || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
