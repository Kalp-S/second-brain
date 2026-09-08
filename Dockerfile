# Multi-stage Python 3.12 Dockerfile for Second Brain PKM RAG
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package resolution
RUN pip install --no-cache-dir uv

# Copy dependencies manifest
COPY pyproject.toml README.md ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv -e .

# Final runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy virtual environment and app code
COPY --from=builder /app/.venv /app/.venv
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY sample_vault/ /app/sample_vault/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
