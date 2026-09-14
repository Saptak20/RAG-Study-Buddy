FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install minimal runtime utilities (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system group and user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch CPU first to avoid heavy CUDA wheels, then install remaining dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Create required persistent storage and cache directories
RUN mkdir -p /app/data/documents /app/data/faiss_index /app/.cache/huggingface

# Pre-download production embedding model into image cache for fast, offline-ready cold starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source code
COPY --chown=appuser:appgroup app /app/app

# Ensure proper ownership of the entire application directory
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Dynamic port binding
EXPOSE 8000

# Container healthcheck using lightweight curl against /health
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

# Start ASGI application respecting dynamic PORT environment variable (Render-compatible)
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
