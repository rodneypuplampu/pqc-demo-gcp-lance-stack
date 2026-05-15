# ============================================================
# Gummaa-Atlas — Dockerfile
# Base: python:3.11-slim  Target: Cloud Run (Gen2)
# ============================================================

FROM python:3.11-slim

# System deps (needed by LanceDB / pyarrow C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure the FUSE mount-point exists so Cloud Storage FUSE
# can bind-mount the LanceDB workspace directory without errors
RUN mkdir -p /app/gummaa_workspace.lance

# Cloud Run injects $PORT at runtime (default 8080)
EXPOSE 8080

# Healthcheck so Cloud Run knows when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-keep-alive", "65"]
