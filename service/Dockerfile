# Reproducible CPU image for the inspection service.
# Build from the repo root:  docker build -t vqc-agent -f service/Dockerfile .
# Run:                        docker run -p 8000:8000 vqc-agent
#
# Requires a trained model under artifacts/perception/ (run `uv run python -m perception.train`
# first); .dockerignore keeps that model but excludes the dataset and runtime DBs.
FROM python:3.11-slim

# uv for locked, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# System libraries that anomalib's OpenCV dependency loads at runtime (absent from -slim).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached) from the lockfile — CPU torch via the pinned index.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# App code + trained model.
COPY . .

# Run as a non-root user; the MES DB and heatmaps are the only runtime-writable paths.
RUN useradd --create-home app \
    && mkdir -p /app/memory /app/artifacts/heatmaps \
    && chown -R app:app /app/memory /app/artifacts
USER app

EXPOSE 8000
# --no-sync: run the already-installed environment without re-syncing (offline / reproducible).
# Confidence threshold / LLM provider are configurable via VQC_* env vars.
CMD ["uv", "run", "--no-sync", "--no-dev", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
