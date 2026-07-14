# ─────────────────────────────────────────────────────────────
# BeeSearch — Application Container
#
# This single container runs the React/FastAPI web app (primary UI),
# the Streamlit UI (secondary), and the CLI side by side.
# The React SPA is built in stage 1 and served by FastAPI at "/" (port 8000).
# It connects to an Ollama server (run separately or via docker-compose).
#
# Build:   docker build -t beesearch .
# Run:     docker run -p 8000:8000 -p 8501:8501 \
#              -e OLLAMA_BASE_URL=http://host.docker.internal:11434 beesearch
#          → React app at http://localhost:8000
#          → Streamlit UI at http://localhost:8501 (still available)
# CLI:     docker run --rm -e OLLAMA_BASE_URL=... beesearch python main.py --check-system
# ─────────────────────────────────────────────────────────────

# ── Stage 1: build the React frontend static assets ────────────
# Built here (rather than requiring a pre-built frontend/dist on the
# host) so a plain `docker build` always produces a complete image.
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python application image ───────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="BeeSearch"
LABEL description="Local-first AI research assistant — React/FastAPI web app (port 8000) + Streamlit UI (port 8501) + CLI"

# ── System dependencies ───────────────────────────────────────
# libffi-dev    : required by some Python C extensions (cryptography, etc.)
# curl          : used in healthcheck
# build-essential: compiler for packages with C extensions
# libgl1        : OpenGL shared library needed by opencv-python (Docling dep)
# libglib2.0-0  : GLib runtime needed by opencv in headless Docker environments
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        curl \
        graphviz \
        espeak-ng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────
# Copy requirements first so Docker layer cache is reused on code-only changes
COPY requirements.txt .
# Install in two layers so the opencv uninstall/reinstall can't mask failures
# in the main pip install step (shell operator precedence with '|| true').
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true \
    && pip install --no-cache-dir opencv-python-headless

# ── Copy application source ───────────────────────────────────
COPY . .

# ── Copy the built React frontend (served by the FastAPI backend) ──
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# ── Create runtime directories ────────────────────────────────
RUN mkdir -p outputs/memory
RUN chmod +x docker-entrypoint.sh

# ── Streamlit configuration ───────────────────────────────────
# These can be overridden with -e at runtime
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_THEME_BASE=dark
# Keep sessions alive during long LLM/PDF analysis runs (default is 600s)
ENV STREAMLIT_SERVER_SESSION_IDLE_SECONDS=3600

# ── Ollama connection (override with -e OLLAMA_BASE_URL=...) ──
# Default points to the companion Ollama service in docker-compose.
# For standalone use, set to http://host.docker.internal:11434 (Mac/Windows)
# or http://172.17.0.1:11434 (Linux bridge network).
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV OLLAMA_MODEL=llama3.2:3b
ENV NUM_CTX=32768

# ── Expose React/FastAPI (primary) + Streamlit (secondary) ───
EXPOSE 8000
EXPOSE 8501

# ── Healthcheck — React/FastAPI is the primary readiness gate ─
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/health && curl -f http://localhost:8501/_stcore/health || exit 1

# ── Default command — React/FastAPI (primary) + Streamlit (secondary) ──
CMD ["./docker-entrypoint.sh"]
