# ─────────────────────────────────────────────────────────────
# ResearchBuddy — Application Container
#
# This container runs the Streamlit web UI.
# It connects to an Ollama server (run separately or via
# docker-compose) for LLM inference.
#
# Build:   docker build -t researchbuddy .
# Run:     docker run -p 8501:8501 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 researchbuddy
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="ResearchBuddy"
LABEL description="Local-first AI research assistant — Systematic Review + Research Notebook"

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        curl \
        graphviz \
        espeak-ng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true \
    && pip install --no-cache-dir opencv-python-headless

# ── Copy application source ───────────────────────────────────
COPY . .

# ── Create runtime directories ────────────────────────────────
RUN mkdir -p outputs/memory

# ── Streamlit configuration ───────────────────────────────────
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_THEME_BASE=dark
ENV STREAMLIT_SERVER_SESSION_IDLE_SECONDS=3600

# ── Ollama connection ─────────────────────────────────────────
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV OLLAMA_MODEL=llama3.2:3b
ENV NUM_CTX=32768

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
