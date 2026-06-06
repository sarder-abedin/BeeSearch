# Agentic Research Assistant

A fully open-source, local-first AI research assistant built with LangGraph, Ollama, and free academic APIs.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-green)](https://ollama.ai)
[![Tests](https://img.shields.io/badge/Tests-496%20passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What is this?

A six-mode AI research assistant that runs 100% locally — no cloud, no API fees, no data leaving your machine. Give it a research goal, a funding call document, a question about your life, your own papers, or any text to polish, and it produces grounded, citation-backed outputs using local LLMs via Ollama.

| Mode | What it does |
|------|-------------|
| 1 — Literature Search | Search arXiv, Semantic Scholar, and CrossRef; synthesise a structured report with APA citations |
| 2 — ProposalGPT | Upload a funding call to a 9-agent pipeline that generates a complete grant proposal: call analysis, win strategy, literature review, 17 sections, budget, compliance check, reviewer simulation, and iterative improvement |
| 3 — Wisdom Mode | Turn any life or research scenario into evidence-backed, actionable scientific insight with per-claim confidence scores |
| 4 — Systematic Review | Run a PRISMA-style systematic review: search, screen, extract evidence, synthesise — with full flow metrics |
| 5 — Research Notebook | NotebookLM-style workspace: build a notebook from your files and web pages, chat with answers grounded only in your sources, run a 7-agent analysis pipeline |
| 6 — Grammar Proofreading | Upload or paste any text and get a professionally polished rewrite — Academic, Professional Email, Formal, or Informal context; per-error explanations, style tips, feedback refinement, and downloadable Markdown output |

---

## Key features

- Fully local — LLMs run on your machine via Ollama; no OpenAI key, no subscriptions
- Real citations — every claim mapped to a real paper from arXiv or Semantic Scholar
- 9-agent ProposalGPT — from funding call to submission-ready 17-section proposal with budget, compliance check, and reviewer scores
- Docling document parsing — Docling is the default parser for all document modes; it starts enabled in the UI and can be disabled with `--no-docling` on the CLI to revert to the lightweight pdfplumber-only parser
- Hybrid RAG with self-reflective grading — dense FAISS and BM25 fused with RRF, then a post-retrieval LLM relevance filter removes irrelevant items before they reach the main LLM; when the Ollama embedding model (`nomic-embed-text`) is unavailable, all document modes automatically fall back to BM25 keyword search during ingestion — no manual configuration needed, just degraded retrieval quality
- Web search — runs in-process via the `ddgs` library (DuckDuckGo); no API key and no separate service required
- Grammar Proofreading — context-aware holistic rewrite with per-error explanations, style tips, and unlimited feedback revision rounds in the UI (click Refine as many times as needed); the CLI supports up to 3 revision rounds; downloadable Markdown output
- Long-term memory — sessions persist across browser restarts for all six modes
- Quality scores — every output is automatically self-evaluated with per-dimension scores (1–5)
- Export — download reports as Markdown, proposals as Word (.docx) or PDF, budgets as CSV, polished text as Markdown
- CLI and UI — full-featured Streamlit web app and `main.py` command-line interface
- 496 tests — comprehensive offline test suite covering all agents, tools, memory systems, SQLite persistence, MCP server, feedback refinement, and self-reflective RAG

---

## Documentation

| Document | Contents |
|----------|---------|
| [Tutorial](docs/tutorial.md) | CLI reference, per-mode examples, flags, smart recommendations, quality scores, self-reflective RAG, style profiles |
| [Overview and Architecture](docs/overview.md) | How it works, all six modes explained, ProposalGPT 9-agent pipeline, Hybrid RAG, Self-Reflective RAG, feedback refinement, memory system |
| [FAQ](docs/FAQ.md) | Common questions about setup, models, memory, citations, exports, Self-Reflective RAG, feedback refinement, and testing |
| [Architecture Details](docs/architecture.md) | LangGraph topology, node descriptions, Self-Reflective RAG grading, feedback agent, engineering decisions |

---

## Installation

Choose Option A (Docker — recommended) for a zero-config single-command setup, or Option B (manual) to manage your own Python environment.

---

## Option A — Docker (Recommended)

Docker bundles Ollama and the Streamlit app into one command. No Python, no virtual environment, no manual Ollama setup required.

### Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Docker | Desktop 4.x or higher (Mac/Windows) or Engine 24 and Compose 2.20 or higher (Linux) | [Get Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| RAM | 8 GB | 16 GB recommended for 8B models |
| Disk | 10 GB free | Model weights and Docker image |

### Step 1 — Install Docker

**Mac and Windows**

Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/), then start it. Wait for the whale icon to appear in your menu bar or taskbar.

**Linux (Ubuntu 22+)**

```bash
sudo apt install docker-compose-plugin
```

For other distros see the [Docker Engine install guide](https://docs.docker.com/engine/install/).

Verify Docker is ready:

```bash
docker --version        # should show 24.x or higher
docker compose version  # should show v2.x or higher
```

### Step 2 — Clone the repository

```bash
git clone https://github.com/sarder-abedin/agentic-research-assistant.git
cd agentic-research-assistant
```

### Step 3 — (Optional) choose your model

The default model is `llama3.2:3b` (2 GB, fast). To use a larger model, create a `.env` file:

```bash
cp .env.example .env
```

Then open `.env` and set your preferred model:

```bash
OLLAMA_MODEL=llama3.1:8b   # change this line
NUM_CTX=32768               # context window in tokens
```

| Model | Size | Context | Best for |
|-------|------|---------|----------|
| `llama3.2:3b` | 2 GB | 32k | Low-RAM machines, quick tests |
| `llama3.1:8b` | 5 GB | 32k | Recommended — best quality/speed balance |
| `mistral-nemo:12b` | 7 GB | 128k | Long documents (needs 16 GB RAM) |
| `gemma2:9b` | 6 GB | 32k | Good alternative to Llama |
| `phi4:14b` | 9 GB | 32k | Highest reasoning quality |

Skip this step to use the default `llama3.2:3b` — you can always switch models later.

### Step 4 — Build and start

> **Apple Silicon Mac (M1/M2/M3)?** See the [Mac section below](#apple-silicon-mac-m1m2m3) — use a different command.

```bash
docker compose up --build
```

This single command does everything on first run:

1. Builds the Python app image (~1.5 GB)
2. Starts the Ollama LLM server
3. Downloads the selected model weights (cached in a Docker volume — only happens once)
4. Starts the Streamlit web UI

Wait until you see this in the logs:

```
research-app | You can now view your Streamlit app in your browser.
research-app |   Local URL: http://localhost:8501
```

> **First run note:** Model download can take several minutes. The app is not ready until the line above appears.

### Step 5 — Open the app

Open [http://localhost:8501](http://localhost:8501) in your browser. You will see the landing page with six mode cards — click any card to open that mode.

### Day-to-day usage

```bash
docker compose up          # start (foreground — Ctrl+C to stop)
docker compose up -d       # start in background
docker compose down        # stop all containers (model weights are kept)
docker compose down --volumes  # stop AND delete model weights (frees disk)
```

To see live logs when running in background:

```bash
docker compose logs -f app     # Streamlit app logs
docker compose logs -f ollama  # Ollama server logs
```

### Switching models

```bash
# Change model before starting (or edit .env)
OLLAMA_MODEL=llama3.1:8b docker compose up

# Pull an additional model into a running stack
docker compose exec ollama ollama pull mistral-nemo:12b
```

### Apple Silicon Mac (M1/M2/M3)

The `ollama/ollama` Docker image runs under x86 emulation on Apple Silicon and fails its healthcheck. Run Ollama natively instead:

```bash
# 1. Install Ollama from https://ollama.com/download (macOS .dmg)
#    It starts automatically on login after installation.

# 2. Pull the model
ollama pull llama3.2:3b

# 3. Start the app (mac.yml is a standalone file — don't combine with docker-compose.yml)
docker compose -f docker-compose.mac.yml up --build
```

`docker-compose.mac.yml` is a standalone file — it starts only the app container. It does not start an Ollama container; instead it connects to the native Ollama on your Mac via `host.docker.internal:11434`.

**Day-to-day Mac commands:**

```bash
# Stop and remove containers
docker compose -f docker-compose.mac.yml down --remove-orphans

# Restart after a code update
git pull
docker compose -f docker-compose.mac.yml up --build

# View logs
docker compose -f docker-compose.mac.yml logs -f app
```

> **Why `--remove-orphans`?** If you previously ran `docker compose up` (without the `-f` flag), stale `research-ollama` or `research-model-init` containers may still exist on the shared network. `--remove-orphans` cleans them up. Safe to always include.

### GPU acceleration (NVIDIA)

First verify your GPU is visible to Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

If that works, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) then start with the GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Persistent data

Your outputs and session memory live on your machine and survive container restarts:

```
outputs/
  memory/
    sessions.db          ← All sessions in one SQLite database (all modes)
  chroma_db/             ← Embedding cache (FAISS + ChromaDB)
  proposal_*.md          ← Full proposal Markdown
  proposal_*.docx/pdf    ← Word and PDF exports
  budget_*.csv           ← Budget exports (Mode 2)
  grammar_*.md           ← Grammar proofreading polished output (CLI)
```

Model weights are in a Docker-managed volume (`ollama_models`) and survive `docker compose down`. They are only deleted with `docker compose down --volumes`.

### Environment variables

All settings are optional. Copy `.env.example` to `.env` and edit as needed:

```bash
OLLAMA_MODEL=llama3.2:3b        # LLM model to pull and use
NUM_CTX=32768                   # LLM context window in tokens
APP_PORT=8501                   # Host port for the Streamlit UI
CROSSREF_EMAIL=you@example.com  # Polite pool for faster CrossRef API
SEMANTIC_SCHOLAR_API_KEY=       # Higher rate limits (free key)
```

Web search uses DuckDuckGo in-process via the `ddgs` library — no API key and no separate service needed.

See [`.env.example`](.env.example) for all available options with descriptions.

---

## Option B — Manual Installation

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.ai) installed on your machine
- 8 GB RAM minimum (16 GB recommended for 8B models)

### Step 1 — Clone the repository

```bash
git clone https://github.com/sarder-abedin/agentic-research-assistant.git
cd agentic-research-assistant
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows
```

Your prompt should now show `(.venv)` to confirm the environment is active.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install Ollama and pull models

**4a — Install Ollama**

Download and install from [https://ollama.ai](https://ollama.ai), then verify:

```bash
ollama --version
```

**4b — Pull a language model** (pick one based on your RAM)

```bash
ollama pull llama3.2:3b       # 2 GB — low-RAM machines
ollama pull llama3.1:8b       # 5 GB — recommended for most machines
ollama pull mistral-nemo:12b  # 7 GB — 128k context for long documents
```

**4c — Pull the embedding model** (required for full Hybrid RAG quality)

```bash
ollama pull nomic-embed-text   # 274 MB — used for dense vector search
```

Without `nomic-embed-text`, all document modes automatically fall back to BM25 keyword search at ingestion time — retrieval still works, but without dense vector ranking. Pull it for full hybrid RAG quality.

**4d — Start the Ollama server** (keep this terminal open)

```bash
ollama serve
```

> **Port conflict?** If you see `bind: address already in use`, use the wrapper script:
> ```bash
> ./scripts/ollama-serve.sh
> ```
> It kills the existing process automatically. Works on macOS and Linux.

### Step 5 — Configure your environment

```bash
cp .env.example .env
```

The two settings you are most likely to change:

```bash
OLLAMA_MODEL=llama3.1:8b   # must match the model you pulled in Step 4b
NUM_CTX=32768               # context window — match your model's max
```

See [`.env.example`](.env.example) for all available options.

### Step 6 — Start the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Next steps

Once the app is running, see the [Tutorial](docs/tutorial.md) for:

- CLI commands and flags for all six modes
- ProposalGPT workflow: from funding call to full proposal in one command
- Grammar Proofreading: style levels, focus areas, and unlimited feedback revision rounds in the UI (up to 3 via CLI)
- Docling parser: enabled by default; use `--no-docling` on the CLI to switch back to pdfplumber
- Per-mode usage examples with real CLI invocations
- Smart hardware recommendations, quality scores, and style profiles
- Self-Reflective RAG: how grading works and how to read the UI expander and CLI table
- Tuning Hybrid RAG and context window settings
