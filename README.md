<p align="center">
  <img src="assets/logo.png" alt="BeeSearch logo" width="160">
</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-green)](https://ollama.ai)
[![License: PolyForm NC](https://img.shields.io/badge/License-PolyForm%20NC-blue)](https://polyformproject.org/licenses/noncommercial/1.0.0)

---

## Table of contents

- [What is BeeSearch?](#what-is-beesearch)
- [The three modes](#the-three-modes)
- [Get started in 3 steps](#get-started-in-3-steps)
- [Choosing an AI model](#choosing-an-ai-model)
- [Using the web interface](#using-the-web-interface)
- [Adjusting AI responses](#adjusting-ai-responses)
- [Settings reference](#settings-reference)
- [For developers](#for-developers)
- [Output files](#output-files)
- [Documentation](#documentation)
- [License](#license)

---

## What is BeeSearch?

BeeSearch is a free AI research tool that runs entirely on your own computer — no internet connection required after setup, no subscriptions, no data leaving your machine. It uses a local AI model to help you search published research papers, analyse documents you upload, and answer research questions with cited sources.

Everything runs locally via [Ollama](https://ollama.ai), an open-source tool that runs AI models on your hardware. BeeSearch automatically picks the right model for your computer.

---

## The three modes

| Mode | What it does |
|------|-------------|
| **1 — Systematic Review** | Searches Google Scholar, arXiv, Semantic Scholar, and CrossRef for papers on your topic; screens them; and produces a formatted review report (Word/PDF) with risk-of-bias ratings, contradiction summaries, citation graphs, trend analysis, and plain-language summaries. |
| **2 — Research Notebook** | Upload your own PDFs, Word docs, or web pages and chat with them. Get cross-document summaries, Q&A, mind maps, knowledge graphs, audio scripts, and more — all grounded in your documents with cited page references. |
| **3 — AI Research Assistant** | Ask a research question in plain English and get a cited answer drawn from published papers — no files to upload, no formal review workflow. |

---

## Get started in 3 steps

The fastest way to run BeeSearch is with Docker — it handles everything automatically (Python, Node.js, the AI model). You only need Docker installed.

**Step 1 — Install Docker**

Download [Docker Desktop](https://docs.docker.com/get-started/get-docker/) for your platform and start it. That's the only prerequisite.

**Step 2 — Download BeeSearch**

```bash
git clone https://github.com/sarder-abedin/BeeSearch.git
cd BeeSearch
cp .env.example .env
```

The `.env` file holds your settings. The defaults work fine to get started.

**Step 3 — Start the app**

```bash
# macOS / Linux
./scripts/start-web.sh

# Windows (Git Bash or WSL)
bash scripts/start-web.sh
```

The first run downloads the AI model (~2 GB) and builds the app — this takes 5–10 minutes. After that, starts take under a minute. The app opens at **http://localhost:8000** automatically. Press **Ctrl-C** to stop.

> **macOS with Ollama already installed natively:**
> Add `OLLAMA_BASE_URL=http://host.docker.internal:11434` to your `.env` file, then run:
> `docker compose -f docker-compose.web.yml up web --build`

> **Want the full stack (React + Streamlit + CLI)?**
> Use `./scripts/start.sh` (Linux/Windows) or `./scripts/start-mac.sh` (macOS) instead.
> Opens **http://localhost:8000** (React); Streamlit is also available at http://localhost:8501.

> **No Docker?** See [Local install (no Docker)](#local-install-no-docker) in the For developers section.

---

## Choosing an AI model

BeeSearch automatically picks the best model for your computer based on available RAM. You can always change it in the **Settings** panel (⚙ button in the top bar).

| RAM | Default model | What to expect |
|-----|--------------|----------------|
| Less than 8 GB | `llama3.2:3b` | Fast responses, works on most laptops |
| 8–16 GB | `llama3.1:8b` | Good quality, recommended for most users |
| 16 GB or more | `mistral-nemo:12b` | Best quality, 128k context window |

To use a different model, pull it with Ollama first:

```bash
# While Docker is running:
docker compose -f docker-compose.web.yml exec ollama ollama pull mistral-nemo:12b

# Or with the full stack:
docker compose exec ollama ollama pull mistral-nemo:12b
```

Then select it in Settings → LLM Model.

---

## Using the web interface

Open **http://localhost:8000** after starting the app.

### Mode 1 — Systematic Review

Click **Systematic Review** on the home page. Enter your research goal, inclusion criteria (e.g. "peer-reviewed studies, human participants"), and exclusion criteria (e.g. "animal studies"). BeeSearch searches multiple databases, screens the results, and produces a full review report.

Output files are saved to the `outputs/` folder in the repository.

### Mode 2 — Research Notebook

Click **Research Notebook** on the home page. Create a notebook, upload your sources (PDFs, Word docs, web URLs), then use the tabs:

| Tab | What it does |
|-----|-------------|
| **Chat** | Ask questions; answers are grounded in your documents with cited page references |
| **Sources** | Upload and manage your documents |
| **Summary** | Cross-document synthesis; drill into any document section by section |
| **FAQ** | Auto-generated Q&A pairs across all your sources |
| **Literature Review** | Academic-style narrative synthesis of your documents |
| **Mind Map** | Visual concept map of the key ideas |
| **Knowledge Graph** | Entity-relationship diagram |
| **Citation Timeline** | Papers cited in your documents, organised by year |
| **Study Comparison** | Side-by-side comparison table of studies |
| **Pipeline** | Runs a 7-step automated analysis (summary → knowledge graph → study guide → podcast script, etc.) |
| **Research Report** | Structured report grounded in your documents, optionally enriched with web or arXiv sources |
| **Explain** | Plain-language explanations of your sources; automatically adapts if you rephrase or say you don't understand |

### Mode 3 — AI Research Assistant

Click **AI Research Assistant** on the home page. Type your question and click **Ask**. BeeSearch searches Google Scholar, arXiv, and Semantic Scholar, reads the results, and writes a cited answer.

---

## Adjusting AI responses

You can change how BeeSearch writes its answers using the **Response Tuning** setting in the Settings panel (⚙ button). This applies to Research Notebook answers, summaries, and all analysis tools.

| Setting | What you get |
|---------|-------------|
| **Precise** | The same question always gives the same answer, word for word. Good for reproducible research. |
| **Focused** *(default)* | Answers stay close to your source material with minimal variation. Recommended for most users. |
| **Balanced** | More natural, varied phrasing while still grounded in your sources. |
| **Creative** | The most varied and exploratory answers. Useful for brainstorming, podcast scripts, and mind maps. |

You can change this setting at any time — it takes effect on your very next question without restarting.

---

## Settings reference

Copy `.env.example` to `.env` before starting. Most users don't need to change anything — BeeSearch picks sensible defaults based on your hardware.

```env
# Address of the Ollama AI server (default works with Docker setup)
OLLAMA_BASE_URL=http://localhost:11434

# Which AI model to use (BeeSearch auto-selects based on your RAM if not set)
OLLAMA_MODEL=llama3.1:8b

# Model used for document search and retrieval
EMBEDDING_MODEL=nomic-embed-text

# How many pages before switching to a lighter PDF parser (lower on machines with < 8 GB RAM)
LARGE_DOC_PAGE_THRESHOLD=50

# Default answer style: precise | focused | balanced | creative
TEMPERATURE_LEVEL=focused

# Vision model for figure captioning in uploaded PDFs (optional)
# Set to an Ollama vision model to caption charts and diagrams automatically.
# Pull first: ollama pull llava:7b   |   then set: VISION_MODEL=llava:7b
# Leave blank (default) to skip figure extraction entirely.
VISION_MODEL=
```

Optional settings for higher API rate limits (leave blank if you don't have these):

```env
SEMANTIC_SCHOLAR_API_KEY=
CROSSREF_EMAIL=your@email.com
```

Optional — LLM observability with [Langfuse](https://langfuse.com) (traces every AI call with prompts, latency, and token counts):

```env
# 1. Start self-hosted Langfuse: docker compose -f docker-compose.langfuse.yml up -d
# 2. Open http://localhost:3000 → create account → Settings → API Keys
# 3. Paste the keys below. Leave blank to disable tracing (no overhead, no errors).
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000
```

---

## For developers

This section covers the CLI, manual installation, running the backend and frontend separately, and other developer tools. If you just want to use BeeSearch, the sections above are all you need.

### Local install (no Docker)

Requires Python 3.10+, Node.js 20+, and [Ollama](https://ollama.ai) installed and running.

```bash
# Pull the AI models
ollama pull llama3.1:8b
ollama pull nomic-embed-text     # required for document search in Research Notebook

# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt

# Start the Streamlit UI
streamlit run app.py

# Or start the web interface (React + backend) — auto-installs npm deps
./scripts/start-react.sh
```

**Windows — PowerShell:**

```powershell
python -m venv .venv
# If you see an execution-policy error, run this once then re-open PowerShell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

> **Optional — Mind Map / Knowledge Graph rendering:** needs the system `dot` binary:
> `apt install graphviz` (Linux), `brew install graphviz` (macOS), or the
> [Graphviz Windows installer](https://graphviz.org/download/).
> Docker users get this automatically — it's already in the image.

### Web interface — manual startup

```bash
# Backend (from repo root, with virtualenv active)
python -m uvicorn backend.app.main:app --reload --port 8000

# Stub out AI calls for UI development (no Ollama needed)
BEESEARCH_MOCK_LLM=1 python -m uvicorn backend.app.main:app --reload --port 8000

# Frontend (in a separate terminal)
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173 — proxies /api/* to the backend automatically
```

### Production build

```bash
cd frontend
npm run build     # type-checks + builds to frontend/dist/
npm run preview   # serves the built output at http://localhost:4173
```

### Tests

```bash
# Full backend suite (root tests/ + backend/tests/), from the repository root
python -m pytest -q

# Frontend
cd frontend
npm run test       # component tests (Vitest)
npm run lint       # ESLint
npx tsc --noEmit   # type-check only
npm run e2e        # Playwright E2E — auto-starts a mock backend and preview server
```

### CLI reference

#### Systematic Literature Review

```bash
# Basic review
python main.py --systematic-review \
  --goal "What is the effect of sleep deprivation on working memory?" \
  --inclusion "Peer-reviewed empirical studies" "Human participants" \
  --exclusion "Animal studies" "Review papers only"

# Generate Word + PDF reports with author info
python main.py --systematic-review --goal "..." \
  --sr-docx --sr-pdf \
  --sr-author "Dr. Jane Smith" --sr-institution "University of Oxford"

# Plain-language summaries (patient / policy / press / all)
python main.py --systematic-review --goal "..." --sr-plain-language all

# Trend analysis + preprint tracking + concept drift
python main.py --systematic-review --goal "..." \
  --sr-trends --sr-preprints --sr-concept-drift

# Print risk-of-bias and contradiction results
python main.py --systematic-review --goal "..." --sr-quality

# Full combined run
python main.py --systematic-review \
  --goal "Efficacy of CBT for treatment-resistant depression" \
  --inclusion "RCTs" "Adult patients" \
  --exclusion "Children" "Open-label studies" \
  --sr-docx --sr-pdf \
  --sr-author "Dr. Smith" --sr-institution "MIT" \
  --sr-plain-language all \
  --sr-trends --sr-preprints --sr-concept-drift
```

#### AI Research Assistant

```bash
# Ask a question, get a literature-grounded answer with citations
python main.py --ask "Does intermittent fasting improve insulin sensitivity in adults?"

# Academic sources only (skip web search)
python main.py --ask "Transformer scaling laws" --no-web
```

#### Research Notebook

```bash
# New notebook
python main.py --notebook --notebook-name "Antibiotic Resistance"

# Open existing notebook
python main.py --notebook --notebook-id <id>

# Add files when opening
python main.py --notebook --notebook-id <id> --files paper.pdf notes.txt

# Document parsing options
python main.py --notebook --files paper.pdf                     # default (Docling)
python main.py --notebook --files paper.pdf --ocr               # Docling + OCR (scanned PDFs)
python main.py --notebook --files paper.pdf --no-docling        # always use pdfplumber
python main.py --notebook --files big.pdf --large-doc-threshold 30

# List all notebooks
python main.py --list-notebooks

# Advanced analysis (one-shot, by notebook ID)
python main.py --notebook-summary <id>
python main.py --notebook-faq <id>
python main.py --notebook-review <id>
python main.py --notebook-audio <id>
python main.py --notebook-mindmap <id>
python main.py --notebook-graph <id>
python main.py --notebook-compare <id> --compare-docs A.pdf B.pdf
python main.py --notebook-timeline <id>
python main.py --notebook-study-table <id>
python main.py --notebook-pipeline <id>

# Response tuning
python main.py --notebook --notebook-id <id> --temperature-level balanced
```

#### Section-by-Section Breakdown

```bash
python cli.py sections <notebook-id> --source paper.pdf
python cli.py sections <notebook-id> --source paper.pdf --level novice
python cli.py sections <notebook-id> --source paper.pdf --review
python cli.py sections <notebook-id> --source paper.pdf --review -o breakdown.md
```

| Flag | Default | Description |
|------|---------|-------------|
| `--source FILENAME` | interactive | Filename substring to match |
| `--level {novice,intermediate,expert}` | `intermediate` | Explanation depth |
| `--review` | off | Add expert reviewer critique per section |
| `-o / --output FILE` | none | Save output to a Markdown file |

#### Interactive notebook slash commands

While in `--notebook` mode:

```
/add <file>            Add a local document
/url <url>             Add a web page
/sources               List all sources
/summary               Cross-document summary
/faq                   FAQ generation
/review                Literature review
/audio                 Audio script + WAV synthesis
/mindmap               Mind map (DOT + PNG + SVG)
/graph                 Knowledge graph
/compare               Compare two sources
/timeline              Citation timeline
/study-table           Study comparison table
/temperature [level]   Show or change response tuning
/quit                  Exit
```

#### Running the CLI inside Docker

```bash
docker compose -f docker-compose.web.yml exec web python main.py --notebook --notebook-name "My Research"
docker compose -f docker-compose.web.yml exec web python main.py --list-notebooks
docker compose -f docker-compose.web.yml exec web bash   # open a shell
```

### MCP Server (optional)

`mcp_servers/research_tools_server.py` exposes BeeSearch's search and notebook tools over the [Model Context Protocol](https://modelcontextprotocol.io), so external MCP clients (Claude Code, Claude Desktop) can call them directly.

```bash
python mcp_servers/research_tools_server.py

# Or with the MCP inspector UI
mcp dev mcp_servers/research_tools_server.py
```

---

## Output files

All outputs are saved to `outputs/`:

| File | Contents |
|------|---------|
| `systematic_review_<id>.md` | Full review report in Markdown |
| `prisma_report_<id>.docx` | Review report as a Word document |
| `prisma_report_<id>.pdf` | Review report as a PDF |
| `summary_patient_<id>.txt` | Plain-language summary for patients |
| `summary_policy_<id>.txt` | Policy brief |
| `summary_press_<id>.txt` | Press release |
| `pipeline_study_guide_<name>.md/docx/pdf` | Study guide from the 7-agent pipeline |
| `pipeline_podcast_<name>.txt` | Podcast script |
| `knowledge_graph_<id>.dot/png/svg` | Knowledge graph |
| `mindmap_<id>.dot/png/svg` | Mind map |
| `citation_timeline_<id>.md` | Papers cited in your documents by year |
| `<name>_sections_<id>.md` | Section-by-section breakdown (CLI `--output`) |

---

## Documentation

Deeper technical documentation:

| Doc | Contents |
|-----|---------|
| [`docs/architecture.md`](docs/architecture.md) | Full pipeline diagrams, state field lists, file map, tech stack |
| [`docs/overview.md`](docs/overview.md) | Condensed architecture overview |
| [`docs/tutorial.md`](docs/tutorial.md) | Step-by-step walkthrough |
| [`docs/FAQ.md`](docs/FAQ.md) | Frequently asked questions |

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for personal, academic, and non-commercial use.
