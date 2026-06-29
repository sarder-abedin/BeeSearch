<p align="center">
  <img src="assets/logo.png" alt="BeeSearch logo" width="160">
</p>
Local AI tools for systematic literature review and source-grounded research notebooks — no cloud, no API fees, no data leaving your machine.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-green)](https://ollama.ai)
[![License: PolyForm NC](https://img.shields.io/badge/License-PolyForm%20NC-blue)](https://polyformproject.org/licenses/noncommercial/1.0.0)

---

## Three modes, fully local

| Mode | What it does |
|------|-------------|
| **1 — Systematic Literature Review** | Full PRISMA pipeline: search Google Scholar + arXiv + Semantic Scholar + CrossRef, LLM abstract screener, inclusion/exclusion screening, PICO evidence extraction, **risk-of-bias (RoB 2 / ROBINS-I)**, **GRADE certainty rating**, **cross-paper contradiction detection**, and narrative synthesis. Generates DOCX/PDF reports, plain-language summaries, citation networks (with **Smart Citation** stance classification and **citation-context** snippets), preprint tracking, trend analysis, evidence maps, and concept drift detection. |
| **2 — Research Notebook** | NotebookLM-style workspace: upload PDFs, DOCX, TXT, or web pages; chat with grounded citations; run a 7-agent analysis pipeline; section-by-section breakdown with audience-level explanations, expert reviewer critique, and claim-based Q&A per section. |
| **3 — AI Research Assistant** | Ask a free-form research question and get an answer grounded in **published literature** with inline, code-rebuilt citations. No documents to upload and no PRISMA workflow — searches Google Scholar, arXiv, Semantic Scholar, and the web, then cites what it actually used. |

---

## Key features

- **Fully local** — LLMs run on your machine via Ollama; no OpenAI key, no subscriptions
- **Google Scholar** — primary search source for SR (no API key, `scholarly` library)
- **Abstract Screener** — LLM scores each paper 0–100 before formal screening
- **PRISMA 2020 reports** — DOCX and PDF with full Methods → Results → Discussion scaffold
- **Plain-language summaries** — general public, policy brief, press release
- **AI Research Assistant (Mode 3)** — free-form question → literature-grounded answer with inline citations, no upload and no PRISMA workflow required (UI tab + `--ask` CLI flag)
- **Risk of Bias** — per-paper Cochrane RoB 2 (trials) / ROBINS-I (observational) assessment across the standard bias domains
- **GRADE certainty** — overall certainty-of-evidence rating (High / Moderate / Low / Very low) across the five GRADE domains, feeding the synthesis narrative
- **Contradiction detection** — flags conflicting findings across included papers with a 0–100 consensus score
- **Smart Citations** — optional LLM classification of each citation-network edge as Supporting / Contrasting / Mentioning (scite.ai-style), colour-coded on the graph
- **Citation Context** — best-effort, open-access-only extraction of the exact sentence where one paper cites another
- **Citation Network** — interactive Pyvis HTML graph of links between included papers, with isolated-paper detection and "gap-finder" suggestions for frequently co-cited papers worth screening
- **Trend Analysis** — CrossRef field-wide year-by-year publication counts + growing/declining classification
- **Evidence Map** — Plotly Population × Intervention bubble chart
- **Concept Drift** — TF-IDF vocabulary shift across 5-year buckets, pure stdlib
- **Hybrid RAG** — FAISS dense + BM25 sparse, fused with Reciprocal Rank Fusion
- **Self-Reflective RAG** — post-retrieval LLM grader removes irrelevant papers/chunks
- **Grounded citations everywhere** — Chat, Literature Review, and Explain all number real source excerpts before the LLM sees them, then rebuild the References list in code from what was actually cited, never from the LLM's own list
- **Research-aware web search** — optional DuckDuckGo search (Chat, Research Report, Explain) re-ranks results toward arXiv/PubMed/IEEE/.edu/.gov and similar research-grade domains
- **7-agent Notebook Pipeline** — ingest → summarize → retrieve → verify_citations → build_kg → study_guide → podcast
- **Section-by-Section Breakdown** — auto-detects document structure (heuristic + LLM fallback); summarises each section at novice / intermediate / expert level; generates claim-based critical questions per section; interactive Q&A anchored to each section
- **Expert Reviewer Mode** — per-section critique modelled on top journal/conference reviews: strengths, weaknesses, limitations, and actionable improvement guidance
- **Adaptive PDF parsing** — Docling (layout-aware, table extraction) with pdfplumber streaming fallback for large PDFs
- **Long-term memory** — all notebooks and SR sessions persist across restarts
- **Quality scores** — every output self-evaluated with per-dimension scores (1–5)
- **Both UI and CLI** — Streamlit web app (`streamlit run app.py`) and `main.py` / `cli.py`

---

## Installation

### Prerequisites

1. **Install [Ollama](https://ollama.ai)** and pull a model:

   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text   # required for Hybrid RAG in Research Notebook
   ```

2. **Clone the repository:**

   ```bash
   git clone https://github.com/sarder-abedin/BeeSearch.git
   cd BeeSearch
   ```

3. **Copy the environment file:**

   ```bash
   cp .env.example .env
   # Edit .env with your preferred model and settings
   ```

> **Optional — Mind Map / Knowledge Graph rendering:** these two Research
> Notebook features render Graphviz DOT diagrams, which needs the system
> `dot` binary in addition to the `graphviz` Python package already in
> `requirements.txt`: `apt install graphviz` (Linux), `brew install graphviz`
> (macOS), or the [Graphviz Windows installer](https://graphviz.org/download/).
> Docker users get this for free — it's already in the image. Without it,
> Mind Map/Knowledge Graph generation fails with a clear error instead of
> crashing the app.

---

### Option A — Virtual Environment

#### macOS / Linux

```bash
# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit UI
streamlit run app.py

# Or use the CLI
python main.py --check-system
python main.py --notebook --notebook-name "My Sources"
```

#### Windows — Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat

pip install -r requirements.txt

streamlit run app.py
```

#### Windows — PowerShell

```powershell
python -m venv .venv

# If you see an execution-policy error, run this once (then re-open PowerShell):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run app.py
```

---

### Option B — Docker

Docker bundles the app **and** an Ollama server together — no separate Ollama install needed.
The same `docker compose up` also starts the React + FastAPI web app (see
[Web App (React + FastAPI)](#web-app-react--fastapi) below) — served by the
same container as Streamlit and the CLI — none of the three interfaces
replaces another.

#### Quick start (all platforms)

Use the platform start script. It builds the images, starts the containers, waits for the app to pass its health-check, then **opens `http://localhost:8501` in your default browser automatically**.

| Platform | Command |
|----------|---------|
| **macOS** (Apple Silicon or Intel) | `./scripts/start-mac.sh` |
| **Linux — CPU** | `./scripts/start.sh` |
| **Linux — GPU (NVIDIA)** | `./scripts/start-gpu.sh` |
| **Windows** (Docker Desktop, Git Bash) | `./scripts/start.sh` |

Add `--build` to force a full image rebuild:

```bash
./scripts/start.sh --build
```
The browser opens at `http://localhost:8501` once the container is healthy (~30–60 s on first run while Docker downloads images and builds layers). On Ctrl-C the script shuts the containers down cleanly.

Alternatively, run the Docker container without automatic Streamit running on the default browser with full build (this is the example of macOS. But, check the .yml file in the repo for other systems):
```bash
docker compose -f docker-compose.mac.yml up --build   
```
After that, copy the URL and paste it to the browser to run the Streamit web-app.  

---

#### Manual Docker commands (advanced)

If you prefer running Docker commands directly:

```bash
# Build and start everything: Ollama + Streamlit/CLI/FastAPI/React app
docker compose up --build

# Start without rebuilding (subsequent runs)
docker compose up

# Stop and remove containers
docker compose down

# Pull a different model while running
docker compose exec ollama ollama pull mistral-nemo:12b
```

This starts two long-running containers: `research-ollama`, and `research-app`
— which serves Streamlit (`http://localhost:8501`), the CLI, and the FastAPI
backend together with the React frontend it serves (`http://localhost:8000`,
interactive docs at `/docs`). Override the host ports with `APP_PORT` /
`BACKEND_PORT` in `.env` if 8501/8000 are already in use.

> **Linux bridge IP** — if the app can't reach Ollama, find the bridge IP with:
> `ip route show default | awk '{print $3}'` (common value: `172.17.0.1`), then:
> ```bash
> OLLAMA_BASE_URL=http://172.17.0.1:11434 docker compose up --build
> ```
> Alternatively, set `network_mode: host` in `docker-compose.yml`.

---

#### Running the CLI inside Docker

Once the containers are running you can execute any `main.py` or `cli.py` command **inside the running app container** without leaving Docker:

```bash
# Open an interactive Research Notebook session
docker compose exec app python main.py --notebook --notebook-name "My Research"

# Run a Systematic Literature Review
docker compose exec app python main.py --systematic-review \
  --goal "Effect of sleep deprivation on working memory" \
  --inclusion "Human participants" "Peer-reviewed" \
  --exclusion "Animal studies"

# List all saved notebooks
docker compose exec app python main.py --list-notebooks

# Section-by-section breakdown (cli.py)
docker compose exec app python cli.py sections <notebook-id> --source paper.pdf

# Open a shell inside the container
docker compose exec app bash
```

> **Tip:** Replace `app` with `research-app` if you're not using `docker compose` from the project root (the container name is `research-app` as set in `docker-compose.yml`).

---

## Web App (React + FastAPI)(Under development)

A React + TypeScript frontend and a FastAPI backend sit alongside the
Streamlit UI and CLI, exposing the same functionality over a REST API.

**Current scope:** Mode 1 (Systematic Review) and Mode 3 (AI Research
Assistant) are fully covered. Mode 2 (Research Notebook) covers the core
workflow — create/rename/delete notebooks, upload sources, and grounded
chat with citations and Self-Reflective RAG status — plus the 7-agent
pipeline, advanced tools (FAQ, mind map, knowledge graph, etc.), and the
Explain tab. Only the Research Report workflow is still Streamlit/CLI-only.
The Streamlit app and CLI keep working exactly as documented above either way.

### Requirements

- Ollama + a pulled model, and the Python dependencies from
  [Prerequisites](#prerequisites) above
- [Node.js](https://nodejs.org) 20+ and npm (bundled with Node)

### Run the backend (FastAPI)

```bash
# From the repository root, same virtual environment / dependencies as the
# CLI and Streamlit app (pip install -r requirements.txt)
python -m uvicorn backend.app.main:app --reload --port 8000
```

The API is now served at `http://localhost:8000` (interactive docs at
`http://localhost:8000/docs`).

To explore the UI without Ollama running, set `BEESEARCH_MOCK_LLM=1` to stub
out all LLM/search calls with canned responses (dev/test only):

```bash
BEESEARCH_MOCK_LLM=1 python -m uvicorn backend.app.main:app --reload --port 8000
```

### Run the frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dev server proxies all `/api/*` requests to
the backend on port 8000 automatically (`frontend/vite.config.ts`) — no
`.env` file or extra configuration needed.

### Production build

```bash
cd frontend
npm run build     # type-checks + builds to frontend/dist/
npm run preview   # serves the built dist/ at http://localhost:4173
```

`preview` serves whatever was last built — re-run `build` after making
changes before previewing.

### Run with Docker

The backend and frontend are built into the same container as Streamlit and
the CLI — see [Option B — Docker](#option-b--docker) above for the full
command reference.

```bash
docker compose up --build
```

- Backend + frontend (FastAPI serves the built React SPA at `/`, alongside
  its `/api/*` routes): `http://localhost:8000`, docs at
  `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

The root `Dockerfile` builds the React app in a `node:20-alpine` stage
(`npm run build`) and copies the static output into the final image, so
there's no separate frontend container, no nginx process, and no
`VITE_API_BASE_URL` or other build-time configuration needed.

### Tests

```bash
# Backend — full suite (root tests/ + backend/tests/), from the repository root
python -m pytest -q

# Frontend, from frontend/
npm run test       # component tests (Vitest)
npm run lint       # ESLint
npx tsc --noEmit   # type-check only
npm run e2e        # Playwright E2E -- auto-starts the backend (mock LLM) and a preview server
```

---

## MCP Server (optional)

`mcp_servers/research_tools_server.py` exposes a subset of BeeSearch's search
and notebook tools (arXiv, Semantic Scholar, CrossRef, web search, notebook
RAG query) over the [Model Context Protocol](https://modelcontextprotocol.io),
so external MCP clients — Claude Code, Claude Desktop — can call them
directly. It's already registered in `.mcp.json` at the repo root, and uses
the same `mcp` dependency pulled in by `requirements.txt`.

```bash
# Run directly
python mcp_servers/research_tools_server.py

# Or with the MCP inspector UI
mcp dev mcp_servers/research_tools_server.py
```

This server is independent of the Streamlit app and CLI — it doesn't share
their session state, and runs fine whether or not either is also running.

---

## CLI Reference

### Systematic Literature Review

```bash
# Basic review
python main.py --systematic-review \
  --goal "What is the effect of sleep deprivation on working memory?" \
  --inclusion "Peer-reviewed empirical studies" "Human participants" \
  --exclusion "Animal studies" "Review papers only"

# Generate DOCX + PDF reports with author info
python main.py --systematic-review --goal "..." \
  --sr-docx --sr-pdf \
  --sr-author "Dr. Jane Smith" --sr-institution "University of Oxford"

# Plain-language summaries (patient / policy / press / all)
python main.py --systematic-review --goal "..." --sr-plain-language all

# Trend analysis + preprint tracking + concept drift
python main.py --systematic-review --goal "..." \
  --sr-trends --sr-preprints --sr-concept-drift

# Print risk-of-bias (RoB 2 / ROBINS-I), GRADE certainty, and contradictions
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

### AI Research Assistant

```bash
# Ask a free-form research question, get a literature-grounded answer with citations
python main.py --ask "Does intermittent fasting improve insulin sensitivity in adults?"

# Academic sources only (skip the web search)
python main.py --ask "Transformer scaling laws" --no-web
```

### Research Notebook

```bash
# New notebook
python main.py --notebook --notebook-name "Antibiotic Resistance"

# Open existing notebook
python main.py --notebook --notebook-id <id>

# Add files when opening
python main.py --notebook --notebook-id <id> --files paper.pdf notes.txt

# Document parsing options
python main.py --notebook --files paper.pdf          # default: Docling (layout-aware)
python main.py --notebook --files paper.pdf --ocr    # Docling + OCR (scanned PDFs)
python main.py --notebook --files paper.pdf --no-docling  # always use pdfplumber
python main.py --notebook --files big.pdf --large-doc-threshold 30  # custom page threshold

# List all notebooks
python main.py --list-notebooks

# Advanced analysis (one-shot)
python main.py --notebook-summary <id>          # cross-document summary
python main.py --notebook-faq <id>              # FAQ generation
python main.py --notebook-review <id>           # literature review
python main.py --notebook-audio <id>            # audio script + WAV
python main.py --notebook-mindmap <id>          # mind map (DOT + PNG + SVG)
python main.py --notebook-graph <id>            # knowledge graph
python main.py --notebook-compare <id> --compare-docs A.pdf B.pdf
python main.py --notebook-timeline <id>         # citation timeline (add --enrich-abstracts for S2 abstracts)
python main.py --notebook-study-table <id>      # study comparison table

# 7-agent pipeline
python main.py --notebook-pipeline <id>
python main.py --notebook-pipeline <id> --pipeline-query "What are the main findings?"

# Response tuning — precise / focused (default) / balanced / creative
python main.py --notebook --notebook-id <id> --temperature-level balanced
```

### Section-by-Section Breakdown (CLI)

```bash
# Basic — intermediate-level breakdown of a source in a notebook
python cli.py sections <notebook-id> --source paper.pdf

# Choose explanation level
python cli.py sections <notebook-id> --source paper.pdf --level novice
python cli.py sections <notebook-id> --source paper.pdf --level expert

# Include expert reviewer critique (strengths / weaknesses / limitations / improvements)
python cli.py sections <notebook-id> --source paper.pdf --review

# Save the full breakdown to a Markdown file
python cli.py sections <notebook-id> --source paper.pdf --review -o breakdown.md

# Interactive: prompts for source selection if --source is omitted
python cli.py sections <notebook-id>
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source FILENAME` | interactive | Filename substring to match against notebook sources |
| `--level {novice,intermediate,expert}` | `intermediate` | Explanation depth |
| `--review` | off | Add expert reviewer critique per section |
| `-o / --output FILE` | none | Save Markdown output to this file |

### Interactive notebook slash commands

Once in `--notebook` mode, type:

```
/add <file>     Add a local document
/url <url>      Add a web page
/sources        List all sources
/summary        Cross-document summary
/faq            FAQ generation
/review         Literature review
/audio          Audio script + WAV synthesis
/mindmap        Mind map (DOT + PNG + SVG)
/graph          Knowledge graph
/compare        Compare two sources
/timeline       Citation timeline
/study-table    Study comparison table
/temperature [level]   Show or change response tuning (see below)
/quit           Exit
```

---

## Response tuning — Temperature levels

Every LLM call in the Research Notebook (Chat answers, Explain, and all
advanced tools — summary, FAQ, literature review, mind map, audio script,
knowledge graph, compare, citation timeline, study table, and the 7-agent
pipeline) is individually tuned: factual answers stay close to your
sources, while the Explain "storyteller" is a bit more exploratory.
**Temperature level** shifts every one of those tunings up or down together,
without changing their relative balance — and it applies on your very next
question, mid-session, with no restart needed.

| Level | What you get |
|-------|-------------|
| **Precise** | Fully deterministic — the same question against the same sources always produces the same answer, word for word. Best for exact reproducibility. |
| **Focused** *(default)* | BeeSearch's original tuning — factual answers, summaries, and extractions stay close to your source material, with minimal wording variation. |
| **Balanced** | More natural, varied phrasing across answers, summaries, and explanations, while still grounded in your sources. |
| **Creative** | The most varied and exploratory phrasing — useful for brainstorming, podcast-style explanations, and mind maps. Written answers may diverge further from exact source wording. |

Chunk/citation grading and faithfulness checks always stay fully
deterministic, no matter which level you pick — only the wording of
generated text is affected.

**In the UI:** use the **Response Tuning** control in the sidebar (under
"LLM Model"). It applies to your next message or generation in any
Research Notebook tab — Chat, Explain, and every advanced tool.

**In the CLI:**

```bash
# Set at session start
python main.py --notebook --notebook-id <id> --temperature-level creative

# Or change anytime during a session
/temperature              # show the current level and what each one means
/temperature balanced     # switch — applies to your next question
```

---

## Research Notebook — UI features

| Tab | What it does |
|-----|-------------|
| **Chat** | Source-grounded conversation with inline citations |
| **Sources** | Upload / manage PDFs, DOCX, TXT, web pages |
| **Summary** | Cross-document synthesis + **Section-by-Section Breakdown** (per-source drill-down at novice / intermediate / expert level, expert reviewer critique, claim-based questions, interactive per-section Q&A) |
| **FAQ** | Auto-generated Q&A pairs across all sources |
| **Literature Review** | Academic-style narrative synthesis |
| **Mind Map** | Visual concept map (DOT + PNG + SVG) |
| **Knowledge Graph** | Entity-relationship graph |
| **Citation Timeline** | Cited works by year, parsed from each source's bibliography, with one-line gists (optional Semantic Scholar abstract enrichment) |
| **Study Comparison** | Side-by-side study table |
| **Pipeline** | 7-agent automated analysis |
| **Research Report** | Structured report grounded in your sources, optionally augmented with arXiv/Semantic Scholar papers and/or a DuckDuckGo web search |
| **Explain** | Conversational, audience-tunable explanations of your sources with numbered citations; automatically searches online when your documents don't sufficiently cover a question; detects when you repeat or rephrase a question (or say "I don't understand") and responds with a different explanation style plus an interactive concept map |

---

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text
NUM_CTX=8192

# PDFs with more pages than this switch from Docling to pdfplumber
# Lower on machines with < 8 GB RAM (e.g. 20 or 30). Set to 0 to always use Docling.
LARGE_DOC_PAGE_THRESHOLD=50

# Research Notebook response tuning: precise | focused | balanced | creative
# (sidebar "Response Tuning" control / CLI /temperature override this per session)
TEMPERATURE_LEVEL=focused
```

---

## Hardware requirements

| RAM | Recommended model |
|-----|------------------|
| < 8 GB | `llama3.2:3b` |
| 8–16 GB | `llama3.1:8b` |
| 16+ GB | `mistral-nemo:12b` (128k context) |

Run `python main.py --check-system` for a hardware-aware recommendation.

---

## Output files

All outputs are saved to `outputs/`:

| File | Contents |
|------|---------|
| `systematic_review_<id>.md` | Full SR report in Markdown |
| `prisma_report_<id>.docx` | PRISMA 2020 Word document |
| `prisma_report_<id>.pdf` | PRISMA 2020 PDF |
| `summary_patient_<id>.txt` | Patient plain-language summary |
| `summary_policy_<id>.txt` | Policy brief |
| `summary_press_<id>.txt` | Press release |
| `pipeline_study_guide_<name>.md/docx/pdf` | Notebook study guide |
| `pipeline_podcast_<name>.txt` | Podcast script |
| `knowledge_graph_<id>.dot/png/svg` | Knowledge graph |
| `mindmap_<id>.dot/png/svg` | Mind map |
| `citation_timeline_<id>.md` | Citation timeline (cited works by year) |
| `<name>_sections_<id>.md` | Section-by-section breakdown (CLI `--output`) |

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for personal, academic, and non-commercial use.
