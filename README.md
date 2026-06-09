<p align="center">
  <img src="assets/logo.png" alt="BeeSearch logo" width="160">
</p>
Local AI tools for systematic literature review and source-grounded research notebooks — no cloud, no API fees, no data leaving your machine.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-green)](https://ollama.ai)
[![License: PolyForm NC](https://img.shields.io/badge/License-PolyForm%20NC-blue)](https://polyformproject.org/licenses/noncommercial/1.0.0)

---

## Two modes, fully local

| Mode | What it does |
|------|-------------|
| **1 — Systematic Literature Review** | Full PRISMA pipeline: search Google Scholar + arXiv + Semantic Scholar + CrossRef, LLM abstract screener, inclusion/exclusion screening, evidence extraction, narrative synthesis. Generates DOCX/PDF reports, plain-language summaries, citation networks, preprint tracking, trend analysis, evidence maps, and concept drift detection. |
| **2 — Research Notebook** | NotebookLM-style workspace: upload PDFs, DOCX, TXT, or web pages; chat with grounded citations; run a 7-agent analysis pipeline; section-by-section breakdown with audience-level explanations, expert reviewer critique, and claim-based Q&A per section. |

---

## Key features

- **Fully local** — LLMs run on your machine via Ollama; no OpenAI key, no subscriptions
- **Google Scholar** — primary search source for SR (no API key, `scholarly` library)
- **Abstract Screener** — LLM scores each paper 0–100 before formal screening
- **PRISMA 2020 reports** — DOCX and PDF with full Methods → Results → Discussion scaffold
- **Plain-language summaries** — patient (8th-grade), policy brief, press release
- **Citation Network** — interactive Pyvis HTML graph of links between included papers
- **Trend Analysis** — CrossRef field-wide year-by-year publication counts + growing/declining classification
- **Evidence Map** — Plotly Population × Intervention bubble chart
- **Concept Drift** — TF-IDF vocabulary shift across 5-year buckets, pure stdlib
- **Hybrid RAG** — FAISS dense + BM25 sparse, fused with Reciprocal Rank Fusion
- **Self-Reflective RAG** — post-retrieval LLM grader removes irrelevant papers/chunks
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

Make sure Ollama is running on your host machine **before** starting the container. The container needs to reach it over the Docker bridge network.

#### macOS

```bash
# Ollama listens on host.docker.internal automatically on Mac
docker compose -f docker-compose.mac.yml up --build
# Open http://localhost:8501
```

#### Linux (CPU)

```bash
# Use the host gateway IP so the container can reach Ollama on the host
OLLAMA_BASE_URL=http://172.17.0.1:11434 docker compose up --build
# Open http://localhost:8501
```

#### Linux (GPU — NVIDIA)

```bash
OLLAMA_BASE_URL=http://172.17.0.1:11434 docker compose -f docker-compose.gpu.yml up --build
```

#### Windows

```powershell
# On Windows, Docker Desktop uses host.docker.internal just like macOS
docker compose up --build
# Open http://localhost:8501
```

> **Tip — finding the bridge IP on Linux:** Run `ip route show default | awk '{print $3}'` or `docker network inspect bridge | grep Gateway`. Common value is `172.17.0.1`. You can also set `network_mode: host` in `docker-compose.yml` so the container shares the host network directly.

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
python main.py --notebook-timeline <id>         # chronological timeline
python main.py --notebook-study-table <id>      # study comparison table

# 7-agent pipeline
python main.py --notebook-pipeline <id>
python main.py --notebook-pipeline <id> --pipeline-query "What are the main findings?"
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
/timeline       Chronological timeline
/study-table    Study comparison table
/quit           Exit
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
| **Timeline** | Chronological event extraction |
| **Study Comparison** | Side-by-side study table |
| **Pipeline** | 7-agent automated analysis |

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
| `timeline_<id>.md` | Chronological timeline |
| `<name>_sections_<id>.md` | Section-by-section breakdown (CLI `--output`) |

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for personal, academic, and non-commercial use.
