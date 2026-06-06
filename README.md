# ResearchBuddy

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
| **2 — Research Notebook** | NotebookLM-style workspace: upload PDFs, DOCX, TXT, or web pages; chat with grounded citations from your sources; run a 7-agent analysis pipeline (summary, citation verification, knowledge graph, study guide, podcast script); advanced tools: FAQ, literature review, mind map, timeline, study comparison table. |

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
- **Adaptive PDF parsing** — Docling (layout-aware, table extraction) for normal documents; pdfplumber streaming fallback for large PDFs to avoid RAM spikes on constrained machines
- **Long-term memory** — all notebooks and SR sessions persist across restarts
- **Quality scores** — every output self-evaluated with per-dimension scores (1–5)
- **Both UI and CLI** — Streamlit web app (`streamlit run app.py`) and `main.py` CLI

---

## Quick start

### Option A — Local Python

```bash
# 1. Install Ollama and pull a model
ollama pull llama3.1:8b
ollama pull nomic-embed-text   # for Hybrid RAG (Research Notebook)

# 2. Clone and install
git clone https://github.com/sarder-abedin/ResearchBuddy.git
cd ResearchBuddy
pip install -r requirements.txt

# 3. Launch the UI
streamlit run app.py

# 4. Or use the CLI
python main.py --check-system
python main.py --systematic-review --goal "Effect of sleep deprivation on working memory"
python main.py --notebook --notebook-name "My Sources"
```

### Option B — Docker

```bash
docker compose up --build
# Open http://localhost:8501
```

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

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for personal, academic, and non-commercial use.
