# ResearchBuddy

A focused, fully local research assistant with two powerful modes:
**Systematic Review** and **Research Notebook** — powered by Ollama LLMs,
hybrid RAG (FAISS + BM25), and a Streamlit UI. No cloud API keys required.

---

## Modes

### Systematic Review
PRISMA-style automated literature review with a 6-node LangGraph pipeline:

| Node | What it does |
|---|---|
| Query Generation | Expands your research question into targeted search queries |
| Literature Search | Searches arXiv + Semantic Scholar + CrossRef |
| Screening | Filters papers by relevance (title/abstract scoring) |
| Evidence Extraction | Pulls key findings, methods, and limitations |
| Synthesis | Writes a structured PRISMA review with citations |
| Evaluation | Self-grades for coverage, rigor, and citation accuracy |

### Research Notebook
NotebookLM-style interactive document workspace with 11 tabs:

| Tab | Feature |
|---|---|
| Chat | Grounded Q&A with hybrid RAG retrieval |
| Summary | Auto-generated executive summary |
| FAQ | AI-generated frequently asked questions |
| Lit Review | Mini literature review from uploaded sources |
| Mind Map | Visual concept map (Mermaid) |
| Audio | Podcast-style audio script |
| Compare | Side-by-side source comparison |
| Graph | Knowledge graph of entities and relationships |
| Timeline | Chronological event extraction |
| Study Table | Key concepts table for learning |
| Pipeline | 7-agent deep analysis pipeline |

---

## Quick Start (Docker — Recommended)

**Prerequisites:** Docker and Docker Compose installed.

```bash
# Clone the repo
git clone https://github.com/sarder-abedin/researchbuddy.git
cd researchbuddy

# Copy environment file and customise if needed
cp .env.example .env

# Start everything (pulls Ollama model on first run)
docker compose up --build
```

Open **http://localhost:8501** in your browser.

The first run downloads the default model (`llama3.2:3b`, ~2 GB). Subsequent
starts reuse the cached weights.

### Change the model

```bash
# Edit .env
OLLAMA_MODEL=mistral-nemo:12b

# Or pull any model into the running container
docker compose exec ollama ollama pull phi4:14b
```

### Stop

```bash
docker compose down          # stop containers (data preserved)
docker compose down -v       # stop and delete model weights volume
```

---

## Local Setup (without Docker)

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com/download) running locally.

```bash
# Install dependencies
pip install -r requirements.txt

# Pull models
ollama pull llama3.2:3b
ollama pull nomic-embed-text   # for hybrid RAG embeddings

# Configure
cp .env.example .env           # edit OLLAMA_BASE_URL if needed

# Run
streamlit run app.py
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Chat/generation model |
| `NUM_CTX` | `32768` | Context window size (tokens) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Local embedding model |
| `HYBRID_TOP_K` | `8` | Chunks returned by hybrid RAG |
| `CHROMA_PERSIST_DIR` | `outputs/chroma` | Embedding cache location |
| `SEMANTIC_SCHOLAR_API_KEY` | *(empty)* | Optional — increases rate limits |
| `CROSSREF_EMAIL` | `researcher@example.com` | Polite pool identifier |

---

## Hardware Recommendations

| RAM | GPU VRAM | Recommended Model |
|---|---|---|
| 8 GB | None (CPU) | `llama3.2:3b` |
| 16 GB | 4–6 GB | `mistral-nemo:12b` or `phi4:14b` |
| 32 GB | 8+ GB | `llama3.1:70b` (Q4) |

For embedding, `nomic-embed-text` runs fine on CPU. Full hybrid RAG
(FAISS + BM25 + ChromaDB) requires ~500 MB RAM beyond the LLM.

---

## Project Structure

```
researchbuddy/
├── app.py                        # Streamlit entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.py               # Pydantic BaseSettings
│   └── hardware.py               # Hardware detection + model recommendations
├── agents/
│   ├── systematic_review_graph.py  # 6-node PRISMA LangGraph
│   ├── systematic_review_nodes.py
│   ├── systematic_review_state.py
│   ├── notebook_graph.py           # Q&A LangGraph (retrieve→answer→save)
│   ├── notebook_nodes.py
│   ├── notebook_state.py
│   ├── notebook_pipeline_graph.py  # 7-agent deep analysis
│   ├── notebook_pipeline_nodes.py
│   ├── notebook_pipeline_state.py
│   ├── notebook_memory.py          # SQLite-backed session memory
│   ├── notebook_advanced.py        # Advanced notebook operations
│   ├── self_reflective_rag.py      # Grade, rewrite, reflect
│   ├── feedback_agent.py           # User feedback loop
│   └── eval_nodes.py               # Evaluation nodes
├── tools/
│   ├── search_tools.py             # arXiv + Semantic Scholar + CrossRef
│   ├── hybrid_store.py             # FAISS + BM25 + RRF hybrid RAG
│   ├── embeddings.py               # OllamaEmbedder
│   ├── document_tools.py           # PDF / DOCX / TXT processor
│   ├── docling_processor.py        # Advanced Docling parser
│   ├── vector_store.py             # ChromaDB manager
│   ├── citation_tools.py           # BibTeX + RIS export
│   ├── export_tools.py             # DOCX + PDF generation
│   ├── session_db.py               # SQLite session storage
│   └── ...
├── ui/
│   ├── landing.py                  # Mode selection landing page
│   ├── sidebar.py                  # Settings sidebar
│   ├── helpers.py                  # Shared UI render helpers
│   ├── theme.py                    # Global CSS + badges
│   └── tabs/
│       ├── systematic_review.py      # Full PRISMA workflow UI
│       └── notebook.py               # 11-tab notebook UI
└── projects/
    ├── mode1_systematic_review.py  # Mode orchestrator
    └── mode2_notebook.py           # Mode orchestrator
```

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Ollama (local, any model) |
| Agent orchestration | LangGraph (StateGraph) |
| Dense retrieval | FAISS + OllamaEmbedder |
| Sparse retrieval | BM25 (rank-bm25) |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) |
| Embedding cache | ChromaDB (persistent) |
| Academic search | arXiv API, Semantic Scholar, CrossRef |
| Web search | DuckDuckGo (ddgs) |
| PDF parsing | pdfplumber + Docling (optional) |
| Session storage | SQLite |
| Export | python-docx, reportlab |
| Containerisation | Docker + docker-compose |

---

## License

 GNU Affero General Public License version 3 (AGPL-3.0)
