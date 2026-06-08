# BeeSearch — Architecture

## System Overview

BeeSearch is a **2-mode, local-first AI research system** built on LangGraph state machines, Ollama LLMs, and Hybrid RAG. All computation runs locally — no cloud LLM, no paid API.

```
┌──────────────────────────────────────────────────────────────┐
│                       User Interfaces                        │
│   Streamlit web UI (app.py)       CLI terminal (main.py)     │
│   Landing page → select mode      --systematic-review /      │
│   (lazy: only selected mode       --notebook                 │
│    code is imported)                                         │
└────────────────────┬─────────────────────────────────────────┘
                     │
           ┌─────────┴─────────┐
           │                   │
   ┌───────▼──────┐   ┌────────▼────────┐
   │   Mode 1     │   │    Mode 2       │
   │  Systematic  │   │  Research       │
   │  Literature  │   │  Notebook       │
   │  Review      │   │                 │
   │  projects/   │   │  projects/      │
   │  mode1_*     │   │  mode2_*        │
   └───────┬──────┘   └────────┬────────┘
           │                   │
           └─────────┬─────────┘
                     │
         ┌───────────▼───────────┐
         │   LangGraph graphs    │
         │   agents/*.py         │
         └───────────┬───────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
 ┌────▼─────┐ ┌──────▼──────┐ ┌───▼──────────────┐
 │ Hybrid   │ │  Academic   │ │   Memory         │
 │ RAG      │ │  Search     │ │  (SQLite WAL)    │
 │ FAISS    │ │  Google     │ │   outputs/       │
 │ BM25     │ │  Scholar    │ │  memory/         │
 │ ChromaDB │ │  arXiv      │ │  sessions.db     │
 │ RRF      │ │  Semantic   │ └──────────────────┘
 │          │ │  Scholar    │
 │ Mode 2   │ │  CrossRef   │
 │ (docs)   │ │             │
 └────┬─────┘ └──────┬──────┘
      │              │
      └──────┬───────┘
             │
 ┌───────────▼──────────────────────────────────┐
 │   Self-Reflective RAG  (agents/              │
 │   self_reflective_rag.py)                    │
 │                                              │
 │   grade_chunks() — Mode 2 (Notebook)        │
 │     batch LLM call grades retrieved chunks  │
 │     < 3 pass → rewrite query + cycle 2      │
 │                                              │
 │   grade_papers() — Mode 1 (SR)              │
 │     batch LLM call grades retrieved papers  │
 │     one-pass filter (no cycle)              │
 │                                              │
 │   Fallback: any failure → all items kept    │
 └───────────────────────┬──────────────────────┘
                         │
             ┌───────────▼──────────┐
             │   Ollama LLM         │
             │   (main reasoning)   │
             └──────────────────────┘
```

---

## Mode 1: Systematic Literature Review

A stateless linear PRISMA pipeline with a suite of on-demand post-synthesis analysis tools. Results are shown in the UI and available for download in Markdown, DOCX, and PDF.

### Core pipeline

```
START
  │
  ▼
[query_generation]
  │  • LLM generates 4–6 varied search queries (broad + narrow + population)
  │  • JSON array parsed from LLM response
  │
  ▼
[literature_search]
  │  • Fans out across 4 sources:
  │      Google Scholar (scholarly, no API key — primary)
  │      arXiv (free preprints)
  │      Semantic Scholar (peer-reviewed, citation counts)
  │      CrossRef (DOI resolution)
  │  • Deduplicates by normalised title slug
  │  • Sorts: peer-reviewed first, then by citation count desc
  │  • abstract_screener runs here: LLM scores each paper 0–100 against
  │    inclusion/exclusion criteria before formal screening
  │  • screener_scores stored in state for UI display
  │  • grade_papers() (Self-Reflective RAG) filters irrelevant papers
  │
  ▼
[screening]
  │  • LLM evaluates each paper against inclusion/exclusion criteria
  │  • Decision: INCLUDE / EXCLUDE with reason
  │  • Records exclusion_reason on excluded papers
  │
  ▼
[evidence_extraction]
  │  • For each included paper (up to 20): one LLM call per paper
  │  • Extracts: study_design, sample_size, key_finding,
  │              quality (High/Medium/Low), relevance_score (1–5)
  │  • Assigns citation_key (<author><year> format)
  │
  ▼
[synthesis]
  │  • Builds prisma_flow dict: identified/screened/eligibility/included/excluded
  │  • LLM call → narrative_synthesis, key_themes, research_gaps,
  │               limitations, conclusion (all inline-cited)
  │
  ▼
[sr_eval]
  │  • Self-evaluation: search_comprehensiveness, screening_rigor,
  │    evidence_quality, synthesis_depth, gap_identification (each 1–5)
  │
 END
```

### On-demand post-synthesis tools

Triggered from the UI (button click) or CLI flags. All are independent and non-blocking — they never re-run the core pipeline.

```
[abstract_screener]          tools/abstract_screener.py
  │  • LLM assigns 0–100 relevance score to each paper
  │  • Verdict: include (≥60) / uncertain (40–59) / exclude (<40)

[citation_network]           tools/citation_network.py
  │  • Queries Semantic Scholar /paper/{id}/references for each included paper
  │  • Builds ego networkx DiGraph (nodes = included papers; edges = citations
  │    between them — ego-only scope, no external expansion)
  │  • Renders interactive Pyvis HTML for the UI

[preprint_tracker]           tools/preprint_tracker.py
  │  • CrossRef title search per included paper
  │  • Status: journal | published (was arXiv) | preprint | retracted
  │  • Flags retraction notices from CrossRef update-policy / relation fields

[trend_analyzer]             tools/trend_analyzer.py
  │  • CrossRef facet API → field-wide publication counts per year
  │  • Supplemented by Semantic Scholar if CrossRef returns < 30 records
  │  • Trend classification: growing | stable | declining | insufficient data

[evidence_map]               tools/evidence_map.py
  │  • Aggregates evidence_table into Population × Intervention cells
  │  • Bubble size = study count, colour = average quality (green/amber/red)
  │  • Primary: Plotly interactive HTML; fallback: matplotlib PNG

[concept_drift]              tools/concept_drift.py
  │  • Groups raw_papers into 5-year buckets
  │  • TF-IDF keyword extraction per bucket (stdlib only, no scikit-learn)
  │  • Classifies terms: rising (+3 rank points) | declining (−3) | stable
  │  • Optional LLM narrative of conceptual shifts

[prisma_report]              tools/prisma_report.py
  │  • DOCX: python-docx — title page, abstract, PRISMA 2020 sections,
  │          evidence table, references (saved to outputs/prisma_report_<id>.docx)
  │  • PDF:  reportlab — same structure, pure-Python, no LibreOffice required
  │          (saved to outputs/prisma_report_<id>.pdf)

[plain_language]             tools/plain_language.py
  │  • patient  — 8th-grade reading level, 4 plain paragraphs, ~350 words
  │  • policy   — 1-page Markdown brief with recommendations (policy-makers)
  │  • press    — inverted-pyramid press release with headline + quote
```

**State type:** `SystematicReviewState` (`agents/systematic_review_state.py`)

State fields: `research_question`, `inclusion_criteria`, `exclusion_criteria`, `model_name`, `num_ctx`, `session_id`, `search_queries`, `raw_papers`, `screener_scores`, `included_papers`, `excluded_papers`, `evidence_table`, `narrative_synthesis`, `key_themes`, `research_gaps`, `conclusion`, `limitations`, `prisma_flow`, `eval_result`, `rag_reflection_info`, `progress_pct`, `status_detail`, `errors`, `preprint_tracking`, `citation_graph_html`, `trend_data`, `evidence_map_data`, `concept_drift_data`

**Output tabs (UI):** Synthesis | Evidence Table | Discovery | Trends & Analysis | Export & Reports

**CLI:**
```bash
# Basic run
python main.py --systematic-review \
  --goal "Effect of sleep deprivation on working memory" \
  --inclusion "Peer-reviewed empirical studies" "Human participants" \
  --exclusion "Animal studies" "Review papers only"

# With all post-run tools
python main.py --systematic-review \
  --goal "Mindfulness-based interventions for anxiety" \
  --sr-docx --sr-pdf \
  --sr-plain-language all \
  --sr-trends --sr-preprints --sr-concept-drift \
  --sr-author "A. Researcher" --sr-institution "Example University"
```

---

## Mode 2: Research Notebook

Two parallel capabilities sharing a common tab in the UI.

### 2a — Q&A Chat (NotebookState)

Single-turn graph invocation per user message. Conversation continuity lives in `NotebookMemory` (SQLite).

```
START
  │
  ▼
[retrieve]
  │  • HybridStore.search() over ingested notebook documents
  │  • FAISS + BM25 + RRF → top-K chunks
  │  • grade_chunks() (Self-Reflective RAG) filters irrelevant chunks
  │  • If < 3 pass: rewrite query + retry (max 2 cycles)
  │  • BM25-only fallback if embedding model not pulled
  │
  ▼
[answer]
  │  • LLM synthesises answer grounded in retrieved chunks
  │  • Inline citations [1], [2], … to source documents
  │  • Proposes 2–3 follow-up questions
  │
  ▼
[save]
  │  • NotebookMemory.add_turn(role="user", content=…)
  │  • NotebookMemory.add_turn(role="assistant", content=…)
  │  • Updates concepts_covered list
  │
 END  (NotebookState → notebooks + notebook_chunks tables in sessions.db)
```

**State type:** `NotebookState` (`agents/notebook_state.py`)

**Memory:** `outputs/memory/sessions.db` — `notebooks` table (meta + conversation) + `notebook_chunks` table (one row per chunk)

### 2b — 7-Agent Pipeline (NotebookPipelineState)

Processes uploaded documents into a structured study package.

```
START
  │
  ▼
[ingest]
  │  • Docling (default) → layout-aware parsing, table extraction,
  │    PPTX/XLSX/HTML/image support → raw text + chunks
  │  • Auto-switch: PDFs > LARGE_DOC_PAGE_THRESHOLD pages (default 50)
  │    use DocumentProcessor instead to avoid ~500 MB Docling ML models
  │  • DocumentProcessor (explicit fallback: --no-docling) → pdfplumber /
  │    python-docx / plain read → page-by-page streaming, low RAM
  │  • OllamaEmbedder → FAISS + ChromaDB cache
  │  • BM25Okapi index built from chunks
  │
  ▼
[summarize]
  │  • LLM generates per-document summaries
  │  • Cross-document synthesis: common themes, contradictions, takeaways
  │
  ▼
[retrieve]
  │  • HybridStore.search() for key concepts and themes
  │  • grade_chunks() filters irrelevant chunks
  │
  ▼
[verify_citations]
  │  • Verifies 5–8 claims against source material
  │  • Confidence per claim: HIGH / MEDIUM / LOW
  │
  ▼
[build_kg]
  │  • Entity–relationship graph extracted from documents
  │  • Graphviz DOT → PNG + SVG
  │
  ▼
[generate_study_guide]
  │  • Key concepts, glossary, Q&A pairs, summary
  │  • Outputs: Markdown + DOCX + PDF
  │
  ▼
[generate_podcast]
  │  • Two-speaker dialogue (HOST: Alex, EXPERT: Dr. Jordan)
  │  • Output: TXT script
  │
 END  (NotebookPipelineState)
```

**State type:** `NotebookPipelineState` (`agents/notebook_pipeline_state.py`)

### Advanced analysis (one-shot tools)

Available from CLI flags and UI tab buttons.

| Feature | What it produces |
|---------|-----------------|
| Cross-document summary | Common themes, contradictions, key takeaways |
| FAQ | 4–16 grounded Q&A pairs |
| Literature review | Formal academic review Markdown |
| Audio script | 300-word script TXT + WAV via pyttsx3 |
| Mind map | DOT + PNG + SVG |
| Knowledge graph | DOT + PNG + SVG |
| Source comparison | Side-by-side Markdown table |
| Timeline | Chronological events table |
| Study comparison | Research method/sample/findings table |

---

## Hybrid RAG Pipeline

```
Document (PDF / DOCX / TXT / HTML / web page)
        │
        ▼
  Parser selection (tools/document_tools.py — get_processor())
  ├── _peek_pdf_pages()  counts pages cheaply before committing
  ├── PDF ≤ LARGE_DOC_PAGE_THRESHOLD pages (default 50, env-configurable)
  │     └── Docling  →  layout-aware parsing, table extraction,
  │                       PPTX/XLSX/HTML/image support
  ├── PDF >  LARGE_DOC_PAGE_THRESHOLD pages  (auto RAM guard)
  │     └── DocumentProcessor  →  pdfplumber page-by-page streaming,
  │                                no ~500 MB ML models loaded
  └── --no-docling flag  →  always DocumentProcessor
  Both paths: clean_text → chunk_text (chunk_size=800, overlap=150)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  OllamaEmbedder                     BM25Okapi
  (tools/embeddings.py)              (rank-bm25)
  • Batched POST /api/embed           • Tokenised chunks
  • Ollama embedding model            • Precomputed IDF weights
  • 768-dim vectors (default)         • No external model needed
        │                                  │
        ▼                                  │
  FAISS IndexFlatIP          ◄─────────────┘
  (in-memory, per session)        Both indexes live in HybridStore
        │                         (tools/hybrid_store.py)
        ▼  (at query time)
  HybridStore.search(query, top_k)
  ├── embed query → FAISS → top-2k dense results (ranked by cosine sim)
  ├── tokenise   → BM25  → top-2k sparse results (ranked by BM25 score)
  └── Reciprocal Rank Fusion (k=60):
        score[doc_id] += 1 / (60 + rank + 1)  for each retriever
        sort by score → top-K unique chunks
        │
        ▼
  ChromaDB (outputs/chroma_db/)
  • Persistent embedding cache — avoids re-embedding same doc
  • On second upload: embeddings loaded from cache, FAISS rebuilt
  • MD5 cache invalidation: content_md5 (MD5 of first 50 000 chars) stored
    per document; if hash differs on re-upload, stale embeddings are
    deleted from ChromaDB before re-embedding (no manual --clear-store needed)
        │
        ▼
  Top-K chunks → Self-Reflective Grading (agents/self_reflective_rag.py)
  • Single batched LLM call grades all chunks for relevance (temperature=0.0)
  • Irrelevant chunks filtered out; if < 3 pass, query is rewritten and a
    second retrieval cycle fires (max 2 cycles total)
  • Any grading failure → original chunks returned unchanged (safe fallback)
        │
        ▼
  Top-K relevant chunks → injected into LLM context window
  • Context capped at ~50% of num_ctx
  • chunk_id deduplication across multiple queries and cycles
```

**Fallback:** If `nomic-embed-text` is not pulled, `HybridStore` falls back to BM25-only automatically. A warning is shown in the UI and CLI.

---

## Memory System

Notebooks persist in `outputs/memory/sessions.db` (SQLite WAL mode):

| Table | Purpose |
|-------|---------|
| `notebooks` | Metadata, source list, conversation history, `concepts_covered` |
| `notebook_chunks` | Chunk text and metadata (never loaded on list calls) |

Embeddings are cached in ChromaDB (`outputs/chroma_db/`) so reopening a notebook does not re-embed.

**Notebook chunks split:** `list_notebooks()` never loads chunk text; `load()` reconstructs the full dict by joining both tables. This avoids loading megabytes of text for a simple session list.

The SR pipeline is stateless — it does not write to SQLite. Results are downloaded directly from the UI Export tab or saved by the CLI to `outputs/`.

---

## Self-Reflective RAG

**Module:** `agents/self_reflective_rag.py`

A post-retrieval relevance filter. After retrieval, a single batched LLM call grades all retrieved items and filters out irrelevant ones before they enter the main LLM context.

| Mode | Retrieved items | Grading function | Cycles |
|------|----------------|-----------------|--------|
| SR (Mode 1) | Academic papers from 4 sources | `grade_papers()` | 1 (one-pass) |
| Notebook Q&A (Mode 2) | Document chunks (HybridStore) | `grade_chunks()` | Up to 2 |
| Notebook Pipeline (Mode 2) | Document chunks (HybridStore) | `grade_chunks()` | Up to 2 |

### `grade_chunks(chunks, query, model_name, num_ctx) → List[bool]`

- **Input:** list of chunk dicts (with `text` key), query string
- **LLM:** `temperature=0.0`, `num_predict=100`, `num_ctx=min(num_ctx, 4096)`
- **Expected response:** `{"grades": [true, false, true, ...]}`
- **Fallback:** any `Exception` or length mismatch → `[True] * len(chunks)`

### `grade_papers(papers, query, model_name, num_ctx) → List[bool]`

- **Input:** `List[Dict]` with at least `title` and `abstract` keys
- **Prompt:** numbered list — each entry: `[N] Title: {title}\nAbstract: {abstract[:300]}`
- **Same LLM settings and fallback as `grade_chunks`**

### `self_reflective_retrieve(store, query, top_k, ...) → Tuple[List[Dict], Dict]`

Orchestrates multi-cycle chunk retrieval for the Notebook.

```
cycle 1:
  chunks = store.search_hybrid(query, k=top_k)
  grades = grade_chunks(chunks, query, ...)
  relevant = [c for c, g in zip(chunks, grades) if g]
  if len(relevant) >= min_relevant → return relevant, metadata

cycle 2 (fires only if cycle 1 passes < 3 items):
  rewritten = rewrite_query(original_query, ...)
  more_chunks = store.search_hybrid(rewritten, k=top_k)
  deduplicate by chunk_id across both cycles
  grade new chunks only
  merge cycle-1 relevant + new relevant
  return merged[:top_k], metadata
```

Safety: any failure → original chunks returned, never raises.

---

## Quality Self-Evaluation

After every pipeline completes, a dedicated eval node makes a single micro LLM call to score output quality. Non-blocking — any failure is caught and silently ignored.

| Mode | Dimensions (each 1–5) |
|------|-----------------------|
| Systematic Review | `search_comprehensiveness`, `screening_rigor`, `evidence_quality`, `synthesis_depth`, `gap_identification` |
| Notebook Q&A | `answer_grounding`, `citation_accuracy`, `relevance` |
| Notebook Pipeline | `summary_quality`, `citation_coverage`, `study_guide_quality` |

Result stored in `state["eval_result"]`. Displayed as a collapsible expander in the UI (colour-coded: 4–5 green, 3 yellow, 1–2 red) and as a Rich table in the CLI.

---

## Feedback Refinement

`agents/feedback_agent.py` — `refine_with_feedback()`

Up to 3 rounds of plain-English feedback after every pipeline output. Each round is one LLM call (`temperature=0.4`). In the UI: collapsible "Refine" expander. In the CLI: `Feedback>` prompt (press Enter to skip).

| Mode | Refined output |
|------|----------------|
| Systematic Review | Narrative synthesis |
| Research Notebook | Study guide |

---

## Hardware Detection

`config/hardware.py` is called at CLI startup and in the Streamlit sidebar.

```
detect_hardware()
  ├── platform.processor(), sys.platform  → cpu, os, arch
  ├── psutil.virtual_memory()             → ram_gb
  └── subprocess("nvidia-smi") / platform.machine()
        → gpu_type: "apple_silicon" | "nvidia" | "cpu"

recommend_config(hw, available_models)
  └── Lookup table: ram_gb × gpu_type × model_size
        → {model, num_ctx, reasoning, hardware_note, can_run, pull_command}
```

The UI sidebar shows only pulled models in the dropdown. Run `python main.py --check-system` for a hardware-aware recommendation.

---

## Engineering Decisions

### Rate-Limit Backoff (`tools/search_tools.py`)

All `@retry` decorators use `retry=retry_if_exception(_is_retryable)` rather than a blanket retry. `_is_retryable()` returns `True` only for HTTP 429/500/502/503/504 and `ConnectionError`/`Timeout`. Wait strategy: `wait_exponential(min=2, max=30)` with `stop_after_attempt(4)`.

### MD5 Embedding Cache Invalidation (`tools/hybrid_store.py`)

`ProcessedDocument` carries a `content_md5` field (MD5 of `raw_text[:50000]`). `HybridStore.add_documents()` compares each document's hash against the manifest. If they differ, `_invalidate_doc_cache(doc_name)` deletes all ChromaDB entries for that filename before re-embedding — no manual `--clear-store` required for modified documents.

### Lazy Tool Imports (`tools/__init__.py`)

The `tools` package uses `__getattr__` for deferred loading. No submodule is imported until the name is first accessed. The loaded value is cached so subsequent accesses are O(1). This ensures importing lightweight tools (e.g. `citation_tools`) does not trigger `faiss`, `chromadb`, or `langchain_ollama`.

---

## File Map

```
BeeSearch/
│
├── app.py                    ← Streamlit entry point; landing page dispatcher
├── main.py                   ← CLI — SR + Notebook modes
│
├── projects/
│   ├── __init__.py           ← PROJECT_REGISTRY {mode1, mode2}
│   ├── mode1_systematic_review.py  ← run(settings) — Systematic Review
│   └── mode2_notebook.py           ← run(settings) — Research Notebook
│
├── ui/
│   ├── sidebar.py            ← render_sidebar() — hardware/model/RAG controls
│   ├── landing.py            ← render_landing() — 2-mode card layout
│   └── tabs/
│       ├── systematic_review.py  ← tab_systematic_review() — 5 tabs
│       └── notebook.py           ← tab_notebook()
│
├── agents/
│   ├── systematic_review_state.py  ← SystematicReviewState TypedDict + factory
│   ├── systematic_review_nodes.py  ← 6 SR nodes
│   ├── systematic_review_graph.py  ← build_systematic_review_graph()
│   │
│   ├── notebook_state.py           ← NotebookState TypedDict
│   ├── notebook_graph.py           ← build_notebook_graph() + run_notebook_turn()
│   ├── notebook_nodes.py           ← retrieve, answer, save nodes
│   ├── notebook_memory.py          ← NotebookMemory (SQLite)
│   ├── notebook_pipeline_state.py  ← NotebookPipelineState TypedDict
│   ├── notebook_pipeline_graph.py  ← build_notebook_pipeline_graph()
│   ├── notebook_pipeline_nodes.py  ← 7 pipeline nodes
│   ├── notebook_advanced.py        ← Advanced notebook features
│   │
│   ├── self_reflective_rag.py  ← grade_chunks(), grade_papers(), self_reflective_retrieve()
│   ├── eval_nodes.py           ← Quality self-evaluation nodes; non-blocking micro LLM call
│   └── feedback_agent.py       ← refine_with_feedback(); up to 3 rounds
│
├── tools/
│   ├── abstract_screener.py    ← LLM 0–100 paper relevance scorer
│   ├── citation_network.py     ← Ego citation graph (networkx + Pyvis HTML)
│   ├── preprint_tracker.py     ← CrossRef preprint / retraction status
│   ├── prisma_report.py        ← PRISMA 2020 DOCX (python-docx) + PDF (reportlab)
│   ├── plain_language.py       ← Patient · Policy brief · Press release
│   ├── trend_analyzer.py       ← CrossRef facet year-count trends
│   ├── evidence_map.py         ← Plotly Population × Intervention bubble chart
│   ├── concept_drift.py        ← TF-IDF keyword shift across 5-year buckets
│   │
│   ├── document_tools.py       ← get_processor() auto-selects Docling or pdfplumber by page count
│   ├── docling_processor.py    ← Advanced Docling parser
│   ├── hybrid_store.py         ← HybridStore: FAISS + ChromaDB + BM25 + RRF
│   ├── embeddings.py           ← OllamaEmbedder (batched /api/embed)
│   ├── search_tools.py         ← GoogleScholarSearcher + arXiv + Semantic Scholar + CrossRef
│   ├── session_db.py           ← SQLite backend: init_db(), pack/unpack, DDL
│   ├── web_loader.py           ← URL → Document
│   ├── export_tools.py         ← DOCX + PDF export
│   ├── citation_tools.py       ← BibTeX + RIS export
│   ├── clarifier.py            ← Socratic clarifying questions
│   └── shutdown.py             ← Safe port release + ChromaDB flush
│
├── config/
│   ├── settings.py             ← Pydantic BaseSettings (env vars)
│   └── hardware.py             ← detect_hardware() + recommend_config()
│
├── outputs/
│   ├── chroma_db/              ← ChromaDB persistent embedding cache
│   ├── memory/
│   │   └── sessions.db         ← SQLite DB for Notebook sessions
│   ├── systematic_review_<id>.md
│   ├── prisma_report_<id>.docx
│   ├── prisma_report_<id>.pdf
│   └── pipeline_study_guide_<name>.md/docx/pdf
│
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Technology Stack

| Layer | Tool | Notes |
|-------|------|-------|
| LLM | Ollama (ChatOllama) | Fully local, Metal/CUDA/CPU |
| Agent Framework | LangGraph ≥ 0.2 | Compiled StateGraph per mode |
| LLM Toolkit | LangChain + langchain-ollama | Prompt templates, ChatOllama |
| Dense Embeddings | OllamaEmbedder → FAISS | In-memory IndexFlatIP |
| Embedding Cache | ChromaDB | Persistent local DB |
| Sparse Retrieval | rank-bm25 (BM25Okapi) | Keyword index, no GPU |
| RAG Fusion | RRF (stdlib only) | k=60, score = Σ 1/(60+rank) |
| Document Parsing | Docling | Default: layout-aware, table extraction |
| PDF Extraction | pdfplumber | Fallback parser (--no-docling) |
| DOCX Extraction | python-docx | Fallback parser; also DOCX export |
| Google Scholar | scholarly | No API key, primary SR source |
| Academic Search | arxiv, requests | arXiv, Semantic Scholar, CrossRef |
| PRISMA Reports | python-docx + reportlab | DOCX + PDF, no LibreOffice |
| Visualisation | Plotly, matplotlib, networkx, pyvis | Evidence map, citation network |
| Concept Drift | stdlib only (no scikit-learn) | TF-IDF + 5-year buckets |
| Audio | pyttsx3 | WAV synthesis from script |
| UI | Streamlit ≥ 1.37 | Web app |
| CLI | Rich ≥ 13 | Terminal panels, tables, Markdown |
| Config | pydantic-settings ≥ 2.0 | Typed env vars |
| Hardware Detection | psutil | Cross-platform RAM/CPU |
| Retry Logic | tenacity | Exponential backoff on API calls |
| Memory | SQLite (stdlib sqlite3) | `sessions.db`; WAL mode |
