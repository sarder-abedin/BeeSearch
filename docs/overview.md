# ResearchBuddy — Technical Overview

ResearchBuddy is a local-first AI tool for two research workflows:

1. **Systematic Literature Review** — PRISMA-compliant pipeline with Google Scholar, abstract screener, and post-synthesis analysis tools
2. **Research Notebook** — NotebookLM-style grounded Q&A with Hybrid RAG and a 7-agent analysis pipeline

Everything runs locally via Ollama. No cloud LLM calls, no API keys (except CrossRef/Semantic Scholar which are free and unauthenticated).

---

## Repository layout

```
ResearchBuddy/
├── app.py                          ← Streamlit entry point (2 modes)
├── main.py                         ← CLI entry point (2 modes)
├── requirements.txt
├── config/
│   ├── settings.py                 ← Pydantic config from .env
│   └── hardware.py                 ← Hardware detection + model recommendation
├── agents/
│   ├── systematic_review_state.py  ← SR TypedDict + create_systematic_review_state()
│   ├── systematic_review_nodes.py  ← 6 PRISMA pipeline nodes
│   ├── systematic_review_graph.py  ← SR LangGraph StateGraph
│   ├── notebook_state.py           ← Notebook TypedDict
│   ├── notebook_memory.py          ← SQLite persistence (NotebookMemory)
│   ├── notebook_nodes.py           ← 3 Q&A nodes: retrieve → answer → save
│   ├── notebook_graph.py           ← Notebook Q&A graph
│   ├── notebook_advanced.py        ← Phase-2 tools: summary, FAQ, review, mindmap…
│   ├── notebook_pipeline_state.py  ← 7-agent pipeline TypedDict
│   ├── notebook_pipeline_nodes.py  ← 7 pipeline nodes
│   ├── notebook_pipeline_graph.py  ← 7-agent pipeline graph
│   ├── self_reflective_rag.py      ← grade_chunks(), grade_papers()
│   ├── eval_nodes.py               ← Quality self-evaluation
│   └── feedback_agent.py           ← Post-output feedback refinement
├── tools/
│   ├── search_tools.py             ← Google Scholar + arXiv + Semantic Scholar + CrossRef
│   ├── abstract_screener.py        ← LLM 0-100 paper relevance scorer
│   ├── citation_network.py         ← Ego citation graph (networkx + Pyvis HTML)
│   ├── preprint_tracker.py         ← CrossRef preprint / retraction status
│   ├── prisma_report.py            ← PRISMA 2020 DOCX (python-docx) + PDF (reportlab)
│   ├── plain_language.py           ← Patient summary · Policy brief · Press release
│   ├── trend_analyzer.py           ← CrossRef facet year-count trends
│   ├── evidence_map.py             ← Plotly Population × Intervention bubble chart
│   ├── concept_drift.py            ← TF-IDF keyword shift (pure stdlib, no scikit-learn)
│   ├── document_tools.py           ← PDF/DOCX/TXT chunking
│   ├── docling_processor.py        ← Advanced Docling parser
│   ├── hybrid_store.py             ← FAISS + BM25 + ChromaDB + RRF (HybridStore)
│   ├── embeddings.py               ← OllamaEmbedder (batched /api/embed calls)
│   ├── export_tools.py             ← DOCX + PDF export (python-docx + ReportLab)
│   ├── citation_tools.py           ← BibTeX + RIS export
│   ├── session_db.py               ← SQLite backend
│   ├── web_loader.py               ← URL → Document
│   ├── clarifier.py                ← Socratic clarifying questions
│   └── shutdown.py                 ← Safe port release + ChromaDB flush
├── ui/
│   ├── landing.py                  ← 2-mode landing page
│   ├── sidebar.py                  ← Hardware detection + model/RAG settings
│   ├── tabs/
│   │   ├── systematic_review.py    ← SR UI (5 tabs)
│   │   └── notebook.py             ← Notebook UI
│   └── theme.py
└── projects/
    ├── mode1_systematic_review.py  ← Mode 1 Streamlit runner
    └── mode2_notebook.py           ← Mode 2 Streamlit runner
```

---

## Mode 1 — Systematic Literature Review

### Core pipeline (6 nodes)

```
query_generation → literature_search → screening → evidence_extraction → synthesis → sr_eval
```

| Node | What happens |
|------|-------------|
| `query_generation` | LLM decomposes the research question into targeted Boolean search queries |
| `literature_search` | Queries **Google Scholar · arXiv · Semantic Scholar · CrossRef**; LLM abstract screener (0–100) pre-ranks papers; deduplicates; applies SR-RAG `grade_papers()` filter |
| `screening` | Each paper assessed against inclusion/exclusion criteria; excluded papers logged with reason |
| `evidence_extraction` | Study design, quality rating (High/Medium/Low), and key finding extracted per included paper |
| `synthesis` | LLM produces narrative synthesis, key themes, research gaps, and conclusion |
| `sr_eval` | Five-dimension quality self-evaluation (1–5 per dimension) |

### State (`SystematicReviewState` TypedDict)

**Core:** `research_question`, `inclusion_criteria`, `exclusion_criteria`, `model_name`, `num_ctx`, `session_id`, `search_queries`, `raw_papers`, `screener_scores`, `included_papers`, `excluded_papers`, `evidence_table`, `narrative_synthesis`, `key_themes`, `research_gaps`, `conclusion`, `limitations`, `prisma_flow`, `eval_result`, `rag_reflection_info`, `progress_pct`, `status_detail`, `errors`

**Post-synthesis:** `preprint_tracking`, `citation_graph_html`, `trend_data`, `evidence_map_data`, `concept_drift_data`

### Post-synthesis on-demand tools

| Tool | File | What it produces |
|------|------|-----------------|
| Abstract Screener | `tools/abstract_screener.py` | 0–100 relevance score + include/uncertain/exclude verdict per paper |
| Citation Network | `tools/citation_network.py` | Pyvis HTML ego-only graph of citation links between included papers |
| Preprint Tracker | `tools/preprint_tracker.py` | Status per paper: journal / published / preprint / retracted |
| PRISMA Report | `tools/prisma_report.py` | DOCX + PDF with Methods → Results → Discussion scaffold |
| Plain-Language | `tools/plain_language.py` | Patient summary · Policy brief · Press release |
| Trend Analyzer | `tools/trend_analyzer.py` | CrossRef facet year counts; growing/declining/stable classification |
| Evidence Map | `tools/evidence_map.py` | Plotly bubble chart (Population × Intervention); matplotlib PNG fallback |
| Concept Drift | `tools/concept_drift.py` | TF-IDF keyword shift across 5-year buckets; optional LLM narrative |

### UI tabs

**Synthesis** | **Evidence Table** | **Discovery** | **Trends & Analysis** | **Export & Reports**

### CLI flags

```
--systematic-review / --sr    Run the pipeline (requires --goal)
--inclusion CRITERIA...        Inclusion criteria (one string each)
--exclusion CRITERIA...        Exclusion criteria (one string each)
--sr-docx                      Generate PRISMA 2020 DOCX
--sr-pdf                       Generate PRISMA 2020 PDF
--sr-plain-language FORMAT     patient / policy / press / all
--sr-trends                    Field-wide CrossRef year-count table
--sr-preprints                 Preprint/retraction status per paper
--sr-concept-drift             Vocabulary shift across 5-year buckets
--sr-author NAME               Author name for title page
--sr-institution NAME          Institution for title page
```

---

## Mode 2 — Research Notebook

### Core Q&A pipeline (3 nodes)

```
retrieve → answer → save
```

| Node | What happens |
|------|-------------|
| `retrieve` | Builds Hybrid RAG index from notebook chunks; FAISS + BM25 + RRF retrieval; `grade_chunks()` filters irrelevant chunks (up to 2 cycles with query rewrite); BM25 fallback if embedding model not pulled |
| `answer` | LLM answers using only retrieved excerpts; cites every claim as `[n]`; proposes 2–3 follow-up questions |
| `save` | Persists Q&A turn to notebook SQLite; updates `concepts_covered` list |

### 7-agent pipeline

```
ingest → summarize → retrieve → verify_citations → build_kg → generate_study_guide → generate_podcast
```

| Agent | Node | What it produces |
|-------|------|-----------------|
| 1 | `ingest` | Loads sources and chunks from NotebookMemory into pipeline state |
| 2 | `summarize` | Per-doc summaries + cross-document synthesis |
| 3 | `retrieve` | Hybrid RAG on focus query; SR-RAG grades chunks |
| 4 | `verify_citations` | Verifies 5–8 claims against source material (HIGH/MEDIUM/LOW confidence) |
| 5 | `build_kg` | Entity–relationship graph → Graphviz DOT |
| 6 | `generate_study_guide` | Key concepts, glossary, Q&A, summary → MD + DOCX + PDF |
| 7 | `generate_podcast` | Two-speaker dialogue (HOST: Alex, EXPERT: Dr. Jordan) → TXT |

### Advanced analysis (one-shot tools)

| Feature | CLI flag | Output |
|---------|----------|--------|
| Cross-document summary | `--notebook-summary <id>` | Markdown (common themes, contradictions, takeaways) |
| FAQ | `--notebook-faq <id>` | 4–16 grounded Q&A pairs |
| Literature review | `--notebook-review <id>` | Formal academic review Markdown |
| Audio script | `--notebook-audio <id>` | 300-word script TXT + WAV via pyttsx3 |
| Mind map | `--notebook-mindmap <id>` | DOT + PNG + SVG |
| Knowledge graph | `--notebook-graph <id>` | DOT + PNG + SVG |
| Source comparison | `--notebook-compare <id> --compare-docs A B` | Side-by-side Markdown table |
| Timeline | `--notebook-timeline <id>` | Chronological events table |
| Study comparison | `--notebook-study-table <id>` | Research method/sample/findings table |
| 7-agent pipeline | `--notebook-pipeline <id>` | All of the above in sequence |

### Persistence

Notebooks are stored in `outputs/memory/sessions.db` (SQLite):
- `notebooks` table — metadata, source list, conversation history, `concepts_covered`
- `notebook_chunks` table — chunk text and metadata (never loaded on list calls)

Embeddings are cached in ChromaDB (`outputs/chroma_db/`) so reopening a notebook does not re-embed.

---

## Hybrid RAG (`tools/hybrid_store.py`)

1. **Dense** — FAISS index on `OllamaEmbedder` vectors (default: `nomic-embed-text`)
2. **Sparse** — BM25 over chunk text (`rank-bm25`)
3. **Fusion** — Reciprocal Rank Fusion (`_rrf_merge()`, k=60)
4. **Grading** — `grade_chunks()` from `self_reflective_rag.py` removes irrelevant chunks; if fewer than 3 pass, the query is rewritten and retrieval is retried (max 2 cycles)

Falls back to BM25-only when the embedding model is not pulled.

---

## Self-Reflective RAG (`agents/self_reflective_rag.py`)

- `grade_chunks(chunks, query)` — used in Notebook retrieval
- `grade_papers(papers, query)` — used in SR literature search

A batched LLM call (`temperature=0.0`, `num_ctx=4096`) returns `{"grades": [true/false, ...]}`. False items are discarded before reaching the main LLM. Any failure silently passes all items.

---

## Quality self-evaluation

`agents/eval_nodes.py` — runs after every pipeline, non-blocking.

| Mode | Dimensions |
|------|-----------|
| Systematic Review | `search_comprehensiveness` · `screening_rigor` · `evidence_quality` · `synthesis_depth` · `gap_identification` |
| Research Notebook Q&A | `answer_grounding` · `citation_accuracy` · `relevance` |
| Research Notebook Pipeline | `summary_quality` · `citation_coverage` · `study_guide_quality` |

---

## Feedback refinement

`agents/feedback_agent.py` — `refine_with_feedback()`

Up to 3 rounds of plain-English feedback after every pipeline output. Each round is one LLM call (`temperature=0.4`). In the UI: collapsible "Refine" expander. In the CLI: `Feedback>` prompt (press Enter to skip).
