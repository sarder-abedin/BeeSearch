# ResearchBuddy — Architecture

## Overview

ResearchBuddy is a Streamlit web application (+ CLI) that orchestrates two LangGraph
agent pipelines powered by a local Ollama LLM and a hybrid RAG stack.

```
┌───────────────────────────────────────────────────────────────┐
│   User Interface (Streamlit / CLI)                             │
│   app.py  ·  ui/landing.py  ·  ui/tabs/  ·  cli.py            │
└───────────────────────────────────────────────────────────────┘
          │                             │
   ┌─────┴────────┐             ┌────┴────────┐
   │ SR Pipeline   │             │ Notebook       │
   │ (6 nodes)     │             │ Pipeline       │
   │               │             │ (3+7 nodes)    │
   └───────────────┘             └───────────────┘
          │                             │
   ┌─────┴──────────────────────┴─────────┐
   │         Shared Services                     │
   │  Ollama LLM  ·  Hybrid RAG  ·  ChromaDB      │
   │  AcademicSearcher  ·  SessionDB              │
   └─────────────────────────────────────────┘
```

---

## Systematic Review Pipeline

```
START
  ↓
query_generation_node
  • LLM generates 4–6 PRISMA-optimised search queries
  ↓
literature_search_node
  • AcademicSearcher searches arXiv + Semantic Scholar + CrossRef
  • Deduplication by title hash
  • Self-Reflective RAG grades papers for relevance
  ↓
screening_node
  • LLM INCLUDE/EXCLUDE per paper vs. inclusion/exclusion criteria
  ↓
evidence_extraction_node
  • LLM extracts: study_design, sample_size, key_finding, quality, relevance_score
  • RoB 2 / ROBINS-I assessment (assess_rob_batch)
  • GRADE evidence grading (grade_evidence_body)
  • Contradiction detection (detect_contradictions)
  ↓
synthesis_node
  • Builds PRISMA flow dict
  • LLM writes 700–1100-word narrative synthesis
  • LLM extracts: key_themes, research_gaps, limitations, conclusion
  ↓
sr_eval_node
  • LLM rates 5 quality dimensions (1–5)
  ↓
END
```

### State schema (`SystematicReviewState`)

| Field | Type | Description |
|-------|------|-------------|
| `research_question` | str | User's PICO question |
| `inclusion_criteria` | List[str] | Per-line criteria |
| `exclusion_criteria` | List[str] | Per-line criteria |
| `search_queries` | List[str] | Generated queries |
| `raw_papers` | List[Dict] | All search results |
| `screened_papers` | List[Dict] | After INCLUDE screening |
| `included_papers` | List[Dict] | Final included set |
| `excluded_papers` | List[Dict] | With exclusion_reason |
| `prisma_flow` | Dict[str,int] | identified/screened/eligibility/included/excluded |
| `evidence_table` | List[Dict] | Per-paper extraction |
| `narrative_synthesis` | str | Main synthesis text |
| `key_themes` | List[str] | Common themes |
| `research_gaps` | List[str] | Identified gaps |
| `limitations` | str | Review limitations |
| `conclusion` | str | Summary conclusion |
| `rob_table` | List[Dict] | RoB 2 / ROBINS-I assessments |
| `grade_results` | Dict | GRADE grading output |
| `contradictions` | List[Dict] | Contradiction groups |
| `pico_extraction` | List[Dict] | Full PICO table |
| `gap_map` | Dict | Categorised gap map |
| `hypotheses` | List[Dict] | Generated hypotheses |
| `sensitivity_results` | Dict | Sensitivity analysis |
| `monitor_state` | Dict | Incremental monitor |
| `preregistration` | str | OSF template text |
| `eval_result` | Dict | Quality scores |

---

## Research Notebook Pipeline

### Core Q&A graph (3 nodes)

```
START → retrieve_node → answer_node → save_node → END

retrieve_node:
  • Hybrid RAG: FAISS dense + BM25 sparse + Reciprocal Rank Fusion
  • Optional: DuckDuckGo web search
  • Self-Reflective RAG re-ranking

answer_node:
  • LLM generates grounded answer with [N] inline citations
  • Extracts citation metadata and suggested follow-up questions

save_node:
  • Appends turn to conversation history in NotebookMemory (SQLite)
```

### 7-agent full pipeline

```
ingest → summarize → retrieve → verify_citations →
build_kg → generate_study_guide → generate_podcast → END
```

---

## Hybrid RAG Stack

```
Query
  │
  ├───► FAISS (dense)  ──► Reciprocal Rank Fusion (RRF)
  └───► BM25  (sparse) ──►           │
                                      ↓
                              Top-k re-ranked chunks
                                      │
                         Self-Reflective RAG grader
                         (filters irrelevant chunks)
                                      ↓
                              Context for LLM answer
```

**Embedding**: `nomic-embed-text` via Ollama  
**Persistence**: ChromaDB (cross-session embedding cache)  
**Storage**: SQLite via `tools/session_db.py` (notebooks, chunks)

---

## New Analysis Modules

### Risk of Bias (`agents/risk_of_bias.py`)
- Detects study design (RCT vs observational) from `study_design` field
- Applies **RoB 2** (5 domains) for RCTs, **ROBINS-I** (7 domains) for observational
- Returns per-domain rating (Low / Some concerns / High) + overall + justification

### GRADE Assessment (`agents/grade_assessment.py`)
- RCTs start at High; observational at Low
- Evaluates 5 domains: risk_of_bias, inconsistency, indirectness, imprecision, publication_bias
- Each domain rated: no concern / -1 / -2 (downgrade levels)
- Returns overall_grade, certainty_statement, domain breakdown

### Contradiction Detector (`agents/contradiction_detector.py`)
- Compares key_finding across all included papers
- Groups conflicting papers into position_a / position_b pairs
- Assigns consensus_score (0–100)

### PRISMA Diagram (`tools/prisma_diagram.py`)
- `generate_prisma_mermaid(flow)` → Mermaid flowchart string
- `generate_prisma_dot(flow)` → Graphviz DOT string
- Rendered in UI via mermaid.js CDN component

### Sensitivity Analysis (`tools/sensitivity_analysis.py`)
- Quality-filter scenarios (High only, High+Medium) — no re-run needed
- Criteria-modified scenarios — full SR re-run with altered inclusion/exclusion
- Returns pct_retained, filtered evidence_table, optional revised conclusion

### Literature Monitor (`tools/literature_monitor.py`)
- Persists search state (queries + known citation keys) to `~/.researchbuddy/monitors/`
- `find_new_papers()` diffs new search results against known keys
- Stable monitor ID derived from MD5 of research question

### Pre-registration (`tools/preregistration.py`)
- `generate_preregistration()` → OSF-compatible Markdown template
- `generate_prisma_checklist()` → PRISMA 2020 item checklist with completion status

### Extraction Table (`tools/extraction_table.py`)
- `extract_structured_row()` → population, intervention, comparator, outcome, finding, limitations
- `extraction_table_to_csv()` and `extraction_table_to_markdown()` for export

### Research Gap Mapper (`tools/research_gaps.py`)
- Five-dimensional analysis: population, methodology, outcome, context, temporal gaps
- Returns priority_gaps list + per-category itemised gaps with priority and rationale

### Hypothesis Generator (`tools/hypothesis_generator.py`)
- Generates PICO-structured, testable hypotheses from identified gaps
- Returns suggested_design, IV, DV, feasibility rating, feasibility note

### BibTeX Importer (`tools/zotero_importer.py`)
- Parses `.bib` files (braced and quoted field syntax)
- Converts entries to `ProcessedDocument` for notebook indexing
- Strips LaTeX markup from field values

### SR → Notebook Bridge (`tools/bridge.py`)
- Converts SR synthesis + evidence table into a `ProcessedDocument`
- Indexes it into any notebook for grounded follow-up Q&A

---

## Docker deployment

```yaml
services:
  researchbuddy-ollama:     # Ollama LLM server
  researchbuddy-model-init: # Pulls llama3.1:8b + nomic-embed-text on startup
  researchbuddy-app:        # Streamlit app on port 8501
```

All services share a Docker network. The app container waits for Ollama to be healthy before starting.

---

## CLI commands

| Command | Description |
|---------|-------------|
| `python cli.py sr "<question>"` | Run systematic review |
| `python cli.py nb --list` | List notebooks |
| `python cli.py nb --new "Name" -q "<question>"` | Create notebook + ask |
| `python cli.py nb --notebook-id <id>` | Interactive REPL |
| `python cli.py bib <id> <file.bib>` | Import BibTeX |
| `python cli.py gap <id> "<question>"` | Map research gaps |
| `python cli.py hyp <id> "<question>"` | Generate hypotheses |

---

## File structure

```
researchbuddy/
├── app.py                   # Streamlit entry point
├── cli.py                   # CLI entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config/
│   ├── settings.py          # App-wide config (Pydantic Settings)
│   └── hardware.py          # GPU/CPU detection
├── agents/
│   ├── systematic_review_state.py
│   ├── systematic_review_graph.py
│   ├── systematic_review_nodes.py
│   ├── risk_of_bias.py      # RoB 2 / ROBINS-I
│   ├── grade_assessment.py  # GRADE
│   ├── contradiction_detector.py
│   ├── notebook_state.py
│   ├── notebook_graph.py
│   ├── notebook_nodes.py
│   ├── notebook_memory.py
│   ├── notebook_advanced.py
│   ├── notebook_pipeline_state.py
│   ├── notebook_pipeline_graph.py
│   ├── notebook_pipeline_nodes.py
│   ├── self_reflective_rag.py
│   ├── feedback_agent.py
│   └── eval_nodes.py
├── tools/
│   ├── search_tools.py      # arXiv, Semantic Scholar, CrossRef
│   ├── hybrid_store.py      # FAISS + BM25 + RRF
│   ├── embeddings.py        # OllamaEmbedder
│   ├── document_tools.py    # DocumentProcessor, ProcessedDocument
│   ├── citation_tools.py    # BibTeX / RIS export
│   ├── export_tools.py      # DOCX, PDF builders
│   ├── web_loader.py        # URL → ProcessedDocument
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── session_db.py        # SQLite notebook storage
│   ├── doi_verifier.py
│   ├── ragchecker_eval.py
│   ├── prisma_diagram.py    # Mermaid + DOT PRISMA
│   ├── sensitivity_analysis.py
│   ├── literature_monitor.py
│   ├── preregistration.py
│   ├── extraction_table.py  # PICO table + CSV
│   ├── research_gaps.py
│   ├── hypothesis_generator.py
│   ├── zotero_importer.py   # BibTeX parser
│   └── bridge.py            # SR → Notebook
├── ui/
│   ├── landing.py
│   ├── sidebar.py
│   ├── theme.py
│   ├── helpers.py
│   └── tabs/
│       ├── systematic_review.py  # 9 tabs (was 4)
│       └── notebook.py           # 14 tabs (was 11)
└── docs/
    ├── tutorial.md
    └── architecture.md
```
