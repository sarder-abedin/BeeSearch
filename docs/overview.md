# BeeSearch — Technical Overview

BeeSearch is a local-first AI tool for three research workflows:

1. **Systematic Literature Review** — PRISMA-compliant pipeline with Google Scholar, abstract screener, risk-of-bias/GRADE/contradiction quality assessment, and post-synthesis analysis tools
2. **Research Notebook** — NotebookLM-style grounded Q&A with Hybrid RAG and a 7-agent analysis pipeline
3. **AI Research Assistant** — stateless, free-form literature-grounded Q&A with code-rebuilt inline citations; no upload, no PRISMA workflow

Everything runs locally via Ollama. No cloud LLM calls, no API keys (except CrossRef/Semantic Scholar which are free and unauthenticated).

---

## Repository layout

```
BeeSearch/
├── app.py                          ← Streamlit entry point (3 modes)
├── main.py                         ← CLI entry point (3 modes)
├── requirements.txt
├── backend/                        ← FastAPI REST API (additive; Mode 1, Mode 3, Mode 2 core)
│   └── app/
│       ├── main.py                 ← App factory, CORS, mock-LLM bootstrap
│       ├── jobs.py                 ← In-memory background job runner
│       ├── routers/, services/, schemas/
├── frontend/                       ← React + TypeScript SPA (Vite)
│   └── src/{api,pages,components}/
├── config/
│   ├── settings.py                 ← Pydantic config from .env
│   └── hardware.py                 ← Hardware detection + model recommendation
├── agents/
│   ├── systematic_review_state.py  ← SR TypedDict + create_systematic_review_state()
│   ├── systematic_review_nodes.py  ← 7 PRISMA pipeline nodes
│   ├── systematic_review_graph.py  ← SR LangGraph StateGraph
│   ├── risk_of_bias.py             ← RoB 2 (RCTs) / ROBINS-I (observational) per-paper assessment
│   ├── grade_assessment.py         ← GRADE certainty-of-evidence rating for the whole body of evidence
│   ├── contradiction_detector.py   ← Cross-paper conflict detection + 0–100 consensus score
│   ├── research_assistant.py       ← Mode 3: stateless search → ground → answer → rebuild citations
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
│   ├── citation_network.py         ← Ego citation graph (networkx + Pyvis HTML); Smart Citations stance classification (Supporting/Contrasting/Mentioning)
│   ├── citation_context.py         ← Best-effort, open-access-only citing-sentence extraction
│   ├── preprint_tracker.py         ← CrossRef preprint / retraction status
│   ├── prisma_report.py            ← PRISMA 2020 DOCX (python-docx) + PDF (reportlab)
│   ├── plain_language.py           ← Patient summary · Policy brief · Press release
│   ├── trend_analyzer.py           ← CrossRef facet year-count trends
│   ├── evidence_map.py             ← Plotly Population × Intervention bubble chart
│   ├── concept_drift.py            ← TF-IDF keyword shift (pure stdlib, no scikit-learn)
│   ├── meta_analysis.py            ← Pooled effect size + forest plot (Plotly / PNG fallback)
│   ├── sensitivity_analysis.py     ← Leave-one-out / subgroup sensitivity scenarios
│   ├── literature_monitor.py       ← Saved-search snapshots + "new papers since last run" diff
│   ├── document_tools.py           ← get_processor(): Docling default, pdfplumber auto-switch for large PDFs
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
│   ├── landing.py                  ← 3-mode landing page
│   ├── sidebar.py                  ← Hardware detection + model/RAG settings
│   ├── tabs/
│   │   ├── systematic_review.py    ← SR UI (4 tabs; Explore tab has 8 deep-dive tools incl. Risk & Certainty, Citation Context)
│   │   ├── notebook.py             ← Notebook UI
│   │   └── research_assistant.py   ← Mode 3 single-screen UI
│   └── theme.py
└── projects/
    ├── mode1_systematic_review.py  ← Mode 1 Streamlit runner
    ├── mode2_notebook.py           ← Mode 2 Streamlit runner
    └── mode3_research_assistant.py ← Mode 3 Streamlit runner
```

---

## Mode 1 — Systematic Literature Review

### Core pipeline (7 nodes)

```
query_generation → literature_search → screening → evidence_extraction → quality_assessment → synthesis → sr_eval
```

| Node | What happens |
|------|-------------|
| `query_generation` | LLM decomposes the research question into targeted Boolean search queries |
| `literature_search` | Queries **Google Scholar · arXiv · Semantic Scholar · CrossRef**; LLM abstract screener (0–100) pre-ranks papers; deduplicates; applies SR-RAG `grade_papers()` filter |
| `screening` | Each paper assessed against inclusion/exclusion criteria; excluded papers logged with reason |
| `evidence_extraction` | PICO fields (population/intervention/comparator/outcome), study design, quality rating (High/Medium/Low), and key finding extracted per included paper (up to `max_evidence_papers`, default 25) |
| `quality_assessment` | Risk of bias per paper (RoB 2 for trials / ROBINS-I for observational studies), GRADE certainty of the whole body of evidence, and cross-paper contradiction detection (0–100 consensus score). Each sub-assessment is independently try/except-wrapped — any failure degrades to an empty result and never blocks the pipeline |
| `synthesis` | LLM produces narrative synthesis, key themes, research gaps, and conclusion — fed by PICO evidence plus the GRADE certainty / risk-of-bias / contradiction results so the prose reflects how certain and how consistent the evidence actually is |
| `sr_eval` | Five-dimension quality self-evaluation (1–5 per dimension) |

### State (`SystematicReviewState` TypedDict)

**Core:** `research_question`, `inclusion_criteria`, `exclusion_criteria`, `model_name`, `num_ctx`, `session_id`, `search_queries`, `raw_papers`, `screener_scores`, `included_papers`, `excluded_papers`, `evidence_table`, `narrative_synthesis`, `key_themes`, `research_gaps`, `conclusion`, `limitations`, `prisma_flow`, `eval_result`, `rag_reflection_info`, `progress_pct`, `status_detail`, `errors`

**Post-synthesis:** `preprint_tracking`, `citation_graph_html`, `trend_data`, `evidence_map_data`, `concept_drift_data`

**Quality assessment:** `rob_table`, `grade_results`, `contradictions`, `max_evidence_papers`, `max_synthesis_papers`, `max_rob_papers`

### Post-synthesis on-demand tools

Picked one at a time from the **Explore** tab's tool radio (`_EXPLORE_TOOLS` in `ui/tabs/systematic_review.py`).

| Tool | File | What it produces |
|------|------|-----------------|
| Citation Network | `tools/citation_network.py` | Pyvis HTML ego-only graph of citation links between included papers, plus an isolated-paper list and gap-finder suggestions for frequently co-cited external papers. Optional **Smart Citations**: `classify_citation_stances()` labels each edge Supporting/Contrasting/Mentioning from the two papers' abstracts, coloured on the graph |
| Citation Context | `tools/citation_context.py` | The exact citing sentence(s) where one included paper cites another, pulled from the citing paper's open-access full text (scite.ai-style). Best-effort, open-access only — fails safe to "unavailable" when no open-access text exists |
| Risk & Certainty | `agents/risk_of_bias.py`, `grade_assessment.py`, `contradiction_detector.py` | Same three assessments as the `quality_assessment` pipeline node, rendered from state (or recomputed on demand for older cached results) |
| Abstract Screener | `tools/abstract_screener.py` | 0–100 relevance score + include/uncertain/exclude verdict per paper |
| Preprint Tracker | `tools/preprint_tracker.py` | Status per paper: journal / published / preprint / retracted |
| Trend Analyzer | `tools/trend_analyzer.py` | CrossRef facet year counts; growing/declining/stable classification |
| Evidence Map | `tools/evidence_map.py` | Plotly bubble chart (Population × Intervention); matplotlib PNG fallback |
| Meta-Analysis | `tools/meta_analysis.py` | Pooled fixed-effect/random-effects estimate + forest plot from per-study effect sizes (editable table, optional LLM draft-from-abstracts pass) |
| Concept Drift | `tools/concept_drift.py` | TF-IDF keyword shift across 5-year buckets; optional LLM narrative |

The **Write-up & Export** tab covers the remaining, always-available outputs:

| Tool | File | What it produces |
|------|------|-----------------|
| PRISMA Report | `tools/prisma_report.py` | DOCX + PDF with Methods → Results → Discussion scaffold |
| Plain-Language | `tools/plain_language.py` | Patient summary · Policy brief · Press release |

### UI tabs

**Synthesis** | **Evidence** | **Explore** | **Write-up & Export**

### CLI flags

```
--systematic-review / --sr    Run the pipeline (requires --goal)
--inclusion CRITERIA...        Inclusion criteria (one string each)
--exclusion CRITERIA...        Exclusion criteria (one string each)
--sr-quality                    Print risk-of-bias / GRADE / contradiction results
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
| 1 | `ingest` | Loads sources; Docling for normal PDFs, pdfplumber auto-selected for large PDFs (> `LARGE_DOC_PAGE_THRESHOLD` pages) |
| 2 | `summarize` | Per-doc summaries + cross-document synthesis |
| 3 | `retrieve` | Hybrid RAG on focus query; SR-RAG grades chunks |
| 4 | `verify_citations` | Verifies 5–8 claims against source material (HIGH/MEDIUM/LOW confidence) |
| 5 | `build_kg` | Entity–relationship graph → Graphviz DOT |
| 6 | `generate_study_guide` | Key concepts, glossary, Q&A, summary → MD + DOCX + PDF |
| 7 | `generate_podcast` | Two-speaker dialogue (HOST: Alex, EXPERT: Dr. Jordan) → TXT |

### Explain / Storyteller pipeline (7 nodes)

```
context_loader → repetition_tracker → source_router → storyteller
  → concept_visualizer → memory_saver → story_eval
```

| Node | What happens |
|------|-------------|
| `context_loader` | Loads conversation history + persisted `document_context` from `StorytellerMemory` (SQLite) |
| `repetition_tracker` | Zero-LLM-call heuristic (word-overlap + confusion phrases like "I don't understand") detects a repeated/rephrased question and, if found, overrides `explanation_style` to one different from what the previous answer used |
| `source_router` | LLM scores document coverage 0–10 for the question; below 6, automatically runs an academic search (arXiv + Semantic Scholar + Google Scholar) and a web search (DuckDuckGo) — no user toggle, unlike Chat/Research Report |
| `storyteller` | Explains at the chosen style/level, citing document excerpts as `[n]` and online results as `[Source n]`; a code-rebuilt References list replaces anything the LLM wrote itself. On a detected repeat, the prompt asks for a genuinely different angle, not reworded text |
| `concept_visualizer` | No-op unless a repeat was detected; otherwise renders an interactive Pyvis concept map (hub-and-spoke HTML) as a second, visual explanation modality. Any failure is a safe no-op |
| `memory_saver` | Persists the turn + extracted concepts + the `explanation_style` actually used to `StorytellerMemory` |
| `story_eval` | Non-blocking quality self-evaluation micro call |

Document excerpts are numbered and page-tagged (`build_numbered_doc_context`) the same way the Q&A pipeline and Literature Review tag theirs, so every `[n]` the storyteller cites resolves to a real chunk and page.

When a user repeats or rephrases a question — or signals confusion directly ("I don't understand", "still lost", …) — the Explain tab automatically switches explanation style (e.g. simple → analogy) and adds an interactive concept map alongside the new explanation. This is always-on, with no UI toggle, matching the tab's existing automatic online-search behavior.

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
| Citation timeline | `--notebook-timeline <id>` (`--enrich-abstracts` for S2 abstracts) | Cited works by year, with one-line gists |
| Study comparison | `--notebook-study-table <id>` | Research method/sample/findings table |
| 7-agent pipeline | `--notebook-pipeline <id>` | All of the above in sequence |

### Document parsing CLI flags

```
--no-docling              Always use pdfplumber (disables Docling ML models)
--ocr                     Enable Docling OCR for scanned PDFs
--large-doc-threshold N   PDFs with more than N pages auto-switch to pdfplumber
                          (default: LARGE_DOC_PAGE_THRESHOLD from settings, usually 50)
```

### Persistence

Notebooks are stored in `outputs/memory/sessions.db` (SQLite):
- `notebooks` table — metadata, source list, conversation history, `concepts_covered`
- `notebook_chunks` table — chunk text and metadata (never loaded on list calls)

Embeddings are cached in ChromaDB (`outputs/chroma_db/`) so reopening a notebook does not re-embed.

---

## Mode 3 — AI Research Assistant

A stateless, single-screen counterpart to the two structured pipelines (`agents/research_assistant.py`). Answers a free-form research question grounded in published literature in general — no document upload, no PRISMA inclusion/exclusion criteria, no persisted session.

### Flow

```
search_literature → build_numbered_sources → LLM answer → build_citations
```

| Step | What happens |
|------|-------------|
| `search_literature` | Queries Google Scholar · arXiv · Semantic Scholar (· CrossRef) plus an optional web search; each backend fails soft so one failing source doesn't sink the answer |
| `build_numbered_sources` | Papers and web results are merged into a single `[n]` numbering namespace, budget-capped per source and in total, tags baked into the context string handed to the LLM |
| LLM answer | Answers grounded only in the numbered context, citing `[n]` inline; explicitly told not to write its own References section |
| `build_citations` | Rebuilds the citation list in code from whichever `[n]` markers the LLM actually used — hallucinated numbers are dropped, never trusted from the LLM's own output |

If no sources are found, the result is marked ungrounded: the answer carries an explicit "general knowledge — verify" caveat and citations are empty.

### UI / CLI

- UI: single-screen tab (`ui/tabs/research_assistant.py`) — question box, "include web results" toggle, answer with inline citations, source list, suggested follow-ups
- CLI: `python main.py --ask "..."` (`--no-web` to skip the web-search backend)

---

## React + FastAPI Web App

A REST API (`backend/`) and a React SPA (`frontend/`) provide an additional
interface, added alongside Streamlit/CLI without modifying either — same
`agents/*`/`projects/*` logic underneath, no parallel business logic.
**Coverage:** Mode 1 and Mode 3 are complete; Mode 2 covers only the core
Q&A workflow (create/upload/chat with citations) — the 7-agent pipeline,
Explain tab, Research Report, and advanced one-shot tools are still
Streamlit/CLI-only. Long-running calls go through an in-memory background
job runner (`backend/app/jobs.py`); the frontend polls for status every
700ms. `BEESEARCH_MOCK_LLM=1` swaps in canned LLM/search responses for
dev/test use, no Ollama required. See the README's "Web App (React +
FastAPI)" section for run commands, and `docs/architecture.md`'s "React +
FastAPI Web App" section for the request-flow diagram.

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

## Web search re-ranking (`tools/search_tools.py`)

`WebSearcher` over-fetches from DuckDuckGo (up to 3× the requested count, capped at 20), deduplicates by URL and normalised title, then applies a stable sort: recognised research domains/TLDs (arxiv.org, PubMed, IEEE Xplore, Nature, ScienceDirect, `.edu`, `.gov`, …) rank first, a short low-signal list (Pinterest, Quora) ranks last, everything else keeps DuckDuckGo's original order. Results are only ever reordered, never dropped, for being non-academic.

---

## Quality self-evaluation

`agents/eval_nodes.py` — runs after every pipeline, non-blocking.

| Mode | Dimensions |
|------|-----------|
| Systematic Review | `search_comprehensiveness` · `screening_rigor` · `evidence_quality` · `synthesis_depth` · `gap_identification` |
| Research Notebook Q&A | `answer_grounding` · `citation_accuracy` · `relevance` |
| Research Notebook Pipeline | `summary_quality` · `citation_coverage` · `study_guide_quality` |
| Explain | `clarity` · `style_adherence` · `overall` |

---

## Feedback refinement

`agents/feedback_agent.py` — `refine_with_feedback()`

Up to 3 rounds of plain-English feedback after every pipeline output. Each round is one LLM call (`temperature=0.4`). In the UI: collapsible "Refine" expander. In the CLI: `Feedback>` prompt (press Enter to skip).
