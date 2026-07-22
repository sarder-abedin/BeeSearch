# BeeSearch — Architecture

## System Overview

BeeSearch is a **3-mode, local-first AI research system** built on LangGraph state machines, Ollama LLMs, and Hybrid RAG. All computation runs locally — no cloud LLM, no paid API.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interfaces                                │
│                                                                             │
│  CLI terminal (main.py)         React SPA (frontend/)                      │
│  --systematic-review /          Browser fetch() →                          │
│  --notebook / --ask             FastAPI (backend/)                          │
│                                 /api/* routes                               │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │  (all three surfaces call the same
                                │   agents/* / projects/* modules)
           ┌────────────────────┼──────────────────────┐
           │                    │                      │
   ┌───────▼──────┐   ┌─────────▼───────┐    ┌────────▼───────┐
   │   Mode 1     │   │    Mode 2       │    │     Mode 3     │
   │  Systematic  │   │  Research       │    │  AI Research   │
   │  Literature  │   │  Notebook       │    │   Assistant    │
   │  Review      │   │                 │    │  (stateless)   │
   │  projects/   │   │  projects/      │    │   projects/    │
   │  mode1_*     │   │  mode2_*        │    │    mode3_*     │
   └───────┬──────┘   └────────┬────────┘    └────────┬───────┘
           │                   │                      │
           └─────────┬─────────┘                      │
                     │                                │
         ┌───────────▼───────────┐                    │ (direct call —
         │   LangGraph graphs    │                    │  no LangGraph,
         │   agents/*.py         │                    │  no Hybrid RAG,
         └───────────┬───────────┘                    │  no Self-Reflective
                     │                                │  RAG, no SQLite
      ┌──────────────┼──────────────┐                 │  memory)
      │              │              │                 │
 ┌────▼─────┐ ┌──────▼──────┐ ┌───▼──────────────┐    │
 │ Hybrid   │ │  Academic   │ │   Memory         │    │
 │ RAG      │ │  Search     │ │  (SQLite WAL)    │    │
 │ FAISS    │ │  Google     │ │   outputs/       │    │
 │ BM25     │ │  Scholar    │ │  memory/         │    │
 │ ChromaDB │ │  arXiv      │ │  sessions.db     │    │
 │ RRF      │ │  Semantic   │ └──────────────────┘    │
 │          │ │  Scholar    │                         │
 │ Mode 2   │ │  CrossRef   │                         │
 │ (docs)   │ │             │                         │
 └────┬─────┘ └──────┬──────┘                         │
      │              │                                │
      └──────┬───────┘                                │
             │                                        │
 ┌───────────▼──────────────────────────────────┐     │
 │   Self-Reflective RAG  (agents/              │     │
 │   self_reflective_rag.py)                    │     │
 │                                              │     │
 │   grade_chunks() — Mode 2 (Notebook)        │      │
 │     batch LLM call grades retrieved chunks  │      │
 │     < 3 pass → rewrite query + cycle 2      │      │
 │                                              │     │
 │   grade_papers() — Mode 1 (SR)              │      │
 │     batch LLM call grades retrieved papers  │      │
 │     one-pass filter (no cycle)              │      │
 │                                              │     │
 │   Fallback: any failure → all items kept    │      │
 └───────────────────────┬──────────────────────┘     │
                         │                            │
             ┌───────────▼──────────┐   ┌─────────────▼────────────┐
             │   Ollama LLM         │   │  Academic Search, then   │
             │   (main reasoning)   │   │   Ollama LLM (direct)    │
             └──────────────────────┘   └──────────────────────────┘
```

Mode 1 and Mode 2 (left and center above) share the LangGraph / Hybrid RAG /
Self-Reflective RAG stack. Mode 3 (right) fans out from the same User Interfaces layer
but is **architecturally separate** — no LangGraph `StateGraph`, no Hybrid RAG, no
Self-Reflective RAG, and no SQLite memory. `run_research_assistant()`
(`agents/research_assistant.py`) is a single stateless function call: Academic Search,
then a grounded Ollama LLM call directly, with citations rebuilt in code from the `[n]`
actually used — no intermediate graph.

All three user-interface surfaces (Streamlit, CLI, React + FastAPI) call the same
`agents/*` / `projects/*` modules — there is no separate business logic for the web
app. The React SPA sends `fetch()` requests to FastAPI routers, each of which is a thin
layer over the shared service modules. See "React + FastAPI Web App" below for the
HTTP-layer diagram, job-polling model, Docker setup, and mock mode.

See "Mode 3: AI Research Assistant" below for source numbering, citation-rebuild, and
CLI/UI details.

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
  │  • For each included paper (up to max_evidence_papers, default 25): one LLM call per paper
  │  • Extracts: population/intervention/comparator/outcome (PICO),
  │              study_design, sample_size, key_finding,
  │              quality (High/Medium/Low), relevance_score (1–5)
  │  • Assigns citation_key (<author><year> format)
  │
  ▼
[quality_assessment]
  │  • Risk of bias per paper — RoB 2 (trials) / ROBINS-I (observational) → rob_table
  │  • GRADE certainty of the body of evidence → grade_results
  │  • Cross-paper contradiction detection (0–100 consensus) → contradictions
  │  • Each sub-assessment fails safe to an empty result (never blocks the pipeline)
  │
  ▼
[synthesis]
  │  • Builds prisma_flow dict: identified/screened/eligibility/included/excluded
  │  • Feeds PICO evidence + GRADE certainty + RoB distribution + contradictions
  │    into the narrative prompt so it reflects certainty and disagreement
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
  │  • network_stats() names isolated papers (no in-corpus citation links)
  │  • find_gap_candidates() surfaces external papers cited by 2+ included
  │    papers but not themselves screened in (gap-finder for screening)
  │  • Optional "Smart Citations": classify_citation_stances() labels each
  │    edge Supporting / Contrasting / Mentioning from the two papers'
  │    abstracts (temperature=0.0); pyvis edges coloured by stance; any
  │    classification failure defaults that edge to neutral Mentioning

[citation_context]           tools/citation_context.py
  │  • Best-effort, open-access-only citing-sentence extraction (scite.ai-style)
  │  • find_citation_mentions() is the pure, tested core: matches the cited
  │    paper's first-author surname + year, or distinctive title tokens,
  │    against sentences in the citing paper's text
  │  • _fetch_fulltext() is the only networked part — open-access PDF
  │    (pdfplumber) or HTML (regex-stripped); fails safe to "unavailable"
  │    when no open-access full text exists

[reference_checking]         agents/risk_of_bias.py, grade_assessment.py,
                              contradiction_detector.py
  │  • Same three assessments as the core pipeline's quality_assessment node,
  │    exposed as an on-demand Explore tool ("Risk & Certainty")
  │  • Renders rob_table / grade_results / contradictions directly from
  │    final_state when the pipeline already computed them; a button
  │    recomputes on demand for older cached results that predate this node

[meta_analysis]               tools/meta_analysis.py
  │  • Pools per-study effect sizes (fixed-effect / random-effects,
  │    generic inverse-variance) into a forest plot
  │  • Editable data_editor for effect/CI/N per study; optional LLM
  │    "draft from abstracts" pass via extract_effect_size_row()
  │  • Heterogeneity stats (I², Q); Plotly forest plot, matplotlib PNG fallback

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

State fields: `research_question`, `inclusion_criteria`, `exclusion_criteria`, `model_name`, `num_ctx`, `session_id`, `search_queries`, `raw_papers`, `screener_scores`, `included_papers`, `excluded_papers`, `evidence_table`, `narrative_synthesis`, `key_themes`, `research_gaps`, `conclusion`, `limitations`, `prisma_flow`, `eval_result`, `rag_reflection_info`, `progress_pct`, `status_detail`, `errors`, `preprint_tracking`, `citation_graph_html`, `trend_data`, `evidence_map_data`, `concept_drift_data`, `rob_table`, `grade_results`, `contradictions`, `max_evidence_papers`, `max_synthesis_papers`, `max_rob_papers`

**Output tabs (UI):** Synthesis | Evidence | Explore | Write-up & Export

**Explore tab tools** (`_EXPLORE_TOOLS` in `ui/tabs/systematic_review.py`, pick one at a time): Citation Network · Citation Context · Risk & Certainty · Preprint Status · Research Trends · Evidence Map · Meta-Analysis · Concept Drift

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

# Print risk-of-bias / GRADE / contradiction results to the console
python main.py --systematic-review \
  --goal "Mindfulness-based interventions for anxiety" \
  --sr-quality
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

### 2c — Explain / Storyteller (StoryState)

Single-turn graph invocation per user message, like 2a — but the source mix (own documents vs. online search) is decided automatically per question rather than fixed up front.

```
START
  │
  ▼
[context_loader]
  │  • Loads conversation history + document_context from StorytellerMemory (SQLite)
  │  • document_context is a flat string with [n] (source: filename, p. X) tags
  │    baked in at session creation — no separate chunk-mapping table needed
  │
  ▼
[repetition_tracker]
  │  • Zero-LLM-call heuristic: Jaccard word-overlap (≥0.4) between this
  │    question and recent prior user questions, OR an explicit confusion-
  │    phrase match ("I don't understand", "still confused", …)
  │  • Requires at least one prior assistant turn — a session's first
  │    message can never be "a repeat"
  │  • On a detected repeat, overrides explanation_style to something
  │    different from the style the previous answer actually used
  │    (_next_explanation_strategy rotates simple→analogy→walkthrough→debate→…)
  │
  ▼
[source_router]
  │  • Fast LLM call (temperature=0) scores 0-10 how well document_context
  │    covers the question
  │  • Score < 6 (_COVERAGE_THRESHOLD): runs AcademicSearcher (arXiv +
  │    Semantic Scholar + Google Scholar) and WebSearcher (DuckDuckGo) —
  │    unconditionally, no user toggle (unlike Chat/Research Report's
  │    opt-in "Auto web search" / "Include web search")
  │
  ▼
[storyteller]
  │  • Explains at the chosen style/level (ELI5 … expert) and length
  │  • Cites document excerpts as [n], online results as [Source n]
  │  • _strip_llm_references_section discards any References list the LLM
  │    wrote itself; _build_references_section rebuilds one from whichever
  │    [n]/[Source n] numbers were actually used in the body
  │  • On a repeat (is_repeat_clarification), the system prompt adds an
  │    instruction to use a genuinely different angle, not just reworded text
  │
  ▼
[concept_visualizer]
  │  • No-op (zero LLM calls) unless is_repeat_clarification is True
  │  • On a repeat: a second LLM call extracts a hub-and-spoke breakdown
  │    {"central", "related": [{"label", "relation"}]} of the concept just
  │    explained, rendered via Pyvis into a self-contained interactive HTML
  │    string (concept_visual_html) — a different modality (diagram vs.
  │    prose) when a different writing style alone may still not land
  │  • Any failure (LLM, JSON parse, pyvis missing) is a safe no-op —
  │    never blocks the primary explanation already produced
  │  • concept_visual_html is ephemeral, like online_results/source_decision
  │    — not persisted to StorytellerMemory, only rendered for this turn
  │
  ▼
[memory_saver]
  │  • StorytellerMemory.add_turn() for user + assistant turns, including
  │    which explanation_style was actually used (so a future repeat can
  │    rotate away from it)
  │  • Extracts and stores newly-covered concepts
  │
  ▼
[story_eval]
  │  • Non-blocking quality self-evaluation micro call
  │
 END  (StoryState → story_sessions table in sessions.db)
```

**State type:** `StoryState` (`agents/story_state.py`) — includes `is_repeat_clarification`,
`repeated_question`, and `concept_visual_html` for the repetition/visualization feature.

**Memory:** `outputs/memory/sessions.db` — `story_sessions` table (`agents/story_memory.py::StorytellerMemory`), independent of the `notebooks` table — deliberately not linked by `notebook_id`, to avoid contaminating the shared `research_docs` vector collection. Each assistant turn also records `explanation_style` (the style actually used, which `repetition_tracker` may have overridden) so later turns can detect what was already tried.

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
| Citation timeline | Cited works by year, parsed from each source's bibliography, with one-line gists (optional Semantic Scholar abstract enrichment) |
| Study comparison | Research method/sample/findings table |

---

## Mode 3: AI Research Assistant

A stateless, single-call counterpart to the two structured pipelines (`agents/research_assistant.py`, surfaced by `ui/tabs/research_assistant.py` and `main.py --ask`). It answers a free-form research question from published literature in general — no upload, no PRISMA criteria.

```
question
   │
   ▼
[search_literature]   AcademicSearcher (Google Scholar · arXiv · Semantic Scholar [· CrossRef]) + optional WebSearcher
   │  • each backend fails soft — a failing source contributes nothing
   ▼
[build_numbered_sources]   one [n] namespace: papers first, then web results; tags baked into the context string
   │
   ▼
[LLM answer]   grounded in the numbered sources, citing [n] inline; told NOT to write its own References
   │
   ▼
[build_citations]   rebuild the citation list in code from the [n] markers actually used (hallucinated numbers dropped)
   │
   ▼
{answer, citations, sources, suggested_questions, grounded}
```

If no sources are retrieved, `grounded` is `False`, the answer carries an explicit "general knowledge — verify" caveat, and `citations` is empty. Citation grounding follows the same code-rebuild-from-what-was-cited pattern as Notebook / Explain / Literature Review.

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
  │                       Tables: Markdown + plain-text stored in chunk metadata
  │                       (content_type="table", table_md=<markdown string>)
  │                       Figures: PictureItem → _extract_figure_chunks()
  │                       → _caption_image() via VISION_MODEL (opt-in;
  │                       blank = skipped; content_type="figure")
  ├── PDF >  LARGE_DOC_PAGE_THRESHOLD pages  (auto RAM guard)
  │     └── DocumentProcessor  →  pdfplumber page-by-page streaming,
  │                                no ~500 MB ML models loaded
  └── --no-docling flag  →  always DocumentProcessor
  Both paths: clean_text → chunk_text (chunk_size=800, overlap=150)
  _flatten_chunks passes content_type + table_md into every retrieved chunk dict
  Context builders label chunks [TABLE] (Markdown body) or [FIGURE] (caption)
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
| Explain | `clarity`, `style_adherence`, `overall` |

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

`config/hardware.py` is called at CLI startup and by the backend `/api/health` endpoint.

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

### Citation Grounding (`notebook_advanced.py`, `story_nodes.py`)

Both Literature Review and the Explain tab once let the LLM cite freely (`[1]`, `[2]`, …) while writing its own References list from a prompt that only gave it one citable number per *document* — the two never matched. Both now follow the same fix as the Chat tab (`notebook_nodes.py::_build_context_block`): number every individual chunk (not document) with its real page tag, bake the tags directly into the persisted context string, and after generation regex-extract whichever numbers the LLM actually used to rebuild an accurate References list in code. The LLM's own References section is always discarded, never trusted. Out-of-range/hallucinated numbers are dropped silently rather than rendered with invented source details. The Explain tab additionally unifies two citation namespaces — `[n]` for document excerpts and `[Source n]` for online search results — into one rebuilt list.

### Research-Domain Re-ranking (`tools/search_tools.py`)

`WebSearcher` wraps plain DuckDuckGo search, which ranks for generic relevance/SEO signals rather than research value. `search()` now over-fetches (up to 3× `max_results`, capped at 20), deduplicates by URL and normalised title, and applies a *stable* sort keyed on `_research_rank_score()`: 0 for a recognised research domain or TLD (arxiv.org, PubMed, IEEE Xplore, Nature, ScienceDirect, `.edu`, `.gov`, …), 2 for a short list of low-signal domains (Pinterest, Quora), 1 for everything else. Ties keep DuckDuckGo's original relative order. Domain matching (`_matches_any_domain`) is dot-boundary-safe (`"fooarxiv.org"` must not match `"arxiv.org"`). Results are only ever reordered, never dropped, for being non-research — a borderline-but-relevant hit is never hidden.

---

## React + FastAPI Web App

The **primary interface** (`backend/` + `frontend/`) exposes Mode 1, Mode 3,
and the core of Mode 2 over a REST API. It calls the exact same `agents/*` / `projects/*`
logic as the CLI — no parallel business logic, no parallel state machine.

**Coverage:** all three modes are fully covered. Mode 1 (Systematic Review)
and Mode 3 (AI Research Assistant) are complete. Mode 2 (Research Notebook)
covers the core Q&A workflow (2a above) — create/rename/delete notebooks,
upload sources, chat with citations and Self-Reflective RAG status — plus
the 7-agent pipeline (2b), the advanced one-shot tools, Explain/Storyteller
(2c), and the Research Report workflow.

```
Browser (React SPA — Vite dev server, a static `vite build` output, or the
         build FastAPI serves directly from `frontend/dist` in Docker)
        │  fetch() — relative paths, proxied to :8000 by Vite (dev/preview)
        │  or served same-origin by FastAPI's StaticFiles mount (Docker only)
        ▼
FastAPI (backend/app/main.py)
  ├── routers/health.py               GET  /api/health
  ├── routers/research_assistant.py   Mode 3 — /api/research-assistant/...
  ├── routers/systematic_review.py    Mode 1 — /api/systematic-review/...
  └── routers/notebook.py             Mode 2 — /api/notebook/...
        │  each router delegates to a services/*_service.py
        ▼
services/*_service.py  →  agents/*.py, projects/*.py   (same modules the CLI calls)
```

**Docker:** `docker compose up --build` is the single command to start the app — it
brings up Ollama and the React/FastAPI web app at **http://localhost:8000**. Apple
Silicon users run `./scripts/start-mac.sh` instead (uses native Ollama). The root
`Dockerfile` is multi-stage: a `node:20-alpine` stage runs `npm run build` for
`frontend/`, then the final `python:3.11-slim` stage copies the built static assets
in, and `backend/app/main.py` mounts them with `StaticFiles(html=True)` at `/`
(registered after all the `/api/*` routers, so it only catches unmatched paths).
The standalone `frontend/Dockerfile` + `frontend/nginx.conf`
(multi-stage Node build → nginx) still work on their own
(`docker build -t beesearch-frontend ./frontend`) but are no longer wired
into the default Compose files.

Long-running calls (an SR pipeline run, a notebook chat turn) go through
`backend/app/jobs.py`'s in-memory, thread-based background job runner: the
endpoint returns a `job_id` immediately (HTTP 202), and the frontend polls
`GET /api/.../jobs/{job_id}` every 700ms until `status` is `done` or `error` —
the same progress reporting the CLI gets from `stream_callback`,
just surfaced over HTTP instead of updating a terminal directly.

**Dev/test-only mock mode:** setting `BEESEARCH_MOCK_LLM=1` before starting
the backend installs stubs (`backend/app/mock_llm.py`, `mock_search.py`) that
replace `ChatOllama` and the search backends with deterministic canned
responses — installed *before* any `agents.*` import, since `ChatOllama` is
bound into other modules' namespaces at import time and patching afterward
would be too late. This is what the Playwright E2E suite (`frontend/e2e/`)
runs against, so it needs neither a reachable Ollama server nor network access.

See the README's "Web App (React + FastAPI)" section for exact run commands.

---

## File Map

```
BeeSearch/
│
├── app.py                    ← Streamlit entry point; landing page dispatcher
├── main.py                   ← CLI — SR + Notebook modes
│
├── backend/                  ← FastAPI REST API (additive, alongside CLI/Streamlit)
│   └── app/
│       ├── main.py           ← App factory, CORS, mock-LLM bootstrap, router includes
│       ├── jobs.py           ← In-memory background job runner (chat/run polling)
│       ├── routers/          ← health, research_assistant, systematic_review, notebook
│       ├── services/         ← thin layer calling straight into agents/*, projects/*
│       └── schemas/          ← Pydantic request/response models
│
├── frontend/                 ← React + TypeScript SPA (Vite), talks to backend/ over REST
│   ├── Dockerfile            ← standalone multi-stage build -> nginx (optional; default Docker build instead serves frontend/dist via the root Dockerfile + FastAPI)
│   ├── src/
│   │   ├── api/              ← fetch wrapper (client.ts) + per-mode API clients
│   │   ├── pages/            ← one page per mode (SystematicReviewPage, NotebookPage, AskPage)
│   │   └── components/       ← mode-specific UI components
│   └── e2e/                  ← Playwright tests (run against the mock-LLM backend)
│
├── projects/
│   ├── __init__.py           ← PROJECT_REGISTRY {mode1, mode2, mode3}
│   ├── mode1_systematic_review.py  ← run(settings) — Systematic Review
│   ├── mode2_notebook.py           ← run(settings) — Research Notebook
│   └── mode3_research_assistant.py ← run(settings) — AI Research Assistant
│
├── ui/
│   ├── sidebar.py            ← render_sidebar() — hardware/model/RAG controls
│   ├── landing.py            ← render_landing() — 3-mode card layout
│   └── tabs/
│       ├── systematic_review.py  ← tab_systematic_review() — 4 result tabs (Explore = 8 tools)
│       ├── research_assistant.py ← tab_research_assistant() — free-form grounded Q&A
│       └── notebook.py           ← tab_notebook()
│
├── agents/
│   ├── systematic_review_state.py  ← SystematicReviewState TypedDict + factory
│   ├── systematic_review_nodes.py  ← 7 SR nodes (incl. quality_assessment)
│   ├── systematic_review_graph.py  ← build_systematic_review_graph()
│   ├── risk_of_bias.py             ← RoB 2 / ROBINS-I per-paper assessment
│   ├── grade_assessment.py         ← GRADE certainty-of-evidence rating
│   ├── contradiction_detector.py   ← cross-paper conflicting-findings detection
│   ├── research_assistant.py       ← run_research_assistant() — Mode 3 search→ground→cite
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
│   ├── story_state.py              ← StoryState TypedDict (Explain tab, internally "Mode 5")
│   ├── story_graph.py              ← build_story_graph() + run_story_turn()
│   ├── story_nodes.py              ← context_loader, repetition_tracker, source_router, storyteller, concept_visualizer, memory_saver nodes
│   ├── story_memory.py             ← StorytellerMemory (SQLite, independent of NotebookMemory)
│   │
│   ├── self_reflective_rag.py  ← grade_chunks(), grade_papers(), self_reflective_retrieve()
│   ├── eval_nodes.py           ← Quality self-evaluation nodes; non-blocking micro LLM call
│   └── feedback_agent.py       ← refine_with_feedback(); up to 3 rounds
│
├── tools/
│   ├── abstract_screener.py    ← LLM 0–100 paper relevance scorer
│   ├── citation_network.py     ← Ego citation graph (networkx + Pyvis HTML) + Smart Citation stances
│   ├── citation_context.py     ← Open-access citing-sentence extraction (best-effort)
│   ├── preprint_tracker.py     ← CrossRef preprint / retraction status
│   ├── prisma_report.py        ← PRISMA 2020 DOCX (python-docx) + PDF (reportlab)
│   ├── plain_language.py       ← Patient · Policy brief · Press release
│   ├── trend_analyzer.py       ← CrossRef facet year-count trends
│   ├── evidence_map.py         ← Plotly Population × Intervention bubble chart
│   ├── concept_drift.py        ← TF-IDF keyword shift across 5-year buckets
│   ├── meta_analysis.py        ← Pooled effect size + forest plot (fixed/random-effects)
│   ├── sensitivity_analysis.py ← Leave-one-out / subgroup sensitivity scenarios (library-level, no UI/CLI hook yet)
│   ├── literature_monitor.py   ← Saved-search snapshots + new-papers-since-last-run diff (library-level, no UI/CLI hook yet)
│   │
│   ├── document_tools.py       ← get_processor() auto-selects Docling or pdfplumber by page count; accepts vision_model param
│   ├── docling_processor.py    ← Advanced Docling parser; table_md extraction; figure captioning via VISION_MODEL (_extract_figure_chunks)
│   ├── hybrid_store.py         ← HybridStore: FAISS + ChromaDB + BM25 + RRF
│   ├── embeddings.py           ← OllamaEmbedder (batched /api/embed)
│   ├── search_tools.py         ← GoogleScholarSearcher + arXiv + Semantic Scholar + CrossRef + WebSearcher (DuckDuckGo, research-domain re-ranked)
│   ├── session_db.py           ← SQLite backend: init_db(), pack/unpack, DDL
│   ├── web_loader.py           ← URL → Document
│   ├── export_tools.py         ← DOCX + PDF export
│   ├── citation_tools.py       ← BibTeX + RIS export
│   ├── clarifier.py            ← Socratic clarifying questions
│   └── shutdown.py             ← Safe port release + ChromaDB flush
│
├── config/
│   ├── settings.py             ← Pydantic BaseSettings (env vars); includes VISION_MODEL for figure captioning
│   ├── hardware.py             ← detect_hardware() + recommend_config()
│   └── observability.py        ← get_langfuse_callbacks() + flush_langfuse() (opt-in LLM tracing)
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
├── tests/                       ← pytest unit tests, one file per concern, e.g.:
│   ├── test_citation_network.py            ← Gap-finder + isolated papers
│   ├── test_literature_review_citations.py ← Citation grounding (Literature Review)
│   ├── test_explain_citation_grounding.py  ← Citation grounding (Explain tab)
│   ├── test_explain_repetition.py          ← Repeated-clarification detection + concept visualizer (Explain tab)
│   ├── test_search_tools.py                ← WebSearcher dedup + research-domain re-rank
│   ├── test_temperature_levels.py          ← Response Tuning (Precise/Focused/Balanced/Creative)
│   ├── test_reference_checking.py          ← RoB/GRADE/contradiction state defaults + quality_assessment_node
│   ├── test_research_assistant.py          ← Mode 3 source numbering, citation rebuild, grounded/ungrounded paths
│   ├── test_citation_stance.py             ← Smart Citations stance parsing + classification + pyvis colouring
│   └── test_citation_context.py            ← Citation-context sentence matching + fulltext-fetch status paths
│
├── docker-compose.yml           ← React + FastAPI web app + Ollama (single `docker compose up --build`)
├── docker-compose.mac.yml       ← Apple Silicon: uses native Ollama, no ollama service
├── docker-compose.gpu.yml       ← GPU override (merge with docker-compose.yml)
├── docker-compose.langfuse.yml  ← Self-hosted Langfuse observability stack (optional)
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
| CLI | Rich ≥ 13 | Terminal panels, tables, Markdown |
| Config | pydantic-settings ≥ 2.0 | Typed env vars |
| Hardware Detection | psutil | Cross-platform RAM/CPU |
| Retry Logic | tenacity | Exponential backoff on API calls |
| Memory | SQLite (stdlib sqlite3) | `sessions.db`; WAL mode |
| Web API | FastAPI + Uvicorn | `backend/` — REST layer over the same agents/projects modules |
| Web Frontend | React 19 + TypeScript + Vite | `frontend/` — SPA; Vite dev-server proxy locally, served statically by FastAPI in Docker |
| Web Frontend Tests | Vitest + Testing Library, Playwright | Component tests; E2E against the mock-LLM backend |
| LLM Observability | Langfuse (optional) | Opt-in tracing of every ChatOllama call — prompts, completions, latency, tokens; self-hosted via `docker-compose.langfuse.yml` or Langfuse Cloud |
