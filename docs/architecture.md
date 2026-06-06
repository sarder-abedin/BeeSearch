# Architecture Deep-Dive

## System Overview

The Agentic Research Assistant is a **multi-mode, local-first AI research system** built on LangGraph state machines, Ollama LLMs, Hybrid RAG, and Self-Reflective RAG. Every computation runs locally — no cloud LLM, no paid API.

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interfaces                             │
│   Streamlit web UI (app.py)    CLI terminal (main.py)           │
│   Landing page → select mode   --goal / --propose /             │
│   (lazy: only selected mode    --wisdom / --systematic-review / │
│    code is imported)            --notebook                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
       ┌─────────────┼──────────────┬──────────────┐
       │             │              │              │
 ┌─────▼──────┐ ┌────▼──────┐ ┌────▼──────┐ ┌────▼──────────────┐ ┌────▼──────────┐
 │  Mode 1    │ │  Mode 2   │ │  Mode 3   │ │  Modes 4 & 5      │ │    Mode 6     │
 │ Literature │ │ProposalGPT│ │  Wisdom   │ │ Systematic Review │ │   Grammar     │
 │  Search    │ │ projects/ │ │  Mode     │ │ & Research        │ │ Proofreading  │
 │ projects/  │ │mode2_*    │ │ projects/ │ │ Notebook          │ │ projects/     │
 │mode1_*     │ │           │ │mode3_*    │ │ projects/mode4-5_*│ │ mode6_grammar │
 └─────┬──────┘ └────┬──────┘ └────┬──────┘ └────┬──────────────┘ └────┬──────────┘
       │             │              │              │
       └─────────────┼──────────────┴──────────────┴──────────────┘
                     │
         ┌───────────▼───────────┐
         │   LangGraph graphs    │  (one compiled StateGraph per mode)
         │   agents/*.py         │
         └───────────┬───────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
 ┌────▼─────┐ ┌──────▼──────┐ ┌───▼─────────────┐
 │ Hybrid   │ │  Academic   │ │   Memory        │
 │ RAG      │ │  Search     │ │  (SQLite WAL)   │
 │ FAISS    │ │  Google     │ │   outputs/      │
 │ BM25     │ │  Scholar    │ │  memory/        │
 │ ChromaDB │ │  arXiv      │ │  sessions.db    │
 │ RRF      │ │  Semantic   │ └─────────────────┘
 │          │ │  Scholar    │
 │ Modes    │ │  CrossRef   │
 │ 1 & 5   │ │  DuckDuckGo │
 │ (docs)  │ │  (ddgs)     │
 └────┬─────┘ └──────┬──────┘
      │              │
      └──────┬───────┘
             │
 ┌───────────▼──────────────────────────────────┐
 │   Self-Reflective RAG  (agents/              │
 │   self_reflective_rag.py)                    │
 │                                              │
 │   grade_chunks() — Modes 1 & 5              │
 │     batch LLM call grades retrieved chunks  │
 │     < 3 pass → rewrite query + cycle 2      │
 │                                              │
 │   grade_papers() — Modes 2, 3, 4            │
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

## Mode 1: Literature Search Graph

A single LangGraph `StateGraph` that searches arXiv, Semantic Scholar, and CrossRef, then synthesises a report with inline citations.

```
START
  │
  ▼
[query_generation]
  │  • LLM decomposes goal → 4–6 sub-queries
  │
  ▼
[academic_search]
  │  • arXiv API (free preprints)
  │  • Semantic Scholar API (peer-reviewed, citation counts)
  │  • CrossRef API (DOI resolution, optional)
  │  • Results sorted by citation count, deduplicated
  │
  ├── web_search enabled? ──► [web_search]
  │                              • DDGS().text(query) — DuckDuckGo in-process
  │                                (tools/search_tools.py, no service needed)
  │                              • Returns [] gracefully on network failure
  │                           ──► merge into academic results
  │
  ▼
[document_analysis]
  │  • HybridStore.search(query) per sub-query:
  │      FAISS dense top-2k + BM25 sparse top-2k
  │      → Reciprocal Rank Fusion (k=60)
  │      → top-K unique chunks (default k=8)
  │
  ▼
[reference_compilation]
  │  • Deduplicate by DOI / title similarity
  │  • Assign sequential [ref_num]
  │  • APA-style formatting
  │
  ▼
[report_generation]
  │  • LLM synthesises: Executive Summary, Key Findings,
  │    Methodology Notes, Detailed Analysis, Conclusions, References
  │
  ▼
[memory_save]          (ResearchMemory → research_sessions table in sessions.db)
  │
 END
```

**State type:** `ResearchState` (`agents/state.py`) — `TypedDict(total=False)`

**Memory:** `outputs/memory/sessions.db` — `research_sessions` table

---

## Mode 2: ProposalGPT Graph

A 9-agent LangGraph pipeline that generates a complete, funder-targeted research proposal from a call document and research goal.

```
START
  │
  ▼
[funding_call_analyzer]
  │  • Parses uploaded call document: funding agency, objectives, eligibility,
  │    budget constraints, evaluation criteria
  │  • Auto-detects budget format from agency name:
  │      Horizon/ERC/MSCA → horizon_europe (25% indirect costs)
  │      Vinnova/VR/Formas → swedish (20% indirect costs)
  │      others            → generic (25% indirect costs)
  │
  ▼
[research_planner]
  │  • Generates title, three objectives, target agency alignment
  │
  ▼
[literature_review_agent]
  │  • Same AcademicSearcher as Mode 1
  │  • Retrieves supporting literature for proposal grounding
  │
  ▼
[proposal_writer]
  │  • Drafts all proposal sections (introduction, methodology,
  │    expected outcomes, dissemination plan)
  │
  ▼
[impact_agent]
  │  • Writes societal, scientific, and economic impact sections
  │
  ▼
[budget_agent]
  │  • Generates line-item budget in agency-appropriate format
  │  • Applies indirect-cost rate from detected budget format
  │
  ▼
[compliance_agent]
  │  • Checks each mandatory section against the call requirements
  │  • Computes compliance_score via fill-rate formula:
  │      score = sections_present / sections_required × 100
  │
  ▼
[reviewer_agent]
  │  • Simulates 5 independent reviewers:
  │      scientific_merit    30%
  │      impact              25%
  │      innovation          20%
  │      implementation      15%
  │      agency_fit          10%
  │  • Weighted aggregate → overall_reviewer_score
  │
  ▼
[improvement_agent]
  │  • Synthesises reviewer feedback into concrete revision suggestions
  │  • Final proposal stored in state
  │
 END  (ProposalGPTState → proposal_sessions table in sessions.db)
```

**State type:** `ProposalGPTState` (`agents/proposal_gpt_state.py`)

**Memory:** `outputs/memory/sessions.db` — `proposal_sessions` table

**CLI flags:** `--propose` + `--call-file`, `--list-proposals`, `--export-proposal SESSION_ID`

---

## Mode 3: Wisdom Mode Graph (conditional routing)

Single-turn model; phase is persisted in `WisdomMemory`. Conversation continuity lives in the JSON file, not in graph state.

```
START
  │
  ▼
[context_loader]
  │  • Load conversation history, document_context, prior wisdom output
  │  • Count clarification_count (assistant turns with is_question=True)
  │  • find_related_sessions(topic_tags) → silently inject up to 3
  │    past wisdom snippets as passive context (no attribution)
  │
  ▼
[route_from_context]  ──── phase == "done"? ───► [wisdom_followup]
  │                                                    │
  │ (phase == "clarifying")                            ▼
  ▼                                             [memory_saver] → END
[clarification]   (temperature=0.6)
  │  • One Socratic question per turn (max 3 rounds)
  │  • If response matches PROCEED_TO_WISDOM (regex, case-insensitive,
  │    anywhere in the text) → phase = "ready_to_generate"
  │
  ▼
[route_after_clarification]
  │
  ├── phase == "ready_to_generate" ──►
  │       [knowledge_search]
  │         │  • LLM generates 3–5 academic queries
  │         │  • AcademicSearcher: arXiv + Semantic Scholar
  │         │  • Sort by citation count, cap at 15 papers
  │         │  • WebSearcher for supplementary web context
  │         ▼
  │       [wisdom_synthesis]   (temperature=0.5)
  │         │  • Formats paper_block + clarification context
  │         │    + passive cross-session snippets (silent)
  │         │  • Returns JSON:
  │         │      deep_understanding, simple_explanation,
  │         │      actionable_takeaways, topic_tags
  │         ▼
  │       [wisdom_validator]   (temperature=0.2)
  │         │  • LLM self-critique: per-claim confidence
  │         │    (High/Medium/Low), consensus label, devil's advocate
  │         │  • Composes assistant_response with confidence summary
  │         ▼
  │       [memory_saver]
  │         │  • save_wisdom(deep_understanding, simple_explanation,
  │         │                actionable_takeaways, validation, topic_tags)
  │         │  • update_phase(session_id, "done")
  │         ▼
  │        END
  │
  └── phase == "clarifying" ──► [memory_saver] → END
                                  (still asking questions)
```

**State type:** `WisdomState` (`agents/wisdom_state.py`)

**Memory:** `outputs/memory/sessions.db` — `wisdom_sessions` + `wisdom_tags` tables

**Cross-session enrichment:** `WisdomMemory.find_related_sessions()` uses an indexed SQL JOIN on the `wisdom_tags` table (individual tag words stored as rows, enabling efficient `WHERE word IN (...)` overlap queries) — no embeddings required. Top-3 overlapping sessions' `wisdom_snippet` (first 400 chars of `deep_understanding`) are silently injected into the synthesis prompt.

---

## Mode 4: Systematic Review Graph (linear)

A stateless linear PRISMA pipeline with a suite of on-demand post-synthesis analysis tools. No persistent memory file; results are shown in the UI and available for download in Markdown, DOCX, and PDF.

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
  │  • AcademicSearcher fans out across 4 sources:
  │      Google Scholar (scholarly, no API key — primary)
  │      arXiv (free preprints)
  │      Semantic Scholar (peer-reviewed, citation counts)
  │      CrossRef (DOI resolution, optional)
  │  • Deduplicates by normalised title slug
  │  • Sorts: peer-reviewed first, then by citation count desc
  │  • abstract_screener runs here: LLM scores each paper 0–100 against
  │    inclusion/exclusion criteria before formal screening
  │  • screener_scores stored in state for UI display
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
  │  • Self-evaluation: comprehensiveness, evidence_quality,
  │    synthesis_quality, methodological_rigour, clinical_utility (1–5)
  │
 END
```

### On-demand post-synthesis tools

Triggered from the UI (button click) or CLI flags. All are independent and non-blocking — they never re-run the core pipeline.

```
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
  │  • Corpus year distribution (instant, from fetched papers)
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

New fields added to state: `screener_scores`, `preprint_tracking`, `citation_graph_html`, `trend_data`, `evidence_map_data`, `concept_drift_data`

**Output tabs (UI):** Synthesis | Evidence Table | Discovery | Trends & Analysis | Export & Reports

**CLI:**
```bash
# Basic run
python main.py --systematic-review \
  --goal "Effect of sleep deprivation on working memory in university students" \
  --inclusion "Peer-reviewed empirical studies" "Human participants" "Published 2010–2024" \
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

## Mode 5: Research Notebook

Two parallel capabilities sharing a common tab in the UI.

### 5a — Q&A Chat (NotebookState)

Single-turn graph invocation per user message. Conversation continuity lives in `NotebookMemory` (JSON).

```
START
  │
  ▼
[retrieve]
  │  • HybridStore.search() over ingested notebook documents
  │  • Returns top-K chunks via RRF
  │
  ▼
[answer]
  │  • LLM synthesises answer grounded in retrieved chunks
  │  • Inline citations to source documents
  │
  ▼
[save]
  │  • NotebookMemory.add_turn(role="user", content=…)
  │  • NotebookMemory.add_turn(role="assistant", content=…)
  │
 END  (NotebookState → notebooks + notebook_chunks tables in sessions.db)
```

**State type:** `NotebookState` (`agents/notebook_state.py`)

**Memory:** `outputs/memory/sessions.db` — `notebooks` table (meta + conversation) + `notebook_chunks` table (one row per chunk)

### 5b — 7-Agent Pipeline (NotebookPipelineState)

Processes uploaded documents into a structured study package.

```
START
  │
  ▼
[ingest]
  │  • Docling (default) → layout-aware parsing, table extraction,
  │    PPTX/XLSX/HTML/image support → raw text + chunks
  │  • DocumentProcessor (fallback with --no-docling) → pdfplumber /
  │    python-docx / plain read → raw text + chunks
  │  • OllamaEmbedder → FAISS + ChromaDB cache
  │  • BM25Okapi index built from chunks
  │
  ▼
[summarize]
  │  • LLM generates per-document and cross-document summaries
  │
  ▼
[retrieve]
  │  • HybridStore.search() for key concepts and themes
  │
  ▼
[verify_citations]
  │  • DOIVerifier.verify() on extracted references
  │  • Flags broken or unresolvable DOIs
  │
  ▼
[build_kg]
  │  • Constructs a knowledge graph of concepts and relationships
  │    extracted from the documents
  │
  ▼
[generate_study_guide]
  │  • LLM generates structured study guide:
  │    key concepts, summaries, practice questions
  │
  ▼
[generate_podcast]
  │  • LLM generates a podcast-style narrative script
  │    for audio-friendly consumption of the material
  │
 END  (NotebookPipelineState)
```

**State type:** `NotebookPipelineState` (`agents/notebook_pipeline_state.py`)

### 5c — Explain Feature (story_graph)

Embedded within the notebook tab. Uses the `story_graph` pipeline to provide interactive science-communication-style explanations of specific concepts from the loaded documents.

```
START → [context_loader] → [storyteller] → [memory_saver] → END
```

**State type:** `StoryState` (`agents/story_state.py`)

**Memory file:** `outputs/memory/story_<session_id>.json`

---

## Mode 6: Grammar Proofreading Graph (linear)

A stateless linear pipeline — no external retrieval. All processing happens locally via the Ollama LLM. `rag_reflection_info` is always `{}`.

```
START
  │
  ▼
[text_loader]
  │  • Strip whitespace; count words and sentences
  │  • Emit informational warning to errors[] if estimated_chars > 60% of num_ctx×4
  │    (no truncation — model's context window governs what fits)
  │
  ▼
[grammar_analysis]  (temperature=0.1)
  │  • System: "You are an expert English proofreader. Return ONLY a JSON array…"
  │  • Identifies grammar/spelling/punctuation errors in raw_text
  │  • Returns: [{"type", "original", "suggestion", "explanation", "severity"}]
  │  • Parse: re.search(r"\[.*\]", raw, re.DOTALL) → json.loads
  │  • Fallback on parse failure: issues_found = []
  │
  ▼
[polish]  (temperature=0.2)  ← PRIMARY OUTPUT
  │  • System prompt: _STYLE_PROMPTS[style_level]
  │      "academic"          — third-person, no contractions, technical precision,
  │                            hedging language, suitable for peer review
  │      "professional_email"— clear structure, professional but warm, active voice
  │      "formal"            — elevated vocabulary, no contractions, authoritative
  │      "informal"          — preserve voice, contractions OK, focus on clarity
  │  • Human: raw_text + issues hints + "---CHANGES--- sentinel" instruction
  │  • Parse on sentinel "---CHANGES---":
  │      before → polished_text  (PRIMARY OUTPUT)
  │      after  → change_summary (Markdown bullet list)
  │  • Revision path (refinement_round > 0):
  │      Prepends: PREVIOUS VERSION + USER FEEDBACK to human message
  │
  ▼
[style_advisor]  (temperature=0.3)
  │  • Skips if neither "style" nor "clarity" in focus_areas AND focus_areas is non-empty
  │  • Skips if both polished_text and raw_text are empty
  │  • Returns: [{"category", "suggestion", "rationale"}]
  │  • Fallback on parse failure: style_suggestions = []
  │
  ▼
[grammar_eval]  (temperature=0.1)
  │  • Skips if polished_text is empty
  │  • Dimensions (each 1–5):
  │      polish_quality — is the polished text genuinely better?
  │      context_fit    — does the style match the requested writing context?
  │      error_coverage — were the detected errors addressed?
  │      fluency        — does the text read naturally and flow well?
  │  • Plus overall (1–5) and summary (one sentence)
  │
 END
```

**State type:** `GrammarState` (`agents/grammar_state.py`)

**Key state fields:**
```python
class GrammarState(TypedDict, total=False):
    raw_text: str               # input text (no hard cap)
    style_level: str            # "academic"|"professional_email"|"formal"|"informal"
    focus_areas: List[str]      # ["grammar","punctuation","spelling","style","clarity"]
    word_count: int
    sentence_count: int
    issues_found: List[Dict]    # [{type, original, suggestion, explanation, severity}]
    polished_text: str          # PRIMARY OUTPUT — fully rewritten, fluent text
    change_summary: str         # Markdown bullet list of what changed and why
    style_suggestions: List[Dict]  # [{category, suggestion, rationale}]
    feedback: str               # non-empty on revision rounds
    refinement_round: int       # 0 = initial, 1+ = revised
    feedback_history: List[Dict]
    eval_result: Dict           # polish_quality, context_fit, error_coverage, fluency
    rag_reflection_info: Dict   # Always {} — no retrieval in Mode 6
```

**Memory:** `outputs/memory/sessions.db` — `grammar_sessions` table

**CLI flags:** `--grammar-check` + `--goal` / `--files`, `--style-level`, `--focus`, `--grammar-session`, `--list-grammar`

---

## Hybrid RAG Pipeline

```
Document (PDF / DOCX / TXT / PPTX / XLSX / HTML / image)
        │
        ▼
  Docling (default parser — tools/document_tools.py)
  ├── Layout-aware PDF parsing, table extraction
  ├── PPTX, XLSX, HTML, and image support
  ├── clean_text    (control-char removal, whitespace normalise)
  └── chunk_text    (sliding window, chunk_size=800, overlap=150)

  DocumentProcessor (fallback with --no-docling)
  ├── extract_text  (pdfplumber / python-docx / plain read)
  ├── clean_text
  └── chunk_text
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
  • python main.py --clear-store  to free all disk space
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

**Fallback:** If `nomic-embed-text` (or the chosen embedding model) is not pulled, `HybridStore` raises `EmbeddingUnavailableError` and the system falls back to BM25 keyword search automatically. A warning banner is shown in the UI and via CLI indicating that dense embeddings are unavailable.

---

## Memory System

All modes persist state in a single SQLite database at `outputs/memory/sessions.db` (WAL mode, `PRAGMA foreign_keys=ON`, `PRAGMA synchronous=NORMAL`). The database is initialised by `tools/session_db.py::init_db()` on first use. Mode 4 (Systematic Review) is stateless and does not write a session; results are downloaded directly from the UI.

| Mode | Class | SQLite table(s) | Key fields |
|------|-------|----------------|------------|
| Literature Search (1) | `ResearchMemory` | `research_sessions` | goal, report, references, key_findings |
| ProposalGPT (2) | `ProposalMemory` | `proposal_sessions` | goal, proposal_markdown, compliance_score, reviewer_score |
| Wisdom (3) | `WisdomMemory` | `wisdom_sessions` + `wisdom_tags` | topic, scenario, phase, conversation, wisdom_output, topic_tags |
| Systematic Review (4) | *(stateless)* | *(none)* | Download `.md` from UI Export tab |
| Research Notebook (5) | `NotebookMemory` | `notebooks` + `notebook_chunks` | name, sources, conversation; chunks in separate table |
| Grammar Proofreading (6) | `GrammarMemory` | `grammar_sessions` | raw_text excerpt, style_level, polished_text, issues_found, eval_result |
| Style Profiles | `StyleMemory` | `style_profiles` | name, name_lower (indexed), sample_documents, analysis, injection_prompt |

**Session discovery:** `list_sessions()` / `list_notebooks()` / `list_profiles()` use `SELECT … ORDER BY updated_at DESC LIMIT ?` — no filesystem glob required.

**Notebook chunks split:** `NotebookMemory` stores source metadata and conversation in `notebooks.meta_json`, but chunk text in a separate `notebook_chunks` table (one row per chunk). `list_notebooks()` never loads chunk text; `load()` reconstructs the full dict by joining both tables. This avoids loading megabytes of text for a simple session list.

**Style Profiles** are not session-scoped — they are named, persistent profiles selected at run time. The profile's `injection_prompt` (~280 words) is appended to system prompts in all prose-generating nodes. When no profile is active the helper returns an empty string — zero overhead. `load_by_name()` uses the indexed `name_lower` column for O(log n) lookup.

**Cross-session enrichment (Wisdom only):** `find_related_sessions(topic_tags, current_id)` — SQL JOIN on `wisdom_tags` table (individual tag words stored as rows), no embedding required.

---

## Writing Style Profiles

Named profiles that capture a user's writing style from sample documents and inject it into every LLM prose call. No fine-tuning — the model is unchanged; only the system prompt grows.

### Analysis pipeline (`tools/style_profiler.py`)

```
upload 2–5 documents
     │
     ▼
_truncate_docs()          ← cap at 3 000 chars/doc, 8 000 chars total
     │
     ▼
LLM call 1 (temp=0.2)     ← extract JSON: {tone_formality, structure_format,
     │                         vocabulary_complexity, citation_evidence}
     ▼
LLM call 2 (temp=0.2)     ← synthesise compact instruction block (~280 words)
     │
     ▼
StyleMemory.create_profile() → style_profiles table in sessions.db
```

### Injection points

| Node | Mode |
|------|------|
| `document_analysis_node` | 1 |
| `report_generation_node` (exec summary) | 1 |
| `proposal_writer_node` | 2 |
| `wisdom_synthesis_node` | 3 |
| `answer_node` | 5 |

The `_style_block(state)` helper in each node file returns `""` when no profile is active — search, retrieval, and routing nodes are unaffected.

---

## Socratic Clarification

Before executing any mode, the system asks 2–3 focused questions tailored to the user's specific goal. Answers are injected into every prose-generating node to sharpen output relevance.

### Flow

```
User enters goal
      │
      ▼
generate_clarifying_questions()      ← tools/clarifier.py
  │  (one LLM call: temp=0.2, num_predict=512)
  │  Fallback: hardcoded per-mode questions if LLM fails
      │
      ▼
UI: radio / text form fields         ← _render_clarification_form() in app.py
CLI: interactive input()             ← _ask_clarifying_questions() in main.py
      │
      ▼
clarifications: Dict[str, str]       ← stored in state (all TypedDicts)
      │
      ▼
_clarification_context(state)        ← helper in each node file
  Returns "" when empty (zero overhead)
  Returns formatted block when answers provided
      │
      ▼
Injected into prose nodes:
  • query_generation_node (nodes.py)          Mode 1
  • document_analysis_node (nodes.py)         Mode 1
  • report_generation_node (nodes.py)         Mode 1
  • proposal_writer_node (proposal_gpt_nodes.py) Mode 2
  • wisdom_synthesis_node (wisdom_nodes.py)   Mode 3
  • answer_node (notebook_nodes.py)           Mode 5
```

### UI pattern (Modes 1–5)
- **Mode 1:** "Clarify Requirements" button below the goal input. Click generates questions as `st.radio()` / `st.text_input()` form widgets. "Reset questions" regenerates them.
- **Mode 2:** Same form appears above "Generate Proposal".
- **Mode 3:** Form appears at session start; answers are stored per `session_id` and carried across every turn in that session.
- **Mode 4:** Not applicable — systematic reviews use explicit inclusion/exclusion criteria text fields instead.
- **Mode 5:** Form appears at session start for the Q&A chat; answers are stored per `session_id`.

### CLI (`--no-clarify`)
Clarification is interactive by default. Pass `--no-clarify` to skip (for scripting):
```bash
python main.py --goal "..." --no-clarify
```

---

## Self-Evaluation Framework

After every mode completes its primary workflow, a dedicated **eval node** makes a single micro LLM call to score the output quality. This is entirely non-blocking — any LLM failure is caught and silently ignored, and the primary output is always returned regardless.

### How it works

```
primary workflow nodes → ... → memory_saver → eval_node → END
```

Each eval node:
1. Builds a compact prompt from key fields of the final state (goal, report excerpt, references count, etc.)
2. Calls the LLM with `temperature=0.1`, `num_predict=300`, `num_ctx=min(session_ctx, 8192)`
3. Extracts the first `{...}` JSON object from the response with `re.search(r"\{.*\}", raw, re.DOTALL)`
4. Stores the result in `state["eval_result"]`

### Dimensions per mode

| Mode | Dimensions (each 1–5) | Also |
|------|-----------------------|------|
| **Literature Search** (Mode 1) | `goal_alignment`, `evidence_quality`, `clarity` | `overall`, `summary` |
| **ProposalGPT** (Mode 2) | `compliance_score`, `overall_reviewer_score`, `proposal_completeness` | `overall`, `summary` |
| **Wisdom** (Mode 3) | `evidence_grounding`, `confidence_calibration`, `actionability` | `overall`, `summary` |
| **Systematic Review** (Mode 4) | `comprehensiveness`, `evidence_quality`, `synthesis_quality`, `methodological_rigour`, `clinical_utility` | `overall`, `summary` |
| **Research Notebook** (Mode 5) | `answer_grounding`, `citation_accuracy`, `relevance` | `overall`, `summary` |
| **Grammar Proofreading** (Mode 6) | `polish_quality`, `context_fit`, `error_coverage`, `fluency` | `overall`, `summary` |

Wisdom eval is skipped silently when `deep_understanding` is empty (clarification turns produce no wisdom yet).

### Graph wiring

| Graph file | Eval node | Wired after |
|------------|-----------|-------------|
| `agents/graph.py` | `research_eval_node` | `report_generation` |
| `agents/proposal_gpt_graph.py` | `proposal_eval_node` | `improvement_agent` |
| `agents/wisdom_graph.py` | `wisdom_eval_node` | `memory_saver` |
| `agents/systematic_review_graph.py` | `sr_eval_node` | `synthesis` |
| `agents/notebook_graph.py` | `notebook_eval_node` | `save` |
| `agents/grammar_graph.py` | `grammar_eval_node` | `style_advisor` |

### UI display

Rendered as a collapsed `st.expander` with a colour-coded score:

```
Quality Score 4/5 — Strong evidence grounding and actionable advice…
  ┌─────────────────────────────────────────────────┐
  │ Evidence Grounding  Confidence Calibration  ...  │
  │      4/5                   4/5              ...  │
  └─────────────────────────────────────────────────┘
  One-sentence summary from the LLM.
```

Score colours: 4–5 · 3 · 1–2 · missing

### CLI display (`main._print_eval_cli`)

Rendered as a Rich table with colour-coded rows — green for 4–5, yellow for 3, red for 1–2. Printed immediately after the mode's primary output.

### Feedback Refinement

After any pipeline completes, users can provide feedback to refine the output. In the Streamlit UI, refinement rounds are unlimited — the user can continue submitting feedback as many times as needed. In the CLI, up to 3 refinement rounds are supported via `agents/feedback_agent.py`. One call to `refine_with_feedback()` takes the current output text and user feedback, then returns an improved version (single non-blocking LLM call, `temperature=0.4`, `num_predict=4096`).

| Mode | Refined output | Feedback entry point |
|------|----------------|---------------------|
| Mode 1 — Literature Search | Research report | "Refine" expander below report / `Feedback>` CLI prompt |
| Mode 2 — ProposalGPT | Full assembled proposal Markdown | Export tab / `Feedback>` prompt after pipeline |
| Mode 3 — Wisdom | Assistant wisdom response | Below last assistant message / `Feedback>` prompt when phase=done |
| Mode 4 — Systematic Review | Narrative synthesis | Synthesis tab / `Feedback>` prompt after results |
| Mode 5 — Research Notebook | Study guide | Study guide tab / `Feedback>` prompt after pipeline |

Refinements are ephemeral in the UI (stored in `st.session_state`). In the CLI, the refined output is written as a separate `*_refined.md` file alongside the original.

---

## Self-Reflective RAG

**Module:** `agents/self_reflective_rag.py`

A post-retrieval relevance filter applied to all 5 modes. After every retrieval call, a single batched LLM call grades all retrieved items for relevance and filters out irrelevant ones before they enter the main LLM context window. This is a **Corrective/Adaptive RAG** variant — it adapts retrieval by either accepting the first cycle's relevant items or rewriting the query and fetching again.

### Coverage

| Mode | Retrieved items | Grading function | Orchestrator |
|------|----------------|-----------------|-------------|
| Mode 1 — Literature Search | Document chunks (HybridStore) | `grade_chunks()` | `self_reflective_retrieve()` — up to 2 cycles |
| Mode 2 — ProposalGPT | Academic papers (arXiv + SS) | `grade_papers()` | Direct call — one-pass |
| Mode 3 — Wisdom | Academic papers (arXiv + SS) | `grade_papers()` | Direct call — one-pass |
| Mode 4 — Systematic Review | Academic papers (pre-screening) | `grade_papers()` | Direct call — one-pass |
| Mode 5 — Research Notebook | Document chunks (HybridStore) | `grade_chunks()` | `self_reflective_retrieve()` — up to 2 cycles |
| Mode 6 — Grammar Proofreading | *(no retrieval)* | *(not applicable)* | — `rag_reflection_info = {}` always |

Modes 2–4 use academic API search (no HybridStore) — one-pass grading is sufficient because API results are already global. Mode 6 has no external retrieval and is excluded from SR-RAG entirely.

### `grade_chunks(chunks, query, model_name, num_ctx) → List[bool]`

Grades document chunks retrieved from HybridStore.

- **Input:** list of chunk dicts (with `text` key), query string
- **Prompt:** numbered list of chunks truncated to 400 chars each
- **LLM:** `temperature=0.0`, `num_predict=100`, `num_ctx=min(num_ctx, 4096)`
- **Expected response:** `{"grades": [true, false, true, ...]}`
- **Parse:** `re.search(r"\{.*\}", response, re.DOTALL)` → `json.loads` → bool array
- **Fallback:** any `Exception` or length mismatch → `[True] * len(chunks)`
- **Never raises**

### `grade_papers(papers, query, model_name, num_ctx) → List[bool]`

Grades academic paper dicts from arXiv / Semantic Scholar.

- **Input:** `List[Dict]` with at least `title` and `abstract` keys
- **Prompt:** numbered list — each entry: `[N] Title: {title}\nAbstract: {abstract[:300]}`
- **Same LLM settings and fallback as `grade_chunks`**

### `rewrite_query(original_query, model_name, num_ctx) → str`

Reformulates a query when too few chunks passed grading in cycle 1.

- **LLM:** `temperature=0.3`, `num_predict=100`
- **Returns:** first non-empty line of LLM response, or `original_query` on any failure

### `self_reflective_retrieve(store, query, top_k, model_name, num_ctx, max_cycles=2, min_relevant=3) → Tuple[List[Dict], Dict]`

Orchestrates multi-cycle chunk retrieval for Modes 1 and 5.

```
cycle 1:
  chunks = store.search_hybrid(query, k=top_k)
  grades = grade_chunks(chunks, query, ...)
  relevant = [c for c, g in zip(chunks, grades) if g]
  if len(relevant) >= min_relevant → return relevant, metadata

cycle 2 (fires only if cycle 1 passes < min_relevant items):
  rewritten = rewrite_query(original_query, ...)
  more_chunks = store.search_hybrid(rewritten, k=top_k)
  deduplicate by chunk_id across both cycles
  grade new chunks only
  merge cycle-1 relevant + new relevant
  return merged[:top_k], metadata
```

Safety guards:
- `store.search_hybrid` raises → return whatever is collected so far
- Empty result after grading → return original cycle-1 chunks (never starve the LLM)
- All grades `True` on a multi-item batch → `grading_skipped=True`, return original (silent LLM failure guard)
- Never raises

**Returns:** `(filtered_chunks, metadata)` where:
```python
metadata = {
    "cycles":            int,        # 1 or 2
    "total_retrieved":   int,        # chunks fetched across all cycles
    "total_relevant":    int,        # chunks that passed grading
    "rewritten_queries": List[str],  # populated if cycle 2 fired
    "grading_skipped":   bool,       # True if all-True silent-failure detected
}
```

### LLM factory: `_reflection_llm`

Mirrors `eval_nodes._eval_llm`. Uses `ChatOllama` with `httpx.Timeout(60.0)` and `num_ctx=min(num_ctx, 4096)` — a deliberately small context window to keep grading fast.

### State storage

`state["rag_reflection_info"]` holds grading metadata per run:

| Mode | Type | Contents |
|------|------|---------|
| Modes 1, 5 | `List[Dict]` | One entry per query: `{query, cycles, total_retrieved, total_relevant, rewritten_queries, grading_skipped}` |
| Modes 2, 3, 4 | `Dict` | `{papers_retrieved, papers_after_grading}` |

Default is `[]` (Modes 1, 5) or `{}` (Modes 2–4). The field is always present on all state TypedDicts.

### UI and CLI display

- **Streamlit UI** (`ui/helpers.py → render_rag_reflection()`) — collapsed expander below the quality score expander; shows retrieved count, relevant count, pass rate, cycle count, and rewritten query if cycle 2 fired
- **CLI** (`main.py → _print_rag_reflection_cli()`) — cyan Rich table printed immediately after the quality score table with the same metrics

Both helpers are no-ops when `rag_reflection_info` is empty — they never error.

### Fallback guarantee

| Condition | Result |
|-----------|--------|
| LLM call raises | All items relevant |
| JSON parse fails | All items relevant |
| Response length wrong | All items relevant |
| All grades `true` (multi-item) | `grading_skipped=True`; original returned |
| Cycle 2 still insufficient | Original cycle-1 relevant items returned |
| Empty input | Empty list, no LLM call |

---

## Web Search (DuckDuckGo)

Web search runs in-process via the `ddgs` library — no separate service or API key required.

```python
# tools/search_tools.py — WebSearcher.search()
from ddgs import DDGS

with DDGS() as ddgs:
    raw = list(ddgs.text(query, max_results=max_results))
return [SearchResult(title=r["title"], url=r["href"],
        snippet=r["body"][:500], source="duckduckgo") for r in raw]
```

- **No API key** — DuckDuckGo's search API is free and requires no account
- **Graceful degradation:** any exception returns `[]` — research continues without web results
- **MCP tool:** also exposed as `web_search(query, max_results)` via the MCP server for external AI clients

---

## MCP Server (`mcp_servers/research_tools_server.py`)

The project ships a [FastMCP](https://github.com/jlowin/fastmcp) server that exposes six research tools over the [Model Context Protocol](https://modelcontextprotocol.io) (JSON-RPC 2.0 over stdio). This lets Claude Desktop, Claude Code, and any MCP-compatible client call the project's search and RAG functions without launching the full Streamlit UI.

```
Claude Desktop / Claude Code
    │  JSON-RPC 2.0 (stdio)
    ▼
mcp_servers/research_tools_server.py  (FastMCP)
    ├── search_arxiv(query, max_results=10)         → List[dict]
    ├── search_semantic_scholar(query, max_results) → List[dict]
    ├── search_crossref(query, max_results)         → List[dict]
    ├── web_search(query, max_results=5)            → List[dict]
    ├── query_notebook(notebook_id, question, top_k=8) → dict
    └── ingest_document(content_base64, filename, notebook_id, ...) → dict
```

Registration in `.mcp.json` (project root):
```json
{
  "mcpServers": {
    "research-tools": {
      "command": "python",
      "args": ["mcp_servers/research_tools_server.py"]
    }
  }
}
```

Start the server manually for testing:
```bash
python mcp_servers/research_tools_server.py
```

---

## Hardware Detection

`config/hardware.py` is called at CLI startup and in the Streamlit sidebar.

```
detect_hardware()
  ├── platform.processor(), sys.platform  → cpu, os, arch
  ├── psutil.virtual_memory()             → ram_gb
  └── subprocess("nvidia-smi") / platform.machine()
        → gpu_type: "apple_silicon" | "nvidia" | "cpu"

get_available_models(ollama_base_url)
  └── GET /api/tags  → list of pulled model names

recommend_config(hw, available_models)
  └── Lookup table: ram_gb × gpu_type × model_size
        → {model, num_ctx, reasoning, hardware_note, can_run, pull_command}
```

The UI sidebar shows only pulled models in the dropdown. The recommended model is highlighted. Models not yet pulled show a `ollama pull <name>` hint.

---

## Engineering Decisions

### Rate-Limit Backoff (`tools/search_tools.py`)

All four `@retry` decorators use `retry=retry_if_exception(_is_retryable)` rather than a blanket retry. `_is_retryable()` returns `True` only for HTTP 429/500/502/503/504 responses and `ConnectionError`/`Timeout` — other failures (bad query, JSON parse error) fail immediately without retry. The wait strategy is `wait_exponential(min=2, max=30)` with `stop_after_attempt(4)`, giving a maximum delay of 2 + 4 + 8 = 14 seconds before the fourth attempt.

### MD5 Embedding Cache Invalidation (`tools/hybrid_store.py`, `tools/document_tools.py`)

`ProcessedDocument` carries a `content_md5` field (MD5 of `raw_text[:50000]`). `HybridStore.add_documents()` compares each document's hash against `self._manifest`. If they differ, `_invalidate_doc_cache(doc_name)` deletes all ChromaDB entries for that filename before re-embedding. This means re-uploading a modified document automatically refreshes its embeddings — no manual `--clear-store` required.

### DOI Verification (`tools/doi_verifier.py`)

`DOIVerifier.verify(doi)` issues a HEAD (falling back to GET) to `https://doi.org/{doi}`. HTTP 404/410 → `valid=False`; 200/301/302 → `valid=True`. Network errors (timeout, DNS failure) → `valid=True` with `reason="Could not verify"` — the system gives the benefit of the doubt rather than incorrectly flagging a valid DOI. `flag_references(refs)` adds a `doi_valid` boolean to each reference dict; warnings are surfaced in the UI References tab.

### BibTeX Key Generation

`_make_bibtex_key()` in `tools/citation_tools.py` uses `first_author.split()[0]` — the **first space-separated token** — as the last name. For the common `"Lastname Initials"` format (`"Vaswani A"`), `split()[0]` produces `"Vaswani"` → key `vaswani2017`. Using `split()[-1]` would produce the initial (`"A"` → `a2017`) — a bug that was fixed by correcting the index.

### Lazy Tool Imports (`tools/__init__.py`)

The `tools` package uses `__getattr__` for deferred loading. No submodule is imported until the name is first accessed. The loaded value is cached via `globals()[name] = value` so subsequent accesses are O(1). This ensures that importing `tools.citation_tools` (pure stdlib) does not trigger the import of `faiss`, `chromadb`, `tenacity`, or `langchain_ollama`.

### Lazy Memory Singletons in Node Files

`story_nodes.py`, `wisdom_nodes.py`, and `proposal_gpt_nodes.py` define memory objects as `None` at module level and populate them on first use via `_get_memory()`. This avoids creating `outputs/memory/` as a side effect of import, and allows test injection via `monkeypatch.setattr(module, "_memory", custom_instance)`.

### Integration Test Strategy

Five graph integration test files cover all modes end-to-end without Ollama.

| File | Nodes tested | Key scenarios |
|------|-------------|---------------|
| `test_integration_story.py` (6) | context_loader → storyteller → memory_saver | JSON parsing, 2-turn memory, concept extraction |
| `test_integration_research.py` (4) | 6-node research graph | mode="search", mocked AcademicSearcher + WebSearcher |
| `test_integration_proposal.py` (4) | proposal_gpt graph | call analysis, budget detection, reviewer simulation |
| `test_integration_wisdom.py` (5) | 7-node wisdom graph | clarifying path, forced-proceed → knowledge_search, phase="done" → wisdom_followup |
| `test_notebook_pipeline.py` | 7-agent notebook pipeline | ingest → summarize → retrieve → verify_citations → build_kg → guides |

Each test patches `ChatOllama` with a stateful `side_effect` counter that returns the correct response type per call, and injects real memory objects pointed at `tmp_path`.

---

## Technology Stack

| Layer | Tool | Version | License | Notes |
|-------|------|---------|---------|-------|
| LLM | Ollama + 14 supported models | ≥ 0.3 | MIT / various | Fully local, Metal/CUDA/CPU |
| Agent Framework | LangGraph | ≥ 0.2 | MIT | Compiled StateGraph per mode |
| LLM Toolkit | LangChain + langchain-ollama | ≥ 0.3 | MIT | Prompt templates, ChatOllama |
| Dense Embeddings | OllamaEmbedder → FAISS | ≥ 1.8 | MIT | In-memory IndexFlatIP |
| Embedding Cache | ChromaDB | ≥ 0.5 | Apache 2.0 | Persistent local DB |
| Sparse Retrieval | rank-bm25 (BM25Okapi) | ≥ 0.2.2 | Apache 2.0 | Keyword index, no GPU |
| RAG Fusion | RRF (custom, stdlib only) | — | — | k=60, score = Σ 1/(60+rank) |
| Document Parsing | Docling | ≥ 2.0 | MIT | Default parser: layout-aware, table extraction, PPTX/XLSX/HTML/images |
| PDF Extraction | pdfplumber | ≥ 0.11 | MIT | Fallback parser (--no-docling) |
| DOCX Extraction | python-docx | ≥ 1.1 | MIT | Fallback parser (--no-docling); also used for DOCX export |
| Academic Search | arxiv | ≥ 2.1 | MIT | arXiv API client |
| Academic Search | requests | ≥ 2.31 | Apache 2.0 | Semantic Scholar + CrossRef |
| Web Search | ddgs (DuckDuckGo) | ≥ 6.0 | MIT | In-process, no API key, no service |
| MCP Server | mcp + FastMCP | ≥ 1.0 | MIT | JSON-RPC 2.0 stdio server; 6 research tools |
| MCP Integration | langchain-mcp-adapters | ≥ 0.1 | MIT | Wire MCP tools into LangGraph agents |
| JSON Serialization | orjson | ≥ 3.10 | Apache 2.0 | 3–5× faster than stdlib json; SQLite BLOB columns |
| Export (DOCX) | python-docx | ≥ 1.1 | MIT | Proposal Word export |
| Export (PDF) | ReportLab | ≥ 4.2 | BSD | Proposal PDF export |
| Citation Export | stdlib `re` | — | — | BibTeX + RIS, no extra deps |
| UI | Streamlit | ≥ 1.37 | Apache 2.0 | Web app |
| CLI | Rich | ≥ 13 | MIT | Terminal panels, tables, Markdown |
| CLI Recommendations | stdlib `difflib`, `re` | — | — | Zero extra deps |
| Config | pydantic-settings | ≥ 2.0 | MIT | Typed env vars with aliases |
| Hardware Detection | psutil | ≥ 5.9 | BSD | Cross-platform RAM/CPU |
| Retry Logic | tenacity | ≥ 8.3 | Apache 2.0 | Exponential backoff on API calls |
| Memory | SQLite (stdlib `sqlite3`) + orjson | — | — | Single `sessions.db`; WAL mode; 9 tables for all modes |

---

## File Map

```
Agentic-Research-Assistant/
│
├── app.py                    ← Streamlit entry point; landing page dispatcher; lazy sub-project loading
├── main.py                   ← CLI — all 6 modes + smart recommendations
│                               flags: --goal, --propose, --call-file, --wisdom, --wisdom-session,
│                               --systematic-review, --notebook, --notebook-name,
│                               --grammar-check, --grammar-session, --list-grammar,
│                               --style-level, --focus,
│                               --list-proposals, --export-proposal SESSION_ID
│
├── projects/                 ← Sub-project modules (one per mode; lazy imports inside run())
│   ├── __init__.py           ← PROJECT_REGISTRY metadata dict
│   ├── _research_runner.py   ← Shared background-thread workflow runner for Mode 1
│   ├── mode1_search.py       ← run(settings) — Literature Search
│   ├── mode2_proposal.py     ← run(settings) — ProposalGPT
│   ├── mode3_wisdom.py       ← run(settings) — Wisdom Mode
│   ├── mode4_systematic_review.py ← run(settings) — Systematic Review
│   ├── mode5_notebook.py     ← run(settings) — Research Notebook
│   └── mode6_grammar.py      ← run(settings) — Grammar Proofreading
│
├── ui/                       ← Streamlit UI modules (split from app.py)
│   ├── helpers.py            ← Shared helpers: uploads, clarification form, citations, references
│   ├── sidebar.py            ← render_sidebar() — hardware/model/RAG/style controls (shared)
│   └── tabs/
│       ├── search.py         ← tab_search()             Mode 1 — Literature Search
│       ├── proposal_gpt.py   ← tab_proposal_gpt()       Mode 2 — ProposalGPT
│       ├── wisdom.py         ← tab_wisdom()             Mode 3 — Wisdom Mode
│       ├── systematic_review.py ← tab_systematic_review() Mode 4 — Systematic Review
│       ├── notebook.py       ← tab_notebook()           Mode 5 — Research Notebook
│       ├── grammar_proofreading.py ← tab_grammar_proofreading() Mode 6 — Grammar
│       └── style_profiles.py ← tab_style_profiles() — accessible from landing page expander
│
├── tests/                    ← pytest test suite (496 tests, 0 external deps)
│   ├── test_citation_tools.py         ← BibTeX/RIS format, key generation, collision handling
│   ├── test_story_memory.py           ← StorytellerMemory CRUD, add_turn, add_concepts
│   ├── test_research_memory.py        ← ResearchMemory save/load/list/delete
│   ├── test_notebook_memory.py        ← NotebookMemory CRUD
│   ├── test_notebook_nodes.py         ← Notebook Q&A node unit tests
│   ├── test_notebook_pipeline.py      ← 7-agent pipeline integration tests
│   ├── test_notebook_advanced.py      ← Advanced notebook features
│   ├── test_clarifier.py              ← Fallback questions, snake_case keys, question marks
│   ├── test_state_factories.py        ← All create_*_state() factories + clarifications field
│   ├── test_call_analyzer.py          ← ProposalGPT call document analysis
│   ├── test_proposal_gpt.py           ← ProposalGPT graph integration tests
│   ├── test_web_loader.py             ← Web content fetching
│   ├── test_integration_story.py      ← Story graph (mocked ChatOllama)
│   ├── test_integration_research.py   ← Research graph (mocked LLM + searchers)
│   ├── test_integration_proposal.py   ← Proposal graph integration
│   ├── test_integration_wisdom.py     ← Wisdom graph (clarify/proceed/followup)
│   ├── test_session_db.py             ← SQLite backend: init_db, pack/unpack, all 9 tables, CRUD (14 tests)
│   └── test_phase3_gaps.py            ← Self-eval, routing regex, streaming, memory saver
│
├── agents/
│   ├── state.py              ← ResearchState TypedDict
│   ├── graph.py              ← build_research_graph() + run_research()  (Mode 1)
│   ├── nodes.py              ← Research nodes (Hybrid RAG + academic search)
│   ├── memory.py             ← ResearchMemory
│   │
│   ├── eval_nodes.py         ← 5 self-evaluation nodes (one per mode); non-blocking micro LLM call
│   ├── feedback_agent.py     ← Mode-agnostic output refinement — `refine_with_feedback()`, `MAX_FEEDBACK_ROUNDS=3`
│   │
│   ├── proposal_gpt_state.py ← ProposalGPTState TypedDict
│   ├── proposal_gpt_graph.py ← build_proposal_gpt_graph() + run_proposal_gpt()  (Mode 2)
│   ├── proposal_gpt_nodes.py ← 9 proposal nodes; budget format detection; compliance score
│   │
│   ├── wisdom_state.py       ← WisdomState TypedDict (+ eval_result field)
│   ├── wisdom_graph.py       ← build_wisdom_graph() + run_wisdom_turn()  (Mode 3)
│   ├── wisdom_nodes.py       ← 7 nodes + 2 routing functions (regex PROCEED_TO_WISDOM)
│   ├── wisdom_memory.py      ← WisdomMemory (wisdom_sessions + wisdom_tags tables) + SQL tag overlap
│   │
│   ├── systematic_review_state.py  ← SystematicReviewState TypedDict + factory
│   ├── systematic_review_nodes.py  ← 6 nodes: query_generation → literature_search →
│   │                                  screening → evidence_extraction → synthesis → sr_eval
│   ├── systematic_review_graph.py  ← build_systematic_review_graph() + run_systematic_review()
│   │
│   ├── notebook_state.py     ← NotebookState TypedDict  (Mode 5 Q&A)
│   ├── notebook_graph.py     ← build_notebook_graph() + run_notebook_turn()  (Mode 5 Q&A)
│   ├── notebook_nodes.py     ← retrieve, answer, save nodes
│   ├── notebook_memory.py    ← NotebookMemory (notebooks + notebook_chunks tables)
│   ├── notebook_pipeline_state.py ← NotebookPipelineState TypedDict  (Mode 5 pipeline)
│   ├── notebook_pipeline_graph.py ← build_notebook_pipeline_graph()
│   ├── notebook_pipeline_nodes.py ← 7 pipeline nodes: ingest → summarize → retrieve →
│   │                                  verify_citations → build_kg → generate_study_guide →
│   │                                  generate_podcast
│   ├── notebook_advanced.py  ← Advanced notebook features (knowledge graph utilities)
│   │
│   ├── story_state.py        ← StoryState TypedDict  (Mode 5 Explain feature)
│   ├── story_graph.py        ← build_story_graph() + run_story_turn()
│   ├── story_nodes.py        ← 3 nodes: context_loader, storyteller, memory_saver
│   ├── story_memory.py       ← StorytellerMemory (story_<id>.json)
│   │
│   ├── grammar_state.py      ← GrammarState TypedDict + create_grammar_state()  (Mode 6)
│   ├── grammar_nodes.py      ← 4 nodes: text_loader, grammar_analysis, polish, style_advisor
│   │                            _STYLE_PROMPTS dict for context-specific rewrites
│   ├── grammar_graph.py      ← build_grammar_graph() + run_grammar_check()  (Mode 6)
│   ├── grammar_memory.py     ← GrammarMemory (grammar_sessions table)
│   │
│   ├── proposal_graph.py     ← Legacy proposal graph (retained for compatibility)
│   ├── proposal_nodes.py     ← Legacy proposal nodes
│   ├── proposal_state.py     ← Legacy ProposalState
│   │
│   └── style_memory.py       ← StyleMemory — named writing style profiles (style_profiles table)
│
├── tools/
│   ├── document_tools.py     ← Docling parser (default) + DocumentProcessor (fallback); content_md5 field
│   ├── embeddings.py         ← OllamaEmbedder (batched /api/embed)
│   ├── hybrid_store.py       ← HybridStore: FAISS + ChromaDB + BM25 + RRF;
│   │                            MD5 cache invalidation via _invalidate_doc_cache()
│   ├── vector_store.py       ← Vector store abstractions
│   ├── search_tools.py       ← AcademicSearcher + WebSearcher (ddgs); tenacity exponential
│   │                            backoff on 429/5xx via retry_if_exception(_is_retryable)
│   ├── session_db.py         ← SQLite backend: _tx(), init_db(), pack/unpack, 9-table DDL
│   ├── web_loader.py         ← Web content fetching
│   ├── export_tools.py       ← build_docx() + build_pdf()
│   ├── citation_tools.py     ← refs_to_bibtex() + refs_to_ris()
│   ├── doi_verifier.py       ← DOIVerifier: HEAD/GET to doi.org; flag_references()
│   ├── call_analyzer.py      ← Funding call document parser (Mode 2)
│   ├── proposal_tools.py     ← Shared proposal utilities
│   ├── funding_templates.py  ← FUNDING_TEMPLATES dict; get_template()
│   ├── style_profiler.py     ← analyse_writing_style() — two-step LLM analysis
│   ├── clarifier.py          ← generate_clarifying_questions() — Socratic pre-run form
│   ├── cli_recommender.py    ← Smart recommendation engine (all 5 modes)
│   └── shutdown.py           ← Graceful shutdown utilities
│
├── config/
│   ├── settings.py           ← Pydantic BaseSettings (env vars)
│   └── hardware.py           ← detect_hardware() + recommend_config()
│
├── mcp_servers/
│   └── research_tools_server.py  ← FastMCP server (6 tools: arXiv, SS, CrossRef, DuckDuckGo,
│                                    query_notebook, ingest_document); run with `python mcp_servers/research_tools_server.py`
│
├── .mcp.json                 ← MCP server registration (Claude Desktop / Claude Code)
│
├── outputs/
│   ├── chroma_db/            ← ChromaDB persistent embedding cache
│   ├── memory/
│   │   └── sessions.db       ← Single SQLite DB for all modes (9 tables, WAL mode)
│   ├── report_<id>.md
│   └── grammar_<id>.md       ← polished proofreading output saved by CLI
│
├── docker-compose.yml        ← 2 services: ollama + app (+ model-init one-shot)
├── .env.example
└── requirements.txt
```
