# Tutorial — Using the Agentic Research Assistant

This guide covers everything you need to use the assistant effectively: CLI commands, per-mode examples, advanced flags, and built-in features.

---

## Table of Contents

1. [Running the Tests](#running-the-tests)
2. [CLI Reference](#cli-reference)
3. [Mode 1 — Literature Search](#mode-1--literature-search)
4. [Mode 2 — ProposalGPT](#mode-2--proposalgpt)
5. [Mode 3 — Wisdom Mode](#mode-3--wisdom-mode)
6. [Mode 4 — Systematic Review](#mode-4--systematic-review)
7. [Mode 5 — Research Notebook](#mode-5--research-notebook)
8. [Mode 6 — Grammar Proofreading](#mode-6--grammar-proofreading)
9. [Smart Recommendations](#smart-recommendations)
10. [Quality Scores (Self-Evaluation)](#quality-scores-self-evaluation)
11. [Feedback & Refinement](#feedback--refinement)
12. [Self-Reflective RAG](#self-reflective-rag)
13. [Writing Style Profiles](#writing-style-profiles)
14. [Socratic Clarification](#socratic-clarification)
15. [Tuning Hybrid RAG (`--embed-model`, `--top-k`)](#tuning-hybrid-rag---embed-model---top-k)
16. [Tuning Context Window (`--num-ctx`)](#tuning-context-window---num-ctx)
17. [Configuration Reference](#configuration-reference)
18. [UI Module Layout](#ui-module-layout)

---

## Running the Tests

```bash
pip install pytest          # one-time — pytest is not in requirements.txt
python -m pytest tests/ -v  # 496 tests, no Ollama required
```

The suite covers citation export (BibTeX/RIS format and key generation), memory CRUD for all six session types, Socratic fallback questions, all state factory functions, graph-level integration smoke tests for all modes (research, proposal_gpt, wisdom, notebook pipeline), `test_phase3_gaps.py` (self-evaluation framework, wisdom routing regex, node-level streaming, proposal memory saver separation, word-level cross-session tag overlap), `test_call_analyzer.py` (ProposalGPT funding call analysis), `test_notebook_pipeline.py` (7-agent pipeline), `test_notebook_advanced.py`, `test_feedback_agent.py` (feedback refinement), `test_self_reflective_rag.py` (self-reflective RAG — 38 tests), and `test_grammar_nodes.py` (Grammar Proofreading nodes, graph integration, memory — 36 tests). All tests run fully offline.

---

## CLI Reference

All six modes are available via `main.py`. **Docker users:** prefix every command with `docker compose exec app`.

```
Usage: python main.py [OPTIONS]

Literature Search (Mode 1)
  --goal, -g TEXT         Research goal or question (required)
  --model MODEL           Ollama LLM model name        (default: llama3.1:8b)
  --num-ctx INT           LLM context window in tokens (default: 32768)
  --embed-model MODEL     Ollama embedding model       (default: nomic-embed-text)
  --top-k INT             Chunks retrieved per query via RRF (default: 8)
  --web                   Also search the web via DuckDuckGo (in-process, no separate service required)
  --output, -o PATH       Save report to this path    (default: outputs/report_<id>.md)

ProposalGPT (Mode 2)
  --propose               Generate a new proposal (requires --goal and --call-file)
  --call-file FILE        Funding call document (PDF/DOCX/TXT) to analyse
  --list-proposals        List all saved ProposalGPT sessions
  --export-proposal SID   Export a saved proposal session to file

Wisdom Mode (Mode 3)
  --wisdom                Start a new Wisdom Mode session
  --wisdom-session SID    Continue an existing session by ID
  --scenario TEXT         Describe your specific situation (prompted if omitted)

Systematic Review (Mode 4)
  --systematic-review     Run a PRISMA systematic review (requires --goal).
                          Searches Google Scholar, arXiv, Semantic Scholar, and CrossRef.
  --inclusion TEXT [...]  One or more inclusion criteria strings
  --exclusion TEXT [...]  One or more exclusion criteria strings
  --sr-docx               Generate a PRISMA 2020-compliant Word document after the run
  --sr-pdf                Generate a PRISMA 2020-compliant PDF after the run
  --sr-author NAME        Author name for the DOCX/PDF title page
  --sr-institution NAME   Institution for the DOCX/PDF title page
  --sr-plain-language FMT Generate a lay-audience summary (patient | policy | press | all)
  --sr-trends             Fetch field-wide year-by-year publication counts from CrossRef
  --sr-preprints          Check each included paper for preprint/published/retracted status
  --sr-concept-drift      Analyse vocabulary evolution across 5-year time buckets

Research Notebook (Mode 5)
  --notebook              Start a new Research Notebook Q&A session
  --notebook-name TEXT    Name for the new notebook session
  --files, -f FILE [...]  Documents to ingest into the notebook

Grammar Proofreading (Mode 6)
  --grammar-check         Run Grammar Proofreading (requires --goal or --files)
  --grammar-session SID   Load an existing grammar session by ID
  --list-grammar          List saved grammar proofreading sessions
  --style-level LEVEL     Writing context: academic | professional_email | formal | informal
                          (default: professional_email)
  --focus AREA [...]      Focus areas: grammar punctuation spelling style clarity
                          (default: all areas)

Document Parsing
  --docling               Use Docling PDF parser (default; BooleanOptionalAction)
  --no-docling            Use the fast pdfplumber-only parser instead of Docling

Writing Style Profiles
  --style-profile NAME    Activate a named style profile for this run
  --create-style-profile  Analyse documents and create a new style profile
  --style-name NAME       Name for the new style profile (used with --create-style-profile)
  --list-style-profiles   List all saved style profiles

Socratic Clarification
  --no-clarify            Skip the pre-run clarifying questions (useful for scripting)

Utilities
  --check-system          Detect hardware, list pulled models, show recommendation
  --list-docs             List documents in the ChromaDB embedding cache
  --clear-store           Delete all ChromaDB embedding collections
  --verbose, -v           Enable debug logging
```

### Document parsing

Docling is the default PDF parser. It provides higher-quality extraction for complex layouts including multi-column text, tables, and figures. To use the faster pdfplumber-only parser instead, pass `--no-docling`:

```bash
# Default: Docling parser (high-quality, handles complex layouts)
python main.py --goal "Extract methods from uploaded papers" --files paper.pdf

# Fast pdfplumber-only parser (lower overhead, simpler documents)
python main.py --no-docling --goal "Quick scan of plain-text PDF" --files simple.pdf
```

### Progress visibility

The UI shows the current pipeline step name and a detail line as processing advances. Examples:

```
Indexing documents...   14 chunks indexed · FAISS + BM25 + RRF
Searching literature... Found 23 papers via Google Scholar · arXiv · Semantic Scholar · CrossRef
```

This makes it easy to follow what the pipeline is doing without any extra configuration.

### Smart error suggestions

If you mistype a flag, the CLI suggests the closest match:

```
$ python main.py --wisdon
Error: unrecognized arguments: '--wisdon'

Did you mean:
  --wisdom

Run python main.py --help for full usage.
```

---

## Mode 1 — Literature Search

Search arXiv, Semantic Scholar, and CrossRef — no local files needed. The pipeline decomposes your goal into sub-queries, fetches papers, fuses them via Hybrid RAG, and synthesises a cited report.

```bash
# Basic literature search
python main.py \
  --goal "Survey of large language models for biomedical NLP"

# Supplement with DuckDuckGo web search (in-process, no extra service needed)
python main.py \
  --goal "Current approaches to protein structure prediction" \
  --web

# Higher-precision embedding model
python main.py \
  --embed-model mxbai-embed-large \
  --goal "Extract all normative requirements from the literature"

# Save report to a specific path
python main.py \
  --goal "Identify statistical methods used in sleep deprivation research" \
  --output ./reports/methods_summary.md

# Skip clarification (useful for scripting)
python main.py \
  --goal "Effect of exercise on cognitive function" \
  --no-clarify
```

**Pipeline:** `query_generation → academic_search → document_analysis → reference_compilation → report_generation → memory_save`

**State type:** `ResearchState` | **Memory:** `sessions.db` → `research_sessions` table

---

## Mode 2 — ProposalGPT

Generate a full, funder-targeted research proposal from a funding call document. A 9-agent pipeline analyses the call, plans the research, reviews the literature, writes every section, estimates a budget, checks compliance, simulates peer review, and generates improvements.

```bash
# Generate a new proposal from a call document
python main.py --propose \
  --goal "Develop an ML framework for antibiotic resistance prediction \
          using whole-genome sequencing data" \
  --call-file horizon_call.pdf

# With a specific model and larger context
python main.py --propose \
  --num-ctx 65536 --model llama3.1:8b \
  --goal "Federated learning for privacy-preserving medical imaging" \
  --call-file vinnova_call.pdf

# List all saved proposal sessions
python main.py --list-proposals

# Export a saved proposal session
python main.py --export-proposal a3f7c9b2
```

### Agent pipeline

Agents run in order: `funding_call_analyzer → research_planner → literature_review_agent → proposal_writer → impact_agent → budget_agent → compliance_agent → reviewer_agent → improvement_agent`

### Budget format auto-detection

The `budget_agent` detects the funding format from the agency name in the call document:

| Agency keywords | Budget format | Indirect cost rate |
|----------------|--------------|-------------------|
| Horizon, ERC, MSCA | `horizon_europe` | 25% |
| Vinnova, VR, Formas | `swedish` | 20% |
| Any other | `generic` | 25% |

### Compliance scoring

The `compliance_agent` computes a fill-rate compliance score:
```
compliance_score = sections_present / sections_required × 100
```

### Virtual peer review

The `reviewer_agent` simulates 5 independent reviewers, each scoring a different dimension:

| Reviewer | Weight |
|---------|--------|
| Scientific merit | 30% |
| Impact | 25% |
| Innovation | 20% |
| Implementation | 15% |
| Agency fit | 10% |

The weighted aggregate produces `overall_reviewer_score`.

**State type:** `ProposalGPTState` | **Memory:** `sessions.db` → `proposal_sessions` table

---

## Mode 3 — Wisdom Mode

Describe a life or scientific scenario and receive validated wisdom: a rigorous scientific explanation, a plain-language version, actionable steps, and per-claim confidence scores — all backed by real academic sources. The agent asks up to 3 Socratic clarifying questions before searching academic databases.

```bash
# New session — scenario prompted interactively
python main.py --wisdom

# Provide scenario inline (skips the interactive prompt)
python main.py --wisdom \
  --scenario "I struggle to make good choices late in the day."

# Continue an existing wisdom session (follow-up Q&A)
python main.py --wisdom-session w9f2b4d1
```

**Session flow:**
1. Context loads from prior session (if resuming)
2. Up to 3 Socratic clarification turns
3. Academic search (arXiv + Semantic Scholar)
4. Wisdom synthesis (deep understanding, plain explanation, actionable steps, topic tags)
5. Wisdom validator adds per-claim confidence (High/Medium/Low) and devil's advocate caveat
6. Session saved; follow-up questions accepted

**Related sessions:** When viewing an active session in the UI, past sessions that share topic-tag word overlap appear as clickable buttons under "Related sessions:" in the left panel. Click any to load that session's wisdom output.

**State type:** `WisdomState` | **Memory:** `sessions.db` → `wisdom_sessions` + `wisdom_tags` tables

---

## Mode 4 — Systematic Review

Run a full PRISMA systematic review from a single command. The agent generates targeted search queries from a PICO-style research question, searches **Google Scholar, arXiv, Semantic Scholar, and CrossRef**, scores each abstract for relevance before screening, screens papers against your inclusion/exclusion criteria, extracts structured evidence, synthesises findings, and provides advanced analysis tools including citation networks, preprint tracking, research trends, evidence maps, concept drift analysis, DOCX/PDF manuscript generation, and plain-language summaries.

### Basic usage

```bash
python main.py --systematic-review \
  --goal "What is the effect of sleep deprivation on working memory?" \
  --inclusion "Peer-reviewed empirical studies" "Human participants" \
  --exclusion "Animal studies" "Review papers only"
```

### Generate a PRISMA 2020 manuscript (Word + PDF)

```bash
python main.py --systematic-review \
  --goal "Effect of exercise on cognitive decline in older adults" \
  --inclusion "RCTs or cohort studies" "Adults aged 60+" \
  --sr-docx --sr-pdf \
  --sr-author "Jane Smith" \
  --sr-institution "University of Example"
# Saves: outputs/prisma_report_<id>.docx  and  outputs/prisma_report_<id>.pdf
```

### Generate plain-language summaries

```bash
# Patient-facing (8th-grade reading level)
python main.py --systematic-review --goal "..." --sr-plain-language patient

# Policy brief (1-page, numbered recommendations)
python main.py --systematic-review --goal "..." --sr-plain-language policy

# Press release (inverted-pyramid, headline + lede)
python main.py --systematic-review --goal "..." --sr-plain-language press

# All three formats
python main.py --systematic-review --goal "..." --sr-plain-language all
# Saves: outputs/summary_patient_<id>.txt, summary_policy_<id>.txt, summary_press_<id>.txt
```

### Run trend analysis and preprint checks

```bash
python main.py --systematic-review \
  --goal "Machine learning for drug discovery" \
  --sr-trends          # year-by-year CrossRef publication counts
  --sr-preprints       # flag unverified preprints / retractions
  --sr-concept-drift   # vocabulary evolution across decades
```

### Combining flags

All post-run tools can be combined freely:

```bash
python main.py --systematic-review \
  --goal "Mindfulness-based interventions for anxiety" \
  --inclusion "RCTs" "Adults" "English language" \
  --exclusion "Animal studies" "Case reports" \
  --sr-docx --sr-pdf \
  --sr-plain-language all \
  --sr-trends \
  --sr-preprints \
  --sr-concept-drift \
  --sr-author "A. Researcher" \
  --sr-institution "Example University" \
  --output ./reports/anxiety_review.md
```

**Progress visibility:**

```
Searching Google Scholar · arXiv · Semantic Scholar · CrossRef... Found 47 papers
Screening papers...   28 included · 19 excluded
Synthesising findings...   done in 42.3s
```

**Output tabs in the UI:**

| Tab | Contents |
|-----|---------|
| Synthesis | Narrative synthesis, key themes, research gaps, conclusion, feedback refinement |
| Evidence Table | Study design, quality (High/Medium/Low), key finding per paper; excluded papers list |
| Discovery | Abstract screener scores with verdict filter; citation network (Pyvis interactive); preprint status |
| Trends & Analysis | Year-by-year trend chart (Plotly); evidence map (Population × Intervention bubble chart); concept drift vocabulary evolution |
| Export & Reports | Markdown download; DOCX + PDF PRISMA 2020 manuscript; plain-language summaries (patient / policy / press) |

**Pipeline:** `query_generation → literature_search (+ abstract screener) → screening → evidence_extraction → synthesis → sr_eval → END`

Post-run tools (triggered on demand in the UI or via CLI flags): `citation_network · preprint_tracker · trend_analyzer · evidence_map · concept_drift · prisma_report · plain_language`

Mode 4 is stateless — no memory file is written. Results are displayed in the UI and available for download.

**State type:** `SystematicReviewState`

---

## Mode 5 — Research Notebook

A two-part research tool for deep engagement with a body of literature.

### Part A — Q&A Chat

Upload documents and ask questions grounded in their content. Answers are inline-cited back to source chunks retrieved via Hybrid RAG.

```bash
# Start a new notebook session
python main.py --notebook --notebook-name "Transformer architectures"

# With documents to ingest
python main.py --notebook \
  --notebook-name "Sleep research" \
  --files paper1.pdf paper2.pdf review.docx
```

**Pipeline:** `retrieve → answer → save → END`

### Part B — 7-Agent Pipeline

Process uploaded documents into a structured study package automatically.

**Pipeline:** `ingest → summarize → retrieve → verify_citations → build_kg → generate_study_guide → generate_podcast`

| Agent | Output |
|-------|--------|
| `ingest` | Chunks and embeds all documents |
| `summarize` | Per-document and cross-document summaries |
| `retrieve` | Identifies key concepts and themes |
| `verify_citations` | DOI-checks all extracted references |
| `build_kg` | Knowledge graph of concepts and relationships |
| `generate_study_guide` | Key concepts, summaries, practice questions |
| `generate_podcast` | Podcast-style narrative script for audio consumption |

### Explain Feature

Embedded within the notebook tab, the Explain feature uses the `story_graph` pipeline to provide interactive science-communication-style explanations of specific concepts from loaded documents. Explanations follow four styles:

| Style | What it does |
|-------|-------------|
| `simple` | Plain-language explanation, no jargon (default) |
| `analogy` | One extended analogy carried throughout the response |
| `walkthrough` | Numbered step-by-step breakdown |
| `debate` | For-and-against structure showing competing perspectives |

**State types:** `NotebookState` (Q&A), `NotebookPipelineState` (pipeline), `StoryState` (Explain)

**Memory:** `sessions.db` → `notebooks` + `notebook_chunks` tables (Q&A sessions), `story_sessions` table (Explain sessions)

---

## Mode 6 — Grammar Proofreading

Paste text or upload a document to receive a professionally rewritten version suited to your writing context. The agent rewrites for clarity, fluency, and correctness — not just spell-check.

### Basic usage

```bash
# Proofread pasted text (professional email context by default)
python main.py --grammar-check \
  --goal "Dear Sir, I am writing with regards to your recent advertisement."

# Academic paper (PDF)
python main.py --grammar-check \
  --files paper_draft.pdf \
  --style-level academic

# Formal document, focus only on grammar and punctuation
python main.py --grammar-check \
  --files contract.docx \
  --style-level formal \
  --focus grammar punctuation

# Informal writing — preserves voice, corrects errors
python main.py --grammar-check \
  --goal "So I been thinking about this for ages and I think its really important." \
  --style-level informal

# All focus areas (default)
python main.py --grammar-check \
  --files blog_post.md \
  --style-level informal
```

### Writing contexts

| `--style-level` | Applied rules |
|----------------|--------------|
| `academic` | Third-person, no contractions, technical precision, hedging language, suitable for peer review |
| `professional_email` | Clear structure (subject → context → action → close), professional but warm, active voice |
| `formal` | Strict formal register, elevated vocabulary, no contractions; for legal/official/policy documents |
| `informal` | Preserves author voice and personality; contractions and colloquialisms acceptable; focus on clarity and natural flow |

### Focus areas

By default all focus areas are active. Narrow them with `--focus`:

```bash
# Grammar and punctuation only
python main.py --grammar-check --goal "..." --focus grammar punctuation

# Style and clarity only
python main.py --grammar-check --goal "..." --focus style clarity
```

Available areas: `grammar`, `punctuation`, `spelling`, `style`, `clarity`

### Output

The CLI prints:
1. **Polished text** — the primary output (fully rewritten)
2. **Change summary** — Markdown bullet list of what changed and why
3. **Issues table** — per-error type, original, suggestion, severity
4. **Style tips** — clarity, conciseness, tone, vocabulary suggestions
5. **Quality score** — 4-dimension self-evaluation (polish_quality, context_fit, error_coverage, fluency)

The polished text is also saved to `outputs/grammar_<session_id>.md`.

### Interactive feedback loop (CLI)

After the initial output, the CLI prompts for feedback. Up to 3 interactive revision rounds are supported in the CLI:

```
─── Feedback round 1 of 3 — press Enter to skip ───
Feedback> Make the tone more formal and add a clearer call-to-action.
```

The agent revises the polished text incorporating the feedback. Press Enter to skip. In the Streamlit UI, revision rounds are unlimited — you can keep refining the output as many times as needed without a round cap.

### Session management

```bash
# List saved grammar sessions
python main.py --list-grammar

# Resume an existing session
python main.py --grammar-session a3f7c9b2
```

### UI walkthrough

1. Navigate to **Mode 6: English Grammar Proofreading** on the landing page
2. Paste text in the text area **or** upload a PDF/DOCX/TXT/MD file
3. Select **Writing context** (Professional Email, Academic, Formal, Informal)
4. Optionally select **Focus areas** (default: all)
5. Click **Proofread**
6. View results in 4 tabs:
   - **Polished Text** — the primary output with MD/TXT download buttons
   - **Issues Found** — change summary + error table
   - **Style Tips** — per-suggestion expanders
   - **Summary** — word count before/after, error counts by type and severity
7. Use the **Refine this output** feedback box below the tabs to request revisions — there is no round limit in the UI

**State type:** `GrammarState` | **Memory:** `sessions.db` → `grammar_sessions` table

---

## Smart Recommendations

The CLI inspects your inputs and results at every stage and prints contextual tips automatically — no configuration needed.

**Pre-run (Mode 1)**

| Situation | Tip |
|-----------|-----|
| Goal mentions "recent", "2024", "latest" without `--web` | Add `--web` to supplement academic results |
| Broad overview goal | Use `--top-k 12` for wider coverage |

**Post-run (Mode 1)**

| Situation | Tip |
|-----------|-----|
| Few papers found, web off | Add `--web` |
| No references cited | Check the goal specificity |
| Short report | Upload documents with `--files` |
| Slow run time | Try `--top-k 4` or `--num-ctx 16384` |

**ProposalGPT (Mode 2)**

| Situation | Tip |
|-----------|-----|
| No `--call-file` provided | Specify a funding call document for better targeting |
| Short `--goal` | More detail improves all generated sections |
| Low compliance score | Reviewer feedback in `improvement_agent` output |

**Wisdom Mode (Mode 3) — per turn**

| Situation | Tip |
|-----------|-----|
| Clarifying phase | N rounds remaining — be specific |
| Last clarification round | Include all context — next answer triggers wisdom generation |
| Wisdom generated | Follow-up questions welcome; resume command shown |

**Example output:**

```
╭─ Pre-Run Recommendations ─────────────────────────────╮
│                                                           │
│ Your goal mentions recent or current work.             │
│    Tip: add --web to supplement academic results.         │
│                                                           │
│ Broad overview goal detected.                          │
│    Tip: --top-k 12 retrieves more chunks for wide         │
│    coverage reports.                                      │
│                                                           │
╰───────────────────────────────────────────────────────────╯
```

---

## Quality Scores (Self-Evaluation)

After every run, a non-blocking micro LLM call scores the output quality on mode-specific dimensions (each 1–5):

| Mode | Dimensions |
|------|-----------|
| Literature Search (Mode 1) | Goal alignment · Evidence quality · Clarity |
| ProposalGPT (Mode 2) | Compliance score · Overall reviewer score · Proposal completeness |
| Wisdom (Mode 3) | Evidence grounding · Confidence calibration · Actionability |
| Systematic Review (Mode 4) | Comprehensiveness · Evidence quality · Synthesis quality · Methodological rigour · Clinical utility |
| Research Notebook (Mode 5) | Answer grounding · Citation accuracy · Relevance |
| Grammar Proofreading (Mode 6) | Polish quality · Context fit · Error coverage · Fluency |

**In the UI** — a collapsed expander appears below each result:
```
Quality Score 4/5 — Strong evidence grounding with actionable advice…
  [ Evidence Grounding 4/5 ]  [ Confidence Calibration 4/5 ]  [ Actionability 5/5 ]
```
Score badges: 4–5 (strong) · 3 (acceptable) · 1–2 (weak)

**In the CLI** — a colour-coded Rich table is printed immediately after the primary output. The eval is always non-blocking — if the LLM call fails for any reason, the primary output is unaffected.

---

## Feedback & Refinement

Every mode supports feedback refinement after the pipeline completes. You are not locked into the first output.

### In the Streamlit UI

1. Run any mode as normal.
2. When results appear, scroll to the bottom — a collapsed **"Refine this output"** expander is shown.
3. Type your feedback instruction (e.g. "Add more detail to the methodology section") and click **Refine**.
4. The refined output appears below. A **"Revision history"** expander shows all previous versions and the feedback that produced each change.
5. You can refine as many times as needed — there is no round limit in the UI.

### In the CLI

After any pipeline finishes printing its results, the CLI prompts:

```
─── Feedback round 1 of 3 — press Enter to skip ───
Feedback>
```

Type your instruction and press Enter. The refined output is printed immediately and saved as a separate file (e.g. `report_abc123_refined.md`). Press Enter without typing to skip refinement. The CLI supports up to 3 interactive feedback rounds.

### What works best

- **Specific instructions**: "Focus the introduction on climate adaptation" works better than "improve it".
- **Citations are preserved**: The agent will not invent or remove references unless you explicitly ask.
- **Structure is preserved**: Markdown headings, tables, and lists are maintained unless you ask to change them.

---

## Self-Reflective RAG

Self-Reflective RAG adds a **post-retrieval relevance filter** to every mode. After each retrieval call, a single batched LLM call grades all retrieved items for relevance before they enter the main LLM context window. This prevents irrelevant chunks or papers from diluting the evidence base.

### How it works

```
retrieve (Hybrid RAG or academic search)
        │
        ▼
grade_chunks() / grade_papers()   ← single LLM call, temperature=0.0
  Returns {"grades": [true, false, ...]} — one bool per item
        │
        ▼
filter → keep only relevant items
        │
        ├─ ≥ 3 relevant? → pass to LLM context
        │
        └─ < 3 relevant (chunk modes only)?
             │
             ▼
           rewrite_query() → cycle 2 retrieval (max 2 cycles total)
             │
             ▼
           merge & deduplicate → pass to LLM context
```

### Coverage by mode

| Mode | What is graded | Function |
|------|----------------|----------|
| Mode 1 — Literature Search | Document chunks | `grade_chunks()` + up to 2 retrieval cycles |
| Mode 2 — ProposalGPT | Academic papers (arXiv + SS) | `grade_papers()` — one-pass filter |
| Mode 3 — Wisdom | Academic papers | `grade_papers()` — one-pass filter |
| Mode 4 — Systematic Review | Papers before screening | `grade_papers()` — reduces per-paper LLM workload |
| Mode 5 — Research Notebook | Document chunks | `grade_chunks()` + up to 2 retrieval cycles |
| Mode 6 — Grammar Proofreading | *(no retrieval)* | Not applicable — `rag_reflection_info = {}` |

### Fallbacks (always safe)

- LLM grading call fails → all items treated as relevant (no filtering)
- All items grade True on a large list → `grading_skipped=True`, original items returned
- Second retrieval cycle yields no new relevant items → original cycle-1 results returned
- The pipeline **always continues** — Self-Reflective RAG never blocks a run

### Viewing grading results

Grading results are surfaced automatically in both the UI and CLI after every run — no extra configuration needed.

**In the Streamlit UI** — a collapsed expander appears directly below the quality score expander for all five modes:
```
Self-Reflective RAG — 6/8 items passed grading (75%)
  [ Retrieved: 8 ]  [ Relevant: 6 ]  [ Pass Rate: 75% ]
  Cycles: 1
```
If a second retrieval cycle fired, the rewritten query is shown inside the expander.

**In the CLI** — a cyan Rich table is printed immediately after the quality score table:
```
┌─────────────── Self-Reflective RAG ───────────────┐
│ Metric          │ Value                            │
│ Retrieved       │ 8                                │
│ Passed grading  │ 6                                │
│ Pass rate       │ 75%                              │
│ Cycles          │ 1                                │
└──────────────────────────────────────────────────┘
```

If `grading_skipped` is true (all items returned true on a large batch — a sign of silent LLM failure), a warning row appears in both the UI and CLI output.

The raw metadata is also available in the LangGraph state as `state["rag_reflection_info"]`. For chunk-based modes (1 & 5) it is a list — one entry per query:
```python
{
    "query": "the search query",
    "cycles": 1,              # how many retrieval cycles ran
    "total_retrieved": 8,     # chunks fetched across all cycles
    "total_relevant": 6,      # chunks that passed grading
    "rewritten_queries": [],  # query rewrites used in cycle 2
    "grading_skipped": False, # True if grading was bypassed
}
```

For paper-based modes (2, 3, 4) it is a single dict:
```python
{
    "papers_retrieved": 10,   # before grading
    "papers_after_grading": 7 # after filtering
}
```

### Performance

Self-Reflective RAG adds **one extra LLM call per retrieval site** (typically 1–3 seconds with a local Ollama model). The grading LLM uses a compact context (`num_ctx=4096`, `num_predict=100`) to stay fast. If grading takes too long or fails, it falls back silently with no impact on the main pipeline.

---

## Writing Style Profiles

Capture your personal writing style from 2–5 of your own documents and apply it to all AI-generated prose — no fine-tuning required.

```bash
# Create a profile from your own papers or reports
python main.py --create-style-profile \
  --style-name "Academic Writing" \
  --files my_paper.pdf thesis_chapter.pdf

# List all saved profiles
python main.py --list-style-profiles

# Apply a profile to a literature search (Mode 1)
python main.py --style-profile "Academic Writing" \
  --goal "Survey of transformer architectures"

# Apply a profile to a proposal (Mode 2)
python main.py --propose \
  --style-profile "Grant Proposals" \
  --goal "ML for antibiotic resistance prediction" \
  --call-file horizon_call.pdf
```

The profile is also selectable from the sidebar in the Streamlit UI. Manage profiles in the **Style Profiles** tab. You can have multiple named profiles (e.g. "Academic Papers", "Grant Proposals", "Industry Reports") and switch between them per run.

---

## Socratic Clarification

Before every run, the assistant generates 2–3 focused questions tailored to your goal. Answers are injected into all prose-generating nodes to sharpen relevance.

**In the UI** — click "Clarify Requirements" below the goal input, answer the form fields, then click Run. A "Reset questions" button regenerates them if your goal changed.

**In the CLI** — questions appear interactively. Type the number for select-type questions, or type your answer for text questions. Press Enter to skip any question.

```bash
# Normal run — clarification prompts appear interactively
python main.py --goal "Compare BERT and GPT-4 on few-shot tasks"

# Skip clarification (useful for scripting)
python main.py --goal "..." --no-clarify

# Works for all modes
python main.py --propose --goal "..." --call-file call.pdf --no-clarify
python main.py --wisdom --no-clarify
python main.py --notebook --no-clarify
```

---

## Tuning Hybrid RAG (`--embed-model`, `--top-k`)

`--embed-model` sets the Ollama model used for dense vector search. `--top-k` controls how many chunks are retrieved per query and merged via Reciprocal Rank Fusion (RRF).

```bash
# Default: nomic-embed-text, 8 chunks per query
python main.py --goal "..."

# Higher-precision embedding
python main.py --embed-model mxbai-embed-large --goal "..."

# More chunks for broad topics
python main.py --top-k 15 --goal "Survey of all methods"

# Fewer chunks for specific factual questions
python main.py --top-k 4 --goal "What is the sample size reported?"
```

| Embedding model | Dimensions | Size | Notes |
|----------------|-----------|------|-------|
| `nomic-embed-text` | 768 | 274 MB | Default — fast, high quality |
| `mxbai-embed-large` | 1024 | 670 MB | Higher precision, slower |
| `bge-m3` | 1024 | 1.2 GB | Multilingual |
| `all-minilm` | 384 | 46 MB | Tiny — for very low RAM |

```bash
# Clear the embedding cache (free disk space)
python main.py --clear-store

# List what is currently in the cache
python main.py --list-docs
```

---

## Tuning Context Window (`--num-ctx`)

Controls how many retrieved chunks fit into a single LLM call. Match to your model's maximum supported context.

| Value | Approx. chunks | Good for |
|-------|----------------|---------|
| `8192` | ~6 | Low-RAM machines, quick tests |
| `16384` | ~12 | Mid-range laptops |
| `32768` | ~24 | Default — good all-round coverage |
| `65536` | ~49 | Broad topics with many relevant sections |
| `131072` | ~98 | `mistral-nemo:12b` (128k native context) |

```bash
# Standard run with default 32k window
python main.py --num-ctx 32768 --goal "Analyse methods"

# Long document with 128k model
python main.py --num-ctx 131072 --model mistral-nemo:12b --top-k 20 \
  --goal "Extract all normative requirements"
```

---

## Configuration Reference

All settings can be set in `.env` (copy from `.env.example`). Environment variables override the defaults shown below.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM model (must be pulled first) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `NUM_CTX` | `32768` | LLM context window in tokens |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for Hybrid RAG dense search |
| `CHROMA_PERSIST_DIR` | `./outputs/chroma_db` | ChromaDB embedding cache location |
| `HYBRID_TOP_K` | `8` | Chunks retrieved per query (dense + BM25 → RRF) |
| `CHUNK_SIZE` | `800` | Characters per document chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `MAX_SEARCH_RESULTS` | `8` | Papers fetched per academic search query |
| `ARXIV_MAX_RESULTS` | `10` | Max arXiv results per query |
| `CROSSREF_EMAIL` | — | Your email for CrossRef polite pool (faster responses) |
| `SEMANTIC_SCHOLAR_API_KEY` | — | Free key for higher Semantic Scholar rate limits |
| `CROSSREF_POLITE_POOL` | — | Same as `CROSSREF_EMAIL` alias |
| `OUTPUT_DIR` | `./outputs` | Directory for reports, proposals, and memory files |
| `APP_PORT` | `8501` | Streamlit UI host port (Docker only) |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |

---

## UI Module Layout

The Streamlit app is split across a `ui/` package so `app.py` stays small:

```
ui/
  helpers.py            ← shared helpers: uploads, clarification form,
  │                        citation downloads, eval result, report display,
  │                        render_feedback_section(), render_rag_reflection()
  sidebar.py            ← hardware/model/RAG/style/session sidebar
  tabs/
    search.py              ← Mode 1 — Literature Search
    proposal_gpt.py        ← Mode 2 — ProposalGPT
    wisdom.py              ← Mode 3 — Wisdom Mode
    systematic_review.py   ← Mode 4 — Systematic Review
    notebook.py            ← Mode 5 — Research Notebook (Q&A + pipeline + Explain)
    grammar_proofreading.py ← Mode 6 — Grammar Proofreading
    style_profiles.py      ← Style Profiles tab
```

Adding a new mode means creating `ui/tabs/<name>.py`, a `projects/modeN_<name>.py` entry point, and a card in `ui/landing.py`.
