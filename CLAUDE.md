# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BeeSearch (repo: ResearchBuddy / BeeSearch) is a local-first AI research app with three
user-facing modes, built on LangGraph + Ollama (no cloud LLM, no API keys required):

- **Mode 1 — Systematic Literature Review**: PRISMA pipeline (Google Scholar, arXiv,
  Semantic Scholar, CrossRef → screening → PICO evidence extraction → quality assessment
  (risk of bias RoB 2/ROBINS-I, GRADE certainty, contradiction detection) → synthesis →
  PRISMA DOCX/PDF + plain-language summaries, trends, citation network with Smart Citation
  stance classification + citation-context snippets, concept drift).
- **Mode 2 — Research Notebook**: NotebookLM-style grounded chat over uploaded
  documents, plus an "Explain" storyteller tab, a Research Report tab, and a
  7-agent analysis pipeline (study guide, podcast, knowledge graph, etc.).
- **Mode 3 — AI Research Assistant**: stateless free-form question answering grounded in
  published literature with code-rebuilt inline citations (`agents/research_assistant.py`,
  `ui/tabs/research_assistant.py`, `main.py --ask`) — no upload, no PRISMA workflow.

All modes are reachable from the Streamlit UI (`app.py`) and the CLI (`main.py`,
plus `cli.py` for the section-by-section breakdown tool).

For deep dives beyond this file: `docs/architecture.md` (full pipeline diagrams,
state field lists, file map, tech stack), `docs/overview.md` (condensed version),
`README.md` (install/usage/CLI reference), `docs/FAQ.md`, `docs/tutorial.md`.

## Commands

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.1:8b
ollama pull nomic-embed-text          # required for Hybrid RAG in Research Notebook

# Run — Streamlit UI
streamlit run app.py

# Run — CLI
python main.py --check-system                          # hardware-aware model recommendation
python main.py --notebook --notebook-name "My Notes"   # Research Notebook session
python main.py --systematic-review --goal "..." \
  --inclusion "Peer-reviewed" --exclusion "Animal studies"
python cli.py sections <notebook-id> --source paper.pdf # section-by-section breakdown

# Docker
./scripts/start.sh        # Linux CPU
./scripts/start-gpu.sh    # Linux NVIDIA GPU
./scripts/start-mac.sh    # macOS
docker compose up --build
```

### Tests

```bash
python -m pytest tests/ -q                                     # full suite
python -m pytest tests/test_temperature_levels.py -q           # one file
python -m pytest tests/test_temperature_levels.py::test_precise_forces_full_determinism -q  # one test
python -m py_compile path/to/file.py                            # syntax check, no deps needed
```

`rich` and `streamlit` may not be installed in some sandboxes even though they're in
`requirements.txt`. If so, `py_compile` is enough for a syntax check; to exercise
`main.py`'s argparse logic, stub `sys.modules["rich"]` (and submodules) with
`MagicMock()` before importing `main`.

## Architecture

### Entry points and dispatch

- `app.py` → `ui/landing.py` → `projects/{mode1_systematic_review,mode2_notebook,mode3_research_assistant}.py::run(settings)`,
  registered in `projects/__init__.py::PROJECT_REGISTRY` (keys `mode1`/`mode2`/`mode3` must
  stay in sync across `PROJECT_REGISTRY`, `app.py::_PROJECT_MODULES`, and `ui/landing.py::_PROJECTS`).
  `ui/tabs/notebook.py` is the large tab container for all of Mode 2 (Chat, Sources, Summary,
  FAQ, Literature Review, Mind Map, Knowledge Graph, Citation Timeline, Study Comparison,
  Pipeline, Research Report, Explain). `ui/tabs/research_assistant.py` is the single-screen
  Mode 3 tab.
- `main.py` → `--systematic-review` / `--notebook` (+ one-shot `--notebook-*` flags) /
  `--ask` (Mode 3) drive the same logic as the UI. SR adds `--sr-quality` to print the
  risk-of-bias / GRADE / contradiction results.

### Internal "Mode N" numbering vs. user-facing modes

Docstrings and comments use internal mode numbers that don't match the README's
"Mode 1 / Mode 2":

- **Mode 7** = user-facing Mode 1 (Systematic Literature Review) — `agents/systematic_review_*.py`
- **Mode 8** = user-facing Mode 2 (Research Notebook) — `agents/notebook_*.py`
- **Mode 5** = old "Research Partner" (storytelling) — `agents/story_*.py`, now surfaced
  as Mode 2's **Explain** tab
- `agents/graph.py` + `agents/state.py` = a separate "Research Report" workflow, also a
  tab inside `ui/tabs/notebook.py`. Both Explain and Research Report degrade gracefully
  (warn + hide the tab) if their modules are missing.

### Per-pipeline file pattern

Every pipeline (SR, Notebook Q&A, Notebook 7-agent pipeline, Explain/story, Research
Report) follows the same layout under `agents/`:

- `*_state.py` — `TypedDict` + `create_*_state(...)` factory that sets all defaults
  (including `temperature_level` for Notebook-related states)
- `*_nodes.py` — node functions (or inlined in `*_graph.py` for smaller pipelines like
  Research Report); each module has a private `_llm` / `_make_llm(...)` ChatOllama factory
- `*_graph.py` — `build_*_graph()` and a `run_*_turn()`/`run_*()` entry point that
  assembles and invokes the LangGraph `StateGraph`
- `*_memory.py` (where persistence applies) — SQLite read/write helpers

When adding a feature to one pipeline, the analogous files in another pipeline are the
best template.

### LLM response tuning (temperature levels)

`tools/temperature_levels.py::apply_temperature_level(base_temperature, level)` is the
single source of truth for the user-tunable "Response Tuning" feature (Precise /
Focused / Balanced / Creative). It's called from the `_llm`/`_make_llm` factories in
`agents/notebook_nodes.py`, `agents/story_nodes.py`, `agents/notebook_advanced.py`, and
`agents/notebook_pipeline_nodes.py`. `level` flows from `state["temperature_level"]` /
`settings["temperature_level"]`, set via the sidebar "Response Tuning" control or
`/temperature <level>` in the CLI. Calls with `base_temperature <= 0.0` (grading /
faithfulness checks) are always forced to `0.0` regardless of level — this is
deliberate, not a bug.

### Citation grounding

Notebook Chat (`notebook_nodes.py::_build_context_block`), Literature Review
(`notebook_advanced.py::_build_numbered_excerpts`), and the Explain tab
(`story_nodes.py::build_numbered_doc_context`) all follow the same pattern: number
every individual chunk (not document) with its real page tag, bake the tag into the
context string handed to the LLM, then after generation regex-rebuild the References
list in code from whichever numbers the LLM actually cited — never trust the LLM's
own self-written references. When adding citations to a new pipeline, follow this
pattern rather than letting the LLM free-write its own References section.

### SR reference checking, Smart Citations, and Mode 3

- **SR `quality_assessment_node`** (`agents/systematic_review_nodes.py`, wired between
  `evidence_extraction` and `synthesis` in `systematic_review_graph.py`) runs three formerly
  dead-code modules and writes `rob_table` / `grade_results` / `contradictions` to state:
  `agents/risk_of_bias.py` (RoB 2 / ROBINS-I per paper), `agents/grade_assessment.py` (GRADE
  certainty of the whole body), `agents/contradiction_detector.py` (cross-paper conflicts +
  0–100 consensus). Each is independently try/except-wrapped — any failure degrades to an
  empty result, never blocks the pipeline (same "safe no-op" philosophy as self-reflective
  RAG). `synthesis_node` feeds these plus PICO fields into its narrative prompt. Surfaced in
  the UI's Explore → *Risk & Certainty* tool and on the CLI via `--sr-quality`. Paper caps are
  configurable via state (`max_evidence_papers`/`max_synthesis_papers`/`max_rob_papers`).
- **Smart Citations** (`tools/citation_network.py::classify_citation_stances`) optionally
  labels each citation-network edge Supporting/Contrasting/Mentioning from the two papers'
  abstracts (`_parse_stance` defaults to neutral Mentioning on any parse failure);
  `network_to_pyvis_html` colours edges by stance. `tools/citation_context.py` is best-effort,
  open-access-only citing-sentence extraction (`find_citation_mentions` is the pure, tested
  core; `_fetch_fulltext` is the only networked part). Both fail safe.
- **Mode 3 — AI Research Assistant** (`agents/research_assistant.py`) is stateless like the SR
  pipeline (no `*_memory.py`, no graph): `run_research_assistant()` does search → number
  sources into one `[n]` namespace → ground LLM answer → rebuild citations in code from the
  `[n]` actually cited. `build_numbered_sources`/`build_citations`/`_strip_llm_references_section`
  are pure and unit-tested; the search backends and ChatOllama are the only external deps.

### Explain tab: repeated-clarification detection + concept visualization

The Explain pipeline (`agents/story_graph.py`) runs `context_loader → repetition_tracker
→ source_router → storyteller → concept_visualizer → memory_saver → story_eval`.
`repetition_tracker_node` and `concept_visualizer_node` (`agents/story_nodes.py`) detect
when a user re-asks the same question or signals confusion ("I don't understand",
"still lost", …), and respond with both a different explanation style and an
interactive concept map — always on, no UI toggle, matching the tab's existing
automatic online-search behavior.

- **Detection is zero-LLM-call and deterministic**: Jaccard word-overlap similarity
  (stopword-stripped, threshold `0.4`, calibrated against real paraphrase pairs) between
  the current question and recent prior user questions, OR a match against
  `_CONFUSION_PHRASES`. Requires at least one prior assistant turn — a session's first
  message can never be "a repeat." An embeddings-based approach was considered and
  rejected: the added latency/fallback-handling would change the node's character for a
  problem word-overlap already solves well enough.
- **Style rotation reuses existing styles** (`simple`, `analogy`, `walkthrough`,
  `debate`) rather than inventing new categories — `_next_explanation_strategy` honors
  the user's current radio selection unless it matches what was already tried last turn,
  in which case it rotates to the next style in `_STYLE_ROTATION`, wrapping around. If
  the previous turn's `explanation_style` is unknown (sessions saved before this feature
  existed), the node keeps the user's current selection rather than guessing.
- **Concept visualization mirrors `tools/citation_network.py::network_to_pyvis_html`**:
  same Pyvis constructor/styling pattern, simplified to a single hub-and-spoke star graph
  (one central concept + up to 6 related nodes) since no graph algorithms run on it.
  Only triggers on a detected repeat — most turns skip its LLM extraction call entirely.
- **Fails safe like the rest of the codebase**: any failure in extraction, JSON parsing,
  or a missing `pyvis` import is caught and never blocks the primary explanation already
  produced by `storyteller_node` — same philosophy as `self_reflective_rag`'s "any
  grading failure is a safe no-op."
- **`concept_visual_html` is ephemeral**, like the pre-existing `online_results`/
  `source_decision` fields — available in `StoryState` for the current turn's UI render
  only, never persisted to `StorytellerMemory` (avoids SQLite bloat from Pyvis HTML
  blobs). `explanation_style` *is* persisted per assistant turn (`StorytellerMemory.
  add_turn(..., explanation_style=...)`) so future turns know what was already tried.

### Hybrid RAG + Self-Reflective RAG

- `tools/hybrid_store.py::HybridStore` — dense FAISS (`IndexFlatIP`, in-memory,
  per-session, no training) + sparse BM25 (`rank-bm25`) + ChromaDB (persistent
  embedding cache with MD5-based invalidation), fused via Reciprocal Rank Fusion (k=60).
  Falls back to BM25-only if the embedding model isn't pulled.
- `agents/self_reflective_rag.py` — post-retrieval LLM grading (`grade_chunks` for
  Notebook, `grade_papers` for SR), always `temperature=0.0`. Notebook retrieval gets up
  to 2 cycles with query rewrite if fewer than 3 chunks pass grading. Any grading
  failure is a safe no-op (all items kept).

### Config and lazy imports

- `config/settings.py::get_settings()` — `lru_cache`'d Pydantic `BaseSettings`
  singleton reading `.env`. New settings need a `Field(default, alias="ENV_VAR_NAME")`.
- `tools/__init__.py` — `__getattr__`-based lazy re-exports (`_EXPORTS` dict); importing
  `tools` does not pull in `faiss`/`chromadb`/`langchain_ollama` until a specific name is
  accessed. Add new public tool functions to `_EXPORTS` rather than importing the
  submodule eagerly.

### Memory

Research Notebook sessions persist in `outputs/memory/sessions.db` (SQLite, WAL mode):
`notebooks` table (metadata + conversation + `concepts_covered`) and `notebook_chunks`
(chunk text, loaded separately so listing notebooks stays cheap). Embeddings are cached
in ChromaDB under `outputs/chroma_db/`. The SR pipeline (Mode 1) is stateless — no DB
writes; outputs go to `outputs/`.
