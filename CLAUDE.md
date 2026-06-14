# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BeeSearch (repo: ResearchBuddy / BeeSearch) is a local-first AI research app with two
user-facing modes, built on LangGraph + Ollama (no cloud LLM, no API keys required):

- **Mode 1 — Systematic Literature Review**: PRISMA pipeline (Google Scholar, arXiv,
  Semantic Scholar, CrossRef → screening → evidence extraction → synthesis → PRISMA
  DOCX/PDF + plain-language summaries, trends, citation network, concept drift).
- **Mode 2 — Research Notebook**: NotebookLM-style grounded chat over uploaded
  documents, plus an "Explain" storyteller tab, a Research Report tab, and a
  7-agent analysis pipeline (study guide, podcast, knowledge graph, etc.).

Both modes are reachable from the Streamlit UI (`app.py`) and the CLI (`main.py`,
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

- `app.py` → `ui/landing.py` → `projects/{mode1_systematic_review,mode2_notebook}.py::run(settings)`,
  registered in `projects/__init__.py::PROJECT_REGISTRY`. `ui/tabs/notebook.py` is the
  large tab container for all of Mode 2 (Chat, Sources, Summary, FAQ, Literature Review,
  Mind Map, Knowledge Graph, Citation Timeline, Study Comparison, Pipeline, Research
  Report, Explain).
- `main.py` → `--systematic-review` / `--notebook` (+ one-shot `--notebook-*` flags)
  drive the same LangGraph graphs as the UI.

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
