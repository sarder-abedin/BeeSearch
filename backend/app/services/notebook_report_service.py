"""backend/app/services/notebook_report_service.py
───────────────────────────────────────────────────
Service layer for Mode 2 Phase E: the Research Report workflow
(``agents/graph.py`` + ``agents/state.py``).

Reuses the existing pipeline unmodified:

- ``agents.notebook_memory.NotebookMemory`` (read-only) plus the shared
  ``rebuild_processed_documents`` helper -- extracted from
  ``ui/tabs/notebook.py``'s own private ``_rebuild_processed_docs`` into
  ``agents.notebook_memory`` so both surfaces call the same code -- to turn
  a notebook's stored chunks into the ``ProcessedDocument`` list
  ``agents.state.create_initial_state`` expects as ``uploaded_docs``.
- ``agents.graph.run_research`` / ``agents.state.create_initial_state`` for
  the workflow itself -- mirrors ``notebook_pipeline_service.py::run_pipeline``'s
  adapter pattern for turning a ``(node_name, final_state)`` callback into
  the job runner's ``(stage, info)`` shape (here, ``run_research``'s
  callback signature is the same shape already).

``mode`` ("document" | "hybrid" | "search") is derived the same way
``ui/tabs/notebook.py::_tab_research_report`` derives it: a notebook with no
sources is always "search" (no notebook content to ground in regardless of
the academic-search toggle); otherwise "hybrid" or "document" depending on
``include_academic``. Re-derived fresh from the notebook's *current* sources
on every call rather than trusted from client input -- there is no
``mode`` request field.

Unlike Phases B-D, ``agents.graph``'s ``_llm`` helper does not call
``tools.temperature_levels.apply_temperature_level`` at all (confirmed by
inspection -- this pipeline predates the Response Tuning feature and was
never wired into it), so this service -- and ``ReportRequest`` -- has no
``temperature_level`` field. That is a pre-existing characteristic of
``agents/graph.py``, not a gap introduced here.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from agents.graph import run_research
from agents.notebook_memory import NotebookMemory, rebuild_processed_documents
from agents.state import create_initial_state
from config.settings import get_settings

from ..schemas.notebook_report import ReportRequest

logger = logging.getLogger(__name__)
cfg = get_settings()

# Mirrors `agents/graph.py::run_research`'s `_steps` list and
# `ui/tabs/notebook.py::_tab_research_report`'s own `_step_labels` dict.
NODE_LABELS: Dict[str, str] = {
    "document_ingestion": "Indexing notebook sources",
    "query_generation": "Generating search queries",
    "academic_search": "Searching arXiv + Semantic Scholar",
    "web_search": "Searching the web",
    "document_analysis": "Analysing sources",
    "reference_compilation": "Compiling references",
    "report_generation": "Generating report",
    "research_eval": "Evaluating quality",
}


def _resolve_mode(notebook: Dict[str, Any], include_academic: bool) -> str:
    """Mirror `_tab_research_report`'s mode derivation exactly: a notebook
    with no sources always searches the literature only, regardless of the
    academic-search toggle."""
    if not notebook.get("sources"):
        return "search"
    return "hybrid" if include_academic else "document"


def _normalize_references(references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stringify each reference's `year` (an int for academic papers via
    `Paper.year`, or the literal "" for web results -- see
    `agents/graph.py::_step_reference_compilation`), since the wire schema
    only declares one type. Computed fresh, not persisted."""
    return [{**r, "year": str(r.get("year") or "")} for r in references]


def build_initial_state(req: ReportRequest, notebook: Dict[str, Any]) -> Dict[str, Any]:
    """Build the initial Research Report state for one run.

    Unlike Phases B-D's `build_initial_state`, every field
    `agents.state.create_initial_state` takes is required (it has no
    internal defaulting of its own), so overrides not provided on the
    request fall back to server config here -- the same values
    `ui/tabs/notebook.py`'s sidebar `settings` dict would otherwise supply.
    """
    return create_initial_state(
        goal=req.goal.strip(),
        uploaded_docs=rebuild_processed_documents(notebook),
        mode=_resolve_mode(notebook, req.include_academic),
        include_web_search=req.include_web,
        model_name=req.model or cfg.ollama_model,
        num_ctx=req.num_ctx or cfg.num_ctx,
        embed_model=req.embed_model or cfg.embedding_model,
    )


def run_report(
    req: ReportRequest, stream_callback: Callable[[str, Dict[str, Any]], None]
) -> Dict[str, Any]:
    """Run the Research Report workflow; returns the raw final-state dict.

    Adapts `run_research`'s `(node_name, state)` callback into the
    `(stage, info)` shape the job runner expects.
    """
    notebook = NotebookMemory().load(req.notebook_id) or {}
    initial_state = build_initial_state(req, notebook)

    def _adapter(node_name: str, state: Dict[str, Any]) -> None:
        stream_callback(node_name, {
            "label": NODE_LABELS.get(node_name, node_name),
            "progress_pct": state.get("progress_pct", 0),
        })

    final_state = run_research(initial_state, stream_callback=_adapter)
    result = dict(final_state)
    result["notebook_id"] = req.notebook_id
    result["mode"] = initial_state["mode"]
    result["references"] = _normalize_references(result.get("references") or [])
    return result
