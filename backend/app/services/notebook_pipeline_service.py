"""backend/app/services/notebook_pipeline_service.py
────────────────────────────────────────────────────
Service layer for Mode 2 Phase B: the 7-agent Research Notebook analysis
pipeline.

Reuses the existing pipeline unmodified:

- ``agents.notebook_pipeline_graph.run_notebook_pipeline`` -- mirrors
  ``notebook_service.py::run_chat_turn``'s pattern of adapting a
  ``(node_name, final_state)`` stream_callback into the small ``(stage, info)``
  shape ``backend.app.jobs.run_in_background`` expects.
- ``agents.notebook_pipeline_state.create_pipeline_state`` for the initial state.
- ``tools.export_tools.build_docx`` / ``build_pdf`` and
  ``agents.notebook_advanced.render_dot_bytes`` for the study-guide and
  knowledge-graph export endpoints -- same helpers ``main.py::_cmd_notebook_pipeline``
  uses to write its own output files.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from agents.notebook_pipeline_graph import run_notebook_pipeline
from agents.notebook_pipeline_state import NotebookPipelineState, create_pipeline_state
from config.settings import get_settings

from ..schemas.notebook_pipeline import PipelineRequest

logger = logging.getLogger(__name__)
cfg = get_settings()

# Mirrors the node names wired in `agents/notebook_pipeline_graph.py::build_notebook_pipeline`.
NODE_LABELS: Dict[str, str] = {
    "ingest": "Agent 1 — Document Ingestion",
    "summarize": "Agent 2 — Summarization",
    "retrieve": "Agent 3 — Retrieval",
    "verify_citations": "Agent 4 — Citation Verification",
    "build_kg": "Agent 5 — Knowledge Graph",
    "generate_study_guide": "Agent 6 — Study Guide",
    "generate_podcast": "Agent 7 — Podcast Script",
    "notebook_pipeline_eval": "Evaluating pipeline quality",
}


def build_initial_state(req: PipelineRequest) -> NotebookPipelineState:
    """Build the initial ``NotebookPipelineState`` for one pipeline run.

    Only forwards optional overrides that were actually provided, deferring
    to ``create_pipeline_state``'s own defaults otherwise -- the same
    "only override when given" pattern as
    ``notebook_service.py::build_initial_state``.
    """
    settings: Dict[str, Any] = {}
    if req.model:
        settings["model"] = req.model
    if req.num_ctx is not None:
        settings["num_ctx"] = req.num_ctx
    if req.embed_model:
        settings["embed_model"] = req.embed_model
    if req.top_k is not None:
        settings["top_k"] = req.top_k
    if req.temperature_level:
        settings["temperature_level"] = req.temperature_level
    return create_pipeline_state(
        notebook_id=req.notebook_id,
        settings=settings,
        query=req.query or "",
    )


def run_pipeline(
    req: PipelineRequest, stream_callback: Callable[[str, Dict[str, Any]], None]
) -> Dict[str, Any]:
    """Run the full 7-agent pipeline; returns the raw final-state dict.

    Adapts ``run_notebook_pipeline``'s ``(node_name, final_state)`` callback
    into the ``(stage, info)`` shape the job runner expects.
    """
    initial_state = build_initial_state(req)

    def _adapter(node_name: str, final_state: Dict[str, Any]) -> None:
        stream_callback(node_name, {
            "label": NODE_LABELS.get(node_name, node_name),
            "progress_pct": final_state.get("progress_pct", 0),
        })

    final_state = run_notebook_pipeline(initial_state, stream_callback=_adapter)
    result = dict(final_state)
    result.setdefault("notebook_id", req.notebook_id)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Exports (sync -- no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def build_knowledge_graph_image(dot: str, fmt: str) -> Tuple[bytes, str]:
    """Render the pipeline's knowledge-graph DOT string to PNG or SVG bytes.

    Returns (image_bytes, error_string) -- mirrors
    ``agents.notebook_advanced.render_dot_bytes`` directly, the same helper
    ``main.py::_cmd_notebook_pipeline`` uses for its own ``.png``/``.svg`` files.
    """
    from agents.notebook_advanced import render_dot_bytes
    return render_dot_bytes(dot, fmt)


def build_study_guide_docx(study_guide: str) -> bytes:
    from tools.export_tools import build_docx
    return build_docx(study_guide, [])


def build_study_guide_pdf(study_guide: str) -> bytes:
    from tools.export_tools import build_pdf
    return build_pdf(study_guide, [])
