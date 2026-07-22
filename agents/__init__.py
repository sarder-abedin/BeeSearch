"""
agents/__init__.py
───────────────────
Public re-export surface for the `agents` package.

Pulls the state/graph/memory entry points for all three LangGraph pipelines —
Systematic Review (Mode 7), Research Notebook (Mode 8) Q&A, and the Notebook
7-agent pipeline — into `agents.*` so callers (UI tabs, `main.py`, CLI tools)
don't need to know each pipeline's submodule layout. Story (Mode 5) and the
Research Report workflow (`agents/graph.py` / `agents/state.py`) are imported
directly by their callers instead of being re-exported here, since both
degrade gracefully (warn + hide their UI tab) if missing.
"""

from agents.systematic_review_state import SystematicReviewState, create_systematic_review_state
from agents.systematic_review_graph import build_systematic_review_graph, run_systematic_review
from agents.notebook_state import NotebookState, create_notebook_state
from agents.notebook_graph import build_notebook_graph, run_notebook_turn
from agents.notebook_memory import NotebookMemory
from agents.notebook_advanced import (
    generate_cross_document_summary,
    generate_faq,
    generate_literature_review,
    generate_mindmap,
    generate_audio_summary,
    compare_sources,
    extract_knowledge_graph,
)
from agents.notebook_pipeline_state import NotebookPipelineState, create_pipeline_state
from agents.notebook_pipeline_graph import build_notebook_pipeline, run_notebook_pipeline

__all__ = [
    "SystematicReviewState", "create_systematic_review_state",
    "build_systematic_review_graph", "run_systematic_review",
    "NotebookState", "create_notebook_state",
    "build_notebook_graph", "run_notebook_turn", "NotebookMemory",
    "generate_cross_document_summary", "generate_faq", "generate_literature_review",
    "generate_mindmap", "generate_audio_summary", "compare_sources",
    "extract_knowledge_graph",
    "NotebookPipelineState", "create_pipeline_state",
    "build_notebook_pipeline", "run_notebook_pipeline",
]
