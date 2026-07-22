"""backend/app/services/notebook_explain_service.py
──────────────────────────────────────────────────────
Service layer for Mode 2 Phase D: the Explain tab (agents/story_*.py's
storyteller pipeline -- internal "Mode 5", surfaced in ui/tabs/notebook.py
as the Explain tab).

Reuses the existing pipeline unmodified:

- ``agents.notebook_memory.NotebookMemory``, fresh-instantiated per call,
  for read-only access to a notebook's chunks/sources -- mirrors
  ``agents.notebook_advanced``'s own access pattern (every one of its 9
  functions does ``mem = NotebookMemory(); notebook = mem.load(notebook_id)``)
  rather than Phase A's shared ``notebook_service`` singleton, since this
  service only ever reads a notebook, never writes one.
- ``agents.story_memory.StorytellerMemory`` for the Explain conversation
  itself, via ``agents.story_nodes.build_numbered_doc_context`` for the
  citation-tagged document context handed to a new session.
- ``agents.story_graph.run_story_turn`` / ``agents.story_state.create_story_state``
  for the turn itself -- mirrors ``notebook_service.py::run_chat_turn``'s
  adapter pattern for turning a ``(node_name, final_state)`` callback into
  the job runner's ``(stage, info)`` shape.

Session identity: rather than invent a separate session-id concept (and a
mapping table) the way Streamlit's ``st.session_state`` tracks one
client-side, the StorytellerMemory session id *is* the notebook id
(``StorytellerMemory.new_session(..., session_id=notebook_id)``) -- a clean
1:1 mapping, since ``story_sessions`` and ``notebooks`` are separate SQLite
tables with no collision risk. ``_ensure_session`` creates that session,
seeded with the notebook's current document context, the first time
Explain is used for a given notebook; later turns reuse it as-is, matching
Streamlit's own behavior of capturing ``document_context`` once at session
creation rather than re-deriving it from the notebook's current sources on
every turn (not a bug to fix here -- preserving existing behavior exactly).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from agents.notebook_memory import NotebookMemory
from agents.story_graph import run_story_turn
from agents.story_memory import StorytellerMemory
from agents.story_nodes import build_numbered_doc_context
from agents.story_state import create_story_state
from config.settings import get_settings
from tools.text_parsing import format_page_label

from ..schemas.notebook_explain import ExplainRequest

logger = logging.getLogger(__name__)
cfg = get_settings()

_story_memory: StorytellerMemory | None = None


def _get_story_memory() -> StorytellerMemory:
    """Return the module-level lazy StorytellerMemory singleton, creating it on first use."""
    global _story_memory
    if _story_memory is None:
        _story_memory = StorytellerMemory()
    return _story_memory


# Mirrors the node names wired in `agents/story_graph.py::build_story_graph`.
NODE_LABELS: Dict[str, str] = {
    "context_loader": "Loading conversation context",
    "repetition_tracker": "Checking for repeated questions",
    "source_router": "Assessing document coverage",
    "storyteller": "Composing explanation",
    "concept_visualizer": "Building concept visualization",
    "memory_saver": "Saving turn",
    "story_eval": "Evaluating response quality",
}


def _normalize_citations(citations: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Stringify each citation's ``n`` (document excerpts are int, online
    sources are already "Source N" strings -- see agents/story_nodes.py::
    _build_citations_list) and add a display-ready ``page_label``, computed
    fresh and never persisted -- same approach as
    notebook_service.py::_with_page_labels."""
    if not citations:
        return []
    return [
        {**c, "n": str(c.get("n", "")), "page_label": format_page_label(c.get("page"))}
        for c in citations
    ]


def _ensure_session(notebook_id: str) -> None:
    """Create this notebook's Explain session on first use, seeded with its
    current document context. No-ops if the session already exists."""
    mem = _get_story_memory()
    if mem.load(notebook_id) is not None:
        return
    notebook = NotebookMemory().load(notebook_id) or {}
    doc_context = build_numbered_doc_context(notebook)
    doc_names = [s.get("filename", "") for s in notebook.get("sources", [])]
    topic = notebook.get("name") or "Untitled Notebook"
    mem.new_session(
        topic=topic,
        document_context=doc_context,
        document_names=doc_names,
        session_id=notebook_id,
    )


def build_initial_state(req: ExplainRequest) -> Dict[str, Any]:
    """Build the initial ``StoryState`` for one Explain turn.

    Only forwards optional overrides that were actually provided, deferring
    to ``create_story_state``'s own defaults otherwise -- the same
    "only override when given" pattern as
    ``notebook_service.py::build_initial_state``.
    """
    kwargs: Dict[str, Any] = {}
    if req.model:
        kwargs["model_name"] = req.model
    if req.num_ctx is not None:
        kwargs["num_ctx"] = req.num_ctx
    if req.temperature_level:
        kwargs["temperature_level"] = req.temperature_level
    return create_story_state(
        user_message=req.message.strip(),
        session_id=req.notebook_id,
        explanation_style=req.explanation_style,
        explanation_level=req.explanation_level,
        **kwargs,
    )


def run_explain_turn(
    req: ExplainRequest, stream_callback: Callable[[str, Dict[str, Any]], None]
) -> Dict[str, Any]:
    """Run one Explain turn; returns the raw final-state dict.

    Ensures the notebook's Explain session exists (seeded with its document
    context) before running, adapts ``run_story_turn``'s
    ``(node_name, final_state)`` callback into the ``(stage, info)`` shape
    the job runner expects, then post-processes citations the same way
    ``notebook_service.py::run_chat_turn`` does.
    """
    _ensure_session(req.notebook_id)
    initial_state = build_initial_state(req)

    def _adapter(node_name: str, final_state: Dict[str, Any]) -> None:
        stream_callback(node_name, {
            "label": NODE_LABELS.get(node_name, node_name),
            "progress_pct": final_state.get("progress_pct", 0),
        })

    final_state = run_story_turn(initial_state, stream_callback=_adapter)
    result = dict(final_state)
    result["citations"] = _normalize_citations(result.get("citations"))
    result["notebook_id"] = req.notebook_id
    result["user_message"] = req.message.strip()
    return result


def get_history(notebook_id: str, max_turns: int = 8) -> List[Dict[str, Any]]:
    """Return this notebook's Explain conversation history (empty if Explain
    has never been used for this notebook -- no session exists yet)."""
    turns = _get_story_memory().get_history(notebook_id, max_turns=max_turns)
    return [{**t, "citations": _normalize_citations(t.get("citations"))} for t in turns]
