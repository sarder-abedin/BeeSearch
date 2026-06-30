"""backend/app/services/notebook_service.py
─────────────────────────────────────────────
Service layer for Mode 2 Phase A (Research Notebook core: notebook CRUD,
source upload/removal, conversation history, chat turns).

Reuses the existing pipeline unmodified:

- ``agents.notebook_memory.NotebookMemory`` for all persistence (same SQLite
  store the Streamlit tab already reads/writes).
- ``agents.notebook_graph.run_notebook_turn`` for chat turns -- mirrors
  ``systematic_review_service.py::run_sr``'s pattern of adapting a
  ``(node_name, final_state)`` stream_callback into the small ``(stage, info)``
  shape ``backend.app.jobs.run_in_background`` expects.
- ``tools.document_tools.DocumentProcessor`` (pdfplumber-based, not Docling)
  for upload parsing -- mirrors ``ui/helpers.py::process_uploads``'s per-file
  pattern (BytesIO -> process_file -> conditional raw_bytes for PDFs).
  Docling (PPTX/XLSX/HTML/images/OCR) remains Streamlit/CLI-only for now;
  this REST endpoint covers the same file types DocumentProcessor itself
  supports (PDF, DOCX, TXT, MD).
- ``tools.hybrid_store._stores`` cache eviction after any source add/remove
  -- mirrors ``ui/tabs/notebook.py``'s ``_index_and_store`` (add) and the
  "Remove"/"Delete notebook" button handlers (remove/delete), so
  ``retrieve_node`` rebuilds its FAISS/BM25 index fresh next turn instead of
  serving a stale one.

``page_label`` (a human-readable "p. N" string) is never persisted -- it's
computed fresh from each citation's raw ``page`` field every time citations
are read (live chat result or stored history), via :func:`_with_page_labels`.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agents.notebook_graph import run_notebook_turn
from agents.notebook_memory import NotebookMemory
from agents.notebook_state import NotebookState, create_notebook_state
from config.settings import get_settings
from tools.document_tools import DocumentProcessor
from tools.hybrid_store import _stores
from tools.text_parsing import format_page_label

from ..schemas.notebook import (
    ChatRequest,
    ConversationTurn,
    CreateNotebookRequest,
    NotebookDetail,
    NotebookSummary,
    SourceMeta,
    UploadSourceResult,
)

logger = logging.getLogger(__name__)
cfg = get_settings()

_memory: NotebookMemory | None = None


def _get_memory() -> NotebookMemory:
    """Return the module-level lazy NotebookMemory singleton, creating it on first use."""
    global _memory
    if _memory is None:
        _memory = NotebookMemory()
    return _memory


def _evict_store(notebook_id: str) -> None:
    """Discard the in-memory hybrid-search index for this notebook so
    ``retrieve_node`` rebuilds it from the freshest chunk set next turn."""
    _stores.pop(f"notebook_{notebook_id}", None)
    _stores.pop(f"notebook_{notebook_id}_bm25", None)


def _with_page_labels(citations: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not citations:
        return citations
    return [{**c, "page_label": format_page_label(c.get("page"))} for c in citations]


# ─────────────────────────────────────────────────────────────────────────────
# Notebook CRUD
# ─────────────────────────────────────────────────────────────────────────────

def _to_summary(nb: Dict[str, Any]) -> NotebookSummary:
    """Adapt a ``NotebookMemory.load()`` dict into a :class:`NotebookSummary`.

    ``list_notebooks()`` already returns this exact shape directly; this
    adapter is only needed right after ``new_notebook()``, whose only
    output is the new id.
    """
    return NotebookSummary(
        notebook_id=nb["notebook_id"],
        name=nb.get("name") or "Untitled",
        source_count=len(nb.get("sources", [])),
        turn_count=len(nb.get("conversation", [])),
        source_names=[s.get("filename", "") for s in nb.get("sources", [])],
        created_at=nb.get("created_at", ""),
        last_modified=nb.get("last_modified", ""),
    )


def create_notebook(req: CreateNotebookRequest) -> NotebookSummary:
    mem = _get_memory()
    nb_id = mem.new_notebook(req.name)
    nb = mem.load(nb_id)
    assert nb is not None  # just created, must exist
    return _to_summary(nb)


def list_notebooks(limit: int = 50) -> List[NotebookSummary]:
    return [NotebookSummary(**nb) for nb in _get_memory().list_notebooks(limit=limit)]


def get_notebook_detail(notebook_id: str) -> Optional[NotebookDetail]:
    nb = _get_memory().load(notebook_id)
    if nb is None:
        return None
    return NotebookDetail(
        notebook_id=nb["notebook_id"],
        name=nb.get("name") or "Untitled",
        source_count=len(nb.get("sources", [])),
        turn_count=len(nb.get("conversation", [])),
        sources=[SourceMeta(**s) for s in nb.get("sources", [])],
        conversation=[
            ConversationTurn(**{**t, "citations": _with_page_labels(t.get("citations"))})
            for t in nb.get("conversation", [])
        ],
        created_at=nb.get("created_at", ""),
        last_modified=nb.get("last_modified", ""),
    )


def notebook_exists(notebook_id: str) -> bool:
    return _get_memory().load(notebook_id) is not None


def delete_notebook(notebook_id: str) -> bool:
    deleted = _get_memory().delete(notebook_id)
    if deleted:
        _stores.pop(f"notebook_{notebook_id}", None)
    return deleted


def rename_notebook(notebook_id: str, new_name: str) -> Optional[NotebookSummary]:
    mem = _get_memory()
    if not mem.rename(notebook_id, new_name):
        return None
    nb = mem.load(notebook_id)
    return _to_summary(nb) if nb else None


def get_history(notebook_id: str, max_turns: int = 8) -> List[ConversationTurn]:
    turns = _get_memory().get_history(notebook_id, max_turns=max_turns)
    return [
        ConversationTurn(**{**t, "citations": _with_page_labels(t.get("citations"))})
        for t in turns
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Source upload / removal
# ─────────────────────────────────────────────────────────────────────────────

def upload_source(
    notebook_id: str,
    filename: str,
    file_bytes: bytes,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> UploadSourceResult:
    """Process one uploaded file and add it to the notebook.

    Raises ``KeyError`` if ``notebook_id`` doesn't exist (router -> 404) and
    lets ``DocumentProcessor.process_file``'s ``ValueError`` (unsupported
    file type) propagate (router -> 400).
    """
    mem = _get_memory()
    if mem.load(notebook_id) is None:
        raise KeyError(notebook_id)

    processor = DocumentProcessor(
        chunk_size=chunk_size if chunk_size is not None else cfg.chunk_size,
        overlap=chunk_overlap if chunk_overlap is not None else cfg.chunk_overlap,
        max_raw_chars=200_000,
        max_pages=300,
    )
    file_obj = io.BytesIO(file_bytes)
    doc = processor.process_file(Path(filename), file_obj=file_obj)
    if Path(filename).suffix.lower() == ".pdf":
        doc.raw_bytes = file_bytes

    if not mem.add_source(notebook_id, doc, source_type="file"):
        return UploadSourceResult(added=False, duplicate=True, source=None)

    if doc.raw_bytes:
        mem.add_source_file(notebook_id, doc.doc_id, doc.filename, doc.raw_bytes)

    _evict_store(notebook_id)

    updated = mem.load(notebook_id) or {}
    source_dict = next(
        (s for s in updated.get("sources", []) if s.get("doc_id") == doc.doc_id), None
    )
    return UploadSourceResult(
        added=True,
        duplicate=False,
        source=SourceMeta(**source_dict) if source_dict else None,
    )


def remove_source(notebook_id: str, doc_id: str) -> bool:
    removed = _get_memory().remove_source(notebook_id, doc_id)
    if removed:
        _evict_store(notebook_id)
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# Chat (background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

# Mirrors the node names wired in `agents/notebook_graph.py::build_notebook_graph`.
NODE_LABELS: Dict[str, str] = {
    "retrieve": "Retrieving relevant sources",
    "answer": "Composing grounded answer",
    "save": "Saving conversation turn",
    "notebook_eval": "Evaluating answer quality",
}


def build_initial_state(req: ChatRequest) -> NotebookState:
    """Build the initial ``NotebookState`` for one chat turn.

    Only forwards optional overrides that were actually provided, deferring
    to ``create_notebook_state``'s own defaults otherwise -- the same
    "only override when given" pattern as
    ``research_assistant_service.build_settings``.
    """
    kwargs: Dict[str, Any] = {}
    if req.model:
        kwargs["model_name"] = req.model
    if req.num_ctx is not None:
        kwargs["num_ctx"] = req.num_ctx
    if req.embed_model:
        kwargs["embed_model"] = req.embed_model
    if req.top_k is not None:
        kwargs["top_k"] = req.top_k
    if req.temperature_level:
        kwargs["temperature_level"] = req.temperature_level
    return create_notebook_state(
        user_message=req.message.strip(),
        notebook_id=req.notebook_id,
        include_web_search=req.include_web_search,
        **kwargs,
    )


def run_chat_turn(
    req: ChatRequest, stream_callback: Callable[[str, Dict[str, Any]], None]
) -> Dict[str, Any]:
    """Run one Research Notebook chat turn; returns the raw final-state dict.

    Adapts ``run_notebook_turn``'s ``(node_name, final_state)`` callback into
    the ``(stage, info)`` shape the job runner expects, then post-processes
    the result's citations to add a display-ready ``page_label`` (computed,
    never persisted -- see module docstring).
    """
    initial_state = build_initial_state(req)

    def _adapter(node_name: str, final_state: Dict[str, Any]) -> None:
        stream_callback(node_name, {
            "label": NODE_LABELS.get(node_name, node_name),
            "progress_pct": final_state.get("progress_pct", 0),
        })

    final_state = run_notebook_turn(initial_state, stream_callback=_adapter)
    result = dict(final_state)
    result["citations"] = _with_page_labels(result.get("citations", [])) or []
    return result
