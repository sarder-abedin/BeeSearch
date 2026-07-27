"""
tools/bridge.py
─────────────────
SR → Notebook bridge: hands a completed Systematic Review off to Mode 2.

Renders the SR state's narrative synthesis, themes, gaps, conclusion, and
evidence table as a single Markdown document, then ingests it into a
Research Notebook (new or existing) through the same `DocumentProcessor` /
`NotebookMemory` path used for uploaded files, so the review becomes
chattable like any other notebook source.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple
logger = logging.getLogger(__name__)

def sr_to_notebook(sr_state: Dict[str, Any], notebook_id: Optional[str], notebook_name: Optional[str], settings: dict) -> Tuple[str, str]:
    """Convert a completed SR state into a Markdown doc and add it to a notebook.

    Creates a new notebook (named from `notebook_name` or the research
    question) when `notebook_id` is falsy, otherwise appends to the
    existing notebook. Invalidates that notebook's cached hybrid-RAG store
    so the next query re-embeds with the new source included.

    Returns:
        `(notebook_id, error)` — `error` is `""` on success, otherwise a
        short message and `notebook_id` is still the (possibly newly
        created) notebook the caller can report back to the user.
    """
    from agents.notebook_memory import NotebookMemory
    from tools.document_tools import DocumentProcessor, ProcessedDocument
    from config.settings import get_settings as _cfg
    from tools.hybrid_store import _stores as _hybrid_stores
    _c = _cfg()
    memory = NotebookMemory()
    if not notebook_id:
        rq = sr_state.get("research_question", "Systematic Review")
        notebook_id = memory.new_notebook(notebook_name or f"SR: {rq[:50]}")
    parts = [f"# Systematic Review: {sr_state.get('research_question','')}", "", "## Narrative Synthesis", sr_state.get("narrative_synthesis", ""), ""]
    if sr_state.get("key_themes"):
        parts += ["## Key Themes"] + [f"- {t}" for t in sr_state["key_themes"]] + [""]
    if sr_state.get("research_gaps"):
        parts += ["## Research Gaps"] + [f"- {g}" for g in sr_state["research_gaps"]] + [""]
    if sr_state.get("conclusion"):
        parts += ["## Conclusion", sr_state["conclusion"], ""]
    if sr_state.get("evidence_table"):
        parts += ["## Evidence Table"] + [f"**[{e.get('citation_key','')}]** {e.get('title','')} ({e.get('year','n.d.')}) — {e.get('key_finding','')}" for e in sr_state["evidence_table"][:15]] + [""]
    content = "\n".join(parts)
    processor = DocumentProcessor(chunk_size=settings.get("chunk_size", _c.chunk_size), overlap=settings.get("chunk_overlap", _c.chunk_overlap))
    try:
        rq = sr_state.get("research_question", "")
        metadata = {"source": "systematic_review", "research_question": rq}
        chunks = processor.chunk_text(content, metadata=metadata)
        processed = ProcessedDocument(content=content, filename=f"systematic_review_{rq[:40].replace(' ','_')}.md", file_type="md", chunks=chunks, metadata=metadata)
        added = memory.add_source(notebook_id, processed, source_type="sr_bridge")
        if added:
            # Drop the cached HybridStore so the new source gets embedded on
            # the notebook's next retrieval instead of being silently missed.
            _hybrid_stores.pop(f"notebook_{notebook_id}", None)
        return notebook_id, ""
    except Exception as e:
        logger.error("SR→Notebook bridge failed: %s", e)
        return notebook_id, str(e)
