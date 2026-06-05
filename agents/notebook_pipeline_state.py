"""
agents/notebook_pipeline_state.py
───────────────────────────────────
State TypedDict for the 7-agent Mode 8 pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class NotebookPipelineState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    notebook_id: str
    query: str
    settings: Dict[str, Any]

    # ── Agent 1 — Document Ingestion ───────────────────────────────────────────
    sources: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    doc_count: int
    ingestion_summary: str

    # ── Agent 2 — Summarization ─────────────────────────────────────────────────
    per_doc_summaries: Dict[str, str]
    cross_summary: str

    # ── Agent 3 — Retrieval ─────────────────────────────────────────────────────
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_mode: str

    # ── Agent 4 — Citation Verification ─────────────────────────────────────────
    verified_citations: List[Dict[str, Any]]
    citation_report: str

    # ── Agent 5 — Knowledge Graph ───────────────────────────────────────────────
    knowledge_graph_dot: str
    kg_data: Dict[str, Any]

    # ── Agent 6 — Study Guide ───────────────────────────────────────────────────
    study_guide: str

    # ── Agent 7 — Podcast Script ────────────────────────────────────────────────
    podcast_script: str

    # ── Workflow control ────────────────────────────────────────────────────────
    errors: List[str]
    completed_steps: List[str]
    current_step: str
    progress_pct: int
    eval_result: Dict[str, Any]
    feedback_history: List[Dict[str, Any]]
    refinement_round: int
    rag_reflection_info: Dict[str, Any]


def create_pipeline_state(
    notebook_id: str,
    settings: Dict[str, Any],
    query: str = "",
) -> NotebookPipelineState:
    """Factory — returns a fully-initialised NotebookPipelineState."""
    return NotebookPipelineState(
        notebook_id=notebook_id,
        query=query,
        settings=settings,
        sources=[],
        chunks=[],
        doc_count=0,
        ingestion_summary="",
        per_doc_summaries={},
        cross_summary="",
        retrieved_chunks=[],
        retrieval_mode="empty",
        verified_citations=[],
        citation_report="",
        knowledge_graph_dot="",
        kg_data={},
        study_guide="",
        podcast_script="",
        errors=[],
        completed_steps=[],
        current_step="",
        progress_pct=0,
        rag_reflection_info={},
    )
