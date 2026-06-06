"""
agents/notebook_pipeline_state.py
───────────────────────────────────
State TypedDict for the 7-agent Mode 8 pipeline.

Each agent reads from this state and returns a partial update; LangGraph
merges the updates automatically between nodes.

Pipeline order
──────────────
  Agent 1 — Document Ingestion     → sources, chunks, ingestion_summary
  Agent 2 — Summarization          → per_doc_summaries, cross_summary
  Agent 3 — Retrieval              → retrieved_chunks, retrieval_mode
  Agent 4 — Citation Verification  → verified_citations, citation_report
  Agent 5 — Knowledge Graph        → knowledge_graph_dot, kg_data
  Agent 6 — Study Guide            → study_guide
  Agent 7 — Podcast Script         → podcast_script
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class NotebookPipelineState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    notebook_id: str
    query: str                            # optional focus query for retrieval
    settings: Dict[str, Any]

    # ── Agent 1 — Document Ingestion ───────────────────────────────────────────
    sources: List[Dict[str, Any]]         # source metadata from NotebookMemory
    chunks: List[Dict[str, Any]]          # all stored text chunks
    doc_count: int
    ingestion_summary: str

    # ── Agent 2 — Summarization ─────────────────────────────────────────────────
    per_doc_summaries: Dict[str, str]     # {filename: summary text}
    cross_summary: str                    # cross-document synthesis

    # ── Agent 3 — Retrieval ─────────────────────────────────────────────────────
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_mode: str                   # "hybrid" | "fallback" | "empty"

    # ── Agent 4 — Citation Verification ─────────────────────────────────────────
    verified_citations: List[Dict[str, Any]]  # [{claim, source_name, confidence, supporting_text}]
    citation_report: str                  # markdown verification table

    # ── Agent 5 — Knowledge Graph ───────────────────────────────────────────────
    knowledge_graph_dot: str              # Graphviz DOT string
    kg_data: Dict[str, Any]              # {"nodes": [...], "edges": [...]}

    # ── Agent 6 — Study Guide ───────────────────────────────────────────────────
    study_guide: str                      # markdown study guide

    # ── Agent 7 — Podcast Script ────────────────────────────────────────────────
    podcast_script: str                   # two-speaker dialogue transcript

    # ── Workflow control ────────────────────────────────────────────────────────
    errors: List[str]
    completed_steps: List[str]
    current_step: str
    progress_pct: int
    eval_result: Dict[str, Any]     # quality self-evaluation scores
    feedback_history: List[Dict[str, Any]]
    refinement_round: int
    rag_reflection_info: Dict[str, Any]  # self-reflective retrieval metadata


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
