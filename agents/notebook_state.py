"""
agents/notebook_state.py
─────────────────────────
State TypedDict for the Research Notebook (Mode 8) workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class NotebookState(TypedDict, total=False):
    # ── User inputs ───────────────────────────────────────────
    user_message: str
    notebook_id: str
    model_name: str
    num_ctx: int
    embed_model: str
    top_k: int
    include_web_search: bool

    # ── Loaded from memory / retrieval ────────────────────────
    conversation_history: List[Dict]
    retrieved_chunks: List[Dict]
    source_count: int
    retrieval_mode: str

    # ── LLM outputs ──────────────────────────────────────────
    assistant_response: str
    citations: List[Dict[str, Any]]
    suggested_questions: List[str]

    # ── Workflow control ──────────────────────────────────────
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    progress_pct: int
    eval_result: Dict[str, Any]
    rag_reflection_info: Dict[str, Any]


def create_notebook_state(
    user_message: str,
    notebook_id: str,
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
    embed_model: str = "nomic-embed-text",
    top_k: int = 8,
    include_web_search: bool = False,
) -> NotebookState:
    """Factory for a single-turn NotebookState."""
    return NotebookState(
        user_message=user_message,
        notebook_id=notebook_id,
        model_name=model_name,
        num_ctx=num_ctx,
        embed_model=embed_model,
        top_k=top_k,
        include_web_search=include_web_search,
        conversation_history=[],
        retrieved_chunks=[],
        source_count=0,
        retrieval_mode="empty",
        assistant_response="",
        citations=[],
        suggested_questions=[],
        current_step="start",
        completed_steps=[],
        errors=[],
        progress_pct=0,
        rag_reflection_info={},
    )
