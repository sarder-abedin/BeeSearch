"""
agents/notebook_state.py
─────────────────────────
State TypedDict for the Research Notebook (Mode 8) workflow.

One graph invocation = one user question against a notebook. The notebook's
sources, chunks, and conversation history live in NotebookMemory (JSON files),
not in this state object — consistent with how StoryState / WisdomState handle
their single-turn workflows.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from tools.temperature_levels import DEFAULT_TEMPERATURE_LEVEL


class NotebookState(TypedDict, total=False):
    # ── User inputs ───────────────────────────────────────────
    user_message: str                  # The current question
    notebook_id: str                   # Links to NotebookMemory on disk
    model_name: str                    # Ollama chat model for this turn
    num_ctx: int                       # LLM context window (tokens)
    embed_model: str                   # Ollama embedding model for retrieval
    top_k: int                         # Number of chunks to retrieve
    include_web_search: bool           # Auto-search Google for each query
    temperature_level: str             # "precise" | "focused" | "balanced" | "creative"

    # ── Loaded from memory / retrieval ────────────────────────
    conversation_history: List[Dict]   # Last N turns from memory
    retrieved_chunks: List[Dict]       # Hybrid-search results for this query
    source_count: int                  # How many sources the notebook holds
    retrieval_mode: str                # "hybrid" | "fallback" | "empty"

    # ── LLM outputs ──────────────────────────────────────────
    assistant_response: str            # The grounded answer text
    citations: List[Dict[str, Any]]    # [{n, doc_name, page, snippet}]
    suggested_questions: List[str]     # 2–3 follow-up questions

    # ── Workflow control ──────────────────────────────────────
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    progress_pct: int
    eval_result: Dict[str, Any]     # quality self-evaluation scores
    rag_reflection_info: Dict[str, Any]  # self-reflective retrieval metadata


def create_notebook_state(
    user_message: str,
    notebook_id: str,
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
    embed_model: str = "nomic-embed-text",
    top_k: int = 8,
    include_web_search: bool = False,
    temperature_level: str = DEFAULT_TEMPERATURE_LEVEL,
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
        temperature_level=temperature_level,
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
