"""
agents/grammar_state.py
────────────────────────
State TypedDict for the Grammar Proofreading Mode (Mode 6) workflow.

One graph invocation = one proofreading run (or one revision round).
Session persistence lives in GrammarMemory (JSON files).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class GrammarState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────
    raw_text: str           # Text to proofread (no hard cap; model context window is the limit)
    session_id: str
    model_name: str
    num_ctx: int
    style_level: str        # "academic" | "professional_email" | "formal" | "informal"
    focus_areas: List[str]  # e.g. ["grammar", "punctuation", "spelling", "style", "clarity"]

    # ── Feedback / revision ───────────────────────────────────
    feedback: str                            # User feedback for current revision round
    refinement_round: int                    # 0 = initial, 1+ = revised
    feedback_history: List[Dict[str, Any]]   # [{round, feedback, previous_polished}]

    # ── Derived in text_loader ────────────────────────────────
    word_count: int
    sentence_count: int

    # ── Outputs ───────────────────────────────────────────────
    issues_found: List[Dict[str, Any]]       # {type, original, suggestion, explanation, severity}
    polished_text: str                        # PRIMARY OUTPUT — fully rewritten, fluent text
    change_summary: str                       # Markdown bullet list of what was changed and why
    style_suggestions: List[Dict[str, Any]]  # {category, suggestion, rationale}

    # ── Quality ───────────────────────────────────────────────
    eval_result: Dict[str, Any]
    rag_reflection_info: Dict[str, Any]      # Always {} — no retrieval in this mode

    # ── Control ───────────────────────────────────────────────
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    progress_pct: int


def create_grammar_state(
    raw_text: str,
    session_id: str,
    model_name: str = "",
    num_ctx: int = 0,
    style_level: str = "professional_email",
    focus_areas: Optional[List[str]] = None,
    feedback: str = "",
    refinement_round: int = 0,
    feedback_history: Optional[List[Dict[str, Any]]] = None,
) -> GrammarState:
    """Factory function for a GrammarState."""
    return GrammarState(
        raw_text=raw_text,
        session_id=session_id,
        model_name=model_name,
        num_ctx=num_ctx,
        style_level=style_level or "professional_email",
        focus_areas=focus_areas or [],
        feedback=feedback,
        refinement_round=refinement_round,
        feedback_history=feedback_history or [],
        word_count=0,
        sentence_count=0,
        issues_found=[],
        polished_text="",
        change_summary="",
        style_suggestions=[],
        eval_result={},
        rag_reflection_info={},
        current_step="start",
        completed_steps=[],
        errors=[],
        progress_pct=0,
    )
