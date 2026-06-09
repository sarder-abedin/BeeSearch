"""
agents/story_state.py
──────────────────────
State TypedDict for the Research Partner (Storytelling) workflow.

One graph invocation = one user turn. Conversation continuity lives
in StorytellerMemory (JSON files), not in this state object — consistent
with how ProposalState handles the proposal workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class StoryState(TypedDict, total=False):
    # ── User inputs ───────────────────────────────────────────
    user_message: str                # The current user message
    explanation_style: str           # "simple" | "analogy" | "walkthrough" | "debate"
    explanation_level: str           # "novice" | "intermediate" | "expert"
    topic: str                       # The research topic being explored
    session_id: str                  # Links to StorytellerMemory on disk
    model_name: str                  # Ollama model for this session
    num_ctx: int                     # LLM context window (tokens)

    # ── Loaded from memory ────────────────────────────────────
    document_context: str            # Truncated raw_text from uploaded docs
    document_names: List[str]        # Filenames of uploaded docs
    conversation_history: List[Dict] # Last 8 turns loaded from memory
    concepts_covered: List[str]      # Concepts already explained in this session

    # ── Socratic Clarifications ───────────────────────────────
    clarifications: Dict[str, str]   # {question_key: answer} from session-start form

    # ── LLM outputs ──────────────────────────────────────────
    assistant_response: str          # The storyteller's response text
    suggested_questions: List[str]   # 2–3 follow-up question suggestions
    new_concepts: List[str]          # Concepts newly introduced in this turn

    # ── Source routing ───────────────────────────────────────
    online_results: List[Dict]             # Academic + web results fetched when docs insufficient
    source_decision: Dict                  # {coverage_score, used_docs, used_online, reason, ...}

    # ── Quality Evaluation ────────────────────────────────────
    eval_result: Dict[str, Any]            # {clarity, style_adherence, overall, summary}

    # ── Workflow control ──────────────────────────────────────
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    progress_pct: int


def create_story_state(
    user_message: str,
    session_id: str,
    topic: str = "",
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
    explanation_style: str = "simple",
    explanation_level: str = "intermediate",
    clarifications: Optional[Dict[str, str]] = None,
) -> StoryState:
    """Factory function for a single-turn StoryState."""
    return StoryState(
        user_message=user_message,
        explanation_style=explanation_style,
        explanation_level=explanation_level,
        topic=topic,
        session_id=session_id,
        model_name=model_name,
        num_ctx=num_ctx,
        clarifications=clarifications or {},
        document_context="",
        document_names=[],
        conversation_history=[],
        concepts_covered=[],
        assistant_response="",
        suggested_questions=[],
        new_concepts=[],
        online_results=[],
        source_decision={},
        eval_result={},
        current_step="start",
        completed_steps=[],
        errors=[],
        progress_pct=0,
    )
