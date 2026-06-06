"""
agents/wisdom_state.py
───────────────────────
State TypedDict for the Wisdom Mode (Mode 6) workflow.

One graph invocation = one user message (same single-turn model as Mode 5).
Wisdom continuity and the generated output live in WisdomMemory (JSON files).

Phase lifecycle
───────────────
  "clarifying"       — agent is asking clarifying questions (max 3 rounds)
  "ready_to_generate"— agent has enough context; triggers knowledge_search branch
  "done"             — wisdom generated; subsequent turns go to wisdom_followup
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class WisdomState(TypedDict, total=False):
    # ── User inputs ───────────────────────────────────────────
    user_message: str          # The current user message
    topic: str                 # High-level topic label
    scenario: str              # Full scenario text (set at session creation)
    session_id: str            # Links to WisdomMemory on disk
    model_name: str
    num_ctx: int

    # ── Loaded from memory ────────────────────────────────────
    phase: str                          # "clarifying" | "ready_to_generate" | "done"
    clarification_count: int            # Assistant question turns so far
    conversation_history: List[Dict]    # Last 8 turns loaded from memory
    document_context: str               # Truncated raw_text from uploaded docs
    document_names: List[str]
    related_sessions: List[Dict]        # Semantically similar past sessions (passive)

    # ── Previously generated wisdom (for follow-up context) ──
    deep_understanding: str
    simple_explanation: str
    actionable_takeaways: List[str]
    wisdom_claims: List[Dict]
    devils_advocate: str
    overall_confidence: str

    # ── Knowledge gathering (set by knowledge_search_node) ───
    search_queries: List[str]
    academic_papers: List[Dict]         # Serialised Paper objects
    web_context: List[Dict]             # [{title, url, snippet}]

    # ── Wisdom synthesis outputs ──────────────────────────────
    topic_tags: List[str]               # For cross-session matching

    # ── Socratic Clarifications ───────────────────────────────
    clarifications: Dict[str, str]   # {question_key: answer} from session-start form

    # ── LLM response ─────────────────────────────────────────
    assistant_response: str

    # ── Quality Evaluation ────────────────────────────────────
    eval_result: Dict[str, Any]            # {evidence_grounding, confidence_calibration, actionability, overall, summary}
    feedback_history: List[Dict[str, Any]]
    refinement_round: int
    rag_reflection_info: Dict[str, Any]  # self-reflective retrieval metadata

    # ── Workflow control ──────────────────────────────────────
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    progress_pct: int


def create_wisdom_state(
    user_message: str,
    session_id: str,
    topic: str = "",
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
    clarifications: Optional[Dict[str, str]] = None,
) -> WisdomState:
    """Factory function for a single-turn WisdomState."""
    return WisdomState(
        user_message=user_message,
        topic=topic,
        scenario="",
        session_id=session_id,
        model_name=model_name,
        num_ctx=num_ctx,
        clarifications=clarifications or {},
        phase="clarifying",
        clarification_count=0,
        conversation_history=[],
        document_context="",
        document_names=[],
        related_sessions=[],
        deep_understanding="",
        simple_explanation="",
        actionable_takeaways=[],
        wisdom_claims=[],
        devils_advocate="",
        overall_confidence="",
        search_queries=[],
        academic_papers=[],
        web_context=[],
        topic_tags=[],
        assistant_response="",
        eval_result={},
        feedback_history=[],
        refinement_round=0,
        rag_reflection_info={},
        current_step="start",
        completed_steps=[],
        errors=[],
        progress_pct=0,
    )
