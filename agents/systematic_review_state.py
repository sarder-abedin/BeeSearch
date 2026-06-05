"""agents/systematic_review_state.py — State for Mode 7 Systematic Review"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict


class SystematicReviewState(TypedDict, total=False):
    # ── User inputs ─────────────────────────────────────────────────────
    research_question: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    search_queries: List[str]
    model_name: str
    num_ctx: int
    session_id: str

    # ── Search results ───────────────────────────────────────────────────
    raw_papers: List[Dict]
    screened_papers: List[Dict]
    included_papers: List[Dict]
    excluded_papers: List[Dict]

    # ── Synthesis outputs ──────────────────────────────────────────────
    prisma_flow: Dict[str, int]
    evidence_table: List[Dict]
    narrative_synthesis: str
    key_themes: List[str]
    research_gaps: List[str]
    limitations: str
    conclusion: str

    # ── Quality & advanced analysis ──────────────────────────────────
    rob_table: List[Dict]                   # Risk-of-bias per paper (RoB 2 / ROBINS-I)
    grade_results: Dict[str, Any]           # GRADE evidence grading output
    contradictions: List[Dict]             # Detected contradictions across papers
    pico_extraction: List[Dict]            # Full PICO structured extraction table
    gap_map: Dict[str, Any]               # Categorised research gap map
    hypotheses: List[Dict]                 # Generated testable hypotheses
    sensitivity_results: Dict[str, Any]   # Sensitivity analysis results
    monitor_state: Dict[str, Any]         # Incremental monitor save state
    preregistration: str                   # OSF pre-registration template

    # ── Self-evaluation ───────────────────────────────────────────────
    eval_result: Dict[str, Any]
    feedback_history: List[Dict[str, Any]]
    refinement_round: int
    rag_reflection_info: Dict[str, Any]

    # ── Workflow control ──────────────────────────────────────────────
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    progress_pct: int


def create_systematic_review_state(
    research_question: str,
    inclusion_criteria: List[str] = None,
    exclusion_criteria: List[str] = None,
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
) -> SystematicReviewState:
    import uuid
    return SystematicReviewState(
        research_question=research_question,
        inclusion_criteria=inclusion_criteria or [],
        exclusion_criteria=exclusion_criteria or [],
        search_queries=[],
        model_name=model_name,
        num_ctx=num_ctx,
        session_id=str(uuid.uuid4())[:8],
        raw_papers=[],
        screened_papers=[],
        included_papers=[],
        excluded_papers=[],
        prisma_flow={},
        evidence_table=[],
        narrative_synthesis="",
        key_themes=[],
        research_gaps=[],
        limitations="",
        conclusion="",
        rob_table=[],
        grade_results={},
        contradictions=[],
        pico_extraction=[],
        gap_map={},
        hypotheses=[],
        sensitivity_results={},
        monitor_state={},
        preregistration="",
        eval_result={},
        feedback_history=[],
        refinement_round=0,
        rag_reflection_info={},
        current_step="start",
        completed_steps=[],
        errors=[],
        progress_pct=0,
    )
