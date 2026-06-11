"""agents/systematic_review_state.py — State for Mode 7 Systematic Review"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict


class SystematicReviewState(TypedDict, total=False):
    # ── User inputs ───────────────────────────────────────────
    research_question: str          # The PICO-style research question
    inclusion_criteria: List[str]   # What papers to include
    exclusion_criteria: List[str]   # What papers to exclude
    search_queries: List[str]       # Generated academic search queries
    model_name: str
    num_ctx: int
    session_id: str
    max_results: int                # Max papers per source per query
    include_crossref: bool          # Also search CrossRef for published articles

    # ── Search results ─────────────────────────────────────────
    raw_papers: List[Dict]          # All papers from initial search
    screened_papers: List[Dict]     # Papers after title/abstract screening
    included_papers: List[Dict]     # Final included papers after criteria check
    excluded_papers: List[Dict]     # Excluded papers with reason

    # ── Synthesis outputs ──────────────────────────────────────
    prisma_flow: Dict[str, int]     # {"identified": N, "screened": N, "included": N, "excluded": N}
    evidence_table: List[Dict]      # [{title, year, design, key_finding, quality}]
    narrative_synthesis: str        # Main synthesis text with citations
    key_themes: List[str]           # Common themes across included papers
    research_gaps: List[str]        # Identified gaps in the literature
    limitations: str                # Limitations of this review
    conclusion: str                 # Summary conclusion

    # ── Quality assessment ─────────────────────────────────────
    eval_result: Dict[str, Any]
    feedback_history: List[Dict[str, Any]]
    refinement_round: int
    rag_reflection_info: Dict[str, Any]  # self-reflective retrieval metadata

    # ── Literature Discovery ───────────────────────────────────
    screener_scores: List[Dict]          # abstract screener results per paper
    preprint_tracking: List[Dict]        # preprint status per included paper
    citation_graph_html: str             # Pyvis HTML string for citation network

    # ── Trend & Analysis ──────────────────────────────────────
    trend_data: Dict[str, Any]           # analyze_trends() output
    evidence_map_data: Dict[str, Any]    # build_evidence_map_data() output
    concept_drift_data: Dict[str, Any]   # detect_concept_drift() output

    # ── Workflow control ───────────────────────────────────────
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
    max_results: int = 8,
    include_crossref: bool = True,
) -> SystematicReviewState:
    import uuid
    return SystematicReviewState(
        research_question=research_question,
        inclusion_criteria=inclusion_criteria or [],
        exclusion_criteria=exclusion_criteria or [],
        search_queries=[],
        model_name=model_name,
        num_ctx=num_ctx,
        max_results=max_results,
        include_crossref=include_crossref,
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
        eval_result={},
        feedback_history=[],
        refinement_round=0,
        rag_reflection_info={},
        screener_scores=[],
        preprint_tracking=[],
        citation_graph_html="",
        trend_data={},
        evidence_map_data={},
        concept_drift_data={},
        current_step="start",
        completed_steps=[],
        errors=[],
        progress_pct=0,
    )
