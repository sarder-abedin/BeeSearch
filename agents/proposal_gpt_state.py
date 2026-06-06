"""
agents/proposal_gpt_state.py
─────────────────────────────
ProposalGPT state TypedDict — shared across all 9 LangGraph agents.

Pipeline order:
  funding_call_analyzer → research_planner → literature_review_agent →
  proposal_writer → impact_agent → budget_agent → compliance_agent →
  reviewer_agent → improvement_agent → END
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ProposalGPTState(TypedDict, total=False):

    # ── Session / config ──────────────────────────────────────────────────────
    session_id: str
    model_name: str
    num_ctx: int
    funding_agency: str          # "Horizon Europe" | "Vinnova" | "NSF" | "Generic"

    # ── INPUT (uploaded by user) ──────────────────────────────────────────────
    funding_call_text: str       # Full text parsed from call PDF/DOCX/URL
    funding_call_filename: str   # Original filename for display
    cv_texts: List[str]          # One entry per uploaded CV
    publication_texts: List[str] # Parsed publication lists
    institution_info: str        # Institution/department description
    partner_profiles: List[str]  # Partner organisation descriptions
    user_ideas: str              # Free-text researcher ideas / positioning
    requirements: str            # Extra instructions (tone, length, focus)

    # ── AGENT 1: Funding Call Analyzer ───────────────────────────────────────
    call_title: str
    call_objectives: List[str]
    eligibility_requirements: List[str]
    evaluation_criteria: List[Dict[str, str]]  # {criterion, weight, description}
    expected_outcomes: List[str]
    budget_constraints: Dict[str, Any]  # {max_budget, currency, duration_months, notes}
    deadline: str
    mandatory_sections: List[str]
    keywords: List[str]
    funding_summary: str         # Funding Opportunity Summary (markdown)
    evaluation_matrix: str       # Formatted markdown table
    compliance_checklist: List[Dict[str, str]]  # {item, status, notes}
    success_factors: str         # Success Factors Report

    # ── AGENT 2: Research Planner / Strategy ─────────────────────────────────
    hidden_priorities: List[str]
    reviewer_expectations: str
    positioning_recommendation: str
    competitive_advantages: List[str]
    proposal_risks: List[Dict[str, str]]  # {risk, likelihood, mitigation}
    proposal_strengths: List[str]
    win_strategy: str            # Win Strategy Report (markdown)
    reviewer_perspective: str    # Reviewer Perspective Analysis
    swot_analysis: str           # SWOT table (markdown)

    # ── AGENT 3: Literature Review ────────────────────────────────────────────
    literature_review: str
    state_of_art: str
    research_gaps: List[str]
    suggested_references: List[Dict[str, str]]  # {title, authors, year, doi, url}

    # ── AGENT 4: Proposal Writer (sections 1–14) ─────────────────────────────
    executive_summary: str
    excellence: str
    scientific_background: str
    objectives: str              # 3–5 SMART objectives
    research_questions: List[str]
    methodology: str
    work_packages: List[Dict[str, Any]]   # {id, title, description, months, lead, tasks}
    deliverables: List[Dict[str, str]]    # {id, title, type, month, wp}
    milestones: List[Dict[str, str]]      # {id, title, month, verification}
    risk_management: str
    consortium_description: str
    management_structure: str
    data_management: str

    # ── AGENT 5: Impact Agent (sections 15–19) ───────────────────────────────
    impact: str
    dissemination: str
    exploitation: str
    ethics: str
    sustainability: str

    # ── AGENT 6: Budget Agent ────────────────────────────────────────────────
    budget_personnel: List[Dict[str, Any]]   # {role, name, months, rate, total}
    budget_equipment: List[Dict[str, Any]]   # {item, cost, justification}
    budget_travel: List[Dict[str, Any]]      # {destination, purpose, cost}
    budget_subcontracting: List[Dict[str, Any]]
    budget_indirect_rate: float              # e.g. 0.25 for 25%
    budget_indirect: float
    budget_total: float
    budget_justification: str
    budget_format: str                       # "horizon_europe" | "swedish" | "generic"
    budget_summary_table: str                # Markdown table

    # ── AGENT 7: Compliance Checker ──────────────────────────────────────────
    compliance_report: str
    missing_sections: List[str]
    keyword_coverage: Dict[str, bool]        # {keyword: covered}
    compliance_score: int                    # 0–100
    page_estimate: int
    compliance_issues: List[str]

    # ── AGENT 8: Reviewer Simulation ─────────────────────────────────────────
    reviewer_scores: Dict[str, Dict[str, Any]]  # {reviewer_type: {section: score, feedback}}
    overall_score: float
    reviewer_report: str

    # ── AGENT 9: Proposal Improvement ────────────────────────────────────────
    improvement_plan: str
    weak_sections: List[str]
    improved_sections: Dict[str, str]        # {section_name: improved_text}

    # ── Pipeline metadata ────────────────────────────────────────────────────
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    progress_pct: int
    eval_result: Dict[str, Any]     # quality self-evaluation scores
    feedback_history: List[Dict[str, Any]]
    refinement_round: int
    rag_reflection_info: Dict[str, Any]  # self-reflective retrieval metadata


def create_proposal_gpt_state(
    funding_call_text: str,
    user_ideas: str = "",
    requirements: str = "",
    funding_agency: str = "Generic",
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
    session_id: str = "",
    cv_texts: Optional[List[str]] = None,
    publication_texts: Optional[List[str]] = None,
    institution_info: str = "",
    partner_profiles: Optional[List[str]] = None,
    funding_call_filename: str = "funding_call",
) -> ProposalGPTState:
    """Factory — creates a fresh ProposalGPT pipeline state."""
    import uuid
    return ProposalGPTState(
        session_id=session_id or str(uuid.uuid4())[:8],
        model_name=model_name,
        num_ctx=num_ctx,
        funding_agency=funding_agency,
        funding_call_text=funding_call_text,
        funding_call_filename=funding_call_filename,
        cv_texts=cv_texts or [],
        publication_texts=publication_texts or [],
        institution_info=institution_info,
        partner_profiles=partner_profiles or [],
        user_ideas=user_ideas,
        requirements=requirements,
        # initialise all list/dict fields so nodes never hit KeyError
        call_objectives=[],
        eligibility_requirements=[],
        evaluation_criteria=[],
        expected_outcomes=[],
        budget_constraints={},
        mandatory_sections=[],
        keywords=[],
        compliance_checklist=[],
        hidden_priorities=[],
        competitive_advantages=[],
        proposal_risks=[],
        proposal_strengths=[],
        research_gaps=[],
        suggested_references=[],
        research_questions=[],
        work_packages=[],
        deliverables=[],
        milestones=[],
        budget_personnel=[],
        budget_equipment=[],
        budget_travel=[],
        budget_subcontracting=[],
        budget_indirect_rate=0.25,
        budget_indirect=0.0,
        budget_total=0.0,
        budget_format=_detect_budget_format(funding_agency),
        missing_sections=[],
        keyword_coverage={},
        compliance_score=0,
        page_estimate=0,
        compliance_issues=[],
        reviewer_scores={},
        overall_score=0.0,
        weak_sections=[],
        improved_sections={},
        completed_steps=[],
        errors=[],
        progress_pct=0,
        rag_reflection_info={},
        eval_result={},
        feedback_history=[],
        refinement_round=0,
    )


def _detect_budget_format(funding_agency: str) -> str:
    agency_lower = funding_agency.lower()
    if any(k in agency_lower for k in ["horizon", "erc", "msca", "europe"]):
        return "horizon_europe"
    if any(k in agency_lower for k in ["vinnova", "vr", "formas", "forte", "swedish", "sverige"]):
        return "swedish"
    return "generic"
