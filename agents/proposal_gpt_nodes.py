"""
agents/proposal_gpt_nodes.py
──────────────────────────────
Nine LangGraph node functions for the ProposalGPT pipeline.

  Agent 1  funding_call_analyzer   — Extract objectives, criteria, keywords
  Agent 2  research_planner        — Win strategy, SWOT, reviewer perspective
  Agent 3  literature_review_agent — RAG-powered literature + gap analysis
  Agent 4  proposal_writer         — Core proposal sections (14 sections)
  Agent 5  impact_agent            — Impact, dissemination, exploitation, ethics
  Agent 6  budget_agent            — Personnel / equipment / travel / indirect
  Agent 7  compliance_agent        — Missing sections, keyword coverage, score
  Agent 8  reviewer_agent          — 5 virtual reviewers (Scientific/Impact/etc.)
  Agent 9  improvement_agent       — Rewrite weak sections from reviewer feedback
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agents.proposal_gpt_state import ProposalGPTState
from config.settings import get_settings

logger = logging.getLogger(__name__)
_cfg = get_settings()


# ── LLM factory ────────────────────────────────────────────────────────────────

def _make_llm(state: ProposalGPTState, temperature: float = 0.3, num_predict: int = 2048):
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=state.get("model_name", _cfg.ollama_model),
        temperature=temperature,
        num_ctx=state.get("num_ctx", 32768),
        num_predict=num_predict,
        base_url=_cfg.ollama_base_url,
    )


def _invoke(llm, system: str, human: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return (resp.content or "").strip()
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return ""


def _parse_json_object(raw: str) -> dict:
    """Extract first JSON object from LLM output."""
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _parse_json_array(raw: str) -> list:
    """Extract first JSON array from LLM output."""
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _progress(step: int) -> int:
    return min(100, round(step / 9 * 100))


def _call_context(state: ProposalGPTState, max_chars: int = 8000) -> str:
    return state.get("funding_call_text", "")[:max_chars]


def _researcher_context(state: ProposalGPTState, max_chars: int = 3000) -> str:
    parts = []
    if state.get("user_ideas"):
        parts.append(f"RESEARCHER IDEAS:\n{state['user_ideas']}")
    if state.get("institution_info"):
        parts.append(f"INSTITUTION:\n{state['institution_info']}")
    for i, cv in enumerate(state.get("cv_texts", [])[:2], 1):
        parts.append(f"CV {i}:\n{cv[:600]}")
    for i, pub in enumerate(state.get("publication_texts", [])[:2], 1):
        parts.append(f"PUBLICATIONS {i}:\n{pub[:600]}")
    return "\n\n".join(parts)[:max_chars]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Funding Call Analyzer
# ══════════════════════════════════════════════════════════════════════════════

def funding_call_analyzer_node(state: ProposalGPTState) -> ProposalGPTState:
    """
    Parse the funding call document and extract:
    - Objectives, eligibility, evaluation criteria, expected outcomes
    - Budget constraints, deadline, mandatory sections, keywords
    - Funding Opportunity Summary, Evaluation Matrix, Compliance Checklist
    """
    errors = list(state.get("errors", []))
    completed = list(state.get("completed_steps", []))

    call_text = _call_context(state, max_chars=10000)
    if not call_text:
        errors.append("No funding call text provided — skipping analysis.")
        return ProposalGPTState(
            errors=errors, completed_steps=completed,
            current_step="funding_call_analyzer", progress_pct=_progress(1),
        )

    llm = _make_llm(state, temperature=0.1, num_predict=3000)

    # ── Extract structured fields ──────────────────────────────────────────
    extract_system = (
        "You are an expert research funding analyst. Extract structured information "
        "from the funding call. Return ONLY a valid JSON object with these keys: "
        "title (str), objectives (list of str), eligibility (list of str), "
        "evaluation_criteria (list of {criterion, weight, description}), "
        "expected_outcomes (list of str), budget_max (str), currency (str), "
        "duration_months (int), deadline (str), mandatory_sections (list of str), "
        "keywords (list of str). "
        "Use exact language from the call. If a field is not found, use empty list/string."
    )
    extract_human = f"FUNDING CALL:\n{call_text}"
    raw_extract = _invoke(llm, extract_system, extract_human)
    extracted = _parse_json_object(raw_extract)

    call_title = extracted.get("title", state.get("funding_call_filename", "Funding Call"))
    call_objectives = extracted.get("objectives", [])
    eligibility = extracted.get("eligibility", [])
    eval_criteria = extracted.get("evaluation_criteria", [])
    expected_outcomes = extracted.get("expected_outcomes", [])
    budget_constraints = {
        "max_budget": extracted.get("budget_max", "Not specified"),
        "currency": extracted.get("currency", "EUR"),
        "duration_months": extracted.get("duration_months", 36),
        "notes": "",
    }
    deadline = extracted.get("deadline", "Not specified")
    mandatory_sections = extracted.get("mandatory_sections", [])
    keywords = extracted.get("keywords", [])

    # ── Funding Opportunity Summary ────────────────────────────────────────
    summary_system = (
        "You are a senior funding advisor. Write a concise Funding Opportunity Summary "
        "in structured markdown (300–400 words). Include: Overview, Strategic Priorities, "
        "Funding Scope, Key Requirements, Success Profile."
    )
    summary_human = (
        f"FUNDING CALL (first 6000 chars):\n{call_text[:6000]}\n\n"
        f"EXTRACTED KEYWORDS: {', '.join(keywords[:10])}"
    )
    funding_summary = _invoke(llm, summary_system, summary_human) or "Summary not generated."

    # ── Evaluation Matrix ──────────────────────────────────────────────────
    if eval_criteria:
        rows = ["| Criterion | Weight | Description | Max Score |",
                "|-----------|--------|-------------|-----------|"]
        for c in eval_criteria:
            rows.append(
                f"| {c.get('criterion','?')} | {c.get('weight','?')} | "
                f"{c.get('description','')[:60]} | — |"
            )
        evaluation_matrix = "\n".join(rows)
    else:
        matrix_system = (
            "Create an evaluation matrix in markdown table format from the funding call. "
            "Columns: Criterion | Weight | Description | Max Score. "
            "Return ONLY the markdown table."
        )
        evaluation_matrix = _invoke(llm, matrix_system, f"CALL:\n{call_text[:4000]}") or "| Criterion | Weight | Description |\n|---|---|---|"

    # ── Compliance Checklist ───────────────────────────────────────────────
    checklist_items = []
    for sec in mandatory_sections:
        checklist_items.append({"item": sec, "status": "⬜ Pending", "notes": ""})
    for kw in keywords[:5]:
        checklist_items.append({"item": f"Keyword: {kw}", "status": "⬜ Pending", "notes": ""})
    if budget_constraints.get("max_budget") and budget_constraints["max_budget"] != "Not specified":
        checklist_items.append({
            "item": f"Budget within {budget_constraints['max_budget']} {budget_constraints.get('currency','')}",
            "status": "⬜ Pending", "notes": ""
        })

    # ── Success Factors Report ─────────────────────────────────────────────
    sf_system = (
        "You are a successful Principal Investigator with 20+ years of experience. "
        "Identify the 5–7 critical success factors for this funding call. "
        "Write a concise Success Factors Report in markdown (200–300 words). "
        "Be specific and actionable."
    )
    success_factors = _invoke(llm, sf_system, f"CALL:\n{call_text[:5000]}") or ""

    completed.append("funding_call_analyzer")
    return ProposalGPTState(
        call_title=call_title,
        call_objectives=call_objectives,
        eligibility_requirements=eligibility,
        evaluation_criteria=eval_criteria,
        expected_outcomes=expected_outcomes,
        budget_constraints=budget_constraints,
        deadline=deadline,
        mandatory_sections=mandatory_sections,
        keywords=keywords,
        funding_summary=funding_summary,
        evaluation_matrix=evaluation_matrix,
        compliance_checklist=checklist_items,
        success_factors=success_factors,
        errors=errors,
        completed_steps=completed,
        current_step="funding_call_analyzer",
        progress_pct=_progress(1),
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Research Planner / Strategy
# ══════════════════════════════════════════════════════════════════════════════

def research_planner_node(state: ProposalGPTState) -> ProposalGPTState:
    """
    Act as an experienced funding evaluator.
    Generate: Win Strategy, SWOT Analysis, Reviewer Perspective, Risk Analysis.
    """
    errors = list(state.get("errors", []))
    completed = list(state.get("completed_steps", []))

    llm = _make_llm(state, temperature=0.4, num_predict=3500)

    call_summary = state.get("funding_summary", _call_context(state, 4000))
    researcher_ctx = _researcher_context(state)
    objectives = state.get("call_objectives", [])
    criteria = state.get("evaluation_criteria", [])
    keywords = state.get("keywords", [])
    agency = state.get("funding_agency", "Generic")

    ctx_block = (
        f"FUNDING CALL SUMMARY:\n{call_summary}\n\n"
        f"EVALUATION CRITERIA: {json.dumps(criteria[:5])}\n"
        f"KEY KEYWORDS: {', '.join(keywords[:10])}\n"
        f"FUNDING AGENCY: {agency}\n\n"
        f"RESEARCHER CONTEXT:\n{researcher_ctx}"
    )

    # ── Hidden priorities + reviewer expectations ──────────────────────────
    priorities_system = (
        "You are an expert research funding strategist with deep knowledge of "
        f"{agency} evaluation panels. "
        "Analyse the funding call and identify hidden priorities (things evaluators value "
        "but don't explicitly state), and what reviewers will expect to see. "
        "Return a JSON object with keys: "
        "hidden_priorities (list of str), reviewer_expectations (str), "
        "positioning (str), competitive_advantages (list of str), "
        "proposal_strengths (list of str)."
    )
    raw_priorities = _invoke(llm, priorities_system, ctx_block)
    priorities_data = _parse_json_object(raw_priorities)

    hidden_priorities = priorities_data.get("hidden_priorities", [])
    reviewer_expectations = priorities_data.get("reviewer_expectations", "")
    positioning = priorities_data.get("positioning", "")
    competitive_advantages = priorities_data.get("competitive_advantages", [])
    proposal_strengths = priorities_data.get("proposal_strengths", [])

    # ── Risk analysis ──────────────────────────────────────────────────────
    risk_system = (
        "Identify 5–7 key proposal risks (reasons it might not get funded) and "
        "mitigation strategies. Return a JSON array of objects with keys: "
        "risk, likelihood (High/Medium/Low), mitigation."
    )
    raw_risks = _invoke(llm, risk_system, ctx_block)
    proposal_risks = _parse_json_array(raw_risks)
    if not proposal_risks:
        proposal_risks = [{"risk": "Insufficient novelty", "likelihood": "Medium",
                           "mitigation": "Emphasise unique approach in excellence section"}]

    # ── Win Strategy Report ────────────────────────────────────────────────
    win_system = (
        "Write a Win Strategy Report for this funding proposal in markdown (700–900 words). "
        "Sections: Executive Positioning, Key Differentiators, Messaging Strategy, "
        "Critical Success Actions, Red Flags to Avoid. "
        "Be specific, actionable, and evidence-based."
    )
    win_strategy = _invoke(llm, win_system, ctx_block) or "Win strategy not generated."

    # ── Reviewer Perspective Analysis ─────────────────────────────────────
    reviewer_sys = (
        f"Write a Reviewer Perspective Analysis for a {agency} reviewer (500–700 words). "
        "Cover: What they look for first, Common rejection reasons, "
        "How to write for this reviewer, Key phrases to include, Tone and style guidance."
    )
    reviewer_perspective = _invoke(llm, reviewer_sys, ctx_block) or ""

    # ── SWOT Analysis ─────────────────────────────────────────────────────
    swot_system = (
        "Generate a SWOT analysis for this proposal opportunity as a markdown table. "
        "Format: | Strengths | Weaknesses | Opportunities | Threats | "
        "with 3–4 bullet points per cell. Return ONLY the markdown table."
    )
    swot_analysis = _invoke(llm, swot_system, ctx_block) or (
        "| Strengths | Weaknesses | Opportunities | Threats |\n"
        "|-----------|------------|---------------|--------|\n"
        "| TBD | TBD | TBD | TBD |"
    )

    completed.append("research_planner")
    return ProposalGPTState(
        hidden_priorities=hidden_priorities,
        reviewer_expectations=reviewer_expectations,
        positioning_recommendation=positioning,
        competitive_advantages=competitive_advantages,
        proposal_risks=proposal_risks,
        proposal_strengths=proposal_strengths,
        win_strategy=win_strategy,
        reviewer_perspective=reviewer_perspective,
        swot_analysis=swot_analysis,
        errors=errors,
        completed_steps=completed,
        current_step="research_planner",
        progress_pct=_progress(2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Literature Review Agent
# ══════════════════════════════════════════════════════════════════════════════

def literature_review_agent_node(state: ProposalGPTState) -> ProposalGPTState:
    """
    Search academic literature (arXiv + Semantic Scholar) and generate:
    - Literature Review section
    - State of the Art section
    - Research Gaps list
    - Suggested References
    """
    errors = list(state.get("errors", []))
    completed = list(state.get("completed_steps", []))

    llm = _make_llm(state, temperature=0.2, num_predict=3500)
    keywords = state.get("keywords", [])
    objectives = state.get("call_objectives", [])
    user_ideas = state.get("user_ideas", "")

    # Build search queries from keywords + objectives
    query_parts = keywords[:5] + [o[:50] for o in objectives[:2]]
    search_query = " ".join(query_parts[:6]) or user_ideas[:80] or "research proposal"

    # ── Academic search (arXiv + Semantic Scholar) ────────────────────────
    papers: list = []
    try:
        from tools.search_tools import search_semantic_scholar, search_arxiv
        ss_papers = search_semantic_scholar(search_query, max_results=6)
        ax_papers = search_arxiv(search_query, max_results=4)
        papers = ss_papers + ax_papers
    except Exception as exc:
        errors.append(f"Literature search failed: {exc}")

    # Grade papers for relevance before building context
    papers_retrieved = len(papers)
    if papers:
        from agents.self_reflective_rag import grade_papers
        paper_dicts = [{"title": getattr(p, "title", ""), "abstract": getattr(p, "abstract", "") or ""} for p in papers]
        grades = grade_papers(
            paper_dicts,
            query=state.get("call_title", "") + " " + " ".join(state.get("keywords", [])[:5]),
            model_name=state.get("model_name", _cfg.ollama_model),
            num_ctx=state.get("num_ctx", _cfg.num_ctx),
        )
        filtered = [p for p, g in zip(papers, grades) if g]
        papers = filtered or papers  # fallback: keep all if none pass

    # Build reference context for LLM
    refs_context = ""
    suggested_references = []
    for p in papers[:8]:
        title = getattr(p, "title", "")
        authors = getattr(p, "authors", [])
        year = getattr(p, "year", "")
        doi = getattr(p, "doi", "") or getattr(p, "arxiv_id", "")
        abstract = (getattr(p, "abstract", "") or "")[:300]
        refs_context += f"\n- [{title}] ({', '.join(authors[:2])}, {year}): {abstract}\n"
        suggested_references.append({
            "title": title,
            "authors": authors[:3],
            "year": str(year),
            "doi": doi,
        })

    call_ctx = _call_context(state, 3000)
    researcher_ctx = _researcher_context(state, 1500)

    base_context = (
        f"FUNDING CALL OBJECTIVES:\n{chr(10).join(objectives[:5])}\n\n"
        f"KEYWORDS: {', '.join(keywords[:10])}\n\n"
        f"RESEARCHER CONTEXT:\n{researcher_ctx}\n\n"
        f"RELEVANT LITERATURE:\n{refs_context or 'No papers found — write based on general knowledge.'}"
    )

    # ── Literature Review section ──────────────────────────────────────────
    lr_system = (
        "You are a senior research scientist writing for a competitive grant proposal. "
        "Write a Literature Review section (900–1200 words) that: "
        "1) Establishes the research domain, 2) Reviews key contributions with inline citations, "
        "3) Identifies what has been achieved and what remains unsolved, "
        "4) Naturally leads to the proposed research. "
        "Use academic tone. Cite papers as [Author, Year]. Use markdown."
    )
    literature_review = _invoke(llm, lr_system, base_context) or "Literature review not generated."

    # ── State of the Art ──────────────────────────────────────────────────
    sota_system = (
        "Write a State of the Art section (600–900 words) that critically assesses "
        "current approaches, highlights limitations, and shows the need for this research. "
        "Be specific and cite the provided literature. Use markdown."
    )
    state_of_art = _invoke(llm, sota_system, base_context) or "State of the art not generated."

    # ── Research Gaps ─────────────────────────────────────────────────────
    gaps_system = (
        "Identify 4–6 specific, concrete research gaps based on the literature review. "
        "Return a JSON array of strings. Each gap should be 1–2 sentences, specific and measurable."
    )
    raw_gaps = _invoke(llm, gaps_system, base_context)
    research_gaps = _parse_json_array(raw_gaps)
    if not research_gaps:
        research_gaps = ["Identified gap: " + g for g in keywords[:3]]

    completed.append("literature_review_agent")
    return ProposalGPTState(
        literature_review=literature_review,
        state_of_art=state_of_art,
        research_gaps=research_gaps,
        suggested_references=suggested_references,
        rag_reflection_info={"papers_retrieved": papers_retrieved,
                             "papers_after_grading": len(papers)},
        errors=errors,
        completed_steps=completed,
        current_step="literature_review_agent",
        progress_pct=_progress(3),
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Proposal Writer (14 core sections)
# ══════════════════════════════════════════════════════════════════════════════

def proposal_writer_node(state: ProposalGPTState) -> ProposalGPTState:
    """
    Generate the 14 core proposal sections:
    Executive Summary, Excellence, Scientific Background, Objectives,
    Research Questions, Methodology, Work Packages, Deliverables,
    Milestones, Risk Management, Consortium, Management, Data Management.
    """
    errors = list(state.get("errors", []))
    completed = list(state.get("completed_steps", []))

    llm = _make_llm(state, temperature=0.4, num_predict=4500)
    agency = state.get("funding_agency", "Generic")
    call_ctx = _call_context(state, 3000)
    researcher_ctx = _researcher_context(state, 2000)
    lit_review = state.get("literature_review", "")[:1500]
    gaps = state.get("research_gaps", [])
    win_strategy = state.get("win_strategy", "")[:800]
    keywords = state.get("keywords", [])
    objectives_call = state.get("call_objectives", [])
    budget_info = state.get("budget_constraints", {})

    proposal_ctx = (
        f"FUNDING AGENCY: {agency}\n"
        f"FUNDING CALL OBJECTIVES: {chr(10).join(objectives_call[:5])}\n"
        f"KEY KEYWORDS: {', '.join(keywords[:10])}\n"
        f"WIN STRATEGY EXCERPT: {win_strategy[:500]}\n"
        f"RESEARCH GAPS: {chr(10).join(gaps[:4])}\n"
        f"LITERATURE REVIEW EXCERPT: {lit_review[:1000]}\n\n"
        f"RESEARCHER CONTEXT:\n{researcher_ctx}"
    )

    # helper to call llm
    def _section(system_prompt: str, extra_ctx: str = "") -> str:
        ctx = proposal_ctx + ("\n\n" + extra_ctx if extra_ctx else "")
        return _invoke(llm, system_prompt, ctx)

    style_note = (
        "Write in the style of an experienced Principal Investigator. "
        f"This is for a {agency} proposal. Use markdown formatting. "
        "Be specific, measurable, and compelling. Avoid generic filler phrases."
    )

    # ── Executive Summary ─────────────────────────────────────────────────
    exec_sys = (
        f"{style_note} Write an Executive Summary (250–350 words) covering: "
        "the research problem, proposed solution, key innovation, expected impact, "
        "and team excellence. Must be compelling to a non-specialist reviewer."
    )
    executive_summary = _section(exec_sys) or "Executive summary not generated."

    # ── Excellence ────────────────────────────────────────────────────────
    excel_sys = (
        f"{style_note} Write the Excellence section (700–1000 words) covering: "
        "beyond-state-of-the-art novelty, scientific/technological innovation, "
        "interdisciplinary approach, and why this team is uniquely positioned. "
        "Directly address the evaluation criteria."
    )
    excellence = _section(excel_sys) or ""

    # ── Scientific Background ─────────────────────────────────────────────
    bg_sys = (
        f"{style_note} Write the Scientific Background (600–900 words). "
        "Explain the fundamental problem, current approaches, and why a new approach is needed. "
        "Build on the literature review to establish credibility."
    )
    scientific_background = _section(bg_sys, f"LITERATURE:\n{lit_review}") or ""

    # ── Objectives ────────────────────────────────────────────────────────
    obj_sys = (
        f"{style_note} Write 4–5 specific, measurable SMART research objectives (400–550 words total). "
        "Each objective should be 2–3 sentences starting with an action verb, with a brief rationale. "
        "Directly address the research gaps. Use numbered markdown list."
    )
    objectives_text = _section(obj_sys) or ""

    # ── Research Questions ─────────────────────────────────────────────────
    rq_sys = (
        "Generate 3–5 precise research questions that the project will answer. "
        "Return a JSON array of strings. Each question should be specific and testable."
    )
    raw_rq = _section(rq_sys)
    research_questions = _parse_json_array(raw_rq)
    if not research_questions:
        research_questions = ["How can " + kw + " be improved?" for kw in keywords[:3]]

    # ── Methodology ───────────────────────────────────────────────────────
    method_sys = (
        f"{style_note} Write the Methodology section (900–1300 words) covering: "
        "research design, data collection, analysis methods, validation approach, "
        "tools and techniques, and how risks are mitigated. "
        "Be specific about methods and justify your choices."
    )
    methodology = _section(method_sys) or ""

    # ── Work Packages ─────────────────────────────────────────────────────
    duration = int(budget_info.get("duration_months", 36))
    wp_sys = (
        f"Generate 4–5 Work Packages for a {duration}-month project. "
        "Return a JSON array of objects with keys: "
        "id (WP1, WP2...), title (str), description (str, 2–3 sentences), "
        "months (str, e.g. M1-M12), lead (str, role), "
        "tasks (list of 3–4 task strings). "
        "WP1 should be coordination/management."
    )
    raw_wps = _section(wp_sys)
    work_packages = _parse_json_array(raw_wps)
    if not work_packages:
        work_packages = [
            {"id": "WP1", "title": "Project Management & Coordination",
             "description": "Overall project governance, reporting, and coordination.",
             "months": "M1-M36", "lead": "Project Manager",
             "tasks": ["Governance", "Reporting", "Communication", "Quality assurance"]},
            {"id": "WP2", "title": "Research & Development",
             "description": "Core research activities.",
             "months": "M1-M30", "lead": "Principal Investigator",
             "tasks": ["Data collection", "Analysis", "Method development", "Validation"]},
        ]

    # ── Deliverables ──────────────────────────────────────────────────────
    del_sys = (
        "Generate 6–8 project deliverables. Return a JSON array of objects with keys: "
        "id (D1.1, D2.1...), title (str), type (Report/Software/Dataset/Prototype), "
        "month (int), wp (WP1, WP2...)."
    )
    raw_dels = _section(del_sys)
    deliverables = _parse_json_array(raw_dels)
    if not deliverables:
        deliverables = [
            {"id": "D1.1", "title": "Data Management Plan", "type": "Report", "month": 3, "wp": "WP1"},
            {"id": "D2.1", "title": "Interim Research Report", "type": "Report", "month": 18, "wp": "WP2"},
            {"id": "D2.2", "title": "Final Research Report", "type": "Report", "month": duration, "wp": "WP2"},
        ]

    # ── Milestones ────────────────────────────────────────────────────────
    ms_sys = (
        "Generate 4–6 project milestones. Return a JSON array of objects with keys: "
        "id (MS1, MS2...), title (str), month (int), verification (str, how to verify)."
    )
    raw_ms = _section(ms_sys)
    milestones = _parse_json_array(raw_ms)
    if not milestones:
        milestones = [
            {"id": "MS1", "title": "Project Kickoff", "month": 1, "verification": "Kickoff meeting minutes"},
            {"id": "MS2", "title": "Mid-term Review", "month": duration // 2, "verification": "Mid-term report approved"},
            {"id": "MS3", "title": "Final Completion", "month": duration, "verification": "Final report submitted"},
        ]

    # ── Risk Management ───────────────────────────────────────────────────
    rm_sys = (
        f"{style_note} Write the Risk Management section (300–450 words). "
        "Include a risk register table (Risk | Likelihood | Impact | Mitigation) "
        "with 5–6 specific risks, and describe the risk monitoring approach."
    )
    risk_management = _section(rm_sys) or ""

    # ── Consortium Description ────────────────────────────────────────────
    partners = state.get("partner_profiles", [])
    partner_ctx = "\n".join(partners[:3]) if partners else "Single-institution project."
    cons_sys = (
        f"{style_note} Write the Consortium Description (200–300 words). "
        "Describe the team's complementary expertise, track record, and why "
        "this consortium is ideally suited to deliver the project."
    )
    consortium_description = _section(cons_sys, f"PARTNERS:\n{partner_ctx}") or ""

    # ── Management Structure ───────────────────────────────────────────────
    mgmt_sys = (
        f"{style_note} Write the Management Structure section (200–250 words). "
        "Cover: governance model, decision-making process, work package leadership, "
        "communication protocols, and quality assurance."
    )
    management_structure = _section(mgmt_sys) or ""

    # ── Data Management ────────────────────────────────────────────────────
    dm_sys = (
        f"{style_note} Write a Data Management Plan section (300–400 words) covering: "
        "data types, collection methods, storage, sharing policy, FAIR principles, "
        "and archiving after project end."
    )
    data_management = _section(dm_sys) or ""

    completed.append("proposal_writer")
    return ProposalGPTState(
        executive_summary=executive_summary,
        excellence=excellence,
        scientific_background=scientific_background,
        objectives=objectives_text,
        research_questions=research_questions,
        methodology=methodology,
        work_packages=work_packages,
        deliverables=deliverables,
        milestones=milestones,
        risk_management=risk_management,
        consortium_description=consortium_description,
        management_structure=management_structure,
        data_management=data_management,
        errors=errors,
        completed_steps=completed,
        current_step="proposal_writer",
        progress_pct=_progress(4),
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Impact Agent
# ══════════════════════════════════════════════════════════════════════════════

def impact_agent_node(state: ProposalGPTState) -> ProposalGPTState:
    """Generate Impact, Dissemination, Exploitation, Ethics, Sustainability sections."""
    errors = list(state.get("errors", []))
    completed = list(state.get("completed_steps", []))

    llm = _make_llm(state, temperature=0.4, num_predict=3000)
    agency = state.get("funding_agency", "Generic")
    executive_summary = state.get("executive_summary", "")[:500]
    keywords = state.get("keywords", [])
    call_ctx = _call_context(state, 2000)

    ctx = (
        f"FUNDING AGENCY: {agency}\n"
        f"PROJECT SUMMARY: {executive_summary}\n"
        f"KEYWORDS: {', '.join(keywords[:8])}\n"
        f"CALL CONTEXT: {call_ctx[:1500]}"
    )

    style = f"You are writing for a {agency} proposal. Write in academic grant style. Use markdown."

    impact = _invoke(llm,
        f"{style} Write the Impact section (600–900 words). Cover: "
        "scientific impact (publications, knowledge), societal impact (who benefits and how), "
        "economic impact (commercialisation potential), policy impact (if applicable), "
        "and a clear pathway to impact timeline.", ctx) or ""

    dissemination = _invoke(llm,
        f"{style} Write the Dissemination & Communication section (400–550 words). "
        "Cover: target audiences, communication channels (publications, conferences, "
        "social media, press), dissemination timeline, and open access policy.", ctx) or ""

    exploitation = _invoke(llm,
        f"{style} Write the Exploitation section (250–350 words). Cover: "
        "IP strategy, commercialisation pathways, spin-out potential, licensing strategy, "
        "and how results will be used after the project ends.", ctx) or ""

    ethics = _invoke(llm,
        f"{style} Write the Ethics section (250–350 words). Cover: "
        "ethical issues (data privacy, dual-use, environmental impact), "
        "how each is addressed, ethical approval status, and compliance frameworks.", ctx) or ""

    sustainability = _invoke(llm,
        f"{style} Write the Sustainability section (200–250 words). Cover: "
        "how the project continues after funding ends, follow-on funding plans, "
        "long-term maintenance of infrastructure/software, and stakeholder engagement.", ctx) or ""

    completed.append("impact_agent")
    return ProposalGPTState(
        impact=impact,
        dissemination=dissemination,
        exploitation=exploitation,
        ethics=ethics,
        sustainability=sustainability,
        errors=errors,
        completed_steps=completed,
        current_step="impact_agent",
        progress_pct=_progress(5),
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — Budget Agent
# ══════════════════════════════════════════════════════════════════════════════

def budget_agent_node(state: ProposalGPTState) -> ProposalGPTState:
    """Generate structured budget: personnel, equipment, travel, indirect + justification."""
    errors = list(state.get("errors", []))
    completed = list(state.get("completed_steps", []))

    llm = _make_llm(state, temperature=0.2, num_predict=2000)
    budget_constraints = state.get("budget_constraints", {})
    budget_format = state.get("budget_format", "generic")
    agency = state.get("funding_agency", "Generic")
    duration = int(budget_constraints.get("duration_months", 36))
    max_budget_str = budget_constraints.get("max_budget", "500000")
    currency = budget_constraints.get("currency", "EUR")
    work_packages = state.get("work_packages", [])
    executive_summary = state.get("executive_summary", "")[:400]

    # parse max budget
    try:
        max_budget = float(re.sub(r'[^0-9.]', '', str(max_budget_str))) or 500000.0
    except (ValueError, TypeError):
        max_budget = 500000.0

    ctx = (
        f"FUNDING AGENCY: {agency}\nBUDGET FORMAT: {budget_format}\n"
        f"MAX BUDGET: {currency} {max_budget:,.0f}\nDURATION: {duration} months\n"
        f"WORK PACKAGES: {json.dumps([w.get('id','') + ' ' + w.get('title','') for w in work_packages])}\n"
        f"PROJECT: {executive_summary}"
    )

    # ── Personnel ─────────────────────────────────────────────────────────
    pers_system = (
        "Generate a realistic personnel budget for this research project. "
        f"Total personnel should be ~60–70% of total budget ({currency} {max_budget:,.0f}). "
        "Return a JSON array of objects with keys: "
        "role (str), months (int), rate_per_month (float), total (float). "
        "Include: Principal Investigator (partial), 1-2 PostDocs, 1 PhD student, "
        "1 Research Engineer/Manager. Use realistic salary rates for the funding agency."
    )
    raw_personnel = _invoke(llm, pers_system, ctx)
    budget_personnel = _parse_json_array(raw_personnel)
    if not budget_personnel:
        monthly = max_budget * 0.65 / (duration * 3.5)
        budget_personnel = [
            {"role": "Principal Investigator (20% FTE)", "months": duration, "rate_per_month": round(monthly * 0.2, 0), "total": round(monthly * 0.2 * duration, 0)},
            {"role": "PostDoc Researcher", "months": min(30, duration), "rate_per_month": round(monthly, 0), "total": round(monthly * min(30, duration), 0)},
            {"role": "PhD Student", "months": duration, "rate_per_month": round(monthly * 0.6, 0), "total": round(monthly * 0.6 * duration, 0)},
        ]

    # ── Equipment ─────────────────────────────────────────────────────────
    equip_system = (
        "Generate an equipment budget (10–15% of total budget). "
        "Return a JSON array of objects with keys: item (str), cost (float), justification (str)."
    )
    raw_equip = _invoke(llm, equip_system, ctx)
    budget_equipment = _parse_json_array(raw_equip)
    if not budget_equipment:
        budget_equipment = [{"item": "Computing infrastructure", "cost": max_budget * 0.08, "justification": "Required for data processing"}]

    # ── Travel ────────────────────────────────────────────────────────────
    travel_system = (
        "Generate a travel budget (~5% of total). "
        "Return a JSON array of objects with keys: destination (str), purpose (str), cost (float)."
    )
    raw_travel = _invoke(llm, travel_system, ctx)
    budget_travel = _parse_json_array(raw_travel)
    if not budget_travel:
        budget_travel = [
            {"destination": "International conferences", "purpose": "Dissemination", "cost": max_budget * 0.03},
            {"destination": "Partner meetings", "purpose": "Collaboration", "cost": max_budget * 0.02},
        ]

    # ── Calculate totals ──────────────────────────────────────────────────
    indirect_rate = 0.25 if budget_format == "horizon_europe" else 0.20
    pers_total = sum(float(p.get("total", 0)) for p in budget_personnel)
    equip_total = sum(float(e.get("cost", 0)) for e in budget_equipment)
    travel_total = sum(float(t.get("cost", 0)) for t in budget_travel)
    direct_total = pers_total + equip_total + travel_total
    indirect = round(pers_total * indirect_rate, 2)
    grand_total = round(direct_total + indirect, 2)

    # Scale to fit within max budget if needed
    if grand_total > max_budget * 1.05:
        scale = (max_budget * 0.98) / grand_total
        for p in budget_personnel:
            p["total"] = round(float(p.get("total", 0)) * scale, 0)
            p["rate_per_month"] = round(float(p.get("rate_per_month", 0)) * scale, 0)
        for e in budget_equipment:
            e["cost"] = round(float(e.get("cost", 0)) * scale, 0)
        for t in budget_travel:
            t["cost"] = round(float(t.get("cost", 0)) * scale, 0)
        pers_total = sum(float(p.get("total", 0)) for p in budget_personnel)
        equip_total = sum(float(e.get("cost", 0)) for e in budget_equipment)
        travel_total = sum(float(t.get("cost", 0)) for t in budget_travel)
        direct_total = pers_total + equip_total + travel_total
        indirect = round(pers_total * indirect_rate, 2)
        grand_total = round(direct_total + indirect, 2)

    # ── Summary Table ─────────────────────────────────────────────────────
    summary_rows = [
        f"| Personnel | {currency} {pers_total:,.0f} |",
        f"| Equipment | {currency} {equip_total:,.0f} |",
        f"| Travel | {currency} {travel_total:,.0f} |",
        f"| Indirect Costs ({int(indirect_rate*100)}%) | {currency} {indirect:,.0f} |",
        f"| **TOTAL** | **{currency} {grand_total:,.0f}** |",
    ]
    budget_summary_table = "| Category | Amount |\n|----------|--------|\n" + "\n".join(summary_rows)

    # ── Budget Justification ──────────────────────────────────────────────
    just_system = (
        f"Write a Budget Justification section (300–400 words) for a {agency} proposal. "
        f"Justify the personnel, equipment, and travel costs as necessary and cost-effective. "
        f"Total budget: {currency} {grand_total:,.0f}. Use markdown."
    )
    budget_justification = _invoke(llm, just_system, ctx + f"\n\nBUDGET SUMMARY:\n{budget_summary_table}") or ""

    completed.append("budget_agent")
    return ProposalGPTState(
        budget_personnel=budget_personnel,
        budget_equipment=budget_equipment,
        budget_travel=budget_travel,
        budget_indirect_rate=indirect_rate,
        budget_indirect=indirect,
        budget_total=grand_total,
        budget_justification=budget_justification,
        budget_summary_table=budget_summary_table,
        errors=errors,
        completed_steps=completed,
        current_step="budget_agent",
        progress_pct=_progress(6),
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 7 — Compliance Checker
# ══════════════════════════════════════════════════════════════════════════════

def compliance_agent_node(state: ProposalGPTState) -> ProposalGPTState:
    """Check mandatory sections, keyword coverage, estimate page count, score compliance."""
    errors = list(state.get("errors", []))
    completed = list(state.get("completed_steps", []))

    mandatory_sections = state.get("mandatory_sections", [])
    keywords = state.get("keywords", [])

    # Collect all generated text
    section_map = {
        "Executive Summary": state.get("executive_summary", ""),
        "Excellence": state.get("excellence", ""),
        "Scientific Background": state.get("scientific_background", ""),
        "Literature Review": state.get("literature_review", ""),
        "State of the Art": state.get("state_of_art", ""),
        "Objectives": state.get("objectives", ""),
        "Methodology": state.get("methodology", ""),
        "Work Packages": json.dumps(state.get("work_packages", [])),
        "Risk Management": state.get("risk_management", ""),
        "Impact": state.get("impact", ""),
        "Dissemination": state.get("dissemination", ""),
        "Exploitation": state.get("exploitation", ""),
        "Ethics": state.get("ethics", ""),
        "Sustainability": state.get("sustainability", ""),
        "Budget Justification": state.get("budget_justification", ""),
        "Consortium Description": state.get("consortium_description", ""),
        "Management Structure": state.get("management_structure", ""),
        "Data Management": state.get("data_management", ""),
    }

    all_text = "\n\n".join(v for v in section_map.values() if v)

    # ── Missing sections ──────────────────────────────────────────────────
    generated_names = {k.lower() for k, v in section_map.items() if v.strip()}
    missing_sections = []
    for ms in mandatory_sections:
        ms_lower = ms.lower()
        if not any(ms_lower in gn or gn in ms_lower for gn in generated_names):
            missing_sections.append(ms)

    # ── Keyword coverage ──────────────────────────────────────────────────
    all_text_lower = all_text.lower()
    keyword_coverage = {}
    for kw in keywords:
        keyword_coverage[kw.lower()] = kw.lower() in all_text_lower

    kw_covered = sum(1 for v in keyword_coverage.values() if v)
    kw_total = max(len(keyword_coverage), 1)

    # ── Page estimate (avg 500 words/page) ────────────────────────────────
    word_count = len(all_text.split())
    page_estimate = max(1, round(word_count / 500))

    # ── Compliance score ──────────────────────────────────────────────────
    filled_count = sum(1 for v in section_map.values() if v.strip())
    fill_pct = round(filled_count / len(section_map) * 100)
    missing_penalty = len(missing_sections) * 10
    section_score = max(0, min(fill_pct, 100 - missing_penalty))
    kw_score = round(kw_covered / kw_total * 100)
    has_budget = 1 if state.get("budget_total", 0) > 0 else 0
    compliance_score = round((section_score * 0.5) + (kw_score * 0.3) + (has_budget * 20))
    compliance_score = min(100, max(0, compliance_score))

    # ── Compliance issues ─────────────────────────────────────────────────
    compliance_issues = []
    if missing_sections:
        compliance_issues.append(f"Missing mandatory sections: {', '.join(missing_sections)}")
    if kw_covered < kw_total * 0.5:
        uncovered = [k for k, v in keyword_coverage.items() if not v][:5]
        compliance_issues.append(f"Low keyword coverage. Missing: {', '.join(uncovered)}")
    if not state.get("budget_total", 0):
        compliance_issues.append("No budget generated.")

    # ── Compliance Report ─────────────────────────────────────────────────
    report_lines = [
        "## Compliance Report\n",
        f"**Compliance Score: {compliance_score}/100**\n",
        f"**Estimated Page Count:** {page_estimate} pages (~{word_count:,} words)\n",
        f"**Keyword Coverage:** {kw_covered}/{kw_total} keywords found\n",
        "",
        "### Section Status",
        "| Section | Status |",
        "|---------|--------|",
    ]
    for section, text in section_map.items():
        status = "✅ Generated" if text.strip() else "❌ Missing"
        report_lines.append(f"| {section} | {status} |")

    if keywords:
        report_lines += ["", "### Keyword Coverage", "| Keyword | Found |", "|---------|-------|"]
        for kw, found in keyword_coverage.items():
            report_lines.append(f"| {kw} | {'✅' if found else '❌'} |")

    if compliance_issues:
        report_lines += ["", "### Issues to Address"]
        for issue in compliance_issues:
            report_lines.append(f"- ⚠️ {issue}")

    compliance_report = "\n".join(report_lines)

    # ── Update checklist status ────────────────────────────────────────────
    checklist = state.get("compliance_checklist", [])
    for item in checklist:
        item_text = item.get("item", "").lower()
        if any(item_text in gn or gn in item_text for gn in generated_names):
            item["status"] = "✅ Complete"
        elif item_text.startswith("keyword:"):
            kw = item_text.replace("keyword:", "").strip()
            item["status"] = "✅ Found" if keyword_coverage.get(kw, False) else "❌ Missing"

    completed.append("compliance_agent")
    return ProposalGPTState(
        compliance_report=compliance_report,
        missing_sections=missing_sections,
        keyword_coverage=keyword_coverage,
        compliance_score=compliance_score,
        page_estimate=page_estimate,
        compliance_issues=compliance_issues,
        compliance_checklist=checklist,
        errors=errors,
        completed_steps=completed,
        current_step="compliance_agent",
        progress_pct=_progress(7),
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 8 — Reviewer Simulation
# ══════════════════════════════════════════════════════════════════════════════

_REVIEWER_PROFILES = {
    "scientific": {
        "name": "Scientific Reviewer",
        "focus": "scientific merit, novelty, methodology rigour, literature grounding",
        "weight": 0.30,
    },
    "impact": {
        "name": "Impact Reviewer",
        "focus": "societal/economic impact, dissemination, exploitation, sustainability",
        "weight": 0.25,
    },
    "innovation": {
        "name": "Innovation Reviewer",
        "focus": "novelty, technological advancement, beyond state-of-the-art",
        "weight": 0.20,
    },
    "implementation": {
        "name": "Implementation Reviewer",
        "focus": "work plan, budget, management, risk mitigation, team competence",
        "weight": 0.15,
    },
    "agency": {
        "name": "Funding Agency Reviewer",
        "focus": "strategic fit with funding priorities, compliance, keywords alignment",
        "weight": 0.10,
    },
}


def reviewer_agent_node(state: ProposalGPTState) -> ProposalGPTState:
    """Simulate 5 virtual reviewers scoring and providing feedback on the proposal."""
    errors = list(state.get("errors", []))
    completed = list(state.get("completed_steps", []))

    llm = _make_llm(state, temperature=0.5, num_predict=1500)
    agency = state.get("funding_agency", "Generic")

    # Build proposal digest for reviewers
    proposal_digest = "\n\n".join([
        f"## Executive Summary\n{state.get('executive_summary', '')[:600]}",
        f"## Excellence\n{state.get('excellence', '')[:500]}",
        f"## Methodology\n{state.get('methodology', '')[:500]}",
        f"## Impact\n{state.get('impact', '')[:400]}",
        f"## Budget Total: {state.get('budget_format', '')} — {state.get('budget_total', 0):,.0f}",
        f"## Compliance Score: {state.get('compliance_score', 0)}/100",
    ])

    reviewer_scores: Dict[str, Dict[str, Any]] = {}
    weighted_total = 0.0

    for reviewer_key, profile in _REVIEWER_PROFILES.items():
        review_system = (
            f"You are a {profile['name']} for a {agency} funding panel. "
            f"Your focus: {profile['focus']}. "
            "Score the following proposal on a scale of 1–5 (5=Excellent). "
            "Return a JSON object with keys: "
            "overall_score (float 1-5), "
            "strengths (list of 3 strings), "
            "weaknesses (list of 3 strings), "
            "suggestions (list of 2-3 improvement suggestions). "
            "Be specific and constructive."
        )
        raw = _invoke(llm, review_system, proposal_digest[:3000])
        review_data = _parse_json_object(raw)

        if not review_data.get("overall_score"):
            review_data = {
                "overall_score": 3.0,
                "strengths": ["Clear research objectives", "Relevant to funding priorities", "Competent team"],
                "weaknesses": ["Novelty could be clearer", "Budget justification needs strengthening", "Impact pathway unclear"],
                "suggestions": ["Add more specific KPIs", "Strengthen the innovation case"],
            }

        reviewer_scores[reviewer_key] = review_data
        weighted_total += float(review_data.get("overall_score", 3.0)) * profile["weight"]

    overall_score = round(min(5.0, max(1.0, weighted_total / sum(p["weight"] for p in _REVIEWER_PROFILES.values()))), 2)

    # ── Reviewer Report ───────────────────────────────────────────────────
    report_lines = [
        "## Reviewer Simulation Report\n",
        f"**Overall Score: {overall_score:.1f} / 5.0** "
        f"({'Excellent' if overall_score >= 4.5 else 'Good' if overall_score >= 3.5 else 'Acceptable' if overall_score >= 2.5 else 'Needs Work'})\n",
        "",
        "| Reviewer | Score | Weight |",
        "|----------|-------|--------|",
    ]
    for key, profile in _REVIEWER_PROFILES.items():
        score = reviewer_scores.get(key, {}).get("overall_score", 3.0)
        report_lines.append(f"| {profile['name']} | {score:.1f}/5.0 | {int(profile['weight']*100)}% |")

    report_lines.append("")
    for key, profile in _REVIEWER_PROFILES.items():
        data = reviewer_scores.get(key, {})
        report_lines += [
            f"### {profile['name']}",
            f"**Score:** {data.get('overall_score', 3.0):.1f}/5.0\n",
            "**Strengths:**",
        ]
        for s in data.get("strengths", []):
            report_lines.append(f"- ✅ {s}")
        report_lines.append("**Weaknesses:**")
        for w in data.get("weaknesses", []):
            report_lines.append(f"- ⚠️ {w}")
        report_lines.append("**Suggestions:**")
        for sg in data.get("suggestions", []):
            report_lines.append(f"- 💡 {sg}")
        report_lines.append("")

    reviewer_report = "\n".join(report_lines)

    completed.append("reviewer_agent")
    return ProposalGPTState(
        reviewer_scores=reviewer_scores,
        overall_score=overall_score,
        reviewer_report=reviewer_report,
        errors=errors,
        completed_steps=completed,
        current_step="reviewer_agent",
        progress_pct=_progress(8),
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 9 — Proposal Improvement Agent
# ══════════════════════════════════════════════════════════════════════════════

def improvement_agent_node(state: ProposalGPTState) -> ProposalGPTState:
    """Identify weak sections from reviewer feedback and rewrite them."""
    errors = list(state.get("errors", []))
    completed = list(state.get("completed_steps", []))

    llm = _make_llm(state, temperature=0.4, num_predict=2500)
    reviewer_scores = state.get("reviewer_scores", {})
    agency = state.get("funding_agency", "Generic")

    # Collect all weaknesses from reviewers
    all_weaknesses: List[str] = []
    all_suggestions: List[str] = []
    for data in reviewer_scores.values():
        all_weaknesses.extend(data.get("weaknesses", []))
        all_suggestions.extend(data.get("suggestions", []))

    # ── Identify weak sections ────────────────────────────────────────────
    section_scores: Dict[str, float] = {}
    section_keywords = {
        "executive_summary": ["executive", "summary", "overview"],
        "excellence": ["novelty", "innovation", "excellence", "beyond state"],
        "methodology": ["methodology", "methods", "approach", "design"],
        "impact": ["impact", "societal", "economic", "benefit"],
        "budget_justification": ["budget", "cost", "financial"],
    }
    for section, keywords in section_keywords.items():
        weaknesses_for_section = sum(
            1 for w in all_weaknesses
            if any(kw in w.lower() for kw in keywords)
        )
        section_scores[section] = weaknesses_for_section

    weak_sections = [s for s, score in sorted(section_scores.items(), key=lambda x: -x[1])
                     if score > 0][:3]

    # ── Improvement Plan ──────────────────────────────────────────────────
    plan_system = (
        "You are a proposal improvement specialist. Based on the reviewer feedback, "
        "write an Improvement Plan (200–300 words) in markdown. "
        "Prioritize the top 3–5 improvements by impact on score. Be specific and actionable."
    )
    plan_ctx = (
        f"REVIEWER WEAKNESSES:\n{chr(10).join(all_weaknesses[:8])}\n\n"
        f"REVIEWER SUGGESTIONS:\n{chr(10).join(all_suggestions[:6])}\n\n"
        f"OVERALL SCORE: {state.get('overall_score', 3.0):.1f}/5.0"
    )
    improvement_plan = _invoke(llm, plan_system, plan_ctx) or "Improvement plan not generated."

    # ── Rewrite weak sections ─────────────────────────────────────────────
    improved_sections: Dict[str, str] = {}
    section_content_map = {
        "executive_summary": state.get("executive_summary", ""),
        "excellence": state.get("excellence", ""),
        "methodology": state.get("methodology", ""),
        "impact": state.get("impact", ""),
        "budget_justification": state.get("budget_justification", ""),
    }

    for section in weak_sections[:2]:  # rewrite top 2 weakest
        original = section_content_map.get(section, "")
        if not original:
            continue

        relevant_feedback = [w for w in all_weaknesses + all_suggestions
                             if any(kw in w.lower() for kw in section_keywords.get(section, []))]

        improve_system = (
            f"You are a senior proposal writer improving a {agency} grant proposal. "
            f"Rewrite the {section.replace('_', ' ').title()} section addressing the reviewer feedback. "
            "Maintain academic grant style. Make it stronger, more specific, and more compelling. "
            "Keep approximately the same length. Use markdown."
        )
        improve_ctx = (
            f"ORIGINAL SECTION:\n{original[:1500]}\n\n"
            f"REVIEWER FEEDBACK TO ADDRESS:\n{chr(10).join(relevant_feedback[:5])}"
        )
        improved = _invoke(llm, improve_system, improve_ctx)
        if improved:
            improved_sections[section] = improved

    completed.append("improvement_agent")
    return ProposalGPTState(
        improvement_plan=improvement_plan,
        weak_sections=weak_sections,
        improved_sections=improved_sections,
        errors=errors,
        completed_steps=completed,
        current_step="improvement_agent",
        progress_pct=100,
    )
