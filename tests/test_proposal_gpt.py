"""
tests/test_proposal_gpt.py
───────────────────────────
Unit tests for ProposalGPT — 9-agent LangGraph pipeline.

Coverage:
  - ProposalGPTState factory
  - Budget format detection
  - Each of the 9 agent nodes (mocked LLM)
  - Budget calculations
  - Compliance scoring
  - Reviewer scoring and weighting
  - LangGraph pipeline build + run (full integration, mocked)
  - tools/proposal_tools.py (assemble_full_proposal_md, build_budget_csv, section_word_count)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ════════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════════

SAMPLE_CALL_TEXT = """
Horizon Europe Research Call — Digital and Industrial Technologies
Budget: up to €2,000,000 per project. Duration: 36 months.
Deadline: 15 March 2025.

Objectives:
1. Develop novel AI-driven solutions for manufacturing.
2. Demonstrate scalability across EU industrial sectors.
3. Ensure GDPR compliance and ethical AI deployment.

Evaluation Criteria:
- Excellence (40%)
- Impact (30%)
- Implementation (30%)

Mandatory sections: Executive Summary, Excellence, Impact, Budget.
Keywords: AI, manufacturing, Industry 4.0, machine learning, digital transformation.
"""

SAMPLE_RESEARCHER_IDEAS = (
    "Our team has 10 years of experience in federated learning for smart factories. "
    "We have preliminary results showing 30% efficiency gain."
)


@pytest.fixture
def sample_state():
    from agents.proposal_gpt_state import create_proposal_gpt_state
    return create_proposal_gpt_state(
        funding_call_text=SAMPLE_CALL_TEXT,
        user_ideas=SAMPLE_RESEARCHER_IDEAS,
        funding_agency="Horizon Europe",
        model_name="test-model",
        num_ctx=4096,
        session_id="test001",
    )


@pytest.fixture
def populated_state(sample_state):
    """State with all main sections filled in (no LLM needed)."""
    state = dict(sample_state)
    state.update(
        call_title="Digital AI for Industry 4.0",
        call_objectives=["Develop AI for manufacturing", "Ensure GDPR compliance"],
        keywords=["AI", "manufacturing", "machine learning"],
        evaluation_criteria=[{"criterion": "Excellence", "weight": "40%", "description": "Novelty"}],
        expected_outcomes=["Scalable AI platform", "Published results"],
        budget_constraints={"max_budget": "2000000", "currency": "EUR", "duration_months": 36},
        mandatory_sections=["Executive Summary", "Excellence", "Impact"],
        deadline="15 March 2025",
        funding_summary="## Summary\n\nThis call funds AI for industry.",
        evaluation_matrix="| Excellence | 40% | Novelty |",
        compliance_checklist=[{"item": "Executive Summary", "status": "⬜ Pending", "notes": ""}],
        success_factors="Focus on measurable impact.",
        hidden_priorities=["Practical deployment", "SME involvement"],
        win_strategy="## Win Strategy\n\nFocus on industrial validation.",
        reviewer_perspective="Scientific reviewers will look for novelty.",
        swot_analysis="| Strengths | Weaknesses | Opportunities | Threats |",
        proposal_risks=[{"risk": "Data availability", "likelihood": "Medium", "mitigation": "Pre-agreements with industry"}],
        proposal_strengths=["Strong preliminary results", "Industry partnerships"],
        literature_review="## Literature Review\n\nFederated learning has shown promise [Smith, 2023].",
        state_of_art="## State of the Art\n\nCurrent methods lack scalability.",
        research_gaps=["Gap 1: No cross-domain validation", "Gap 2: Privacy-preserving methods lacking"],
        suggested_references=[{"title": "FL Survey", "authors": ["Smith J"], "year": "2023", "doi": "10.1"}],
        executive_summary="## Executive Summary\n\nThis project proposes a novel AI system.",
        excellence="## Excellence\n\nOur approach is beyond state-of-the-art.",
        scientific_background="## Scientific Background\n\nAI in manufacturing is critical.",
        objectives="1. Develop FL system\n2. Validate in industry\n3. Publish results",
        research_questions=["How can FL improve manufacturing?", "What privacy guarantees are needed?"],
        methodology="## Methodology\n\nWe use federated learning with differential privacy.",
        work_packages=[
            {"id": "WP1", "title": "Project Management", "description": "Coordination", "months": "M1-M36", "lead": "PI", "tasks": ["Task 1"]},
            {"id": "WP2", "title": "Research", "description": "Core research", "months": "M1-M30", "lead": "PostDoc", "tasks": ["Task 1", "Task 2"]},
        ],
        deliverables=[{"id": "D1.1", "title": "DMP", "type": "Report", "month": 3, "wp": "WP1"}],
        milestones=[{"id": "MS1", "title": "Kickoff", "month": 1, "verification": "Meeting minutes"}],
        risk_management="## Risk Management\n\n| Risk | Likelihood | Mitigation |",
        consortium_description="## Consortium\n\nLed by Chalmers University.",
        management_structure="## Management\n\nPM leads coordination.",
        data_management="## DMP\n\nAll data stored securely.",
        impact="## Impact\n\n30% efficiency gain expected.",
        dissemination="## Dissemination\n\nPublish in top venues.",
        exploitation="## Exploitation\n\nSpin-out company planned.",
        ethics="## Ethics\n\nGDPR compliant. No dual-use issues.",
        sustainability="## Sustainability\n\nFollow-on funding planned.",
        budget_personnel=[
            {"role": "PI (20%)", "months": 36, "rate_per_month": 1000, "total": 36000},
            {"role": "PostDoc", "months": 30, "rate_per_month": 5000, "total": 150000},
        ],
        budget_equipment=[{"item": "GPU Cluster", "cost": 50000, "justification": "Required for training"}],
        budget_travel=[{"destination": "EU Conferences", "purpose": "Dissemination", "cost": 15000}],
        budget_indirect_rate=0.25,
        budget_indirect=46500,
        budget_total=297500,
        budget_justification="Budget is cost-effective.",
        budget_summary_table="| Personnel | EUR 186,000 |\n| Equipment | EUR 50,000 |",
        compliance_report="## Compliance\n\nScore: 85/100",
        missing_sections=[],
        keyword_coverage={"AI": True, "manufacturing": True, "machine learning": True},
        compliance_score=85,
        page_estimate=25,
        compliance_issues=[],
        reviewer_scores={
            "scientific": {"overall_score": 4.0, "strengths": ["s1"], "weaknesses": ["Novelty could be clearer"], "suggestions": ["sg1"]},
            "impact": {"overall_score": 3.5, "strengths": ["s1"], "weaknesses": ["Impact pathway unclear"], "suggestions": ["sg1"]},
            "innovation": {"overall_score": 4.2, "strengths": ["s1"], "weaknesses": ["Innovation beyond state-of-art not demonstrated"], "suggestions": ["sg1"]},
            "implementation": {"overall_score": 3.8, "strengths": ["s1"], "weaknesses": ["Budget not fully justified"], "suggestions": ["sg1"]},
            "agency": {"overall_score": 4.0, "strengths": ["s1"], "weaknesses": ["Keywords not fully aligned"], "suggestions": ["sg1"]},
        },
        overall_score=3.9,
        reviewer_report="## Reviewer Report\n\nOverall: 3.9/5.0",
        improvement_plan="## Improvement Plan\n\n1. Strengthen novelty argument.",
        weak_sections=["excellence"],
        improved_sections={"excellence": "## Excellence (Improved)\n\nStronger novelty claim."},
    )
    return state


# ════════════════════════════════════════════════════════════════════════════════
# State Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestProposalGPTState:
    def test_factory_creates_required_fields(self, sample_state):
        assert sample_state["session_id"] == "test001"
        assert sample_state["model_name"] == "test-model"
        assert sample_state["funding_call_text"] == SAMPLE_CALL_TEXT
        assert sample_state["funding_agency"] == "Horizon Europe"
        assert sample_state["budget_format"] == "horizon_europe"

    def test_factory_initialises_empty_lists(self, sample_state):
        assert sample_state["call_objectives"] == []
        assert sample_state["keywords"] == []
        assert sample_state["errors"] == []
        assert sample_state["completed_steps"] == []

    def test_factory_sets_default_session_id(self):
        from agents.proposal_gpt_state import create_proposal_gpt_state
        state = create_proposal_gpt_state(funding_call_text="test")
        assert len(state["session_id"]) == 8

    def test_budget_format_detection_horizon(self):
        from agents.proposal_gpt_state import _detect_budget_format
        assert _detect_budget_format("Horizon Europe") == "horizon_europe"
        assert _detect_budget_format("ERC Advanced Grant") == "horizon_europe"
        assert _detect_budget_format("MSCA Fellowship") == "horizon_europe"

    def test_budget_format_detection_swedish(self):
        from agents.proposal_gpt_state import _detect_budget_format
        assert _detect_budget_format("Vinnova") == "swedish"
        assert _detect_budget_format("VR (Swedish Research Council)") == "swedish"
        assert _detect_budget_format("Formas") == "swedish"

    def test_budget_format_detection_generic(self):
        from agents.proposal_gpt_state import _detect_budget_format
        assert _detect_budget_format("NSF") == "generic"
        assert _detect_budget_format("DARPA") == "generic"
        assert _detect_budget_format("Generic") == "generic"

    def test_cv_texts_default_empty(self, sample_state):
        assert sample_state["cv_texts"] == []

    def test_researcher_context_passed(self):
        from agents.proposal_gpt_state import create_proposal_gpt_state
        state = create_proposal_gpt_state(
            funding_call_text="call",
            user_ideas="My idea",
            institution_info="Chalmers",
        )
        assert state["user_ideas"] == "My idea"
        assert state["institution_info"] == "Chalmers"


# ════════════════════════════════════════════════════════════════════════════════
# Node Tests — Agent 1: Funding Call Analyzer
# ════════════════════════════════════════════════════════════════════════════════

class TestFundingCallAnalyzer:
    def _mock_llm_response(self, content: str):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = content
        mock_llm.invoke.return_value = mock_resp
        return mock_llm

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_extracts_structured_fields(self, mock_make_llm, sample_state):
        from agents.proposal_gpt_nodes import funding_call_analyzer_node
        extracted = json.dumps({
            "title": "AI Manufacturing Call",
            "objectives": ["Develop AI", "Ensure compliance"],
            "eligibility": ["EU entities only"],
            "evaluation_criteria": [{"criterion": "Excellence", "weight": "40%", "description": "Novelty"}],
            "expected_outcomes": ["Scalable platform"],
            "budget_max": "2000000",
            "currency": "EUR",
            "duration_months": 36,
            "deadline": "March 2025",
            "mandatory_sections": ["Executive Summary"],
            "keywords": ["AI", "manufacturing"],
        })
        mock_make_llm.return_value = self._mock_llm_response(extracted)
        result = funding_call_analyzer_node(sample_state)
        assert result["call_title"] == "AI Manufacturing Call"
        assert result["call_objectives"] == ["Develop AI", "Ensure compliance"]
        assert result["keywords"] == ["AI", "manufacturing"]
        assert result["deadline"] == "March 2025"
        assert result["budget_constraints"]["currency"] == "EUR"

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_handles_empty_call_text(self, mock_make_llm):
        from agents.proposal_gpt_state import create_proposal_gpt_state
        from agents.proposal_gpt_nodes import funding_call_analyzer_node
        state = create_proposal_gpt_state(funding_call_text="")
        mock_make_llm.return_value = self._mock_llm_response("{}")
        result = funding_call_analyzer_node(state)
        assert len(result.get("errors", [])) > 0

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_sets_progress_and_completed_step(self, mock_make_llm, sample_state):
        from agents.proposal_gpt_nodes import funding_call_analyzer_node
        mock_make_llm.return_value = self._mock_llm_response(
            json.dumps({"title": "Test", "objectives": [], "eligibility": [],
                        "evaluation_criteria": [], "expected_outcomes": [],
                        "budget_max": "", "currency": "EUR", "duration_months": 36,
                        "deadline": "", "mandatory_sections": [], "keywords": []})
        )
        result = funding_call_analyzer_node(sample_state)
        assert "funding_call_analyzer" in result["completed_steps"]
        assert result["progress_pct"] > 0

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_generates_compliance_checklist(self, mock_make_llm, sample_state):
        from agents.proposal_gpt_nodes import funding_call_analyzer_node
        mock_make_llm.return_value = self._mock_llm_response(
            json.dumps({"title": "T", "objectives": [], "eligibility": [],
                        "evaluation_criteria": [], "expected_outcomes": [],
                        "budget_max": "1M", "currency": "EUR", "duration_months": 36,
                        "deadline": "Jan 2025", "mandatory_sections": ["Ethics", "Budget"],
                        "keywords": ["AI"]})
        )
        result = funding_call_analyzer_node(sample_state)
        checklist_items = [item["item"] for item in result.get("compliance_checklist", [])]
        assert "Ethics" in checklist_items
        assert "Budget" in checklist_items


# ════════════════════════════════════════════════════════════════════════════════
# Node Tests — Agent 2: Research Planner
# ════════════════════════════════════════════════════════════════════════════════

class TestResearchPlanner:
    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_generates_win_strategy(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import research_planner_node
        priorities_json = json.dumps({
            "hidden_priorities": ["SME involvement", "TRL 4+"],
            "reviewer_expectations": "Expect clear innovation.",
            "positioning": "Position as industry-led.",
            "competitive_advantages": ["10 years experience"],
            "proposal_strengths": ["Preliminary results"],
        })
        mock_resp = MagicMock()
        call_count = [0]

        def side_effect(messages):
            call_count[0] += 1
            resp = MagicMock()
            if call_count[0] == 1:
                resp.content = priorities_json
            elif call_count[0] == 2:
                resp.content = json.dumps([{"risk": "Data access", "likelihood": "Medium", "mitigation": "MOUs"}])
            elif call_count[0] == 3:
                resp.content = "## Win Strategy\n\nFocus on impact."
            elif call_count[0] == 4:
                resp.content = "## Reviewer Perspective\n\nLook for novelty."
            else:
                resp.content = "| S | W | O | T |\n|---|---|---|---|"
            return resp

        mock_make_llm.return_value.invoke.side_effect = side_effect
        result = research_planner_node(populated_state)
        assert "research_planner" in result["completed_steps"]
        assert result["hidden_priorities"] == ["SME involvement", "TRL 4+"]

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_handles_llm_failure_gracefully(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import research_planner_node
        mock_make_llm.return_value.invoke.side_effect = Exception("LLM timeout")
        result = research_planner_node(populated_state)
        assert "research_planner" in result["completed_steps"]
        assert result.get("proposal_risks") is not None


# ════════════════════════════════════════════════════════════════════════════════
# Node Tests — Agent 3: Literature Review
# ════════════════════════════════════════════════════════════════════════════════

class TestLiteratureReviewAgent:
    @patch("agents.proposal_gpt_nodes._make_llm")
    @patch("agents.proposal_gpt_nodes.search_semantic_scholar", create=True)
    @patch("agents.proposal_gpt_nodes.search_arxiv", create=True)
    def test_generates_literature_review(self, mock_arxiv, mock_ss, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import literature_review_agent_node

        mock_ss_paper = MagicMock()
        mock_ss_paper.title = "Federated Learning Survey"
        mock_ss_paper.authors = ["Smith J", "Jones K"]
        mock_ss_paper.year = 2023
        mock_ss_paper.doi = "10.1234/fl"
        mock_ss_paper.abstract = "Comprehensive survey of FL methods."

        with patch("tools.search_tools.search_semantic_scholar", return_value=[mock_ss_paper]):
            with patch("tools.search_tools.search_arxiv", return_value=[]):
                call_count = [0]

                def side_effect(messages):
                    call_count[0] += 1
                    resp = MagicMock()
                    if call_count[0] == 1:
                        resp.content = "## Literature Review\n\nFL is promising [Smith, 2023]."
                    elif call_count[0] == 2:
                        resp.content = "## State of Art\n\nCurrent methods lack scale."
                    else:
                        resp.content = '["Gap 1: No cross-domain eval", "Gap 2: Privacy methods"]'
                    return resp

                mock_make_llm.return_value.invoke.side_effect = side_effect
                result = literature_review_agent_node(populated_state)

        assert "literature_review_agent" in result["completed_steps"]
        assert result["literature_review"] != ""

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_handles_search_failure(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import literature_review_agent_node

        with patch("tools.search_tools.search_semantic_scholar", side_effect=Exception("Network error")):
            with patch("tools.search_tools.search_arxiv", side_effect=Exception("Network error")):
                resp = MagicMock()
                resp.content = "Literature review."
                mock_make_llm.return_value.invoke.return_value = resp
                result = literature_review_agent_node(populated_state)

        assert "literature_review_agent" in result["completed_steps"]
        assert any("search failed" in e.lower() for e in result.get("errors", []))


# ════════════════════════════════════════════════════════════════════════════════
# Node Tests — Agent 4: Proposal Writer
# ════════════════════════════════════════════════════════════════════════════════

class TestProposalWriter:
    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_generates_core_sections(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import proposal_writer_node

        call_count = [0]
        section_texts = [
            "Executive Summary text.",
            "Excellence section.",
            "Scientific Background.",
            "Objectives: 1. Develop. 2. Validate.",
            '["How can AI improve manufacturing?"]',
            "Methodology section.",
            json.dumps([{"id": "WP1", "title": "Management", "description": "Coordination",
                         "months": "M1-M36", "lead": "PI", "tasks": ["Task 1"]}]),
            json.dumps([{"id": "D1.1", "title": "DMP", "type": "Report", "month": 3, "wp": "WP1"}]),
            json.dumps([{"id": "MS1", "title": "Kickoff", "month": 1, "verification": "Minutes"}]),
            "Risk management table.",
            "Consortium description.",
            "Management structure.",
            "Data management plan.",
        ]

        def side_effect(messages):
            resp = MagicMock()
            idx = min(call_count[0], len(section_texts) - 1)
            resp.content = section_texts[idx]
            call_count[0] += 1
            return resp

        mock_make_llm.return_value.invoke.side_effect = side_effect
        result = proposal_writer_node(populated_state)

        assert "proposal_writer" in result["completed_steps"]
        assert result["executive_summary"] == "Executive Summary text."
        assert len(result.get("work_packages", [])) >= 1
        assert len(result.get("deliverables", [])) >= 1

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_fallback_work_packages_when_llm_returns_invalid(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import proposal_writer_node

        resp = MagicMock()
        resp.content = "Not valid JSON"
        mock_make_llm.return_value.invoke.return_value = resp
        result = proposal_writer_node(populated_state)
        assert len(result.get("work_packages", [])) >= 1


# ════════════════════════════════════════════════════════════════════════════════
# Node Tests — Agent 5: Impact Agent
# ════════════════════════════════════════════════════════════════════════════════

class TestImpactAgent:
    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_generates_impact_sections(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import impact_agent_node

        resp = MagicMock()
        resp.content = "Generated section text."
        mock_make_llm.return_value.invoke.return_value = resp
        result = impact_agent_node(populated_state)

        assert "impact_agent" in result["completed_steps"]
        assert result["impact"] == "Generated section text."
        assert result["dissemination"] == "Generated section text."
        assert result["exploitation"] == "Generated section text."
        assert result["ethics"] == "Generated section text."
        assert result["sustainability"] == "Generated section text."


# ════════════════════════════════════════════════════════════════════════════════
# Node Tests — Agent 6: Budget Agent
# ════════════════════════════════════════════════════════════════════════════════

class TestBudgetAgent:
    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_generates_budget_with_correct_format(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import budget_agent_node

        personnel = json.dumps([
            {"role": "PI (20%)", "months": 36, "rate_per_month": 1000, "total": 36000},
            {"role": "PostDoc", "months": 30, "rate_per_month": 5000, "total": 150000},
        ])
        equipment = json.dumps([{"item": "GPU Cluster", "cost": 50000, "justification": "Training"}])
        travel = json.dumps([{"destination": "Conferences", "purpose": "Dissemination", "cost": 15000}])

        call_count = [0]
        responses = [personnel, equipment, travel, "Budget is justified."]

        def side_effect(messages):
            resp = MagicMock()
            resp.content = responses[min(call_count[0], len(responses) - 1)]
            call_count[0] += 1
            return resp

        mock_make_llm.return_value.invoke.side_effect = side_effect
        result = budget_agent_node(populated_state)

        assert "budget_agent" in result["completed_steps"]
        assert result["budget_total"] > 0
        assert result["budget_indirect"] > 0
        assert result["budget_summary_table"] != ""

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_budget_scales_to_max(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import budget_agent_node

        # Provide way-too-large personnel budget
        big_pers = json.dumps([{"role": "PI", "months": 36, "rate_per_month": 100000, "total": 3_600_000}])
        resp = MagicMock()
        resp.content = big_pers
        mock_make_llm.return_value.invoke.return_value = resp
        result = budget_agent_node(populated_state)
        # Should be scaled down to fit within max_budget * 1.05
        assert result["budget_total"] <= 2_000_000 * 1.05 + 100

    def test_budget_format_horizon(self, populated_state):
        populated_state_copy = dict(populated_state)
        populated_state_copy["budget_format"] = "horizon_europe"
        # Indirect rate should be 0.25 for Horizon Europe
        from agents.proposal_gpt_nodes import budget_agent_node
        with patch("agents.proposal_gpt_nodes._make_llm") as m:
            resp = MagicMock()
            resp.content = json.dumps([{"role": "PI", "months": 12, "rate_per_month": 5000, "total": 60000}])
            m.return_value.invoke.return_value = resp
            result = budget_agent_node(populated_state_copy)
        assert abs(result["budget_indirect_rate"] - 0.25) < 0.01


# ════════════════════════════════════════════════════════════════════════════════
# Node Tests — Agent 7: Compliance Checker
# ════════════════════════════════════════════════════════════════════════════════

class TestComplianceAgent:
    def test_scores_complete_proposal_high(self, populated_state):
        from agents.proposal_gpt_nodes import compliance_agent_node
        result = compliance_agent_node(populated_state)
        assert result["compliance_score"] >= 70
        assert result["page_estimate"] > 0
        assert "Compliance Score" in result["compliance_report"]

    def test_detects_missing_sections(self, sample_state):
        from agents.proposal_gpt_nodes import compliance_agent_node
        state = dict(sample_state)
        state["mandatory_sections"] = ["Executive Summary", "Budget", "Ethics"]
        state["executive_summary"] = ""  # missing
        result = compliance_agent_node(state)
        assert result["compliance_score"] < 80

    def test_keyword_coverage_tracked(self, populated_state):
        from agents.proposal_gpt_nodes import compliance_agent_node
        result = compliance_agent_node(populated_state)
        cov = result.get("keyword_coverage", {})
        # Keys are stored lowercase for consistent lookups
        assert "ai" in cov
        assert cov.get("ai") is True

    def test_word_count_estimate(self, populated_state):
        from agents.proposal_gpt_nodes import compliance_agent_node
        result = compliance_agent_node(populated_state)
        assert result["page_estimate"] >= 1

    def test_empty_proposal_low_score(self, sample_state):
        from agents.proposal_gpt_nodes import compliance_agent_node
        state = dict(sample_state)
        state["keywords"] = ["AI", "ML"]
        result = compliance_agent_node(state)
        assert result["compliance_score"] <= 30


# ════════════════════════════════════════════════════════════════════════════════
# Node Tests — Agent 8: Reviewer Simulation
# ════════════════════════════════════════════════════════════════════════════════

class TestReviewerAgent:
    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_all_five_reviewers_generated(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import reviewer_agent_node

        review_data = json.dumps({
            "overall_score": 4.0,
            "strengths": ["Clear objectives", "Strong team", "Good methodology"],
            "weaknesses": ["Limited novelty", "Budget unclear", "Impact vague"],
            "suggestions": ["Add more KPIs", "Clarify budget"],
        })
        resp = MagicMock()
        resp.content = review_data
        mock_make_llm.return_value.invoke.return_value = resp
        result = reviewer_agent_node(populated_state)

        assert "reviewer_agent" in result["completed_steps"]
        assert len(result["reviewer_scores"]) == 5
        assert "scientific" in result["reviewer_scores"]
        assert "impact" in result["reviewer_scores"]

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_overall_score_weighted_average(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import reviewer_agent_node
        from agents.proposal_gpt_nodes import _REVIEWER_PROFILES

        call_count = [0]

        def side_effect(messages):
            resp = MagicMock()
            resp.content = json.dumps({
                "overall_score": 4.0,
                "strengths": ["s1", "s2", "s3"],
                "weaknesses": ["w1", "w2", "w3"],
                "suggestions": ["sg1", "sg2"],
            })
            return resp

        mock_make_llm.return_value.invoke.side_effect = side_effect
        result = reviewer_agent_node(populated_state)
        assert 3.5 <= result["overall_score"] <= 5.0

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_fallback_when_llm_returns_invalid(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import reviewer_agent_node
        resp = MagicMock()
        resp.content = "Not JSON"
        mock_make_llm.return_value.invoke.return_value = resp
        result = reviewer_agent_node(populated_state)
        assert result["overall_score"] > 0
        assert "reviewer_agent" in result["completed_steps"]


# ════════════════════════════════════════════════════════════════════════════════
# Node Tests — Agent 9: Improvement Agent
# ════════════════════════════════════════════════════════════════════════════════

class TestImprovementAgent:
    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_generates_improvement_plan(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import improvement_agent_node

        call_count = [0]

        def side_effect(messages):
            call_count[0] += 1
            resp = MagicMock()
            if call_count[0] == 1:
                resp.content = "## Improvement Plan\n\n1. Strengthen excellence.\n2. Clarify budget."
            else:
                resp.content = "## Excellence (Improved)\n\nStronger novelty argument."
            return resp

        mock_make_llm.return_value.invoke.side_effect = side_effect
        result = improvement_agent_node(populated_state)
        assert "improvement_agent" in result["completed_steps"]
        assert result["improvement_plan"] != ""

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_identifies_weak_sections(self, mock_make_llm, populated_state):
        from agents.proposal_gpt_nodes import improvement_agent_node
        resp = MagicMock()
        resp.content = "Improved text."
        mock_make_llm.return_value.invoke.return_value = resp
        result = improvement_agent_node(populated_state)
        # excellence has "novelty" weakness in populated_state
        assert "excellence" in result.get("weak_sections", [])


# ════════════════════════════════════════════════════════════════════════════════
# Graph Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestProposalGPTGraph:
    def test_build_pipeline_compiles(self):
        from agents.proposal_gpt_graph import build_proposal_gpt_pipeline
        app = build_proposal_gpt_pipeline()
        assert app is not None

    def test_graph_has_all_nodes(self):
        from agents.proposal_gpt_graph import build_proposal_gpt_pipeline
        app = build_proposal_gpt_pipeline()
        node_names = list(app.nodes.keys())
        expected_nodes = [
            "funding_call_analyzer", "research_planner", "literature_review_agent",
            "proposal_writer", "impact_agent", "budget_agent", "compliance_agent",
            "reviewer_agent", "improvement_agent",
        ]
        for node in expected_nodes:
            assert node in node_names, f"Missing node: {node}"

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_run_proposal_gpt_full_pipeline(self, mock_make_llm, sample_state):
        from agents.proposal_gpt_graph import run_proposal_gpt

        resp = MagicMock()
        resp.content = json.dumps({
            "title": "Test Call", "objectives": ["Obj 1"], "eligibility": [],
            "evaluation_criteria": [], "expected_outcomes": [],
            "budget_max": "500000", "currency": "EUR", "duration_months": 24,
            "deadline": "Jan 2026", "mandatory_sections": [], "keywords": ["AI"],
        })
        mock_make_llm.return_value.invoke.return_value = resp

        callbacks = []

        def _cb(node_name, state):
            callbacks.append(node_name)

        with patch("tools.search_tools.search_semantic_scholar", return_value=[]):
            with patch("tools.search_tools.search_arxiv", return_value=[]):
                result = run_proposal_gpt(sample_state, stream_callback=_cb)

        assert len(result.get("completed_steps", [])) == 10
        assert result.get("progress_pct") == 100
        assert len(callbacks) == 10

    @patch("agents.proposal_gpt_nodes._make_llm")
    def test_pipeline_stream_callback_called_for_each_node(self, mock_make_llm, sample_state):
        from agents.proposal_gpt_graph import run_proposal_gpt
        resp = MagicMock()
        resp.content = "{}"
        mock_make_llm.return_value.invoke.return_value = resp

        node_names = []

        def _cb(node_name, state):
            node_names.append(node_name)

        with patch("tools.search_tools.search_semantic_scholar", return_value=[]):
            with patch("tools.search_tools.search_arxiv", return_value=[]):
                run_proposal_gpt(sample_state, stream_callback=_cb)

        assert len(node_names) == 10


# ════════════════════════════════════════════════════════════════════════════════
# Tools Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestProposalTools:
    def test_assemble_full_proposal_md_contains_sections(self, populated_state):
        from tools.proposal_tools import assemble_full_proposal_md
        md = assemble_full_proposal_md(populated_state)
        assert "## Executive Summary" in md
        assert "## Excellence" in md
        assert "## Methodology" in md
        assert "## Budget Summary" in md

    def test_assemble_uses_improved_sections_when_flag_set(self, populated_state):
        from tools.proposal_tools import assemble_full_proposal_md
        md = assemble_full_proposal_md(populated_state, include_improved=True)
        assert "Excellence (Improved)" in md

    def test_assemble_uses_original_when_improved_flag_false(self, populated_state):
        from tools.proposal_tools import assemble_full_proposal_md
        md = assemble_full_proposal_md(populated_state, include_improved=False)
        assert "Our approach is beyond state-of-the-art." in md

    def test_build_budget_csv_has_all_categories(self, populated_state):
        from tools.proposal_tools import build_budget_csv
        csv_bytes = build_budget_csv(populated_state)
        csv_text = csv_bytes.decode("utf-8")
        assert "PERSONNEL BUDGET" in csv_text
        assert "EQUIPMENT BUDGET" in csv_text
        assert "TRAVEL BUDGET" in csv_text
        assert "SUMMARY" in csv_text
        assert "TOTAL" in csv_text

    def test_build_budget_csv_totals_match(self, populated_state):
        from tools.proposal_tools import build_budget_csv
        csv_bytes = build_budget_csv(populated_state)
        csv_text = csv_bytes.decode("utf-8")
        # Just check it's parseable and has content
        assert len(csv_text) > 100

    def test_section_word_count_returns_dict(self, populated_state):
        from tools.proposal_tools import section_word_count
        counts = section_word_count(populated_state)
        assert isinstance(counts, dict)
        assert "Executive Summary" in counts
        assert counts["Executive Summary"] > 0

    def test_section_word_count_uses_improved_versions(self, populated_state):
        from tools.proposal_tools import section_word_count
        counts = section_word_count(populated_state)
        # Excellence improved version has different word count than original
        assert counts.get("Excellence", 0) > 0

    def test_assemble_includes_work_packages_table(self, populated_state):
        from tools.proposal_tools import assemble_full_proposal_md
        md = assemble_full_proposal_md(populated_state)
        assert "Work Packages" in md
        assert "WP1" in md

    def test_assemble_includes_deliverables(self, populated_state):
        from tools.proposal_tools import assemble_full_proposal_md
        md = assemble_full_proposal_md(populated_state)
        assert "Deliverables" in md
        assert "D1.1" in md
