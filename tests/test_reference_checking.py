"""
tests/test_reference_checking.py
─────────────────────────────────────
Unit tests for the re-wired reference-checking features (Phase 0):

  - agents/risk_of_bias.py        — RoB 2 / ROBINS-I per-paper assessment
  - agents/grade_assessment.py    — GRADE certainty-of-evidence rating
  - agents/contradiction_detector.py — cross-paper contradiction detection
  - agents/systematic_review_nodes.py::quality_assessment_node — the pipeline
    node that runs all three and writes rob_table/grade_results/contradictions
  - agents/systematic_review_state.py — new state fields default correctly

ChatOllama is mocked throughout — no Ollama server or network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.contradiction_detector import detect_contradictions
from agents.grade_assessment import grade_evidence_body
from agents.risk_of_bias import assess_risk_of_bias, assess_rob_batch
from agents.systematic_review_nodes import quality_assessment_node
from agents.systematic_review_state import create_systematic_review_state


def _mock_llm(content: str):
    """Build a MagicMock standing in for ChatOllama whose .invoke().content is `content`."""
    llm = MagicMock()
    llm.invoke.return_value.content = content
    return llm


# ── state factory ────────────────────────────────────────────────────────────

def test_state_factory_defaults_reference_checking_fields():
    """The restored rob_table/grade_results/contradictions fields default empty, not missing."""
    state = create_systematic_review_state("Does X affect Y?")
    assert state["rob_table"] == []
    assert state["grade_results"] == {}
    assert state["contradictions"] == []
    # Phase 2 configurable caps are present with sensible defaults.
    assert state["max_evidence_papers"] == 25
    assert state["max_synthesis_papers"] == 20
    assert state["max_rob_papers"] == 15


# ── risk of bias ─────────────────────────────────────────────────────────────

def test_assess_risk_of_bias_selects_rob2_for_trials():
    """An RCT study design routes to RoB 2 and returns the parsed domain ratings."""
    paper = {"title": "A Trial", "study_design": "Randomised Controlled Trial",
             "citation_key": "smith2020", "key_finding": "x", "abstract": "y"}
    fake = _mock_llm('{"randomisation_process":"Low","overall":"Low","justification":"ok"}')
    with patch("agents.risk_of_bias.ChatOllama", return_value=fake):
        result = assess_risk_of_bias(paper, "llama3.1:8b", 4096)
    assert result["tool"] == "RoB 2"
    assert result["overall"] == "Low"
    assert result["citation_key"] == "smith2020"


def test_assess_risk_of_bias_selects_robins_for_observational():
    """A cohort study routes to ROBINS-I (not RoB 2)."""
    paper = {"title": "A Cohort", "study_design": "Prospective Cohort", "citation_key": "lee2019"}
    fake = _mock_llm('{"bias_due_to_confounding":"High","overall":"High","justification":"z"}')
    with patch("agents.risk_of_bias.ChatOllama", return_value=fake):
        result = assess_risk_of_bias(paper, "llama3.1:8b", 4096)
    assert result["tool"] == "ROBINS-I"


def test_assess_rob_batch_degrades_failures_without_aborting():
    """A per-paper failure yields a 'Some concerns' placeholder; the batch still completes."""
    papers = [
        {"title": "Good", "study_design": "RCT", "citation_key": "a"},
        {"title": "Bad", "study_design": "RCT", "citation_key": "b"},
    ]

    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("LLM blew up")
        return _mock_llm('{"overall":"Low","justification":"ok"}')

    with patch("agents.risk_of_bias.ChatOllama", side_effect=flaky):
        results = assess_rob_batch(papers, "llama3.1:8b", 4096)

    assert len(results) == 2
    assert results[0]["overall"] == "Low"
    assert results[1]["overall"] == "Some concerns"  # placeholder for the failed one


# ── GRADE ────────────────────────────────────────────────────────────────────

def test_grade_evidence_body_empty_table_short_circuits():
    """No evidence → empty dict, no LLM call."""
    assert grade_evidence_body([], "rq", [], "llama3.1:8b", 4096) == {}


def test_grade_evidence_body_starts_high_with_rcts():
    """An RCT in the table starts GRADE at High and returns the parsed rating."""
    evidence = [{"citation_key": "a", "study_design": "RCT", "quality": "High", "key_finding": "x"}]
    fake = _mock_llm('{"starting_level":"High","domains":{},"overall_grade":"Moderate",'
                     '"summary":"s","certainty_statement":"Based on 1 study..."}')
    with patch("agents.grade_assessment.ChatOllama", return_value=fake):
        result = grade_evidence_body(evidence, "rq", [], "llama3.1:8b", 4096)
    assert result["overall_grade"] == "Moderate"
    assert result["starting_level"] == "High"


# ── contradictions ───────────────────────────────────────────────────────────

def test_detect_contradictions_needs_two_papers():
    """Fewer than two papers can't contradict — returns [] with no LLM call."""
    assert detect_contradictions([{"citation_key": "a"}], "rq", "llama3.1:8b", 4096) == []


def test_detect_contradictions_parses_json_array():
    """A well-formed JSON array of contradictions is returned as-is."""
    evidence = [
        {"citation_key": "a", "title": "A", "key_finding": "X works"},
        {"citation_key": "b", "title": "B", "key_finding": "X fails"},
    ]
    fake = _mock_llm('[{"claim":"Does X work?","position_a":{"description":"yes","papers":["a"]},'
                     '"position_b":{"description":"no","papers":["b"]},"consensus_score":20,'
                     '"explanation":"they disagree"}]')
    with patch("agents.contradiction_detector.ChatOllama", return_value=fake):
        result = detect_contradictions(evidence, "rq", "llama3.1:8b", 4096)
    assert len(result) == 1
    assert result[0]["consensus_score"] == 20


# ── quality_assessment_node (pipeline integration) ───────────────────────────

def test_quality_assessment_node_empty_evidence_is_safe_noop():
    """With no evidence_table the node writes empty results and never calls the LLM."""
    state = {"evidence_table": [], "research_question": "rq", "model_name": "m", "num_ctx": 4096}
    out = quality_assessment_node(state)
    assert out["rob_table"] == []
    assert out["grade_results"] == {}
    assert out["contradictions"] == []
    assert out["progress_pct"] == 80


def test_quality_assessment_node_aggregates_three_assessments():
    """The node calls all three assessors and surfaces their results into state."""
    state = {
        "evidence_table": [{"citation_key": "a", "study_design": "RCT", "key_finding": "x"}],
        "research_question": "rq", "model_name": "m", "num_ctx": 4096,
    }
    with patch("agents.risk_of_bias.assess_rob_batch", return_value=[{"citation_key": "a", "overall": "Low"}]), \
         patch("agents.grade_assessment.grade_evidence_body", return_value={"overall_grade": "High"}), \
         patch("agents.contradiction_detector.detect_contradictions", return_value=[{"claim": "c"}]):
        out = quality_assessment_node(state)
    assert out["rob_table"] == [{"citation_key": "a", "overall": "Low"}]
    assert out["grade_results"] == {"overall_grade": "High"}
    assert out["contradictions"] == [{"claim": "c"}]


def test_quality_assessment_node_one_failure_does_not_sink_others():
    """If GRADE blows up, RoB and contradictions still come through (safe no-op per assessment)."""
    state = {
        "evidence_table": [{"citation_key": "a", "study_design": "RCT", "key_finding": "x"}],
        "research_question": "rq", "model_name": "m", "num_ctx": 4096,
    }
    with patch("agents.risk_of_bias.assess_rob_batch", return_value=[{"citation_key": "a", "overall": "High"}]), \
         patch("agents.grade_assessment.grade_evidence_body", side_effect=RuntimeError("boom")), \
         patch("agents.contradiction_detector.detect_contradictions", return_value=[]):
        out = quality_assessment_node(state)
    assert out["rob_table"] == [{"citation_key": "a", "overall": "High"}]
    assert out["grade_results"] == {}  # failed → empty, not raised
