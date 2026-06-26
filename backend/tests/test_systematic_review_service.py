"""backend/tests/test_systematic_review_service.py
───────────────────────────────────────────────────────
Unit tests for backend/app/services/systematic_review_service.py.

Mocks every tool/agent function at its *defining* module (e.g.
``tools.citation_network.build_citation_network``) rather than on the
service module's own namespace -- the service deliberately imports each of
them lazily (``from tools.xxx import yyy`` inside the function body, not at
module top level, see the service module's docstring), so patching the
source module is what the lazy import actually picks up at call time.

``run_sr`` is the one exception: ``run_systematic_review`` and
``create_systematic_review_state`` are imported eagerly at the top of the
service module, so those are patched on the service module's own namespace
-- the same pattern as ``test_research_assistant_service.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.schemas.systematic_review import (
    ExportRequest,
    GrammarCheckRequest,
    MetaAnalysisPoolRequest,
    MetaAnalysisRow,
    SRRequest,
)
from backend.app.services import systematic_review_service as svc

# ─────────────────────────────────────────────────────────────────────────────
# build_initial_state
# ─────────────────────────────────────────────────────────────────────────────


def test_build_initial_state_omits_optional_overrides_when_unset():
    req = SRRequest(
        research_question="Does X help Y?",
        inclusion_criteria=["Peer-reviewed"],
        exclusion_criteria=["Animal studies"],
    )
    state = svc.build_initial_state(req)
    assert state["research_question"] == "Does X help Y?"
    assert state["inclusion_criteria"] == ["Peer-reviewed"]
    assert state["exclusion_criteria"] == ["Animal studies"]
    # Defers to create_systematic_review_state's own defaults.
    assert state["model_name"] == "llama3.1:8b"
    assert state["num_ctx"] == 32768
    assert state["max_results"] == 8
    assert state["include_crossref"] is True


def test_build_initial_state_includes_overrides_when_provided():
    req = SRRequest(
        research_question="  Does X help Y?  ",
        model="mistral:7b",
        num_ctx=4096,
        max_results=3,
        include_crossref=False,
    )
    state = svc.build_initial_state(req)
    assert state["research_question"] == "Does X help Y?"  # stripped
    assert state["model_name"] == "mistral:7b"
    assert state["num_ctx"] == 4096
    assert state["max_results"] == 3
    assert state["include_crossref"] is False


def test_build_initial_state_defaults_criteria_to_empty_lists():
    req = SRRequest(research_question="q")
    state = svc.build_initial_state(req)
    assert state["inclusion_criteria"] == []
    assert state["exclusion_criteria"] == []


# ─────────────────────────────────────────────────────────────────────────────
# run_sr
# ─────────────────────────────────────────────────────────────────────────────


def test_run_sr_adapts_stream_callback_and_returns_final_state():
    req = SRRequest(research_question="q")
    fake_final_state = {"progress_pct": 100, "status_detail": "done", "evidence_table": []}

    def fake_run(initial_state, stream_callback):
        stream_callback("evidence_extraction", {"progress_pct": 42, "status_detail": "3/7 papers"})
        return fake_final_state

    cb = MagicMock()
    with patch.object(svc, "run_systematic_review", side_effect=fake_run):
        result = svc.run_sr(req, cb)

    assert result == fake_final_state
    cb.assert_called_once_with(
        "evidence_extraction",
        {
            "label": "Extracting evidence from papers",
            "progress_pct": 42,
            "status_detail": "3/7 papers",
        },
    )


def test_run_sr_unlabeled_node_falls_back_to_raw_node_name():
    req = SRRequest(research_question="q")

    def fake_run(initial_state, stream_callback):
        stream_callback("some_future_node", {})
        return {}

    cb = MagicMock()
    with patch.object(svc, "run_systematic_review", side_effect=fake_run):
        svc.run_sr(req, cb)

    cb.assert_called_once_with(
        "some_future_node", {"label": "some_future_node", "progress_pct": 0, "status_detail": ""}
    )


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_model
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_model_prefers_explicit_options_override():
    model, num_ctx = svc._resolve_model(
        {"model_name": "from-state", "num_ctx": 1000}, {"model": "from-options", "num_ctx": 2000}
    )
    assert (model, num_ctx) == ("from-options", 2000)


def test_resolve_model_falls_back_to_final_state_then_hardcoded_defaults():
    model, num_ctx = svc._resolve_model({"model_name": "from-state", "num_ctx": 1000}, {})
    assert (model, num_ctx) == ("from-state", 1000)

    model, num_ctx = svc._resolve_model({}, {})
    assert (model, num_ctx) == ("llama3.1:8b", 32768)


# ─────────────────────────────────────────────────────────────────────────────
# Explore tools
# ─────────────────────────────────────────────────────────────────────────────


def test_explore_citation_network_with_no_included_papers_returns_error_dict():
    cb = MagicMock()
    result = svc.run_explore_tool("citation_network", {"included_papers": []}, {}, cb)
    assert result["html"] == ""
    assert "error" in result


def test_explore_citation_network_reuses_existing_html_without_rebuilding():
    final_state = {"included_papers": [{"title": "p1"}], "citation_graph_html": "<div>cached</div>"}
    with patch("tools.citation_network.build_citation_network") as mock_build:
        result = svc.run_explore_tool("citation_network", final_state, {}, MagicMock())
    mock_build.assert_not_called()
    assert result["html"] == "<div>cached</div>"


def test_explore_citation_network_builds_fresh_and_classifies_stances_when_requested():
    final_state = {"included_papers": [{"title": "p1"}, {"title": "p2"}], "model_name": "m", "num_ctx": 999}
    fake_graph = MagicMock()
    fake_graph.number_of_edges.return_value = 1
    with patch("tools.citation_network.build_citation_network", return_value=(fake_graph, {"meta": 1}, {})) as mock_build, \
         patch("tools.citation_network.network_to_pyvis_html", return_value="<html>net</html>") as mock_html, \
         patch("tools.citation_network.network_stats", return_value={"nodes": 2}) as mock_stats, \
         patch("tools.citation_network.find_gap_candidates", return_value=[]) as mock_gaps, \
         patch("tools.citation_network.classify_citation_stances", return_value={"Supporting": 1}) as mock_stances:
        result = svc.run_explore_tool("citation_network", final_state, {"classify_stances": True}, MagicMock())

    mock_build.assert_called_once_with([{"title": "p1"}, {"title": "p2"}], max_papers=25)
    mock_stances.assert_called_once_with(fake_graph, {"meta": 1}, "m", 999)
    assert result == {
        "html": "<html>net</html>",
        "stats": {"nodes": 2},
        "gap_candidates": [],
        "stance_counts": {"Supporting": 1},
    }
    mock_html.assert_called_once()
    mock_stats.assert_called_once()
    mock_gaps.assert_called_once()


def test_explore_citation_context_requires_both_indices():
    with pytest.raises(ValueError, match="citing_idx"):
        svc.run_explore_tool("citation_context", {"included_papers": [{}, {}]}, {"citing_idx": 0}, MagicMock())


def test_explore_citation_context_rejects_out_of_range_indices():
    with pytest.raises(ValueError, match="out of range"):
        svc.run_explore_tool(
            "citation_context", {"included_papers": [{}]}, {"citing_idx": 0, "cited_idx": 5}, MagicMock()
        )


def test_explore_citation_context_rejects_identical_indices():
    with pytest.raises(ValueError, match="must differ"):
        svc.run_explore_tool(
            "citation_context", {"included_papers": [{}, {}]}, {"citing_idx": 1, "cited_idx": 1}, MagicMock()
        )


def test_explore_citation_context_happy_path():
    papers = [{"title": "citing"}, {"title": "cited"}]
    with patch("tools.citation_context.extract_citation_context", return_value={"snippet": "..."}) as mock_extract:
        result = svc.run_explore_tool(
            "citation_context", {"included_papers": papers}, {"citing_idx": 0, "cited_idx": 1}, MagicMock()
        )
    mock_extract.assert_called_once_with(papers[0], papers[1])
    assert result == {"snippet": "..."}


def test_explore_reference_checking_with_no_evidence_table_returns_none_source():
    result = svc.run_explore_tool("reference_checking", {"evidence_table": []}, {}, MagicMock())
    assert result["source"] == "none"


def test_explore_reference_checking_reuses_existing_results():
    final_state = {
        "evidence_table": [{"citation_key": "a"}],
        "rob_table": [{"citation_key": "a", "overall": "Low"}],
        "grade_results": {"overall_grade": "High"},
        "contradictions": [],
    }
    with patch("agents.risk_of_bias.assess_rob_batch") as mock_rob:
        result = svc.run_explore_tool("reference_checking", final_state, {}, MagicMock())
    mock_rob.assert_not_called()
    assert result["source"] == "existing"
    assert result["rob_table"] == [{"citation_key": "a", "overall": "Low"}]


def test_explore_reference_checking_recomputes_when_nothing_cached():
    final_state = {"evidence_table": [{"citation_key": "a"}], "research_question": "q"}
    with patch("agents.risk_of_bias.assess_rob_batch", return_value=[{"citation_key": "a"}]) as mock_rob, \
         patch("agents.grade_assessment.grade_evidence_body", return_value={"overall_grade": "Moderate"}) as mock_grade, \
         patch("agents.contradiction_detector.detect_contradictions", return_value=[]) as mock_contra:
        result = svc.run_explore_tool("reference_checking", final_state, {}, MagicMock())

    mock_rob.assert_called_once()
    mock_grade.assert_called_once()
    mock_contra.assert_called_once()
    assert result["source"] == "recomputed"
    assert result["grade_results"] == {"overall_grade": "Moderate"}


def test_explore_preprint_status_with_no_included_papers_returns_empty():
    result = svc.run_explore_tool("preprint_status", {"included_papers": []}, {}, MagicMock())
    assert result == {"results": [], "summary": {}}


def test_explore_preprint_status_reuses_existing_tracking():
    final_state = {"included_papers": [{"title": "p1"}], "preprint_tracking": [{"status": "published"}]}
    with patch("tools.preprint_tracker.track_preprints") as mock_track, \
         patch("tools.preprint_tracker.preprint_summary", return_value={"published": 1}) as mock_summary:
        result = svc.run_explore_tool("preprint_status", final_state, {}, MagicMock())
    mock_track.assert_not_called()
    mock_summary.assert_called_once_with([{"status": "published"}])
    assert result == {"results": [{"status": "published"}], "summary": {"published": 1}}


def test_explore_preprint_status_recomputes_when_not_tracked_yet():
    final_state = {"included_papers": [{"title": "p1"}]}
    with patch("tools.preprint_tracker.track_preprints", return_value=[{"status": "preprint"}]) as mock_track, \
         patch("tools.preprint_tracker.preprint_summary", return_value={"preprint": 1}):
        result = svc.run_explore_tool("preprint_status", final_state, {}, MagicMock())
    mock_track.assert_called_once_with([{"title": "p1"}])
    assert result["results"] == [{"status": "preprint"}]


def test_explore_research_trends_builds_chart_data_and_html():
    final_state = {"research_question": "q", "search_queries": ["q1"], "included_papers": [{"title": "p"}]}
    fig = MagicMock()
    fig.to_html.return_value = "<div>chart</div>"
    with patch("tools.trend_analyzer.analyze_trends", return_value={"trend": 1}) as mock_analyze, \
         patch("tools.trend_analyzer.trend_to_chart_data", return_value={"chart": 1}), \
         patch("tools.trend_analyzer.build_trend_figure", return_value=fig):
        result = svc.run_explore_tool("research_trends", final_state, {}, MagicMock())

    mock_analyze.assert_called_once_with(
        research_question="q", search_queries=["q1"], corpus_papers=[{"title": "p"}]
    )
    assert result["trend_data"] == {"trend": 1}
    assert result["chart_data"] == {"chart": 1}
    assert result["html"] == "<div>chart</div>"


def test_explore_research_trends_html_is_none_when_plotly_missing():
    """build_trend_figure raises ImportError when plotly isn't installed -- this
    sandbox doesn't have it, so this exercises the real (unmocked) ImportError path."""
    final_state = {"research_question": "q", "search_queries": [], "included_papers": []}
    with patch("tools.trend_analyzer.analyze_trends", return_value={}), \
         patch("tools.trend_analyzer.trend_to_chart_data", return_value={}):
        result = svc.run_explore_tool("research_trends", final_state, {}, MagicMock())
    assert result["html"] is None


def test_explore_concept_drift_with_no_papers_returns_empty_defaults():
    result = svc.run_explore_tool("concept_drift", {"raw_papers": []}, {}, MagicMock())
    assert result == {
        "buckets": {}, "rising_terms": [], "declining_terms": [], "stable_terms": [], "llm_analysis": "",
    }


def test_explore_concept_drift_happy_path_resolves_model():
    final_state = {"raw_papers": [{"title": "p"}], "model_name": "m", "num_ctx": 555}
    with patch("tools.concept_drift.detect_concept_drift", return_value={"rising_terms": ["ai"]}) as mock_drift:
        result = svc.run_explore_tool("concept_drift", final_state, {}, MagicMock())
    mock_drift.assert_called_once_with(papers=[{"title": "p"}], model_name="m", num_ctx=555)
    assert result == {"rising_terms": ["ai"]}


def test_explore_evidence_map_delegates_to_build_evidence_map():
    final_state = {"evidence_table": [{"study_design": "RCT"}]}
    with patch.object(svc, "build_evidence_map", wraps=svc.build_evidence_map) as mock_build:
        svc.run_explore_tool("evidence_map", final_state, {}, MagicMock())
    mock_build.assert_called_once_with([{"study_design": "RCT"}])


def test_explore_meta_analysis_returns_seeded_rows():
    final_state = {
        "evidence_table": [{"citation_key": "a1", "authors": ["Smith, J."], "year": 2021, "sample_size": "N=50"}]
    }
    result = svc.run_explore_tool("meta_analysis", final_state, {}, MagicMock())
    assert result["rows"][0]["citation_key"] == "a1"
    assert result["rows"][0]["n"] == 50


def test_run_explore_tool_unknown_tool_raises():
    with pytest.raises(ValueError, match="Unknown explore tool"):
        svc.run_explore_tool("not_a_real_tool", {}, {}, MagicMock())


def test_run_explore_tool_reports_progress_via_stream_callback():
    cb = MagicMock()
    svc.run_explore_tool("evidence_map", {"evidence_table": []}, {}, cb)
    assert cb.call_count == 2
    assert cb.call_args_list[0].args[1]["progress_pct"] == 0
    assert cb.call_args_list[1].args[1]["progress_pct"] == 100


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Map
# ─────────────────────────────────────────────────────────────────────────────


def test_build_evidence_map_empty_table_returns_no_html():
    result = svc.build_evidence_map([])
    assert result["map_data"]["total_studies"] == 0
    assert result["html"] is None


def test_build_evidence_map_nonempty_table_builds_html():
    evidence_table = [{"study_design": "RCT", "quality": "High", "population": "Adults"}]
    with patch("tools.evidence_map.build_evidence_map_data", return_value={"total_studies": 1}), \
         patch("tools.evidence_map.evidence_map_to_plotly_html", return_value="<div>map</div>") as mock_html:
        result = svc.build_evidence_map(evidence_table)
    mock_html.assert_called_once_with({"total_studies": 1})
    assert result == {"map_data": {"total_studies": 1}, "html": "<div>map</div>"}


def test_build_evidence_map_html_none_when_plotly_missing():
    """Real (unmocked) ImportError path -- plotly isn't installed in this sandbox."""
    result = svc.build_evidence_map([{"study_design": "RCT", "quality": "High", "population": "Adults"}])
    assert result["html"] is None
    assert result["map_data"]["total_studies"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Analysis
# ─────────────────────────────────────────────────────────────────────────────


def test_seed_meta_analysis_builds_rows_from_evidence_table():
    evidence_table = [{"citation_key": "a1", "authors": ["Smith, J."], "year": 2021, "sample_size": "N=50"}]
    rows = svc.seed_meta_analysis(evidence_table)
    assert rows == [
        {"citation_key": "a1", "label": "Smith (2021)", "effect": None, "ci_low": None, "ci_high": None, "n": 50}
    ]


def test_get_measure_labels_returns_known_measures():
    labels = svc.get_measure_labels()
    assert "OR" in labels
    assert isinstance(labels, dict)


def test_draft_meta_analysis_rows_merges_llm_drafts_and_reports_progress():
    final_state = {
        "included_papers": [{"citation_key": "a1", "abstract": "..."}],
        "evidence_table": [],
        "model_name": "m",
        "num_ctx": 111,
    }
    rows = [{"citation_key": "a1", "label": "A (2021)", "effect": None, "ci_low": None, "ci_high": None, "n": None}]
    cb = MagicMock()
    with patch(
        "tools.meta_analysis.extract_effect_size_row",
        return_value={"found": True, "effect": 1.5, "ci_low": 1.1, "ci_high": 2.0, "n": 50},
    ) as mock_extract:
        result = svc.draft_meta_analysis_rows(final_state, rows, "OR", {}, cb)

    mock_extract.assert_called_once_with({"citation_key": "a1", "abstract": "..."}, "OR", "m", 111)
    assert result[0]["effect"] == 1.5
    assert result[0]["n"] == 50
    cb.assert_called_once_with("meta_analysis_draft", {"label": "Drafting effect sizes", "progress_pct": 100})


def test_draft_meta_analysis_rows_skips_rows_with_no_matching_paper():
    final_state = {"included_papers": [], "evidence_table": []}
    rows = [{"citation_key": "missing", "effect": None, "ci_low": None, "ci_high": None, "n": None}]
    with patch("tools.meta_analysis.extract_effect_size_row") as mock_extract:
        result = svc.draft_meta_analysis_rows(final_state, rows, "OR", {}, MagicMock())
    mock_extract.assert_not_called()
    assert result[0]["effect"] is None


def test_draft_meta_analysis_rows_does_not_mutate_input_rows():
    final_state = {"included_papers": [{"citation_key": "a1"}], "evidence_table": []}
    original_rows = [{"citation_key": "a1", "effect": None, "ci_low": None, "ci_high": None, "n": None}]
    with patch(
        "tools.meta_analysis.extract_effect_size_row",
        return_value={"found": True, "effect": 9.9, "ci_low": 1, "ci_high": 2, "n": 10},
    ):
        svc.draft_meta_analysis_rows(final_state, original_rows, "OR", {}, MagicMock())
    assert original_rows[0]["effect"] is None


def test_pool_meta_analysis_runs_real_math_and_skips_forest_html_without_plotly():
    req = MetaAnalysisPoolRequest(
        rows=[
            MetaAnalysisRow(citation_key="a", label="A", effect=1.5, ci_low=1.1, ci_high=2.0, n=50),
            MetaAnalysisRow(citation_key="b", label="B", effect=1.2, ci_low=0.8, ci_high=1.8, n=30),
        ],
        measure="OR",
        model="random",
    )
    result = svc.pool_meta_analysis(req)
    assert result["result"]["ok"] is True
    assert result["forest_html"] is None  # plotly not installed in this sandbox


def test_pool_meta_analysis_builds_forest_html_when_available():
    req = MetaAnalysisPoolRequest(
        rows=[MetaAnalysisRow(citation_key="a", label="A", effect=1.5, ci_low=1.1, ci_high=2.0, n=50)],
        measure="OR",
    )
    with patch("tools.meta_analysis.run_meta_analysis", return_value={"ok": True, "pooled_effect": 1.5}), \
         patch("tools.meta_analysis.meta_analysis_to_forest_plotly", return_value="<div>forest</div>"):
        result = svc.pool_meta_analysis(req)
    assert result["forest_html"] == "<div>forest</div>"


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────


def test_build_markdown_includes_research_question_and_flow():
    final_state = {"prisma_flow": {"identified": 5, "included": 2}, "evidence_table": []}
    md = svc.build_markdown("Does X help Y?", final_state)
    assert "Does X help Y?" in md
    assert "PRISMA Flow" in md


def test_build_docx_forwards_author_and_institution():
    req = ExportRequest(author="A. Researcher", institution="Big University")
    with patch("tools.prisma_report.generate_prisma_docx", return_value=b"DOCX") as mock_gen:
        result = svc.build_docx({"foo": "bar"}, req)
    mock_gen.assert_called_once_with({"foo": "bar"}, author="A. Researcher", institution="Big University")
    assert result == b"DOCX"


def test_build_pdf_forwards_author_and_institution():
    req = ExportRequest(author="A. Researcher", institution="Big University")
    with patch("tools.prisma_report.generate_prisma_pdf", return_value=b"PDF") as mock_gen:
        result = svc.build_pdf({"foo": "bar"}, req)
    mock_gen.assert_called_once_with({"foo": "bar"}, author="A. Researcher", institution="Big University")
    assert result == b"PDF"


# ─────────────────────────────────────────────────────────────────────────────
# Plain-language summaries
# ─────────────────────────────────────────────────────────────────────────────


def test_generate_plain_language_summary_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unknown summary format"):
        svc.generate_plain_language_summary({}, "not_a_format", None, None)


@pytest.mark.parametrize(
    "fmt,tool_fn,expected_key",
    [
        ("patient", "generate_patient_summary", "patient"),
        ("policy", "generate_policy_brief", "policy"),
        ("press", "generate_press_release", "press"),
    ],
)
def test_generate_plain_language_summary_single_format(fmt, tool_fn, expected_key):
    final_state = {"model_name": "m", "num_ctx": 123}
    with patch(f"tools.plain_language.{tool_fn}", return_value="summary text") as mock_fn:
        result = svc.generate_plain_language_summary(final_state, fmt, None, None)
    mock_fn.assert_called_once_with(final_state, "m", 123)
    assert result == {expected_key: "summary text"}


def test_generate_plain_language_summary_all_formats():
    final_state = {}
    with patch(
        "tools.plain_language.generate_all_summaries",
        return_value={"patient": "p", "policy": "po", "press": "pr"},
    ) as mock_all:
        result = svc.generate_plain_language_summary(final_state, "all", "override-model", 999)
    mock_all.assert_called_once_with(final_state, "override-model", 999)
    assert result == {"patient": "p", "policy": "po", "press": "pr"}


# ─────────────────────────────────────────────────────────────────────────────
# Guided templates + grammar check
# ─────────────────────────────────────────────────────────────────────────────


def test_list_templates_returns_all_four_presets():
    templates = svc.list_templates()
    assert {t["key"] for t in templates} == {
        "clinical_rct", "cs_survey", "qual_synthesis", "scoping_review",
    }


def test_check_grammar_forwards_request_fields_with_default_num_ctx():
    req = GrammarCheckRequest(text="teh quick fox", context_hint="research question")
    with patch(
        "tools.grammar_check.check_and_fix_grammar",
        return_value={"original": "teh quick fox", "corrected": "the quick fox", "changed": True},
    ) as mock_check:
        result = svc.check_grammar(req)
    mock_check.assert_called_once_with(
        "teh quick fox", model_name="", num_ctx=8192, context_hint="research question"
    )
    assert result["changed"] is True


def test_check_grammar_forwards_explicit_model_and_num_ctx_overrides():
    req = GrammarCheckRequest(text="q", model="mistral:7b", num_ctx=2048)
    with patch("tools.grammar_check.check_and_fix_grammar", return_value={}) as mock_check:
        svc.check_grammar(req)
    mock_check.assert_called_once_with("q", model_name="mistral:7b", num_ctx=2048, context_hint="")
