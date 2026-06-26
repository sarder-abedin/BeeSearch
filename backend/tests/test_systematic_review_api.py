"""backend/tests/test_systematic_review_api.py
───────────────────────────────────────────────────
API-level tests for the Mode 1 (Systematic Literature Review) HTTP endpoints.

The main pipeline run is exercised end-to-end through the real
router → service → job-runner → polling stack (mocking only
``run_systematic_review`` at the service module's own namespace, the same
boundary ``test_research_assistant_api.py`` uses for Mode 3's pipeline
entry point).

Every other endpoint operates on an *already-completed* review, so rather
than re-running the full mocked pipeline for each one, these tests fabricate
a finished/errored/running ``Job`` directly via ``backend.app.jobs`` --
exactly mirroring how the Streamlit tab and the router itself treat
``final_state`` as a plain dict once a review is done. Tool-level mocks
(citation network, meta-analysis, plain-language, prisma_report, …) are
patched at their *defining* module, matching the service's lazy
function-local imports -- see ``test_systematic_review_service.py``.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import jobs as jobs_module
from backend.app.main import app

_BASE = "/api/systematic-review"


def _poll_until_terminal(client: TestClient, path: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    data: dict = {}
    while time.monotonic() < deadline:
        r = client.get(path)
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"{path} did not reach a terminal state: {data}")


def _finished_job(result: dict) -> str:
    job = jobs_module.create_job()
    job.status = "done"
    job.result = result
    return job.id


def _errored_job(error: str) -> str:
    job = jobs_module.create_job()
    job.status = "error"
    job.error = error
    return job.id


def _running_job() -> str:
    job = jobs_module.create_job()
    job.status = "running"
    return job.id


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline run
# ─────────────────────────────────────────────────────────────────────────────

def test_run_blank_research_question_returns_422(client: TestClient):
    r = client.post(f"{_BASE}/run", json={"research_question": "   "})
    assert r.status_code == 422


def test_run_negative_max_results_returns_422(client: TestClient):
    r = client.post(f"{_BASE}/run", json={"research_question": "q", "max_results": 0})
    assert r.status_code == 422


def test_get_unknown_job_returns_404(client: TestClient):
    r = client.get(f"{_BASE}/jobs/does-not-exist")
    assert r.status_code == 404


def test_run_round_trip_returns_done_with_result(client: TestClient):
    fake_final_state = {
        "research_question": "Does X help Y?",
        "prisma_flow": {"identified": 5, "screened": 5, "eligibility": 3, "included": 2, "excluded": 3},
        "evidence_table": [],
        "narrative_synthesis": "X helps Y.",
        "conclusion": "X helps Y.",
        "progress_pct": 100,
    }

    def fake_run(initial_state, stream_callback=None):
        if stream_callback:
            stream_callback("synthesis", {**fake_final_state, "progress_pct": 90})
        return fake_final_state

    with patch(
        "backend.app.services.systematic_review_service.run_systematic_review",
        side_effect=fake_run,
    ):
        r = client.post(
            f"{_BASE}/run",
            json={"research_question": "Does X help Y?", "inclusion_criteria": ["Peer-reviewed"]},
        )
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        data = _poll_until_terminal(client, f"{_BASE}/jobs/{job_id}")

    assert data["status"] == "done"
    assert data["error"] is None
    assert data["result"]["narrative_synthesis"] == "X helps Y."
    assert data["result"]["prisma_flow"]["included"] == 2


def test_run_forwards_optional_overrides_to_initial_state(client: TestClient):
    captured: dict = {}

    def fake_run(initial_state, stream_callback=None):
        captured.update(initial_state)
        return dict(initial_state)

    with patch(
        "backend.app.services.systematic_review_service.run_systematic_review",
        side_effect=fake_run,
    ):
        r = client.post(
            f"{_BASE}/run",
            json={
                "research_question": "q",
                "model": "mistral:7b",
                "num_ctx": 4096,
                "max_results": 3,
                "include_crossref": False,
            },
        )
        _poll_until_terminal(client, f"{_BASE}/jobs/{r.json()['job_id']}")

    assert captured["model_name"] == "mistral:7b"
    assert captured["num_ctx"] == 4096
    assert captured["max_results"] == 3
    assert captured["include_crossref"] is False


def test_run_pipeline_exception_surfaces_as_job_error(client: TestClient):
    with patch(
        "backend.app.services.systematic_review_service.run_systematic_review",
        side_effect=RuntimeError("search backend exploded"),
    ):
        r = client.post(f"{_BASE}/run", json={"research_question": "q"})
        data = _poll_until_terminal(client, f"{_BASE}/jobs/{r.json()['job_id']}")

    assert data["status"] == "error"
    assert "search backend exploded" in data["error"]
    assert data["result"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Explore tools (generic background-job dispatch)
# ─────────────────────────────────────────────────────────────────────────────

def test_trigger_explore_tool_unknown_job_returns_404(client: TestClient):
    r = client.post(f"{_BASE}/jobs/does-not-exist/explore/evidence_map", json={})
    assert r.status_code == 404


def test_trigger_explore_tool_errored_job_returns_409_with_message(client: TestClient):
    job_id = _errored_job("boom")
    r = client.post(f"{_BASE}/jobs/{job_id}/explore/evidence_map", json={})
    assert r.status_code == 409
    assert "boom" in r.json()["detail"]


def test_trigger_explore_tool_unfinished_job_returns_409(client: TestClient):
    job_id = _running_job()
    r = client.post(f"{_BASE}/jobs/{job_id}/explore/evidence_map", json={})
    assert r.status_code == 409


def test_trigger_explore_tool_invalid_tool_name_returns_422(client: TestClient):
    job_id = _finished_job({})
    r = client.post(f"{_BASE}/jobs/{job_id}/explore/not_a_real_tool", json={})
    assert r.status_code == 422


def test_get_tool_job_status_unknown_job_returns_404(client: TestClient):
    r = client.get(f"{_BASE}/tool-jobs/does-not-exist")
    assert r.status_code == 404


def test_explore_tool_round_trip_evidence_map(client: TestClient):
    job_id = _finished_job(
        {"evidence_table": [{"study_design": "RCT", "quality": "High", "population": "Adults"}]}
    )
    r = client.post(f"{_BASE}/jobs/{job_id}/explore/evidence_map", json={})
    assert r.status_code == 202
    data = _poll_until_terminal(client, f"{_BASE}/tool-jobs/{r.json()['job_id']}")
    assert data["status"] == "done"
    assert data["result"]["map_data"]["total_studies"] == 1


def test_explore_tool_round_trip_citation_network_no_papers_returns_error_in_result(client: TestClient):
    job_id = _finished_job({"included_papers": []})
    r = client.post(f"{_BASE}/jobs/{job_id}/explore/citation_network", json={})
    data = _poll_until_terminal(client, f"{_BASE}/tool-jobs/{r.json()['job_id']}")
    assert data["status"] == "done"
    assert "error" in data["result"]


def test_explore_tool_round_trip_meta_analysis_seed(client: TestClient):
    job_id = _finished_job(
        {"evidence_table": [{"citation_key": "a1", "authors": ["Smith, J."], "year": 2021, "sample_size": "N=50"}]}
    )
    r = client.post(f"{_BASE}/jobs/{job_id}/explore/meta_analysis", json={})
    data = _poll_until_terminal(client, f"{_BASE}/tool-jobs/{r.json()['job_id']}")
    assert data["status"] == "done"
    assert data["result"]["rows"][0]["citation_key"] == "a1"


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Map (sync)
# ─────────────────────────────────────────────────────────────────────────────

def test_get_evidence_map_unknown_job_404(client: TestClient):
    r = client.get(f"{_BASE}/jobs/does-not-exist/evidence-map")
    assert r.status_code == 404


def test_get_evidence_map_empty_table_returns_no_html(client: TestClient):
    job_id = _finished_job({"evidence_table": []})
    r = client.get(f"{_BASE}/jobs/{job_id}/evidence-map")
    assert r.status_code == 200
    data = r.json()
    assert data["map_data"]["total_studies"] == 0
    assert data["html"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Analysis: seed -> draft -> pool
# ─────────────────────────────────────────────────────────────────────────────

def test_seed_meta_analysis_unknown_job_404(client: TestClient):
    r = client.get(f"{_BASE}/jobs/does-not-exist/meta-analysis/seed")
    assert r.status_code == 404


def test_seed_meta_analysis_endpoint(client: TestClient):
    job_id = _finished_job(
        {"evidence_table": [{"citation_key": "a1", "authors": ["Smith, J."], "year": 2021, "sample_size": "N=50"}]}
    )
    r = client.get(f"{_BASE}/jobs/{job_id}/meta-analysis/seed")
    assert r.status_code == 200
    data = r.json()
    assert data["rows"][0]["citation_key"] == "a1"
    assert data["rows"][0]["n"] == 50
    assert "OR" in data["measure_labels"]


def test_draft_meta_analysis_round_trip(client: TestClient):
    job_id = _finished_job({"included_papers": [{"citation_key": "a1", "abstract": "..."}], "evidence_table": []})
    body = {
        "rows": [{"citation_key": "a1", "label": "A (2021)", "effect": None, "ci_low": None, "ci_high": None, "n": None}],
        "measure": "OR",
    }
    with patch(
        "tools.meta_analysis.extract_effect_size_row",
        return_value={"found": True, "effect": 1.5, "ci_low": 1.1, "ci_high": 2.0, "n": 50},
    ):
        r = client.post(f"{_BASE}/jobs/{job_id}/meta-analysis/draft", json=body)
        assert r.status_code == 202
        data = _poll_until_terminal(client, f"{_BASE}/tool-jobs/{r.json()['job_id']}")

    assert data["status"] == "done"
    assert data["result"]["rows"][0]["effect"] == 1.5
    assert data["result"]["rows"][0]["n"] == 50


def test_draft_meta_analysis_unknown_job_404(client: TestClient):
    r = client.post(f"{_BASE}/jobs/does-not-exist/meta-analysis/draft", json={"rows": []})
    assert r.status_code == 404


def test_pool_meta_analysis_endpoint_runs_real_math(client: TestClient):
    body = {
        "rows": [
            {"citation_key": "a", "label": "A", "effect": 1.5, "ci_low": 1.1, "ci_high": 2.0, "n": 50},
            {"citation_key": "b", "label": "B", "effect": 1.2, "ci_low": 0.8, "ci_high": 1.8, "n": 30},
        ],
        "measure": "OR",
        "model": "random",
    }
    r = client.post(f"{_BASE}/meta-analysis/pool", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["result"]["ok"] is True
    assert data["forest_html"] is None  # plotly not installed in this sandbox


# ─────────────────────────────────────────────────────────────────────────────
# Export: Markdown / DOCX / PDF
# ─────────────────────────────────────────────────────────────────────────────

def test_export_markdown_unknown_job_404(client: TestClient):
    r = client.get(f"{_BASE}/jobs/does-not-exist/export/markdown")
    assert r.status_code == 404


def test_export_markdown_happy_path(client: TestClient):
    job_id = _finished_job(
        {"research_question": "Does X help Y?", "prisma_flow": {"identified": 5, "included": 2}, "evidence_table": []}
    )
    r = client.get(f"{_BASE}/jobs/{job_id}/export/markdown")
    assert r.status_code == 200
    assert "Does X help Y?" in r.text
    assert r.headers["content-type"].startswith("text/markdown")


def test_export_docx_happy_path(client: TestClient):
    job_id = _finished_job({"research_question": "q"})
    with patch("tools.prisma_report.generate_prisma_docx", return_value=b"FAKE-DOCX-BYTES") as mock_gen:
        r = client.post(f"{_BASE}/jobs/{job_id}/export/docx", json={"author": "A. Researcher"})
    assert r.status_code == 200
    assert r.content == b"FAKE-DOCX-BYTES"
    assert "wordprocessingml" in r.headers["content-type"]
    assert "systematic_review.docx" in r.headers["content-disposition"]
    mock_gen.assert_called_once_with({"research_question": "q"}, author="A. Researcher", institution="")


def test_export_docx_unknown_job_404(client: TestClient):
    r = client.post(f"{_BASE}/jobs/does-not-exist/export/docx", json={})
    assert r.status_code == 404


def test_export_docx_missing_dependency_returns_clean_500():
    """Real (unmocked) ImportError path -- python-docx isn't installed in this
    sandbox. Uses a raw TestClient(raise_server_exceptions=False) since the
    default `client` fixture re-raises exceptions instead of letting the
    app's registered global exception handler convert them to a 500."""
    job_id = _finished_job({"research_question": "q"})
    raw_client = TestClient(app, raise_server_exceptions=False)
    r = raw_client.post(f"{_BASE}/jobs/{job_id}/export/docx", json={})
    assert r.status_code == 500
    assert "python-docx" in r.json()["detail"]


def test_export_pdf_happy_path(client: TestClient):
    job_id = _finished_job({"research_question": "q"})
    with patch("tools.prisma_report.generate_prisma_pdf", return_value=b"FAKE-PDF-BYTES") as mock_gen:
        r = client.post(f"{_BASE}/jobs/{job_id}/export/pdf", json={})
    assert r.status_code == 200
    assert r.content == b"FAKE-PDF-BYTES"
    assert r.headers["content-type"] == "application/pdf"
    mock_gen.assert_called_once_with({"research_question": "q"}, author="", institution="")


# ─────────────────────────────────────────────────────────────────────────────
# Plain-language summaries (background job)
# ─────────────────────────────────────────────────────────────────────────────

def test_trigger_plain_language_summary_unknown_job_404(client: TestClient):
    r = client.post(f"{_BASE}/jobs/does-not-exist/plain-language-summary", json={})
    assert r.status_code == 404


def test_plain_language_summary_round_trip_all_formats(client: TestClient):
    job_id = _finished_job({"model_name": "m", "num_ctx": 123})
    with patch(
        "tools.plain_language.generate_all_summaries",
        return_value={"patient": "p", "policy": "po", "press": "pr"},
    ):
        r = client.post(f"{_BASE}/jobs/{job_id}/plain-language-summary", json={})
        assert r.status_code == 202
        data = _poll_until_terminal(client, f"{_BASE}/tool-jobs/{r.json()['job_id']}")

    assert data["status"] == "done"
    assert data["result"] == {"patient": "p", "policy": "po", "press": "pr"}


def test_plain_language_summary_single_format(client: TestClient):
    job_id = _finished_job({"model_name": "m", "num_ctx": 123})
    with patch("tools.plain_language.generate_patient_summary", return_value="patient text"):
        r = client.post(f"{_BASE}/jobs/{job_id}/plain-language-summary", json={"format": "patient"})
        data = _poll_until_terminal(client, f"{_BASE}/tool-jobs/{r.json()['job_id']}")

    assert data["result"] == {"patient": "patient text"}


# ─────────────────────────────────────────────────────────────────────────────
# Guided templates + grammar-check gate
# ─────────────────────────────────────────────────────────────────────────────

def test_list_templates_endpoint_returns_four_presets(client: TestClient):
    r = client.get(f"{_BASE}/templates")
    assert r.status_code == 200
    keys = {t["key"] for t in r.json()}
    assert keys == {"clinical_rct", "cs_survey", "qual_synthesis", "scoping_review"}


def test_grammar_check_endpoint_forwards_fields(client: TestClient):
    with patch(
        "tools.grammar_check.check_and_fix_grammar",
        return_value={"original": "teh quick fox", "corrected": "the quick fox", "changed": True},
    ) as mock_check:
        r = client.post(
            f"{_BASE}/grammar-check", json={"text": "teh quick fox", "context_hint": "research question"}
        )
    assert r.status_code == 200
    assert r.json() == {"original": "teh quick fox", "corrected": "the quick fox", "changed": True}
    mock_check.assert_called_once_with(
        "teh quick fox", model_name="", num_ctx=8192, context_hint="research question"
    )


def test_grammar_check_no_ollama_fails_safe_and_returns_original(client: TestClient):
    """Real (unmocked) path -- no Ollama/network reachable in this sandbox, so
    check_and_fix_grammar's own try/except should return the text unchanged
    rather than raising."""
    r = client.post(f"{_BASE}/grammar-check", json={"text": "Does sleeep help memroy?"})
    assert r.status_code == 200
    data = r.json()
    assert data["original"] == "Does sleeep help memroy?"
    assert data["corrected"] == "Does sleeep help memroy?"
    assert data["changed"] is False
