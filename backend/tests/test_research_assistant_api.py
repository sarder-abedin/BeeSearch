"""backend/tests/test_research_assistant_api.py
───────────────────────────────────────────────────
API-level tests for the Mode 3 (AI Research Assistant) HTTP endpoints.

Mocks at the same boundary as tests/test_research_assistant.py
(agents.research_assistant.search_literature / .ChatOllama) so these tests
exercise the real run_research_assistant pipeline through the full
router → service → job-runner → polling stack, with no network or LLM calls.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _two_call_llm(answer: str, followups_json: str = "[]") -> MagicMock:
    """Mock ChatOllama whose two invoke() calls return the answer, then follow-ups."""
    llm = MagicMock()
    msg1 = MagicMock()
    msg1.content = answer
    msg2 = MagicMock()
    msg2.content = followups_json
    llm.invoke.side_effect = [msg1, msg2]
    return llm


def _poll_until_terminal(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    data = {}
    while time.monotonic() < deadline:
        r = client.get(f"/api/research-assistant/jobs/{job_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal state: {data}")


def test_health_endpoint(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "mock_llm": False}


def test_ask_blank_question_returns_422(client: TestClient):
    r = client.post("/api/research-assistant/ask", json={"question": "   "})
    assert r.status_code == 422


def test_ask_missing_question_field_returns_422(client: TestClient):
    r = client.post("/api/research-assistant/ask", json={})
    assert r.status_code == 422


def test_get_unknown_job_returns_404(client: TestClient):
    r = client.get("/api/research-assistant/jobs/does-not-exist")
    assert r.status_code == 404


def test_ask_grounded_round_trip_returns_citations_and_sources(client: TestClient):
    found = {
        "papers": [{
            "title": "Sleep & Memory", "authors": ["A Smith"], "year": 2020,
            "abstract": "sleep helps memory", "url": "http://p1", "source": "arxiv",
        }],
        "web_results": [],
    }
    llm = _two_call_llm("Sleep improves memory [1].", '["What about naps?"]')

    with patch("agents.research_assistant.search_literature", return_value=found), \
         patch("agents.research_assistant.ChatOllama", return_value=llm):
        r = client.post("/api/research-assistant/ask", json={"question": "Does sleep help memory?"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        data = _poll_until_terminal(client, job_id)

    assert data["status"] == "done"
    assert data["error"] is None
    result = data["result"]
    assert result["grounded"] is True
    assert result["academic_count"] == 1
    assert [c["n"] for c in result["citations"]] == [1]
    assert result["sources"][0]["title"] == "Sleep & Memory"
    assert result["suggested_questions"] == ["What about naps?"]


def test_ask_no_sources_is_ungrounded_with_no_citations(client: TestClient):
    found = {"papers": [], "web_results": []}
    llm = _two_call_llm("Generally, sleep is thought to help memory.", "[]")

    with patch("agents.research_assistant.search_literature", return_value=found), \
         patch("agents.research_assistant.ChatOllama", return_value=llm):
        r = client.post("/api/research-assistant/ask", json={"question": "Does sleep help memory?"})
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    result = data["result"]
    assert result["grounded"] is False
    assert result["citations"] == []
    assert result["answer"]


def test_ask_include_web_false_is_forwarded_to_search_literature(client: TestClient):
    mock_search = MagicMock(return_value={"papers": [], "web_results": []})
    llm = _two_call_llm("Generally, X.", "[]")

    with patch("agents.research_assistant.search_literature", mock_search), \
         patch("agents.research_assistant.ChatOllama", return_value=llm):
        r = client.post(
            "/api/research-assistant/ask", json={"question": "q", "include_web": False}
        )
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    assert mock_search.call_args.kwargs["include_web"] is False


def test_ask_pipeline_exception_surfaces_as_job_error(client: TestClient):
    with patch(
        "agents.research_assistant.search_literature",
        side_effect=RuntimeError("search backend exploded"),
    ):
        r = client.post("/api/research-assistant/ask", json={"question": "q"})
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "error"
    assert "search backend exploded" in data["error"]
    assert data["result"] is None
