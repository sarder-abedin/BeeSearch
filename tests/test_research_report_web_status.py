"""
tests/test_research_report_web_status.py
───────────────────────────────────────────
Unit tests for agents/graph.py::_step_web_search's `web_search_status` field.

A Research Report with "Include web search" enabled but zero [Web N] references
in the end looks identical whether the toggle was off, the search ran and found
nothing, or it ran and failed outright. `web_search_status`
("disabled"/"ok"/"empty"/"error") lets the UI distinguish these so it can show a
banner when the toggle was on but didn't actually contribute anything.

Pure stdlib — no network access; WebSearcher is mocked.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.graph import _step_web_search
from tools.search_tools import WebResult


def test_disabled_when_include_web_search_false():
    """Web search never runs when the toggle is off — status is "disabled", not "empty"."""
    state = {"include_web_search": False, "goal": "g", "search_queries": ["g"]}
    result = _step_web_search(state)
    assert result["web_search_status"] == "disabled"
    assert result["web_results"] == []


def test_ok_when_results_found():
    """A successful search with results yields status "ok"."""
    state = {"include_web_search": True, "goal": "g", "search_queries": ["g"]}
    with patch("tools.search_tools.WebSearcher") as mock_cls:
        mock_cls.return_value.search.return_value = [WebResult(title="t", url="u", snippet="s")]
        result = _step_web_search(state)
    assert result["web_search_status"] == "ok"
    assert len(result["web_results"]) == 1


def test_empty_when_search_runs_but_finds_nothing():
    """Search runs cleanly but returns no results — status is "empty", distinct from "disabled"."""
    state = {"include_web_search": True, "goal": "g", "search_queries": ["g"]}
    with patch("tools.search_tools.WebSearcher") as mock_cls:
        mock_cls.return_value.search.return_value = []
        result = _step_web_search(state)
    assert result["web_search_status"] == "empty"
    assert result["web_results"] == []


def test_error_when_search_raises():
    """A WebSearcher exception is caught (no crash) and reported as status "error"."""
    state = {"include_web_search": True, "goal": "g", "search_queries": ["g"]}
    with patch("tools.search_tools.WebSearcher") as mock_cls:
        mock_cls.return_value.search.side_effect = Exception("boom")
        result = _step_web_search(state)
    assert result["web_search_status"] == "error"
    assert result["web_results"] == []
