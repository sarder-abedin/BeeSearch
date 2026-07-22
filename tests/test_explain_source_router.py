"""
tests/test_explain_source_router.py
─────────────────────────────────────
Unit tests for agents/story_nodes.py::source_router_node's `search_attempted` flag.

The Explain tab only showed an "online sources used" banner when search actually
found results, with no distinction between "coverage was already fine, search
never ran" and "coverage was low, search ran, but found nothing" — both rendered
the same plain "Answered from your documents" caption, hiding the fact that the
system tried (and failed) to fill a real gap. `search_attempted` tracks whether
low coverage triggered a search at all, independent of whether it found
anything, so the UI can render a distinct "searched but found nothing" banner.

ChatOllama and the academic/web searchers are mocked — no network access.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.story_nodes import source_router_node
from tools.search_tools import WebResult


def test_search_attempted_false_when_coverage_high():
    """High coverage never triggers a search — search_attempted matches used_online (both False)."""
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = '{"score": 8, "reason": "covers it well"}'
        result = source_router_node({"user_message": "q", "document_context": "some context"})
    decision = result["source_decision"]
    assert decision["search_attempted"] is False
    assert decision["used_online"] is False


def test_search_attempted_true_when_coverage_low_and_nothing_found():
    """Low coverage triggers a search; when it finds nothing, search_attempted stays True even though used_online is False."""
    with patch("agents.story_nodes.ChatOllama") as mock_chat, \
         patch("tools.search_tools.AcademicSearcher") as mock_acad, \
         patch("tools.search_tools.WebSearcher") as mock_web:
        mock_chat.return_value.invoke.return_value.content = '{"score": 3, "reason": "barely covers it"}'
        mock_acad.return_value.search.return_value = []
        mock_web.return_value.search.return_value = []
        result = source_router_node({"user_message": "q", "document_context": "some context"})
    decision = result["source_decision"]
    assert decision["search_attempted"] is True
    assert decision["used_online"] is False


def test_search_attempted_true_when_coverage_low_and_results_found():
    """Low coverage triggers a search; when it finds results, both search_attempted and used_online are True."""
    with patch("agents.story_nodes.ChatOllama") as mock_chat, \
         patch("tools.search_tools.AcademicSearcher") as mock_acad, \
         patch("tools.search_tools.WebSearcher") as mock_web:
        mock_chat.return_value.invoke.return_value.content = '{"score": 3, "reason": "barely covers it"}'
        mock_acad.return_value.search.return_value = []
        mock_web.return_value.search.return_value = [
            WebResult(title="t", url="https://example.com", snippet="s")
        ]
        result = source_router_node({"user_message": "q", "document_context": "some context"})
    decision = result["source_decision"]
    assert decision["search_attempted"] is True
    assert decision["used_online"] is True


def test_search_attempted_true_when_no_documents():
    """No document context at all forces coverage_score=0 (no LLM call) — search is always attempted."""
    with patch("tools.search_tools.AcademicSearcher") as mock_acad, \
         patch("tools.search_tools.WebSearcher") as mock_web:
        mock_acad.return_value.search.return_value = []
        mock_web.return_value.search.return_value = []
        result = source_router_node({"user_message": "q", "document_context": ""})
    assert result["source_decision"]["search_attempted"] is True
