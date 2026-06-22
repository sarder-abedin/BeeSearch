"""
tests/test_notebook_web_query_rewrite.py
───────────────────────────────────────────
Unit tests for agents/notebook_nodes.py::_build_web_query.

Conversational questions like "Explain the limitation of this work" are sent
verbatim to DuckDuckGo when "Auto web search" is on. A search engine has no
idea what "this work" refers to, so the search technically succeeds (200 OK,
N results) while returning results irrelevant to the actual notebook topic —
the LLM then correctly declines to cite them, and web search looks broken
even though nothing crashed or returned empty. _build_web_query rewrites the
question into a self-contained query grounded in the notebook's own retrieved
content before it reaches the search engine.

ChatOllama is mocked — no network access or Ollama server required.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.notebook_nodes import _build_web_query


def test_returns_raw_query_unchanged_when_no_notebook_chunks():
    """No notebook context to ground a rewrite in — use the raw question, and skip the LLM call entirely."""
    with patch("agents.notebook_nodes.ChatOllama") as mock_chat:
        result = _build_web_query({}, "Explain the limitation of this work", [])
    assert result == "Explain the limitation of this work"
    mock_chat.assert_not_called()


def test_rewrites_query_using_notebook_chunks_as_context():
    """With notebook chunks available, the LLM's rewritten (and stripped) query is used."""
    chunks = [{"text": "This paper studies Age of Information in UAV-assisted networks."}]
    with patch("agents.notebook_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = (
            "Age of Information UAV network limitations state of the art"
        )
        result = _build_web_query({}, "Explain the limitation of this work", chunks)
    assert result == "Age of Information UAV network limitations state of the art"


def test_strips_wrapping_quotes_from_rewritten_query():
    """LLMs commonly wrap a generated query in quotes; those must not reach the search engine verbatim."""
    chunks = [{"text": "Some grounding text."}]
    with patch("agents.notebook_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = '"UAV network limitations"'
        result = _build_web_query({}, "Explain the limitation of this work", chunks)
    assert result == "UAV network limitations"


def test_falls_back_to_raw_query_when_rewrite_fails():
    """A rewrite-call failure must not break web search — fall back to the original question."""
    chunks = [{"text": "Some grounding text."}]
    with patch("agents.notebook_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.side_effect = Exception("boom")
        result = _build_web_query({}, "Explain the limitation of this work", chunks)
    assert result == "Explain the limitation of this work"


def test_falls_back_to_raw_query_when_rewrite_is_empty():
    """An empty/whitespace-only rewrite must not produce a useless empty search query."""
    chunks = [{"text": "Some grounding text."}]
    with patch("agents.notebook_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = "   "
        result = _build_web_query({}, "Explain the limitation of this work", chunks)
    assert result == "Explain the limitation of this work"
