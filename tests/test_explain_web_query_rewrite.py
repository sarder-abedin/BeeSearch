"""
tests/test_explain_web_query_rewrite.py
─────────────────────────────────────────
Unit tests for agents/story_nodes.py::_build_web_query.

Same failure mode as the Notebook Chat tab (see test_notebook_web_query_rewrite.py),
but on the Explain tab's source_router_node: a conversational question like "What's
the state of the art compared to this approach?" means nothing to DuckDuckGo/arXiv/
Semantic Scholar on its own — they have no idea what "this approach" refers to, so
the search comes back technically non-empty but contextually irrelevant.
_build_web_query rewrites the question into a self-contained query grounded in the
uploaded document context before it reaches any search engine.

ChatOllama is mocked — no network access or Ollama server required.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.story_nodes import _build_web_query


def test_returns_raw_query_unchanged_when_no_document_context():
    """No document context to ground a rewrite in — use the raw question, and skip the LLM call entirely."""
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        result = _build_web_query("What's the state of the art vs this approach?", "", "llama3.1:8b", 4096)
    assert result == "What's the state of the art vs this approach?"
    mock_chat.assert_not_called()


def test_rewrites_query_using_document_context():
    """With document context available, the LLM's rewritten (and stripped) query is used."""
    doc_context = "This paper studies Age of Information in UAV-assisted networks."
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = (
            "Age of Information UAV network state of the art comparison"
        )
        result = _build_web_query("What's the state of the art vs this approach?", doc_context, "llama3.1:8b", 4096)
    assert result == "Age of Information UAV network state of the art comparison"


def test_strips_wrapping_quotes_from_rewritten_query():
    """LLMs commonly wrap a generated query in quotes; those must not reach the search engine verbatim."""
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = '"UAV network limitations"'
        result = _build_web_query("Explain the limitation of this work", "Some grounding text.", "llama3.1:8b", 4096)
    assert result == "UAV network limitations"


def test_falls_back_to_raw_query_when_rewrite_fails():
    """A rewrite-call failure must not break online search — fall back to the original question."""
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.side_effect = Exception("boom")
        result = _build_web_query("Explain the limitation of this work", "Some grounding text.", "llama3.1:8b", 4096)
    assert result == "Explain the limitation of this work"


def test_falls_back_to_raw_query_when_rewrite_is_empty():
    """An empty/whitespace-only rewrite must not produce a useless empty search query."""
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = "   "
        result = _build_web_query("Explain the limitation of this work", "Some grounding text.", "llama3.1:8b", 4096)
    assert result == "Explain the limitation of this work"
