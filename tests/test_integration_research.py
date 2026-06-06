"""tests/test_integration_research.py — Smoke tests for the research graph end-to-end."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import agents.nodes as nodes_module
from agents.graph import run_research
from agents.state import create_initial_state


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_fake_invoke():
    """
    Stateful fake for ChatOllama.invoke across the research graph.

    Call 1  → query_generation:   JSON list of search queries
    Call 2  → key-findings / doc analysis: JSON list (safe for any JSON parser)
    Call 3+ → report_generation:  markdown report
    """
    call_count = 0

    def _invoke(messages):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if call_count == 1:
            resp.content = (
                '["attention mechanisms in transformers", '
                '"BERT self-attention", "NLP deep learning"]'
            )
        elif call_count == 2:
            resp.content = (
                '["Transformers rely on attention mechanisms", '
                '"BERT achieves state-of-the-art performance"]'
            )
        else:
            resp.content = (
                "# Research Report\n\n"
                "Transformer models have revolutionised NLP via attention.\n\n"
                "## Key Findings\n\n"
                "1. Attention allows models to focus on relevant tokens.\n"
                "2. BERT demonstrates transfer learning effectiveness."
            )
        return resp

    return _invoke


def _mock_searcher():
    """Return a MagicMock AcademicSearcher whose search() always returns []."""
    m = MagicMock()
    m.search.return_value = []
    return m


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestResearchGraphSmoke:
    def test_report_is_populated(self, monkeypatch):
        """A search-mode run with mocked LLM produces a non-empty report."""
        monkeypatch.setattr(nodes_module, "_academic", _mock_searcher())
        monkeypatch.setattr(nodes_module, "_web", _mock_searcher())

        with patch("agents.nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_fake_invoke()
            MockOllama.return_value = instance

            state = create_initial_state(
                goal="Explain attention mechanisms in transformer models",
                mode="search",
            )
            result = run_research(state)

        assert result.get("report", "") != ""

    def test_completed_steps_includes_report_generation(self, monkeypatch):
        """report_generation must appear in completed_steps after a full run."""
        monkeypatch.setattr(nodes_module, "_academic", _mock_searcher())
        monkeypatch.setattr(nodes_module, "_web", _mock_searcher())

        with patch("agents.nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_fake_invoke()
            MockOllama.return_value = instance

            state = create_initial_state(goal="BERT pre-training", mode="search")
            result = run_research(state)

        steps = result.get("completed_steps", [])
        assert "report_generation" in steps

    def test_no_uploaded_docs_no_crash(self, monkeypatch):
        """search mode with no documents must complete without raising."""
        monkeypatch.setattr(nodes_module, "_academic", _mock_searcher())
        monkeypatch.setattr(nodes_module, "_web", _mock_searcher())

        with patch("agents.nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_fake_invoke()
            MockOllama.return_value = instance

            state = create_initial_state(
                goal="Survey of federated learning",
                mode="search",
                uploaded_docs=[],
            )
            result = run_research(state)

        assert result is not None

    def test_errors_empty_on_happy_path(self, monkeypatch):
        """No errors should be accumulated on a normal mocked run."""
        monkeypatch.setattr(nodes_module, "_academic", _mock_searcher())
        monkeypatch.setattr(nodes_module, "_web", _mock_searcher())

        with patch("agents.nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_fake_invoke()
            MockOllama.return_value = instance

            state = create_initial_state(goal="Meta-learning in NLP", mode="search")
            result = run_research(state)

        assert result.get("errors", []) == []
