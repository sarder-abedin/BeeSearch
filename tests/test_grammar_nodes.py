"""
tests/test_grammar_nodes.py
────────────────────────────
Unit tests for Grammar Proofreading Mode (Mode 6) nodes and graph.

Mocking strategy: patch("agents.grammar_nodes.ChatOllama") so no real Ollama
calls are made. Each test configures the mock's .invoke().content to return
a specific string and then asserts on the returned state dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.grammar_state import GrammarState, create_grammar_state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_ollama(response_text: str):
    """Return a patched ChatOllama class whose instance returns response_text."""
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MagicMock(content=response_text)
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


def _base_state(**kwargs) -> GrammarState:
    s = create_grammar_state(
        raw_text="She dont like cats and she go to school everyday.",
        session_id="test-session",
        model_name="llama3.1:8b",
        num_ctx=4096,
        style_level="informal",
        focus_areas=[],
    )
    s.update(kwargs)
    return s


# ── TestTextLoaderNode ────────────────────────────────────────────────────────

class TestTextLoaderNode:
    def test_basic_word_count(self):
        from agents.grammar_nodes import text_loader_node
        state = _base_state(raw_text="Hello world. How are you?")
        result = text_loader_node(state)
        assert result["word_count"] == 5
        assert result["sentence_count"] >= 2
        assert result["progress_pct"] == 10
        assert "text_loader" in result["completed_steps"]

    def test_strips_whitespace(self):
        from agents.grammar_nodes import text_loader_node
        state = _base_state(raw_text="  leading and trailing spaces.  ")
        result = text_loader_node(state)
        assert not result["raw_text"].startswith(" ")
        assert not result["raw_text"].endswith(" ")

    def test_empty_text(self):
        from agents.grammar_nodes import text_loader_node
        state = _base_state(raw_text="")
        result = text_loader_node(state)
        assert result["word_count"] == 0
        assert result["sentence_count"] == 0

    def test_long_text_warning(self, monkeypatch):
        from agents.grammar_nodes import text_loader_node
        # Force a very small context limit so the warning triggers
        monkeypatch.setattr("agents.grammar_nodes.cfg", MagicMock(
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.1:8b",
            num_ctx=10,  # tiny limit
        ))
        state = _base_state(raw_text="word " * 100, num_ctx=10)
        result = text_loader_node(state)
        assert any("long" in e.lower() or "context" in e.lower() for e in result.get("errors", []))

    def test_no_truncation(self):
        from agents.grammar_nodes import text_loader_node
        long_text = "word " * 6000
        state = _base_state(raw_text=long_text)
        result = text_loader_node(state)
        # Text must NOT be truncated
        assert len(result["raw_text"]) >= len(long_text.strip()) - 5


# ── TestGrammarAnalysisNode ───────────────────────────────────────────────────

class TestGrammarAnalysisNode:
    def test_valid_json_array(self):
        from agents.grammar_nodes import grammar_analysis_node
        issues = [
            {"type": "grammar", "original": "dont", "suggestion": "don't",
             "explanation": "Missing apostrophe.", "severity": "error"}
        ]
        mock_cls = _make_mock_ollama(json.dumps(issues))
        with patch("agents.grammar_nodes.ChatOllama", mock_cls):
            state = _base_state()
            result = grammar_analysis_node(state)
        assert len(result["issues_found"]) == 1
        assert result["issues_found"][0]["type"] == "grammar"
        assert result["progress_pct"] == 35

    def test_json_in_markdown_fence(self):
        from agents.grammar_nodes import grammar_analysis_node
        issues = [{"type": "spelling", "original": "teh", "suggestion": "the",
                   "explanation": "Typo.", "severity": "error"}]
        response = f"```json\n{json.dumps(issues)}\n```"
        mock_cls = _make_mock_ollama(response)
        with patch("agents.grammar_nodes.ChatOllama", mock_cls):
            result = grammar_analysis_node(_base_state())
        assert len(result["issues_found"]) == 1

    def test_malformed_json_returns_empty(self):
        from agents.grammar_nodes import grammar_analysis_node
        mock_cls = _make_mock_ollama("This is not JSON at all.")
        with patch("agents.grammar_nodes.ChatOllama", mock_cls):
            result = grammar_analysis_node(_base_state())
        assert result["issues_found"] == []
        assert any("analysis" in e.lower() or "output" in e.lower()
                   for e in result.get("errors", []))

    def test_llm_exception_returns_empty(self):
        from agents.grammar_nodes import grammar_analysis_node
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = RuntimeError("LLM unavailable")
        with patch("agents.grammar_nodes.ChatOllama", MagicMock(return_value=mock_instance)):
            result = grammar_analysis_node(_base_state())
        assert result["issues_found"] == []
        assert result.get("errors")

    def test_empty_raw_text_skips_llm(self):
        from agents.grammar_nodes import grammar_analysis_node
        with patch("agents.grammar_nodes.ChatOllama") as mock_cls:
            result = grammar_analysis_node(_base_state(raw_text=""))
        mock_cls.assert_not_called()
        assert result["issues_found"] == []


# ── TestPolishNode ────────────────────────────────────────────────────────────

class TestPolishNode:
    def test_sentinel_splits_correctly(self):
        from agents.grammar_nodes import polish_node
        polished = "She doesn't like cats and goes to school every day."
        changes = "- Fixed contraction\n- Fixed verb agreement"
        response = f"{polished}\n---CHANGES---\n{changes}"
        mock_cls = _make_mock_ollama(response)
        with patch("agents.grammar_nodes.ChatOllama", mock_cls):
            result = polish_node(_base_state())
        assert result["polished_text"] == polished
        assert "Fixed" in result["change_summary"]
        assert result["progress_pct"] == 65

    def test_no_sentinel_full_response_as_polished(self):
        from agents.grammar_nodes import polish_node
        response = "She doesn't like cats and she goes to school every day."
        mock_cls = _make_mock_ollama(response)
        with patch("agents.grammar_nodes.ChatOllama", mock_cls):
            result = polish_node(_base_state())
        assert result["polished_text"] == response
        assert result["change_summary"] == ""

    def test_llm_error_returns_original(self):
        from agents.grammar_nodes import polish_node
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = RuntimeError("timeout")
        with patch("agents.grammar_nodes.ChatOllama", MagicMock(return_value=mock_instance)):
            state = _base_state()
            result = polish_node(state)
        assert result["polished_text"] == state.get("raw_text")
        assert result.get("errors")

    def test_revision_path_includes_feedback(self):
        from agents.grammar_nodes import polish_node
        response = "Revised version.\n---CHANGES---\n- Incorporated feedback"
        mock_cls = _make_mock_ollama(response)
        with patch("agents.grammar_nodes.ChatOllama", mock_cls):
            state = _base_state(
                refinement_round=1,
                feedback="Make it more formal",
                polished_text="Previous polished version.",
            )
            result = polish_node(state)
        # Verify feedback was recorded in history
        assert len(result["feedback_history"]) == 1
        assert result["feedback_history"][0]["feedback"] == "Make it more formal"

    def test_context_specific_style_used(self):
        from agents.grammar_nodes import polish_node, _STYLE_PROMPTS
        # Verify all style levels have prompts
        for level in ("academic", "professional_email", "formal", "informal"):
            assert level in _STYLE_PROMPTS
            assert len(_STYLE_PROMPTS[level]) > 50


# ── TestStyleAdvisorNode ──────────────────────────────────────────────────────

class TestStyleAdvisorNode:
    def test_returns_suggestions_when_style_in_focus(self):
        from agents.grammar_nodes import style_advisor_node
        tips = [{"category": "clarity", "suggestion": "Simplify sentence.", "rationale": "Too long."}]
        mock_cls = _make_mock_ollama(json.dumps(tips))
        with patch("agents.grammar_nodes.ChatOllama", mock_cls):
            state = _base_state(
                focus_areas=["style"],
                polished_text="Some polished text here.",
            )
            result = style_advisor_node(state)
        assert len(result["style_suggestions"]) == 1

    def test_skips_when_style_not_in_focus_areas(self):
        from agents.grammar_nodes import style_advisor_node
        with patch("agents.grammar_nodes.ChatOllama") as mock_cls:
            state = _base_state(focus_areas=["grammar", "spelling"])
            result = style_advisor_node(state)
        mock_cls.assert_not_called()
        assert result["style_suggestions"] == []

    def test_runs_when_focus_areas_empty(self):
        from agents.grammar_nodes import style_advisor_node
        tips = [{"category": "tone", "suggestion": "Use active voice.", "rationale": "Clearer."}]
        mock_cls = _make_mock_ollama(json.dumps(tips))
        with patch("agents.grammar_nodes.ChatOllama", mock_cls):
            state = _base_state(focus_areas=[], polished_text="Some text.")
            result = style_advisor_node(state)
        assert len(result["style_suggestions"]) == 1

    def test_parse_failure_returns_empty_list(self):
        from agents.grammar_nodes import style_advisor_node
        mock_cls = _make_mock_ollama("not valid json")
        with patch("agents.grammar_nodes.ChatOllama", mock_cls):
            state = _base_state(focus_areas=[], polished_text="Some text.")
            result = style_advisor_node(state)
        assert result["style_suggestions"] == []

    def test_skips_when_no_polished_text_and_no_raw_text(self):
        from agents.grammar_nodes import style_advisor_node
        with patch("agents.grammar_nodes.ChatOllama") as mock_cls:
            state = _base_state(focus_areas=[], polished_text="", raw_text="")
            result = style_advisor_node(state)
        mock_cls.assert_not_called()


# ── TestGrammarEvalNode ───────────────────────────────────────────────────────

class TestGrammarEvalNode:
    def test_returns_four_dimensions_plus_overall(self):
        from agents.eval_nodes import grammar_eval_node
        eval_json = {
            "polish_quality": 4,
            "context_fit": 5,
            "error_coverage": 4,
            "fluency": 4,
            "overall": 4,
            "summary": "Well-polished output.",
        }
        mock_cls = _make_mock_ollama(json.dumps(eval_json))
        with patch("agents.eval_nodes.ChatOllama", mock_cls):
            state = dict(_base_state(polished_text="She doesn't like cats."))
            result = grammar_eval_node(state)
        assert result["eval_result"]["overall"] == 4
        assert "polish_quality" in result["eval_result"]
        assert result["progress_pct"] == 100

    def test_skips_when_no_polished_text(self):
        from agents.eval_nodes import grammar_eval_node
        with patch("agents.eval_nodes.ChatOllama") as mock_cls:
            state = dict(_base_state(polished_text=""))
            result = grammar_eval_node(state)
        mock_cls.assert_not_called()
        assert result["eval_result"] == {}

    def test_llm_error_returns_empty(self):
        from agents.eval_nodes import grammar_eval_node
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = RuntimeError("LLM timeout")
        with patch("agents.eval_nodes.ChatOllama", MagicMock(return_value=mock_instance)):
            state = dict(_base_state(polished_text="Some text."))
            result = grammar_eval_node(state)
        assert result["eval_result"] == {}


# ── TestRunGrammarCheck (full graph) ─────────────────────────────────────────

class TestRunGrammarCheck:
    def _run_with_mocks(self, issues_json="[]", polished_response="Polished.\n---CHANGES---\n- Fixed it",
                        tips_json="[]", eval_json=None):
        """Run the full graph with all LLM calls mocked."""
        if eval_json is None:
            eval_json = {"polish_quality": 4, "context_fit": 4,
                         "error_coverage": 4, "fluency": 4, "overall": 4, "summary": "Good."}

        responses = iter([
            issues_json,         # grammar_analysis
            polished_response,   # polish
            tips_json,           # style_advisor
            json.dumps(eval_json),  # grammar_eval
        ])

        def make_invoke():
            def _invoke(messages):
                return MagicMock(content=next(responses, "{}"))
            return _invoke

        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = make_invoke()

        from agents.grammar_graph import run_grammar_check
        initial = create_grammar_state(
            raw_text="She dont like cats.",
            session_id="test-full",
            style_level="informal",
        )

        with patch("agents.grammar_nodes.ChatOllama", MagicMock(return_value=mock_instance)), \
             patch("agents.eval_nodes.ChatOllama", MagicMock(return_value=mock_instance)):
            return run_grammar_check(initial)

    def test_all_completed_steps_present(self):
        final = self._run_with_mocks()
        steps = final.get("completed_steps", [])
        for expected in ("text_loader", "grammar_analysis", "polish", "style_advisor", "grammar_eval"):
            assert expected in steps, f"Missing step: {expected}"

    def test_rag_reflection_always_empty(self):
        final = self._run_with_mocks()
        assert final.get("rag_reflection_info") == {}

    def test_polished_text_populated(self):
        final = self._run_with_mocks()
        assert final.get("polished_text") == "Polished."

    def test_change_summary_populated(self):
        final = self._run_with_mocks()
        assert "Fixed" in final.get("change_summary", "")

    def test_progress_reaches_100(self):
        final = self._run_with_mocks()
        assert final.get("progress_pct") == 100


# ── TestGrammarState ─────────────────────────────────────────────────────────

class TestGrammarState:
    def test_factory_defaults(self):
        s = create_grammar_state("Hello.", "sid")
        assert s["raw_text"] == "Hello."
        assert s["session_id"] == "sid"
        assert s["style_level"] == "professional_email"
        assert s["focus_areas"] == []
        assert s["issues_found"] == []
        assert s["polished_text"] == ""
        assert s["rag_reflection_info"] == {}
        assert s["refinement_round"] == 0
        assert s["feedback_history"] == []
        assert s["progress_pct"] == 0
        assert s["completed_steps"] == []
        assert s["errors"] == []

    def test_custom_style_level(self):
        s = create_grammar_state("Text.", "sid", style_level="academic")
        assert s["style_level"] == "academic"

    def test_revision_round_fields(self):
        s = create_grammar_state(
            "Text.", "sid",
            refinement_round=2,
            feedback="Be more concise",
            feedback_history=[{"round": 1, "feedback": "x", "previous_polished": "y"}],
        )
        assert s["refinement_round"] == 2
        assert s["feedback"] == "Be more concise"
        assert len(s["feedback_history"]) == 1


# ── TestGrammarMemory ─────────────────────────────────────────────────────────

class TestGrammarMemory:
    def test_new_session_and_load(self, tmp_path):
        from agents.grammar_memory import GrammarMemory
        mem = GrammarMemory(db_path=tmp_path / "sessions.db")
        sid = mem.new_session(raw_text="Hello world.", style_level="academic")
        data = mem.load(sid)
        assert data is not None
        assert data["session_id"] == sid
        assert data["style_level"] == "academic"

    def test_save_result(self, tmp_path):
        from agents.grammar_memory import GrammarMemory
        mem = GrammarMemory(db_path=tmp_path / "sessions.db")
        sid = mem.new_session(raw_text="Hello.")
        state = {
            "raw_text": "Hello.",
            "polished_text": "Hello there.",
            "issues_found": [{"type": "style"}],
            "style_suggestions": [],
            "eval_result": {"overall": 4},
            "word_count": 1,
            "refinement_round": 0,
            "feedback_history": [],
            "change_summary": "",
        }
        mem.save_result(sid, state)
        data = mem.load(sid)
        assert data["polished_text"] == "Hello there."
        assert data["eval_result"]["overall"] == 4

    def test_list_sessions(self, tmp_path):
        from agents.grammar_memory import GrammarMemory
        mem = GrammarMemory(db_path=tmp_path / "sessions.db")
        sid1 = mem.new_session(raw_text="First text.")
        sid2 = mem.new_session(raw_text="Second text.")
        sessions = mem.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert sid1 in ids and sid2 in ids

    def test_load_nonexistent(self, tmp_path):
        from agents.grammar_memory import GrammarMemory
        mem = GrammarMemory(db_path=tmp_path / "sessions.db")
        assert mem.load("nonexistent") is None

    def test_delete(self, tmp_path):
        from agents.grammar_memory import GrammarMemory
        mem = GrammarMemory(db_path=tmp_path / "sessions.db")
        sid = mem.new_session(raw_text="Delete me.")
        assert mem.delete(sid) is True
        assert mem.load(sid) is None
