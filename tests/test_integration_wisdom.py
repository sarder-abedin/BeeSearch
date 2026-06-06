"""tests/test_integration_wisdom.py — Smoke tests for the wisdom graph end-to-end."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import agents.wisdom_nodes as wisdom_nodes_module
from agents.wisdom_graph import run_wisdom_turn
from agents.wisdom_memory import WisdomMemory
from agents.wisdom_state import create_wisdom_state


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_clarifying_invoke():
    """Fake invoke that always returns a clarifying question (keeps phase=clarifying)."""
    def _invoke(messages):
        resp = MagicMock()
        resp.content = (
            "PROCEED: No\n\n"
            "How long have you been experiencing chronic stress? "
            "Is it work-related or personal?"
        )
        return resp
    return _invoke


def _make_knowledge_search_invoke():
    """
    Stateful fake for a forced-proceed run (clarification_count >= 3).

    The graph path is:
      context_loader → clarification → knowledge_search →
      wisdom_synthesis → wisdom_validator → memory_saver

    Call 1  → clarification node — must start with PROCEED_TO_WISDOM
    Call 2  → knowledge_search query generation (JSON array)
    Call 3  → wisdom_synthesis deep_understanding
    Call 4  → wisdom_synthesis simple_explanation
    Call 5  → wisdom_synthesis actionable_takeaways (JSON list)
    Call 6  → wisdom_validator (JSON claims)
    Call 7+ → safe fallback
    """
    call_count = 0

    def _invoke(messages):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if call_count == 1:
            # Trigger the PROCEED_TO_WISDOM branch in clarification_node
            resp.content = (
                "PROCEED_TO_WISDOM\n"
                "I have enough context — searching the scientific literature now..."
            )
        elif call_count == 2:
            resp.content = (
                '["chronic stress cortisol memory", '
                '"stress hippocampus neuroplasticity", "HPA axis cognitive function"]'
            )
        elif call_count == 3:
            resp.content = (
                "Chronic stress elevates cortisol, which impairs hippocampal neurogenesis "
                "and working memory consolidation over time."
            )
        elif call_count == 4:
            resp.content = (
                "Think of cortisol as a fire alarm: helpful in short bursts, "
                "damaging if it never stops ringing."
            )
        elif call_count == 5:
            resp.content = (
                '["Prioritise sleep to allow memory consolidation", '
                '"Try 10-minute daily mindfulness to lower baseline cortisol", '
                '"Schedule short recovery breaks between demanding tasks"]'
            )
        elif call_count == 6:
            resp.content = (
                '{"overall_confidence": "High", "claims": ['
                '{"claim": "Cortisol impairs memory", "confidence": "High",'
                ' "consensus": "Well-established in literature"}], '
                '"devils_advocate": "Individual variation in stress tolerance is large."}'
            )
        else:
            resp.content = "That is a great follow-up question."
        return resp

    return _invoke


def _make_followup_invoke():
    """Fake for the follow-up path (phase=done)."""
    def _invoke(messages):
        resp = MagicMock()
        resp.content = (
            "Great follow-up. The relationship between cortisol and sleep is bidirectional: "
            "stress disrupts sleep, and sleep deprivation raises cortisol further."
        )
        return resp
    return _invoke


def _mock_searcher():
    m = MagicMock()
    m.search.return_value = []
    return m


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def mem(tmp_path):
    return WisdomMemory(db_path=tmp_path / "sessions.db")


@pytest.fixture()
def session_id(mem):
    return mem.new_session(topic="chronic stress and memory")


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestWisdomGraphSmoke:
    def test_assistant_response_is_populated(self, mem, session_id, monkeypatch):
        """First turn must produce a non-empty assistant_response."""
        monkeypatch.setattr(wisdom_nodes_module, "_memory", mem)
        monkeypatch.setattr(wisdom_nodes_module, "_academic", _mock_searcher())
        monkeypatch.setattr(wisdom_nodes_module, "_web", _mock_searcher())

        with patch("agents.wisdom_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_clarifying_invoke()
            MockOllama.return_value = instance

            state = create_wisdom_state(
                user_message="I feel stressed all the time, is that bad for memory?",
                session_id=session_id,
                topic="chronic stress and memory",
            )
            result = run_wisdom_turn(state)

        assert result.get("assistant_response", "") != ""

    def test_memory_saver_always_runs(self, mem, session_id, monkeypatch):
        """memory_saver must appear in completed_steps on every turn."""
        monkeypatch.setattr(wisdom_nodes_module, "_memory", mem)
        monkeypatch.setattr(wisdom_nodes_module, "_academic", _mock_searcher())
        monkeypatch.setattr(wisdom_nodes_module, "_web", _mock_searcher())

        with patch("agents.wisdom_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_clarifying_invoke()
            MockOllama.return_value = instance

            state = create_wisdom_state(
                user_message="Tell me about stress",
                session_id=session_id,
                topic="chronic stress and memory",
            )
            result = run_wisdom_turn(state)

        assert "memory_saver" in result.get("completed_steps", [])

    def test_no_session_id_completes_without_error(self, monkeypatch):
        """Graph must complete even when session_id is empty (no memory read/write)."""
        monkeypatch.setattr(wisdom_nodes_module, "_academic", _mock_searcher())
        monkeypatch.setattr(wisdom_nodes_module, "_web", _mock_searcher())

        with patch("agents.wisdom_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_clarifying_invoke()
            MockOllama.return_value = instance

            state = create_wisdom_state(
                user_message="What does chronic stress do to the brain?",
                session_id="",
                topic="stress and brain",
            )
            result = run_wisdom_turn(state)

        assert result is not None
        assert result.get("assistant_response", "") != ""

    def test_knowledge_search_path_when_forced_proceed(self, mem, session_id, monkeypatch):
        """
        When clarification_count >= 3 the graph must route through knowledge_search.

        We add 3 clarifying assistant turns via mem.add_turn() (which stores
        is_question=True at the top level of each turn, as wisdom_nodes expects).
        """
        monkeypatch.setattr(wisdom_nodes_module, "_memory", mem)
        monkeypatch.setattr(wisdom_nodes_module, "_academic", _mock_searcher())
        monkeypatch.setattr(wisdom_nodes_module, "_web", _mock_searcher())

        # Build 3 completed clarification rounds so force_proceed triggers
        for i in range(3):
            mem.add_turn(session_id, "user", f"user answer {i}")
            mem.add_turn(
                session_id, "assistant", f"Clarifying question {i}?",
                metadata={"is_question": True},
            )

        with patch("agents.wisdom_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_knowledge_search_invoke()
            MockOllama.return_value = instance

            state = create_wisdom_state(
                user_message="I sleep poorly and feel exhausted every day.",
                session_id=session_id,
                topic="chronic stress and memory",
            )
            result = run_wisdom_turn(state)

        assert "knowledge_search" in result.get("completed_steps", [])

    def test_followup_path_when_phase_done(self, mem, session_id, monkeypatch):
        """
        When phase == 'done' the graph routes to wisdom_followup, not clarification.

        We use mem.save_wisdom() to set phase='done' and inject wisdom_output,
        then mem.update_phase() would be redundant since save_wisdom sets phase.
        """
        monkeypatch.setattr(wisdom_nodes_module, "_memory", mem)
        monkeypatch.setattr(wisdom_nodes_module, "_academic", _mock_searcher())
        monkeypatch.setattr(wisdom_nodes_module, "_web", _mock_searcher())

        mem.save_wisdom(
            session_id=session_id,
            deep_understanding="Cortisol impairs memory formation.",
            simple_explanation="Stress is like an alarm that never stops.",
            actionable_takeaways=["Sleep more", "Exercise regularly"],
            validation={"overall_confidence": "High", "claims": [], "devils_advocate": ""},
            papers=[],
            queries=[],
            topic_tags=["stress", "memory"],
        )

        with patch("agents.wisdom_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_followup_invoke()
            MockOllama.return_value = instance

            state = create_wisdom_state(
                user_message="Does poor sleep make the stress worse?",
                session_id=session_id,
                topic="chronic stress and memory",
            )
            result = run_wisdom_turn(state)

        assert "wisdom_followup" in result.get("completed_steps", [])
        assert result.get("assistant_response", "") != ""
