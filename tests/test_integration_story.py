"""tests/test_integration_story.py — Smoke tests for the story graph end-to-end."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import agents.story_nodes as story_nodes_module
from agents.story_graph import run_story_turn
from agents.story_memory import StorytellerMemory
from agents.story_state import create_story_state


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_fake_invoke():
    """
    Return a stateful callable that behaves like ChatOllama.invoke.

    First call → main storyteller response (contains suggested_questions JSON).
    Subsequent calls → concept extraction micro-call response.
    """
    call_count = 0

    def _invoke(messages):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if call_count == 1:
            resp.content = (
                "Attention mechanisms allow a model to weigh the importance of "
                "different tokens when producing an output. Think of it as a spotlight "
                "that can focus on multiple places at once.\n\n"
                '{"suggested_questions": ["What is multi-head attention?", '
                '"How does softmax work?", "What are the limitations of attention?"]}'
            )
        else:
            resp.content = '["attention mechanism", "spotlight"]'
        return resp

    return _invoke


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def mem(tmp_path):
    return StorytellerMemory(db_path=tmp_path / "sessions.db")


@pytest.fixture()
def session_id(mem):
    return mem.new_session(topic="Attention Mechanisms")


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestStoryGraphSmoke:
    def test_assistant_response_is_populated(self, mem, session_id, monkeypatch):
        """Graph run with mocked LLM: assistant_response must be non-empty."""
        monkeypatch.setattr(story_nodes_module, "_memory", mem)

        with patch("agents.story_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_fake_invoke()
            MockOllama.return_value = instance

            state = create_story_state(
                user_message="Explain attention mechanisms",
                session_id=session_id,
                topic="Attention Mechanisms",
            )
            result = run_story_turn(state)

        assert result["assistant_response"] != ""
        assert "attention" in result["assistant_response"].lower()

    def test_suggested_questions_parsed_from_json_block(self, mem, session_id, monkeypatch):
        """Suggested questions JSON at end of response must be extracted into list."""
        monkeypatch.setattr(story_nodes_module, "_memory", mem)

        call_count = 0

        def fake_invoke(messages):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.content = (
                "Some explanation.\n\n"
                '{"suggested_questions": ["Q1?", "Q2?", "Q3?"]}'
                if call_count == 1
                else '["concept a"]'
            )
            return resp

        with patch("agents.story_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = fake_invoke
            MockOllama.return_value = instance

            state = create_story_state(
                user_message="What is softmax?",
                session_id=session_id,
                topic="Attention Mechanisms",
            )
            result = run_story_turn(state)

        assert result["suggested_questions"] == ["Q1?", "Q2?", "Q3?"]

    def test_memory_saved_with_two_turns(self, mem, session_id, monkeypatch):
        """After one graph run, memory must contain one user turn + one assistant turn."""
        monkeypatch.setattr(story_nodes_module, "_memory", mem)

        with patch("agents.story_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_fake_invoke()
            MockOllama.return_value = instance

            user_msg = "Tell me about softmax"
            state = create_story_state(
                user_message=user_msg,
                session_id=session_id,
                topic="Attention Mechanisms",
            )
            run_story_turn(state)

        session = mem.load(session_id)
        assert session is not None
        convo = session["conversation"]
        assert len(convo) == 2
        assert convo[0]["role"] == "user"
        assert convo[0]["content"] == user_msg
        assert convo[1]["role"] == "assistant"
        assert convo[1]["content"] != ""

    def test_all_nodes_appear_in_completed_steps(self, mem, session_id, monkeypatch):
        """context_loader, storyteller, and memory_saver must all run."""
        monkeypatch.setattr(story_nodes_module, "_memory", mem)

        with patch("agents.story_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_fake_invoke()
            MockOllama.return_value = instance

            state = create_story_state(
                user_message="Hello",
                session_id=session_id,
                topic="Test",
            )
            result = run_story_turn(state)

        steps = result.get("completed_steps", [])
        assert "context_loader" in steps
        assert "storyteller" in steps
        assert "memory_saver" in steps

    def test_no_session_id_completes_without_error(self, monkeypatch):
        """Graph should run without error even when session_id is empty."""
        # Without injecting _memory, the lazy getter will create one —
        # but memory_saver_node short-circuits when session_id is "".
        with patch("agents.story_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _make_fake_invoke()
            MockOllama.return_value = instance

            state = create_story_state(
                user_message="Explain embeddings",
                session_id="",
                topic="Embeddings",
            )
            result = run_story_turn(state)

        assert result is not None
        assert result.get("assistant_response", "") != ""

    def test_second_turn_loads_history_from_memory(self, mem, session_id, monkeypatch):
        """After one turn, a second turn should load the saved conversation as history."""
        monkeypatch.setattr(story_nodes_module, "_memory", mem)

        call_count = 0

        def fake_invoke(messages):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.content = (
                f"Turn {(call_count + 1) // 2} answer.\n\n"
                '{"suggested_questions": ["A?", "B?", "C?"]}'
                if call_count % 2 == 1
                else '[]'
            )
            return resp

        base_opts = dict(session_id=session_id, topic="Attention Mechanisms")

        with patch("agents.story_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = fake_invoke
            MockOllama.return_value = instance

            run_story_turn(create_story_state(user_message="First question", **base_opts))
            result = run_story_turn(create_story_state(user_message="Second question", **base_opts))

        # After two turns, memory has 4 entries; second turn must have seen history
        assert result["conversation_history"] != []
        session = mem.load(session_id)
        assert len(session["conversation"]) == 4
