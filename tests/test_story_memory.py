"""tests/test_story_memory.py — Unit tests for agents/story_memory.py"""

import pytest

from agents.story_memory import StorytellerMemory


@pytest.fixture()
def mem(tmp_path):
    """StorytellerMemory backed by a temp SQLite database."""
    return StorytellerMemory(db_path=tmp_path / "sessions.db")


# ── new_session ───────────────────────────────────────────────────────────────

class TestNewSession:
    def test_returns_session_id_string(self, mem):
        sid = mem.new_session(topic="Transformers")
        assert isinstance(sid, str) and len(sid) > 0

    def test_session_persisted(self, mem):
        sid = mem.new_session(topic="Attention")
        assert mem.load(sid) is not None

    def test_topic_stored(self, mem):
        sid = mem.new_session(topic="Quantum Computing")
        data = mem.load(sid)
        assert data["topic"] == "Quantum Computing"

    def test_document_context_stored(self, mem):
        sid = mem.new_session(topic="T", document_context="context text")
        data = mem.load(sid)
        assert data["document_context"] == "context text"

    def test_document_names_stored(self, mem):
        sid = mem.new_session(topic="T", document_names=["a.pdf", "b.pdf"])
        data = mem.load(sid)
        assert data["document_names"] == ["a.pdf", "b.pdf"]

    def test_empty_conversation_on_creation(self, mem):
        sid = mem.new_session(topic="T")
        data = mem.load(sid)
        assert data["conversation"] == []


# ── load ──────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_load_returns_none_for_missing_session(self, mem):
        assert mem.load("nonexistent_id") is None

    def test_load_returns_dict(self, mem):
        sid = mem.new_session(topic="Test")
        assert isinstance(mem.load(sid), dict)


# ── add_turn ──────────────────────────────────────────────────────────────────

class TestAddTurn:
    def test_user_turn_appended(self, mem):
        sid = mem.new_session(topic="T")
        mem.add_turn(sid, "user", "Hello?")
        data = mem.load(sid)
        assert len(data["conversation"]) == 1
        assert data["conversation"][0]["role"] == "user"
        assert data["conversation"][0]["content"] == "Hello?"

    def test_assistant_turn_with_questions(self, mem):
        sid = mem.new_session(topic="T")
        mem.add_turn(sid, "assistant", "Hi!", suggested_questions=["Q1", "Q2"])
        data = mem.load(sid)
        assert data["conversation"][0]["suggested_questions"] == ["Q1", "Q2"]

    def test_multiple_turns_ordered(self, mem):
        sid = mem.new_session(topic="T")
        mem.add_turn(sid, "user", "First")
        mem.add_turn(sid, "assistant", "Second")
        data = mem.load(sid)
        assert data["conversation"][0]["content"] == "First"
        assert data["conversation"][1]["content"] == "Second"

    def test_turn_has_timestamp(self, mem):
        sid = mem.new_session(topic="T")
        mem.add_turn(sid, "user", "msg")
        data = mem.load(sid)
        assert "timestamp" in data["conversation"][0]


# ── add_concepts ──────────────────────────────────────────────────────────────

class TestAddConcepts:
    def test_concepts_added(self, mem):
        sid = mem.new_session(topic="T")
        mem.add_concepts(sid, ["attention", "softmax"])
        data = mem.load(sid)
        assert "attention" in data["concepts_covered"]
        assert "softmax" in data["concepts_covered"]

    def test_no_duplicate_concepts(self, mem):
        sid = mem.new_session(topic="T")
        mem.add_concepts(sid, ["attention"])
        mem.add_concepts(sid, ["attention", "relu"])
        data = mem.load(sid)
        assert data["concepts_covered"].count("attention") == 1

    def test_empty_list_is_safe(self, mem):
        sid = mem.new_session(topic="T")
        mem.add_concepts(sid, [])
        data = mem.load(sid)
        assert data["concepts_covered"] == []


# ── list_sessions ─────────────────────────────────────────────────────────────

class TestListSessions:
    def test_empty_when_no_sessions(self, mem):
        assert mem.list_sessions() == []

    def test_returns_one_entry_per_session(self, mem):
        mem.new_session(topic="A")
        mem.new_session(topic="B")
        assert len(mem.list_sessions()) == 2

    def test_entries_have_required_keys(self, mem):
        mem.new_session(topic="T")
        entry = mem.list_sessions()[0]
        for key in ("session_id", "topic", "created_at", "turn_count"):
            assert key in entry

    def test_turn_count_reflects_conversations(self, mem):
        sid = mem.new_session(topic="T")
        mem.add_turn(sid, "user", "Hi")
        mem.add_turn(sid, "assistant", "Hello")
        entry = mem.list_sessions()[0]
        assert entry["turn_count"] == 2


# ── delete ────────────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_removes_session(self, mem):
        sid = mem.new_session(topic="T")
        assert mem.delete(sid) is True
        assert mem.load(sid) is None

    def test_load_returns_none_after_delete(self, mem):
        sid = mem.new_session(topic="T")
        mem.delete(sid)
        assert mem.load(sid) is None

    def test_delete_nonexistent_is_safe(self, mem):
        mem.delete("ghost_id")  # should not raise
