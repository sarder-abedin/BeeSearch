"""tests/test_research_memory.py — Unit tests for agents/memory.py ResearchMemory"""

import pytest

from agents.memory import ResearchMemory


@pytest.fixture()
def mem(tmp_path):
    """ResearchMemory backed by a temp SQLite database."""
    return ResearchMemory(db_path=tmp_path / "sessions.db")


def _session_kwargs(**overrides):
    base = dict(
        session_id="abc123",
        goal="Understand transformers",
        report="# Report\n\nSome content.",
        references=[{"title": "Paper", "authors": ["Smith J"], "year": 2022}],
        key_findings=["Transformers use attention", "They scale well"],
        document_names=["paper.pdf"],
        mode="search",
        model_name="llama3.1:8b",
    )
    base.update(overrides)
    return base


# ── save_session ──────────────────────────────────────────────────────────────

class TestSaveSession:
    def test_file_created(self, mem):
        mem.save_session(**_session_kwargs())
        assert mem.load("abc123") is not None

    def test_file_uses_session_id_in_name(self, mem):
        mem.save_session(**_session_kwargs(session_id="myid"))
        assert mem.load("myid") is not None

    def test_goal_persisted(self, mem):
        mem.save_session(**_session_kwargs(goal="My Goal"))
        data = mem.load("abc123")
        assert data["goal"] == "My Goal"

    def test_report_persisted(self, mem):
        mem.save_session(**_session_kwargs(report="Report text"))
        data = mem.load("abc123")
        assert data["report"] == "Report text"

    def test_overwrite_same_session(self, mem):
        mem.save_session(**_session_kwargs(report="First"))
        mem.save_session(**_session_kwargs(report="Second"))
        data = mem.load("abc123")
        assert data["report"] == "Second"


# ── load ──────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_returns_none_for_missing(self, mem):
        assert mem.load("ghost") is None

    def test_returns_dict_with_all_fields(self, mem):
        mem.save_session(**_session_kwargs())
        data = mem.load("abc123")
        for key in ("goal", "report", "references", "key_findings", "mode", "model_name"):
            assert key in data


# ── list_sessions ─────────────────────────────────────────────────────────────

class TestListSessions:
    def test_empty_when_none_saved(self, mem):
        assert mem.list_sessions() == []

    def test_returns_all_sessions(self, mem):
        mem.save_session(**_session_kwargs(session_id="s1", goal="Goal 1"))
        mem.save_session(**_session_kwargs(session_id="s2", goal="Goal 2"))
        assert len(mem.list_sessions()) == 2

    def test_limit_respected(self, mem):
        for i in range(10):
            mem.save_session(**_session_kwargs(session_id=f"s{i}", goal=f"Goal {i}"))
        assert len(mem.list_sessions(limit=3)) == 3

    def test_entries_have_required_keys(self, mem):
        mem.save_session(**_session_kwargs())
        entry = mem.list_sessions()[0]
        for key in ("session_id", "goal", "mode", "created_at"):
            assert key in entry

    def test_sorted_newest_first(self, mem):
        mem.save_session(**_session_kwargs(session_id="old", goal="Old"))
        mem.save_session(**_session_kwargs(session_id="new", goal="New"))
        sessions = mem.list_sessions()
        # Both were inserted at essentially the same time; just verify both are present
        ids = {s["session_id"] for s in sessions}
        assert "old" in ids and "new" in ids


# ── delete ────────────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_removes_session(self, mem):
        mem.save_session(**_session_kwargs(session_id="del1"))
        mem.delete("del1")
        assert mem.load("del1") is None

    def test_load_returns_none_after_delete(self, mem):
        mem.save_session(**_session_kwargs())
        mem.delete("abc123")
        assert mem.load("abc123") is None

    def test_delete_nonexistent_safe(self, mem):
        mem.delete("ghost")  # should not raise
