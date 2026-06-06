"""tests/test_session_db.py — Unit tests for the SQLite session storage layer."""
import pytest
from pathlib import Path
from tools.session_db import init_db, _tx, pack, unpack


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test_sessions.db"
    init_db(db_path)
    return db_path


def test_init_db_creates_tables(db):
    with _tx(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    expected = {
        "grammar_sessions", "wisdom_sessions", "wisdom_tags", "story_sessions",
        "style_profiles", "proposal_sessions", "research_sessions",
        "notebooks", "notebook_chunks",
    }
    assert expected.issubset(tables)


def test_pack_unpack_roundtrip():
    obj = {"key": "value", "nums": [1, 2, 3], "nested": {"a": True}}
    assert unpack(pack(obj)) == obj


def test_pack_unpack_none():
    assert unpack(None) == {}


def test_grammar_session_roundtrip(db):
    from agents.grammar_memory import GrammarMemory
    mem = GrammarMemory(db_path=db)
    sid = mem.new_session(raw_text="Hello world.", style_level="informal")
    mem.save_result(sid, {
        "raw_text": "Hello world.",
        "style_level": "informal",
        "polished_text": "Hello, world.",
        "issues_found": [{"type": "punctuation"}],
        "style_suggestions": [],
        "eval_result": {"overall": 4},
        "word_count": 2,
        "refinement_round": 0,
        "feedback_history": [],
    })
    loaded = mem.load(sid)
    assert loaded is not None
    assert loaded["polished_text"] == "Hello, world."
    assert loaded["style_level"] == "informal"


def test_grammar_list_sessions(db):
    from agents.grammar_memory import GrammarMemory
    mem = GrammarMemory(db_path=db)
    for i in range(3):
        sid = mem.new_session(raw_text=f"Text {i}")
        mem.save_result(sid, {"polished_text": f"Polished {i}", "issues_found": [], "word_count": 2,
                               "style_level": "formal", "style_suggestions": [], "eval_result": {},
                               "raw_text": f"Text {i}", "refinement_round": 0, "feedback_history": []})
    sessions = mem.list_sessions()
    assert len(sessions) == 3
    assert all("session_id" in s for s in sessions)
    assert all("created_at" in s for s in sessions)


def test_grammar_delete(db):
    from agents.grammar_memory import GrammarMemory
    mem = GrammarMemory(db_path=db)
    sid = mem.new_session()
    assert mem.delete(sid) is True
    assert mem.load(sid) is None
    assert mem.delete(sid) is False


def test_notebook_roundtrip(db):
    from agents.notebook_memory import NotebookMemory
    from unittest.mock import MagicMock
    mem = NotebookMemory(db_path=db)
    nb_id = mem.new_notebook("Test Notebook")
    nb = mem.load(nb_id)
    assert nb is not None
    assert nb["name"] == "Test Notebook"
    assert nb["chunks"] == []
    assert nb["sources"] == []


def test_notebook_add_source(db):
    from agents.notebook_memory import NotebookMemory
    from unittest.mock import MagicMock
    mem = NotebookMemory(db_path=db)
    nb_id = mem.new_notebook("Paper Notebook")

    # Build a mock ProcessedDocument
    chunk = MagicMock()
    chunk.chunk_id = "doc1_c0"
    chunk.doc_id = "doc1"
    chunk.doc_name = "paper.pdf"
    chunk.page_num = 1
    chunk.chunk_index = 0
    chunk.text = "Some text content."
    doc = MagicMock()
    doc.doc_id = "doc1"
    doc.filename = "paper.pdf"
    doc.file_type = "PDF"
    doc.total_pages = 1
    doc.total_chunks = 1
    doc.content_md5 = "abc123"
    doc.chunks = [chunk]

    result = mem.add_source(nb_id, doc)
    assert result is True

    nb = mem.load(nb_id)
    assert len(nb["sources"]) == 1
    assert len(nb["chunks"]) == 1
    assert nb["chunks"][0]["text"] == "Some text content."

    # Duplicate check
    result2 = mem.add_source(nb_id, doc)
    assert result2 is False


def test_notebook_remove_source(db):
    from agents.notebook_memory import NotebookMemory
    from unittest.mock import MagicMock
    mem = NotebookMemory(db_path=db)
    nb_id = mem.new_notebook("NB")

    chunk = MagicMock(chunk_id="d1_c0", doc_id="d1", doc_name="a.pdf",
                      page_num=1, chunk_index=0, text="text")
    doc = MagicMock(doc_id="d1", filename="a.pdf", file_type="PDF",
                    total_pages=1, total_chunks=1, content_md5="x", chunks=[chunk])
    mem.add_source(nb_id, doc)

    assert mem.remove_source(nb_id, "d1") is True
    nb = mem.load(nb_id)
    assert nb["chunks"] == []
    assert nb["sources"] == []


def test_notebook_conversation(db):
    from agents.notebook_memory import NotebookMemory
    mem = NotebookMemory(db_path=db)
    nb_id = mem.new_notebook("Chat NB")
    mem.add_turn(nb_id, "user", "What is this about?")
    mem.add_turn(nb_id, "assistant", "It is about X.", citations=[{"n": 1}])
    history = mem.get_history(nb_id, max_turns=10)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["citations"] == [{"n": 1}]


def test_wisdom_find_related(db):
    from agents.wisdom_memory import WisdomMemory
    mem = WisdomMemory(db_path=db)
    s1 = mem.new_session(topic="Sleep and cognition")
    mem.save_wisdom(s1, "deep", "simple", ["act1"], {}, [], [], ["sleep", "cognition"])
    s2 = mem.new_session(topic="Sleep deprivation")
    mem.save_wisdom(s2, "deep2", "simple2", [], {}, [], [], ["sleep", "memory"])
    s3 = mem.new_session(topic="Exercise")
    mem.save_wisdom(s3, "deep3", "simple3", [], {}, [], [], ["exercise", "fitness"])

    # Should find s1 and s2 (both have "sleep"), not s3
    related = mem.find_related_sessions(["sleep", "focus"], current_session_id="other")
    ids = [r["session_id"] for r in related]
    assert s1 in ids
    assert s2 in ids
    assert s3 not in ids


def test_style_load_by_name(db):
    from agents.style_memory import StyleMemory
    from unittest.mock import patch, MagicMock
    mem = StyleMemory(db_path=db)

    # Directly write a profile bypassing create_profile (avoids Ollama call)
    from tools.session_db import _tx, pack
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with _tx(db) as conn:
        conn.execute(
            "INSERT INTO style_profiles VALUES (?,?,?,?,?,?,?)",
            ("sp_test", now, now, "Academic Writing", "academic writing", 1,
             pack({"profile_id": "sp_test", "name": "Academic Writing",
                   "injection_prompt": "Write academically.", "analysis": {}}))
        )

    profile = mem.load_by_name("Academic Writing")
    assert profile is not None
    assert profile["injection_prompt"] == "Write academically."

    profile_ci = mem.load_by_name("academic writing")
    assert profile_ci is not None


def test_proposal_roundtrip(db):
    from agents.memory import ProposalMemory
    mem = ProposalMemory(db_path=db)
    sid = mem.new_session(goal="Build a robot", model="llama3.1:8b")
    mem.save_proposal(sid, "# Proposal\nContent here.", [{"ref_num": 1, "apa": "Smith 2020"}],
                      title="Robot Project")
    loaded = mem.load(sid)
    assert loaded is not None
    assert loaded["proposal_markdown"] == "# Proposal\nContent here."
    assert loaded["title"] == "Robot Project"

    mem.add_revision(sid, "Make it shorter", "# Short Proposal", None)
    loaded2 = mem.load(sid)
    assert loaded2["proposal_markdown"] == "# Short Proposal"
    assert len(loaded2["revision_history"]) == 1


def test_research_session(db):
    from agents.memory import ResearchMemory
    mem = ResearchMemory(db_path=db)
    mem.save_session("sid1", "What is ML?", "Report text", [{"ref_num": 1}],
                     ["Finding 1"], ["paper.pdf"], "research", "llama3.1")
    loaded = mem.load("sid1")
    assert loaded["goal"] == "What is ML?"
    sessions = mem.list_sessions()
    assert len(sessions) == 1
