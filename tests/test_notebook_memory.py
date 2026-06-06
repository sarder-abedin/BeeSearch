"""tests/test_notebook_memory.py — Unit tests for agents/notebook_memory.py"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest

from agents.notebook_memory import NotebookMemory


# ── Lightweight stand-ins for ProcessedDocument / DocumentChunk ─────────────────

@dataclass
class _Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_num: int
    chunk_index: int
    text: str


@dataclass
class _Doc:
    doc_id: str
    filename: str
    file_type: str
    total_pages: int
    total_chunks: int
    content_md5: str
    chunks: List[_Chunk] = field(default_factory=list)


def _make_doc(doc_id="d1", filename="paper.pdf", n_chunks=3) -> _Doc:
    chunks = [
        _Chunk(
            chunk_id=f"{doc_id}_c{i}",
            doc_id=doc_id,
            doc_name=filename,
            page_num=i,
            chunk_index=i,
            text=f"chunk {i} of {filename}",
        )
        for i in range(n_chunks)
    ]
    return _Doc(
        doc_id=doc_id,
        filename=filename,
        file_type="PDF",
        total_pages=n_chunks,
        total_chunks=n_chunks,
        content_md5=f"md5_{doc_id}",
        chunks=chunks,
    )


@pytest.fixture
def mem(tmp_path):
    return NotebookMemory(db_path=tmp_path / "sessions.db")


# ── Notebook lifecycle ──────────────────────────────────────────────────────────

class TestNotebookLifecycle:
    def test_new_notebook_returns_id(self, mem):
        nb_id = mem.new_notebook("My Notebook")
        assert isinstance(nb_id, str) and nb_id

    def test_new_notebook_persists(self, mem):
        nb_id = mem.new_notebook("My Notebook")
        data = mem.load(nb_id)
        assert data is not None
        assert data["name"] == "My Notebook"
        assert data["sources"] == []
        assert data["chunks"] == []
        assert data["conversation"] == []

    def test_blank_name_defaults(self, mem):
        nb_id = mem.new_notebook("   ")
        assert mem.load(nb_id)["name"] == "Untitled Notebook"

    def test_load_missing_returns_none(self, mem):
        assert mem.load("does_not_exist") is None

    def test_list_notebooks_newest_first(self, mem):
        a = mem.new_notebook("A")
        b = mem.new_notebook("B")
        ids = [n["notebook_id"] for n in mem.list_notebooks()]
        assert set(ids) == {a, b}

    def test_list_includes_counts(self, mem):
        nb_id = mem.new_notebook("Counts")
        mem.add_source(nb_id, _make_doc())
        mem.add_turn(nb_id, "user", "hi")
        summary = next(n for n in mem.list_notebooks() if n["notebook_id"] == nb_id)
        assert summary["source_count"] == 1
        assert summary["turn_count"] == 1

    def test_rename(self, mem):
        nb_id = mem.new_notebook("Old")
        assert mem.rename(nb_id, "New") is True
        assert mem.load(nb_id)["name"] == "New"

    def test_rename_missing_returns_false(self, mem):
        assert mem.rename("nope", "X") is False

    def test_delete(self, mem):
        nb_id = mem.new_notebook("Temp")
        assert mem.delete(nb_id) is True
        assert mem.load(nb_id) is None

    def test_delete_missing_returns_false(self, mem):
        assert mem.delete("nope") is False


# ── Source management ───────────────────────────────────────────────────────────

class TestSources:
    def test_add_source_stores_metadata_and_chunks(self, mem):
        nb_id = mem.new_notebook("S")
        assert mem.add_source(nb_id, _make_doc(n_chunks=3)) is True
        data = mem.load(nb_id)
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "paper.pdf"
        assert len(data["chunks"]) == 3
        assert data["chunks"][0]["text"] == "chunk 0 of paper.pdf"

    def test_add_source_to_missing_notebook_returns_false(self, mem):
        assert mem.add_source("nope", _make_doc()) is False

    def test_duplicate_source_skipped(self, mem):
        nb_id = mem.new_notebook("Dup")
        doc = _make_doc(doc_id="same")
        assert mem.add_source(nb_id, doc) is True
        assert mem.add_source(nb_id, doc) is False  # same doc_id
        assert len(mem.load(nb_id)["sources"]) == 1

    def test_add_source_url_type(self, mem):
        nb_id = mem.new_notebook("U")
        mem.add_source(nb_id, _make_doc(doc_id="web1", filename="example.com"),
                       source_type="url", url="https://example.com")
        s = mem.load(nb_id)["sources"][0]
        assert s["source_type"] == "url"
        assert s["url"] == "https://example.com"

    def test_remove_source_drops_chunks(self, mem):
        nb_id = mem.new_notebook("R")
        mem.add_source(nb_id, _make_doc(doc_id="a", n_chunks=2))
        mem.add_source(nb_id, _make_doc(doc_id="b", filename="b.pdf", n_chunks=3))
        assert mem.remove_source(nb_id, "a") is True
        data = mem.load(nb_id)
        assert [s["doc_id"] for s in data["sources"]] == ["b"]
        assert all(c["doc_id"] == "b" for c in data["chunks"])
        assert len(data["chunks"]) == 3

    def test_remove_missing_source_returns_false(self, mem):
        nb_id = mem.new_notebook("R")
        assert mem.remove_source(nb_id, "ghost") is False


# ── Conversation ────────────────────────────────────────────────────────────────

class TestConversation:
    def test_add_turn_appends(self, mem):
        nb_id = mem.new_notebook("C")
        mem.add_turn(nb_id, "user", "Question?")
        mem.add_turn(nb_id, "assistant", "Answer [1].",
                     citations=[{"n": 1, "doc_name": "p.pdf", "page": 2}],
                     suggested_questions=["Next?"])
        conv = mem.load(nb_id)["conversation"]
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["citations"][0]["doc_name"] == "p.pdf"
        assert conv[1]["suggested_questions"] == ["Next?"]

    def test_add_turn_missing_notebook_noop(self, mem):
        mem.add_turn("nope", "user", "x")  # must not raise

    def test_get_history_limits_turns(self, mem):
        nb_id = mem.new_notebook("H")
        for i in range(12):
            mem.add_turn(nb_id, "user", f"msg {i}")
        history = mem.get_history(nb_id, max_turns=8)
        assert len(history) == 8
        assert history[-1]["content"] == "msg 11"

    def test_get_history_missing_returns_empty(self, mem):
        assert mem.get_history("nope") == []
