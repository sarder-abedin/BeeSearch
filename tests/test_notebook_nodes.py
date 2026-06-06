"""tests/test_notebook_nodes.py
─────────────────────────────────
Unit + integration tests for the Research Notebook (Mode 8) nodes and graph.

Covers:
  - _rebuild_docs_from_chunks: reconstruction from stored chunks
  - _build_context_block / _split_suggested_questions / _extract_citations
  - retrieve_node: empty notebook, hybrid path, fallback on embed failure
  - answer_node: no sources, no chunks, grounded answer with citations
  - full graph run via run_notebook_turn: response + citations + memory save
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import agents.notebook_nodes as nb_nodes
from agents.notebook_graph import run_notebook_turn
from agents.notebook_memory import NotebookMemory
from agents.notebook_nodes import (
    _build_context_block,
    _extract_citations,
    _rebuild_docs_from_chunks,
    _split_suggested_questions,
    answer_node,
    retrieve_node,
)
from agents.notebook_state import create_notebook_state


# ── Fixtures ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    return NotebookMemory(db_path=tmp_path / "sessions.db")


def _seed_notebook(mem, n_sources=1, n_chunks=3):
    """Create a notebook and add mock sources + chunks via add_source."""
    from unittest.mock import MagicMock
    nb_id = mem.new_notebook("Test Notebook")
    for s in range(n_sources):
        doc_id = f"doc{s}"
        fname = f"file{s}.pdf"
        chunks = []
        for c in range(n_chunks):
            ch = MagicMock()
            ch.chunk_id = f"{doc_id}_c{c}"
            ch.doc_id = doc_id
            ch.doc_name = fname
            ch.page_num = c + 1
            ch.chunk_index = c
            ch.text = f"content {c} from {fname}"
            chunks.append(ch)
        doc = MagicMock()
        doc.doc_id = doc_id
        doc.filename = fname
        doc.file_type = "PDF"
        doc.total_pages = n_chunks
        doc.total_chunks = n_chunks
        doc.content_md5 = f"md5_{doc_id}"
        doc.chunks = chunks
        mem.add_source(nb_id, doc)
    return nb_id


# ── Pure helpers ───────────────────────────────────────────────────────────────

class TestPureHelpers:
    def test_rebuild_docs_groups_by_document(self, mem):
        nb_id = _seed_notebook(mem, n_sources=2, n_chunks=3)
        docs = _rebuild_docs_from_chunks(mem.load(nb_id))
        assert len(docs) == 2
        for d in docs:
            assert len(d.chunks) == 3
            assert d.content_md5 == f"md5_{d.doc_id}"
            assert d.chunks[0].chunk_id.startswith(d.doc_id)

    def test_build_context_block_numbers_sources(self):
        chunks = [
            {"doc_name": "a.pdf", "page_num": 2, "text": "alpha"},
            {"doc_name": "b.pdf", "page_num": 5, "text": "beta"},
        ]
        block = _build_context_block(chunks)
        assert "[1]" in block and "[2]" in block
        assert "a.pdf" in block and "p.2" in block
        assert "alpha" in block and "beta" in block

    def test_split_suggested_questions_extracts_tail(self):
        raw = 'The answer is X [1].\n\n{"suggested_questions": ["Q1?", "Q2?"]}'
        body, qs = _split_suggested_questions(raw)
        assert body == "The answer is X [1]."
        assert qs == ["Q1?", "Q2?"]

    def test_split_suggested_questions_none_present(self):
        raw = "Just an answer with no JSON."
        body, qs = _split_suggested_questions(raw)
        assert body == raw
        assert qs == []

    def test_split_suggested_questions_malformed_json_recovered(self):
        raw = 'Answer.\n\n{"suggested_questions": ["Q1?", "Q2?"] trailing junk'
        _, qs = _split_suggested_questions(raw)
        assert qs == ["Q1?", "Q2?"]

    def test_extract_citations_only_cited_numbers(self):
        chunks = [
            {"doc_name": "a.pdf", "page_num": 1, "text": "alpha text"},
            {"doc_name": "b.pdf", "page_num": 2, "text": "beta text"},
            {"doc_name": "c.pdf", "page_num": 3, "text": "gamma text"},
        ]
        answer = "Claim one [1] and claim three [3]."
        cites = _extract_citations(answer, chunks)
        assert [c["n"] for c in cites] == [1, 3]
        assert cites[0]["doc_name"] == "a.pdf"
        assert cites[1]["page"] == 3

    def test_extract_citations_out_of_range_ignored(self):
        chunks = [{"doc_name": "a.pdf", "page_num": 1, "text": "x"}]
        cites = _extract_citations("Bogus [5] reference.", chunks)
        assert cites == []


# ── retrieve_node ────────────────────────────────────────────────────────────────

class TestRetrieveNode:
    def test_missing_notebook_records_error(self, mem, monkeypatch):
        monkeypatch.setattr(nb_nodes, "_memory", mem)
        state = create_notebook_state("q", notebook_id="ghost")
        out = retrieve_node(state)
        assert out["retrieval_mode"] == "empty"
        assert any("not found" in e for e in out["errors"])

    def test_empty_notebook_returns_no_chunks(self, mem, monkeypatch):
        monkeypatch.setattr(nb_nodes, "_memory", mem)
        nb_id = mem.new_notebook("Empty")
        out = retrieve_node(create_notebook_state("q", notebook_id=nb_id))
        assert out["retrieved_chunks"] == []
        assert out["retrieval_mode"] == "empty"
        assert out["source_count"] == 0

    def test_hybrid_path_uses_store(self, mem, monkeypatch):
        monkeypatch.setattr(nb_nodes, "_memory", mem)
        nb_id = _seed_notebook(mem, n_sources=1, n_chunks=3)

        fake_chunks = [
            {"chunk_id": "doc0_c1", "doc_name": "file0.pdf", "page_num": 2,
             "chunk_index": 1, "text": "content 1 from file0.pdf"}
        ]
        fake_store = MagicMock()
        fake_store.is_indexed.return_value = True
        fake_store.search_hybrid.return_value = fake_chunks
        # react_retrieve is imported inside retrieve_node; mock at source to avoid Ollama call
        mock_react = MagicMock(return_value=(fake_chunks, {}))
        with patch("agents.notebook_nodes.get_or_create_store", return_value=fake_store), \
             patch("agents.self_reflective_rag.react_retrieve", mock_react):
            out = retrieve_node(create_notebook_state("question", notebook_id=nb_id, top_k=4))

        # retrieval_mode is "react" when FAISS index present, "bm25" otherwise
        assert out["retrieval_mode"] in ("react", "bm25")
        assert len(out["retrieved_chunks"]) >= 1
        # react_retrieve must have been called with the mocked store
        mock_react.assert_called_once()
        assert mock_react.call_args[0][0] is fake_store

    def test_rebuilds_index_when_not_indexed(self, mem, monkeypatch):
        monkeypatch.setattr(nb_nodes, "_memory", mem)
        nb_id = _seed_notebook(mem, n_sources=1, n_chunks=2)

        fake_store = MagicMock()
        fake_store.is_indexed.return_value = False
        fake_store.search_hybrid.return_value = []
        with patch("agents.notebook_nodes.get_or_create_store", return_value=fake_store):
            retrieve_node(create_notebook_state("q", notebook_id=nb_id))

        fake_store.add_documents.assert_called_once()
        # add_documents should receive rebuilt docs reconstructed from chunks
        docs_arg = fake_store.add_documents.call_args[0][0]
        assert len(docs_arg) == 1
        assert len(docs_arg[0].chunks) == 2

    def test_fallback_on_embedding_runtimeerror(self, mem, monkeypatch):
        monkeypatch.setattr(nb_nodes, "_memory", mem)
        nb_id = _seed_notebook(mem, n_sources=1, n_chunks=5)

        fake_store = MagicMock()
        fake_store.is_indexed.return_value = True
        fake_store.search_hybrid.side_effect = RuntimeError("model not pulled")
        with patch("agents.notebook_nodes.get_or_create_store", return_value=fake_store):
            out = retrieve_node(create_notebook_state("q", notebook_id=nb_id, top_k=3))

        assert out["retrieval_mode"] == "fallback"
        assert len(out["retrieved_chunks"]) == 3   # first k stored chunks
        assert any("ollama pull" in e.lower() for e in out["errors"])


# ── answer_node ──────────────────────────────────────────────────────────────────

class TestAnswerNode:
    def test_no_sources_message_no_llm(self):
        state = create_notebook_state("q", notebook_id="x")
        state["source_count"] = 0
        state["retrieved_chunks"] = []
        out = answer_node(state)
        assert "no sources" in out["assistant_response"].lower()
        assert out["citations"] == []

    def test_sources_but_no_chunks_message(self):
        state = create_notebook_state("q", notebook_id="x")
        state["source_count"] = 2
        state["retrieved_chunks"] = []
        out = answer_node(state)
        assert "couldn't find" in out["assistant_response"].lower()

    def test_grounded_answer_with_citations(self):
        state = create_notebook_state("What method is used?", notebook_id="x")
        state["source_count"] = 1
        state["retrieved_chunks"] = [
            {"doc_name": "paper.pdf", "page_num": 3, "text": "We use a CNN classifier."},
            {"doc_name": "paper.pdf", "page_num": 4, "text": "Trained with Adam."},
        ]

        fake = MagicMock()
        resp = MagicMock()
        resp.content = ('The study uses a CNN [1] trained with Adam [2].\n\n'
                        '{"suggested_questions": ["What dataset?", "What accuracy?"]}')
        fake.invoke.return_value = resp

        with patch("agents.notebook_nodes.ChatOllama", return_value=fake):
            out = answer_node(state)

        assert "CNN" in out["assistant_response"]
        assert "suggested_questions" not in out["assistant_response"]
        assert [c["n"] for c in out["citations"]] == [1, 2]
        assert out["suggested_questions"] == ["What dataset?", "What accuracy?"]

    def test_llm_exception_returns_error_text(self):
        state = create_notebook_state("q", notebook_id="x")
        state["source_count"] = 1
        state["retrieved_chunks"] = [{"doc_name": "a.pdf", "page_num": 1, "text": "x"}]

        fake = MagicMock()
        fake.invoke.side_effect = RuntimeError("connection refused")
        with patch("agents.notebook_nodes.ChatOllama", return_value=fake):
            out = answer_node(state)
        assert "Error generating answer" in out["assistant_response"]
        assert out["citations"] == []


# ── Full graph ───────────────────────────────────────────────────────────────────

class TestNotebookGraphSmoke:
    def test_full_run_saves_two_turns(self, mem, monkeypatch):
        monkeypatch.setattr(nb_nodes, "_memory", mem)
        nb_id = _seed_notebook(mem, n_sources=1, n_chunks=3)

        fake_store = MagicMock()
        fake_store.is_indexed.return_value = True
        fake_store.search_hybrid.return_value = [
            {"chunk_id": "doc0_c0", "doc_name": "file0.pdf", "page_num": 1,
             "chunk_index": 0, "text": "content 0 from file0.pdf"},
        ]

        fake_llm = MagicMock()
        resp = MagicMock()
        resp.content = ('The source says content zero [1].\n\n'
                        '{"suggested_questions": ["More?"]}')
        fake_llm.invoke.return_value = resp

        with patch("agents.notebook_nodes.get_or_create_store", return_value=fake_store), \
             patch("agents.notebook_nodes.ChatOllama", return_value=fake_llm):
            result = run_notebook_turn(
                create_notebook_state("What does it say?", notebook_id=nb_id)
            )

        assert result["assistant_response"] != ""
        assert result["citations"][0]["doc_name"] == "file0.pdf"
        assert "retrieve" in result["completed_steps"]
        assert "answer" in result["completed_steps"]
        assert "save" in result["completed_steps"]

        conv = mem.load(nb_id)["conversation"]
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"
        assert conv[1]["citations"][0]["page"] == 1

    def test_full_run_empty_notebook_prompts_for_sources(self, mem, monkeypatch):
        monkeypatch.setattr(nb_nodes, "_memory", mem)
        nb_id = mem.new_notebook("Empty")

        # No store/LLM should be needed, but patch to be safe.
        with patch("agents.notebook_nodes.get_or_create_store") as mock_store:
            result = run_notebook_turn(
                create_notebook_state("anything", notebook_id=nb_id)
            )
            mock_store.assert_not_called()  # empty notebook short-circuits before store

        assert "no sources" in result["assistant_response"].lower()
