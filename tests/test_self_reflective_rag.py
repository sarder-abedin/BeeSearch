"""tests/test_self_reflective_rag.py — Unit tests for agents/self_reflective_rag.py"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from agents.self_reflective_rag import (
    grade_chunks,
    grade_papers,
    rewrite_query,
    self_reflective_retrieve,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm(content: str):
    """Return a mock ChatOllama whose .invoke() returns content."""
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def _chunks(n: int):
    return [{"chunk_id": str(i), "text": f"chunk text {i}"} for i in range(n)]


def _papers(n: int):
    return [{"title": f"Paper {i}", "abstract": f"Abstract {i}"} for i in range(n)]


# ── TestGradeChunks ────────────────────────────────────────────────────────────

class TestGradeChunks:
    @patch("agents.self_reflective_rag.ChatOllama")
    def test_all_relevant(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm('{"grades": [true, true, true]}')
        result = grade_chunks(_chunks(3), "quantum computing")
        assert result == [True, True, True]

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_mixed_relevance(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm('{"grades": [true, false, true]}')
        result = grade_chunks(_chunks(3), "quantum computing")
        assert result == [True, False, True]

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_parse_failure_returns_all_true(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm("sorry, I cannot grade these")
        result = grade_chunks(_chunks(3), "query")
        assert result == [True, True, True]

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_llm_error_returns_all_true(self, MockChatOllama):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionRefusedError("Ollama not running")
        MockChatOllama.return_value = mock_llm
        result = grade_chunks(_chunks(4), "query")
        assert result == [True, True, True, True]

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_length_mismatch_returns_all_true(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm('{"grades": [true, false]}')
        result = grade_chunks(_chunks(4), "query")
        assert result == [True, True, True, True]

    def test_empty_input_returns_empty(self):
        result = grade_chunks([], "query")
        assert result == []

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_markdown_fenced_json_parsed(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm(
            '```json\n{"grades": [true, false]}\n```'
        )
        result = grade_chunks(_chunks(2), "query")
        assert result == [True, False]


# ── TestGradePapers ────────────────────────────────────────────────────────────

class TestGradePapers:
    @patch("agents.self_reflective_rag.ChatOllama")
    def test_all_relevant(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm('{"grades": [true, true]}')
        result = grade_papers(_papers(2), "machine learning")
        assert result == [True, True]

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_mixed_relevance(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm('{"grades": [false, true, false]}')
        result = grade_papers(_papers(3), "neural networks")
        assert result == [False, True, False]

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_parse_failure_returns_all_true(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm("not a json response")
        result = grade_papers(_papers(2), "query")
        assert result == [True, True]

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_llm_error_returns_all_true(self, MockChatOllama):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("timeout")
        MockChatOllama.return_value = mock_llm
        result = grade_papers(_papers(3), "query")
        assert result == [True, True, True]

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_length_mismatch_returns_all_true(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm('{"grades": [true]}')
        result = grade_papers(_papers(3), "query")
        assert result == [True, True, True]

    def test_empty_input_returns_empty(self):
        result = grade_papers([], "query")
        assert result == []

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_markdown_fenced_json_parsed(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm(
            "```\n{\"grades\": [true, false, true]}\n```"
        )
        result = grade_papers(_papers(3), "query")
        assert result == [True, False, True]


# ── TestRewriteQuery ──────────────────────────────────────────────────────────

class TestRewriteQuery:
    @patch("agents.self_reflective_rag.ChatOllama")
    def test_success(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm("improved query for retrieval")
        result = rewrite_query("original query")
        assert result == "improved query for retrieval"

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_llm_error_returns_original(self, MockChatOllama):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionRefusedError("Ollama not running")
        MockChatOllama.return_value = mock_llm
        result = rewrite_query("original query")
        assert result == "original query"

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_empty_response_returns_original(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm("   \n\n  ")
        result = rewrite_query("original query")
        assert result == "original query"

    @patch("agents.self_reflective_rag.ChatOllama")
    def test_multiline_returns_first_line(self, MockChatOllama):
        MockChatOllama.return_value = _make_llm("first line\nsecond line\nthird line")
        result = rewrite_query("original query")
        assert result == "first line"


# ── TestSelfReflectiveRetrieve ────────────────────────────────────────────────

class TestSelfReflectiveRetrieve:
    def _make_store(self, return_value_or_sequence):
        """
        Mock store. Pass a list-of-dicts for a single call returning that list,
        or a list-of-lists for multiple successive calls (one list per call).
        """
        store = MagicMock()
        if (
            isinstance(return_value_or_sequence, list)
            and return_value_or_sequence
            and isinstance(return_value_or_sequence[0], list)
        ):
            store.search_hybrid.side_effect = return_value_or_sequence
        else:
            store.search_hybrid.return_value = return_value_or_sequence
        return store

    @patch("agents.self_reflective_rag.grade_chunks")
    def test_one_cycle_sufficient(self, mock_grade):
        """≥3 relevant chunks on cycle 1 → no cycle 2, correct metadata."""
        chunks = _chunks(5)
        store = self._make_store(chunks)
        mock_grade.return_value = [True, True, True, False, False]

        result, meta = self_reflective_retrieve(store, "query", top_k=5)
        assert len(result) == 3
        assert meta["cycles"] == 1
        assert meta["grading_skipped"] is False
        assert mock_grade.call_count == 1  # no cycle 2

    @patch("agents.self_reflective_rag.rewrite_query")
    @patch("agents.self_reflective_rag.grade_chunks")
    def test_triggers_cycle_2(self, mock_grade, mock_rewrite):
        """< min_relevant on cycle 1 → cycle 2 fires."""
        cycle1 = _chunks(3)
        cycle2 = [{"chunk_id": "99", "text": "new chunk"}]
        store = self._make_store([cycle1, cycle2])
        # cycle 1: only 1 relevant → triggers cycle 2
        # cycle 2 (new chunks only): 1 relevant
        mock_grade.side_effect = [[False, False, True], [True]]
        mock_rewrite.return_value = "rewritten query"

        result, meta = self_reflective_retrieve(store, "query", top_k=10, min_relevant=3)
        assert meta["cycles"] == 2
        assert "rewritten query" in meta["rewritten_queries"]
        assert meta["total_relevant"] >= 1

    @patch("agents.self_reflective_rag.rewrite_query")
    @patch("agents.self_reflective_rag.grade_chunks")
    def test_deduplication_across_cycles(self, mock_grade, mock_rewrite):
        """Chunks returned by cycle 2 that share chunk_id with cycle 1 are dropped."""
        cycle1 = [{"chunk_id": "a", "text": "text a"}, {"chunk_id": "b", "text": "text b"}]
        # cycle 2 returns the same chunk ids + one new one
        cycle2 = [
            {"chunk_id": "a", "text": "text a"},  # duplicate
            {"chunk_id": "c", "text": "text c"},  # new
        ]
        store = self._make_store([cycle1, cycle2])
        mock_grade.side_effect = [[True, False], [True]]  # cycle1: 1 relevant; new chunk: 1 relevant
        mock_rewrite.return_value = "rewritten"

        result, meta = self_reflective_retrieve(store, "query", top_k=10, min_relevant=3)
        result_ids = [c["chunk_id"] for c in result]
        assert "a" in result_ids
        assert result_ids.count("a") == 1  # no duplicate

    @patch("agents.self_reflective_rag.grade_chunks")
    def test_max_cycles_respected(self, mock_grade):
        """max_cycles=1 → no cycle 2 even if < min_relevant pass."""
        store = self._make_store(_chunks(2))
        mock_grade.return_value = [False, True]

        result, meta = self_reflective_retrieve(store, "query", top_k=5, max_cycles=1, min_relevant=3)
        assert meta["cycles"] == 1

    @patch("agents.self_reflective_rag.rewrite_query")
    @patch("agents.self_reflective_rag.grade_chunks")
    def test_zero_pass_returns_original_cycle1(self, mock_grade, mock_rewrite):
        """If no chunks pass grading on either cycle, original cycle-1 chunks are returned."""
        cycle1 = _chunks(3)
        cycle2 = [{"chunk_id": "99", "text": "new"}]
        store = self._make_store([cycle1, cycle2])
        mock_grade.side_effect = [[False, False, False], [False]]
        mock_rewrite.return_value = "rewritten"

        result, meta = self_reflective_retrieve(store, "query", top_k=10, min_relevant=3)
        # safety: returns original cycle-1 chunks
        assert result == cycle1

    @patch("agents.self_reflective_rag.grade_chunks")
    def test_all_true_grades_skipped(self, mock_grade):
        """All-True grades on multi-chunk list → grading_skipped=True, original returned."""
        chunks = _chunks(4)
        store = self._make_store(chunks)
        mock_grade.return_value = [True, True, True, True]

        result, meta = self_reflective_retrieve(store, "query", top_k=4)
        assert meta["grading_skipped"] is True
        assert result == chunks

    def test_search_hybrid_exception_graceful(self):
        """search_hybrid raising → returns ([], metadata) without raising."""
        store = MagicMock()
        store.search_hybrid.side_effect = RuntimeError("index not built")

        result, meta = self_reflective_retrieve(store, "query", top_k=5)
        assert result == []
        assert meta["cycles"] == 0

    @patch("agents.self_reflective_rag.grade_chunks")
    def test_metadata_keys_complete(self, mock_grade):
        """All expected metadata keys are always present."""
        store = self._make_store(_chunks(2))
        mock_grade.return_value = [True, True]

        _, meta = self_reflective_retrieve(store, "query", top_k=5)
        assert "cycles" in meta
        assert "total_retrieved" in meta
        assert "total_relevant" in meta
        assert "rewritten_queries" in meta
        assert "grading_skipped" in meta


# ── TestStateFields ───────────────────────────────────────────────────────────

class TestStateFields:
    def test_research_state_has_rag_reflection_info(self):
        from agents.state import ResearchState
        assert "rag_reflection_info" in ResearchState.__annotations__

    def test_notebook_state_has_rag_reflection_info(self):
        from agents.notebook_state import NotebookState
        assert "rag_reflection_info" in NotebookState.__annotations__

    def test_notebook_pipeline_state_has_rag_reflection_info(self):
        from agents.notebook_pipeline_state import NotebookPipelineState
        assert "rag_reflection_info" in NotebookPipelineState.__annotations__

    def test_proposal_gpt_state_has_rag_reflection_info(self):
        from agents.proposal_gpt_state import ProposalGPTState
        assert "rag_reflection_info" in ProposalGPTState.__annotations__

    def test_wisdom_state_has_rag_reflection_info(self):
        from agents.wisdom_state import WisdomState
        assert "rag_reflection_info" in WisdomState.__annotations__

    def test_systematic_review_state_has_rag_reflection_info(self):
        from agents.systematic_review_state import SystematicReviewState
        assert "rag_reflection_info" in SystematicReviewState.__annotations__

    def test_research_state_factory_default(self):
        from agents.state import create_initial_state
        s = create_initial_state("goal")
        assert s["rag_reflection_info"] == []

    def test_notebook_state_factory_default(self):
        from agents.notebook_state import create_notebook_state
        s = create_notebook_state("msg", "nb1")
        assert s["rag_reflection_info"] == {}

    def test_notebook_pipeline_state_factory_default(self):
        from agents.notebook_pipeline_state import create_pipeline_state
        s = create_pipeline_state("nb1", {})
        assert s["rag_reflection_info"] == {}

    def test_proposal_gpt_state_factory_default(self):
        from agents.proposal_gpt_state import create_proposal_gpt_state
        s = create_proposal_gpt_state("call text")
        assert s["rag_reflection_info"] == {}

    def test_wisdom_state_factory_default(self):
        from agents.wisdom_state import create_wisdom_state
        s = create_wisdom_state("msg", "sid")
        assert s["rag_reflection_info"] == {}

    def test_systematic_review_state_factory_default(self):
        from agents.systematic_review_state import create_systematic_review_state
        s = create_systematic_review_state("question")
        assert s["rag_reflection_info"] == {}
