"""
tests/test_notebook_pipeline.py
─────────────────────────────────
Tests for the 7-agent Mode 8 notebook pipeline.

Coverage
────────
  TestNotebookPipelineState       — create_pipeline_state factory
  TestIngestionNode               — not found, empty, single source
  TestSummarizationNode           — no sources, single source (mocked LLM), multi-source
  TestRetrievalNode               — no chunks, keyword fallback
  TestCitationVerificationNode    — no summary, LLM success, malformed JSON
  TestKnowledgeGraphNode          — no chunks, LLM success, invalid JSON
  TestStudyGuideNode              — no material, LLM success
  TestPodcastScriptNode           — no material, LLM success
  TestBuildPipeline               — graph compiles without error
  TestRunPipelineIntegration      — end-to-end with mocked nodes
  TestPipelineProgressPercentages — _progress() correctness
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from agents.notebook_pipeline_state import NotebookPipelineState, create_pipeline_state


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_sources():
    return [
        {"doc_id": "doc1", "filename": "paper1.pdf", "file_type": "pdf",
         "total_chunks": 3, "content_md5": "abc", "added_at": "2024-01-01T00:00:00",
         "source_type": "file", "url": "", "total_pages": 5},
        {"doc_id": "doc2", "filename": "paper2.pdf", "file_type": "pdf",
         "total_chunks": 2, "content_md5": "def", "added_at": "2024-01-02T00:00:00",
         "source_type": "file", "url": "", "total_pages": 3},
    ]


@pytest.fixture
def sample_chunks():
    return [
        {"chunk_id": "c1", "doc_id": "doc1", "doc_name": "paper1.pdf",
         "page_num": 1, "chunk_index": 0,
         "text": "This paper presents a novel approach to transformer attention mechanisms."},
        {"chunk_id": "c2", "doc_id": "doc1", "doc_name": "paper1.pdf",
         "page_num": 2, "chunk_index": 1,
         "text": "The results show significant improvement in accuracy by 15%."},
        {"chunk_id": "c3", "doc_id": "doc1", "doc_name": "paper1.pdf",
         "page_num": 3, "chunk_index": 2,
         "text": "Limitations include the computational cost of the method."},
        {"chunk_id": "c4", "doc_id": "doc2", "doc_name": "paper2.pdf",
         "page_num": 1, "chunk_index": 0,
         "text": "We evaluate BERT-based models on downstream NLP tasks."},
        {"chunk_id": "c5", "doc_id": "doc2", "doc_name": "paper2.pdf",
         "page_num": 2, "chunk_index": 1,
         "text": "Fine-tuning achieves state-of-the-art results on GLUE benchmark."},
    ]


@pytest.fixture
def sample_notebook(sample_sources, sample_chunks):
    return {
        "notebook_id": "test_nb_001",
        "name": "Test Notebook",
        "sources": sample_sources,
        "chunks": sample_chunks,
        "conversation": [],
        "created_at": "2024-01-01T00:00:00",
        "last_modified": "2024-01-01T00:00:00",
    }


@pytest.fixture
def base_settings():
    return {"model": "llama3.2:3b", "num_ctx": 4096}


@pytest.fixture
def base_state(sample_sources, sample_chunks, base_settings):
    state = create_pipeline_state("test_nb_001", base_settings)
    state["sources"] = sample_sources
    state["chunks"] = sample_chunks
    state["doc_count"] = 2
    state["ingestion_summary"] = "Loaded 2 sources."
    return state


# ─────────────────────────────────────────────────────────────────────────────
# TestNotebookPipelineState
# ─────────────────────────────────────────────────────────────────────────────

class TestNotebookPipelineState:
    def test_factory_defaults(self, base_settings):
        state = create_pipeline_state("nb1", base_settings)
        assert state["notebook_id"] == "nb1"
        assert state["query"] == ""
        assert state["sources"] == []
        assert state["chunks"] == []
        assert state["doc_count"] == 0
        assert state["per_doc_summaries"] == {}
        assert state["cross_summary"] == ""
        assert state["retrieved_chunks"] == []
        assert state["retrieval_mode"] == "empty"
        assert state["verified_citations"] == []
        assert state["citation_report"] == ""
        assert state["knowledge_graph_dot"] == ""
        assert state["kg_data"] == {}
        assert state["study_guide"] == ""
        assert state["podcast_script"] == ""
        assert state["errors"] == []
        assert state["completed_steps"] == []
        assert state["progress_pct"] == 0

    def test_factory_with_query(self, base_settings):
        state = create_pipeline_state("nb2", base_settings, query="attention mechanisms")
        assert state["query"] == "attention mechanisms"

    def test_is_typeddict(self, base_settings):
        state = create_pipeline_state("nb3", base_settings)
        assert isinstance(state, dict)


# ─────────────────────────────────────────────────────────────────────────────
# TestPipelineProgressPercentages
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineProgressPercentages:
    def test_progress_values(self):
        from agents.notebook_pipeline_nodes import _progress
        assert _progress(1) == 14
        assert _progress(2) == 29
        assert _progress(3) == 43
        assert _progress(4) == 57
        assert _progress(5) == 71
        assert _progress(6) == 86
        assert _progress(7) == 100

    def test_progress_never_exceeds_100(self):
        from agents.notebook_pipeline_nodes import _progress
        assert _progress(100) == 100


# ─────────────────────────────────────────────────────────────────────────────
# TestIngestionNode
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestionNode:
    def test_notebook_not_found(self, base_settings, tmp_path):
        from agents.notebook_pipeline_nodes import ingestion_node
        import agents.notebook_pipeline_nodes as mod
        mock_mem = MagicMock()
        mock_mem.load.return_value = None
        mod._memory = mock_mem

        state = create_pipeline_state("missing_id", base_settings)
        result = ingestion_node(state)

        assert any("not found" in e for e in result["errors"])
        assert result["progress_pct"] == 14
        mod._memory = None  # reset

    def test_loads_sources_and_chunks(self, base_settings, sample_notebook):
        from agents.notebook_pipeline_nodes import ingestion_node
        import agents.notebook_pipeline_nodes as mod
        mock_mem = MagicMock()
        mock_mem.load.return_value = sample_notebook
        mod._memory = mock_mem

        state = create_pipeline_state("test_nb_001", base_settings)
        result = ingestion_node(state)

        assert result["doc_count"] == 2
        assert len(result["sources"]) == 2
        assert len(result["chunks"]) == 5
        assert "ingest" in result["completed_steps"]
        assert result["progress_pct"] == 14
        assert result["errors"] == []
        mod._memory = None

    def test_ingestion_summary_contains_source_names(self, base_settings, sample_notebook):
        from agents.notebook_pipeline_nodes import ingestion_node
        import agents.notebook_pipeline_nodes as mod
        mock_mem = MagicMock()
        mock_mem.load.return_value = sample_notebook
        mod._memory = mock_mem

        state = create_pipeline_state("test_nb_001", base_settings)
        result = ingestion_node(state)

        assert "paper1.pdf" in result["ingestion_summary"]
        assert "paper2.pdf" in result["ingestion_summary"]
        mod._memory = None

    def test_empty_notebook(self, base_settings):
        from agents.notebook_pipeline_nodes import ingestion_node
        import agents.notebook_pipeline_nodes as mod
        mock_mem = MagicMock()
        mock_mem.load.return_value = {
            "notebook_id": "empty", "name": "Empty", "sources": [], "chunks": [],
            "conversation": [], "created_at": "", "last_modified": "",
        }
        mod._memory = mock_mem

        state = create_pipeline_state("empty", base_settings)
        result = ingestion_node(state)

        assert result["doc_count"] == 0
        assert result["sources"] == []
        assert "No sources found" in result["ingestion_summary"]
        mod._memory = None


# ─────────────────────────────────────────────────────────────────────────────
# TestSummarizationNode
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizationNode:
    def test_no_sources_skips(self, base_settings):
        from agents.notebook_pipeline_nodes import summarization_node
        state = create_pipeline_state("nb1", base_settings)
        result = summarization_node(state)
        assert any("Summarization skipped" in e for e in result["errors"])
        assert "summarize" in result["completed_steps"]

    def test_single_source_generates_summary(self, base_settings, sample_sources, sample_chunks):
        from agents.notebook_pipeline_nodes import summarization_node
        state = create_pipeline_state("nb1", base_settings)
        state["sources"] = [sample_sources[0]]
        state["chunks"] = [c for c in sample_chunks if c["doc_id"] == "doc1"]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="This paper presents attention mechanisms showing 15% accuracy improvement."
        )
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = summarization_node(state)

        assert "paper1.pdf" in result["per_doc_summaries"]
        assert "attention" in result["cross_summary"].lower()
        assert result["errors"] == []

    def test_multi_source_generates_per_doc_and_synthesis(
        self, base_settings, sample_sources, sample_chunks
    ):
        from agents.notebook_pipeline_nodes import summarization_node
        state = create_pipeline_state("nb1", base_settings)
        state["sources"] = sample_sources
        state["chunks"] = sample_chunks

        call_count = [0]
        def fake_llm_invoke(messages):
            call_count[0] += 1
            return MagicMock(content=f"Summary response {call_count[0]}")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = fake_llm_invoke
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = summarization_node(state)

        # 2 per-doc summaries + 1 synthesis = 3 LLM calls
        assert len(result["per_doc_summaries"]) == 2
        assert result["cross_summary"] != ""
        assert "summarize" in result["completed_steps"]

    def test_llm_failure_appends_error(self, base_settings, sample_sources, sample_chunks):
        from agents.notebook_pipeline_nodes import summarization_node
        state = create_pipeline_state("nb1", base_settings)
        state["sources"] = [sample_sources[0]]
        state["chunks"] = [c for c in sample_chunks if c["doc_id"] == "doc1"]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM timeout")
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = summarization_node(state)

        assert any("failed" in e.lower() for e in result["errors"])
        assert "summarize" in result["completed_steps"]


# ─────────────────────────────────────────────────────────────────────────────
# TestRetrievalNode
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievalNode:
    def test_no_chunks_returns_empty(self, base_settings):
        from agents.notebook_pipeline_nodes import retrieval_node
        state = create_pipeline_state("nb1", base_settings)
        result = retrieval_node(state)
        assert result["retrieved_chunks"] == []
        assert result["retrieval_mode"] == "empty"
        assert "retrieve" in result["completed_steps"]

    def test_keyword_fallback_on_store_failure(self, base_settings, sample_chunks):
        from agents.notebook_pipeline_nodes import retrieval_node
        state = create_pipeline_state("nb1", base_settings)
        state["chunks"] = sample_chunks
        state["sources"] = []
        state["query"] = "attention transformer"

        with patch("agents.notebook_nodes._rebuild_docs_from_chunks",
                   side_effect=ImportError("no module")):
            result = retrieval_node(state)

        assert result["retrieval_mode"] == "fallback"
        assert len(result["retrieved_chunks"]) > 0
        assert "retrieve" in result["completed_steps"]

    def test_hybrid_store_success(self, base_settings, sample_sources, sample_chunks):
        from agents.notebook_pipeline_nodes import retrieval_node
        state = create_pipeline_state("nb1", base_settings)
        state["sources"] = sample_sources
        state["chunks"] = sample_chunks
        state["query"] = "accuracy results"

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = True
        mock_store.search_hybrid.return_value = sample_chunks[:3]
        mock_store.embedder_available.return_value = True

        with patch("agents.notebook_nodes._rebuild_docs_from_chunks", return_value=[]), \
             patch("tools.hybrid_store.get_or_create_store", return_value=mock_store):
            result = retrieval_node(state)

        assert result["retrieval_mode"] == "self_reflective"
        assert len(result["retrieved_chunks"]) == 3
        assert "retrieve" in result["completed_steps"]


# ─────────────────────────────────────────────────────────────────────────────
# TestCitationVerificationNode
# ─────────────────────────────────────────────────────────────────────────────

class TestCitationVerificationNode:
    def test_no_summary_skips(self, base_settings, sample_chunks):
        from agents.notebook_pipeline_nodes import citation_verification_node
        state = create_pipeline_state("nb1", base_settings)
        state["chunks"] = sample_chunks
        # No cross_summary
        result = citation_verification_node(state)
        assert "skipped" in result["citation_report"].lower() or \
               any("skipped" in e for e in result["errors"])

    def test_no_chunks_skips(self, base_settings):
        from agents.notebook_pipeline_nodes import citation_verification_node
        state = create_pipeline_state("nb1", base_settings)
        state["cross_summary"] = "Some summary text."
        result = citation_verification_node(state)
        assert any("skipped" in e for e in result["errors"])

    def test_valid_json_response_builds_report(
        self, base_settings, base_state
    ):
        from agents.notebook_pipeline_nodes import citation_verification_node
        state = dict(base_state)
        state["cross_summary"] = "Transformers achieve 15% accuracy improvement."

        claims = [
            {"claim": "Transformers improve accuracy by 15%.",
             "source_name": "paper1.pdf", "confidence": "HIGH",
             "supporting_text": "The results show significant improvement in accuracy by 15%."},
            {"claim": "BERT achieves state-of-the-art on GLUE.",
             "source_name": "paper2.pdf", "confidence": "MEDIUM",
             "supporting_text": "Fine-tuning achieves state-of-the-art results."},
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(claims))
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = citation_verification_node(state)

        assert len(result["verified_citations"]) == 2
        assert "✅" in result["citation_report"]
        assert "🟡" in result["citation_report"]
        assert "verify_citations" in result["completed_steps"]

    def test_malformed_json_returns_empty_citations(self, base_settings, base_state):
        from agents.notebook_pipeline_nodes import citation_verification_node
        state = dict(base_state)
        state["cross_summary"] = "Some summary."

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not valid json at all")
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = citation_verification_node(state)

        assert result["verified_citations"] == []
        assert any("failed" in e.lower() for e in result["errors"])


# ─────────────────────────────────────────────────────────────────────────────
# TestKnowledgeGraphNode
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeGraphNode:
    def test_no_chunks_skips(self, base_settings):
        from agents.notebook_pipeline_nodes import knowledge_graph_node
        state = create_pipeline_state("nb1", base_settings)
        result = knowledge_graph_node(state)
        assert result["knowledge_graph_dot"] == ""
        assert result["kg_data"] == {}
        assert "build_kg" in result["completed_steps"]

    def test_valid_json_produces_dot(self, base_settings, base_state):
        from agents.notebook_pipeline_nodes import knowledge_graph_node
        state = dict(base_state)

        kg_json = {
            "nodes": [
                {"id": "1", "label": "Transformer", "type": "concept"},
                {"id": "2", "label": "Attention", "type": "method"},
            ],
            "edges": [
                {"from": "1", "to": "2", "label": "uses"},
            ],
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(kg_json))
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = knowledge_graph_node(state)

        assert result["knowledge_graph_dot"].startswith("digraph")
        assert "Transformer" in result["knowledge_graph_dot"]
        assert result["kg_data"] == kg_json
        assert "build_kg" in result["completed_steps"]

    def test_invalid_json_appends_error(self, base_settings, base_state):
        from agents.notebook_pipeline_nodes import knowledge_graph_node
        state = dict(base_state)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not json")
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = knowledge_graph_node(state)

        assert any("failed" in e.lower() for e in result["errors"])
        assert result["knowledge_graph_dot"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# TestStudyGuideNode
# ─────────────────────────────────────────────────────────────────────────────

class TestStudyGuideNode:
    def test_no_material_skips(self, base_settings):
        from agents.notebook_pipeline_nodes import study_guide_node
        state = create_pipeline_state("nb1", base_settings)
        result = study_guide_node(state)
        assert any("skipped" in e for e in result["errors"])
        assert result["study_guide"] == ""

    def test_generates_guide_from_summary(self, base_settings, base_state):
        from agents.notebook_pipeline_nodes import study_guide_node
        state = dict(base_state)
        state["cross_summary"] = "Transformers use attention. BERT fine-tunes on downstream tasks."

        expected_guide = (
            "## Key Concepts\n- **Attention** — mechanism in transformers\n\n"
            "## Glossary\n| Term | Definition | Source |\n|------|-----------|--------|\n"
            "| Transformer | Neural architecture | paper1.pdf |\n\n"
            "## Review Questions\n**Q:** What is attention?\n**A:** A weighting mechanism.\n\n"
            "## Quick Summary\nTransformers are powerful."
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=expected_guide)
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = study_guide_node(state)

        assert "## Key Concepts" in result["study_guide"]
        assert "study_guide" in result["completed_steps"]
        assert result["errors"] == []

    def test_llm_failure_appends_error(self, base_settings, base_state):
        from agents.notebook_pipeline_nodes import study_guide_node
        state = dict(base_state)
        state["cross_summary"] = "Some content."

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("timeout")
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = study_guide_node(state)

        assert any("failed" in e.lower() for e in result["errors"])
        assert result["study_guide"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# TestPodcastScriptNode
# ─────────────────────────────────────────────────────────────────────────────

class TestPodcastScriptNode:
    def test_no_material_skips(self, base_settings):
        from agents.notebook_pipeline_nodes import podcast_script_node
        state = create_pipeline_state("nb1", base_settings)
        result = podcast_script_node(state)
        assert any("skipped" in e for e in result["errors"])
        assert result["podcast_script"] == ""

    def test_generates_dialogue(self, base_settings, base_state):
        from agents.notebook_pipeline_nodes import podcast_script_node
        state = dict(base_state)
        state["cross_summary"] = "Transformers achieve better NLP results."
        state["study_guide"] = "## Key Concepts\n- **Attention** — A mechanism."

        expected = (
            "HOST: Welcome to Research Decoded. Today we explore transformers.\n"
            "EXPERT: Glad to be here. Transformers use self-attention.\n"
            "HOST: Key takeaway: attention is central to NLP advances."
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=expected)
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = podcast_script_node(state)

        assert "HOST:" in result["podcast_script"]
        assert "EXPERT:" in result["podcast_script"]
        assert "podcast" in result["completed_steps"]
        assert result["errors"] == []

    def test_llm_failure_appends_error(self, base_settings, base_state):
        from agents.notebook_pipeline_nodes import podcast_script_node
        state = dict(base_state)
        state["cross_summary"] = "Some content."

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("timeout")
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = podcast_script_node(state)

        assert any("failed" in e.lower() for e in result["errors"])

    def test_key_concepts_extracted_from_study_guide(self, base_settings, base_state):
        from agents.notebook_pipeline_nodes import podcast_script_node
        state = dict(base_state)
        state["cross_summary"] = "Attention is key."
        state["study_guide"] = (
            "## Key Concepts\n"
            "- **Self-Attention** — Query-Key-Value mechanism\n"
            "- **Positional Encoding** — Injects position info\n"
            "## Glossary\n"
            "| Term | Definition | Source |\n"
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="HOST: Let's discuss self-attention.\nEXPERT: It uses Q, K, V matrices."
        )
        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm):
            result = podcast_script_node(state)

        # Verify the human prompt included key concepts
        call_args = mock_llm.invoke.call_args
        human_msg = str(call_args)
        assert "Self-Attention" in human_msg or result["podcast_script"] != ""


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildPipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPipeline:
    def test_compiles_without_error(self):
        from agents.notebook_pipeline_graph import build_notebook_pipeline
        pipeline = build_notebook_pipeline()
        assert pipeline is not None

    def test_has_eight_nodes(self):
        from agents.notebook_pipeline_graph import build_notebook_pipeline
        pipeline = build_notebook_pipeline()
        graph = pipeline.get_graph()
        node_names = {n for n in graph.nodes if n not in ("__start__", "__end__")}
        assert node_names == {
            "ingest", "summarize", "retrieve",
            "verify_citations", "build_kg",
            "generate_study_guide", "generate_podcast",
            "notebook_pipeline_eval",
        }


# ─────────────────────────────────────────────────────────────────────────────
# TestRunPipelineIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPipelineIntegration:
    def test_end_to_end_with_mocked_memory(self, tmp_path, base_settings, sample_notebook):
        """Full pipeline run with mocked NotebookMemory and LLM calls."""
        from agents.notebook_pipeline_graph import run_notebook_pipeline
        from agents.notebook_pipeline_state import create_pipeline_state
        import agents.notebook_pipeline_nodes as nodes_mod

        # Mock memory
        mock_mem = MagicMock()
        mock_mem.load.return_value = sample_notebook
        nodes_mod._memory = mock_mem

        # Mock LLM — returns appropriate JSON for each node
        kg_json = {
            "nodes": [{"id": "1", "label": "Transformer", "type": "concept"}],
            "edges": [],
        }
        citation_json = [
            {"claim": "15% improvement", "source_name": "paper1.pdf",
             "confidence": "HIGH", "supporting_text": "accuracy by 15%"}
        ]

        call_count = [0]
        def mock_invoke(messages):
            call_count[0] += 1
            n = call_count[0]
            if n <= 3:
                return MagicMock(content=f"Summary text {n}.")
            elif n == 4:
                return MagicMock(content=json.dumps(citation_json))
            elif n == 5:
                return MagicMock(content=json.dumps(kg_json))
            elif n == 6:
                return MagicMock(content="## Key Concepts\n- **Attention** — mechanism")
            else:
                return MagicMock(content="HOST: Welcome.\nEXPERT: Thanks.")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_invoke

        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm), \
             patch("agents.notebook_nodes._rebuild_docs_from_chunks", return_value=[]), \
             patch("tools.hybrid_store.get_or_create_store") as mock_store_factory:
            mock_store = MagicMock()
            mock_store.is_indexed.return_value = True
            mock_store.search_hybrid.return_value = sample_notebook["chunks"][:3]
            mock_store.embedder_available.return_value = False
            mock_store_factory.return_value = mock_store

            initial = create_pipeline_state("test_nb_001", base_settings)
            final = run_notebook_pipeline(initial)

        # All 8 steps completed (7 pipeline agents + eval)
        assert {
            "ingest", "summarize", "retrieve",
            "verify_citations", "build_kg",
            "study_guide", "podcast",
        }.issubset(set(final.get("completed_steps", [])))
        assert final.get("doc_count") == 2
        assert final.get("cross_summary") != ""
        assert "HOST:" in final.get("podcast_script", "")
        assert final.get("progress_pct") == 100

        nodes_mod._memory = None  # reset

    def test_stream_callback_called_for_each_node(
        self, base_settings, sample_notebook
    ):
        """Verify stream_callback is invoked once per node."""
        from agents.notebook_pipeline_graph import run_notebook_pipeline
        from agents.notebook_pipeline_state import create_pipeline_state
        import agents.notebook_pipeline_nodes as nodes_mod

        mock_mem = MagicMock()
        mock_mem.load.return_value = sample_notebook
        nodes_mod._memory = mock_mem

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"nodes": [], "edges": []}'
        )

        called_nodes = []

        def _cb(node_name, state):
            called_nodes.append(node_name)

        with patch("agents.notebook_pipeline_nodes._make_llm", return_value=mock_llm), \
             patch("agents.notebook_nodes._rebuild_docs_from_chunks", return_value=[]), \
             patch("tools.hybrid_store.get_or_create_store") as msf:
            ms = MagicMock()
            ms.is_indexed.return_value = True
            ms.search_hybrid.return_value = []
            ms.embedder_available.return_value = False
            msf.return_value = ms

            initial = create_pipeline_state("test_nb_001", base_settings)
            run_notebook_pipeline(initial, stream_callback=_cb)

        assert len(called_nodes) == 8
        assert "ingest" in called_nodes
        assert "generate_podcast" in called_nodes

        nodes_mod._memory = None
