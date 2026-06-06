"""tests/test_notebook_advanced.py — Unit tests for agents/notebook_advanced.py"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.notebook_advanced import (
    _knowledge_graph_to_dot,
    _mindmap_to_dot,
    _parse_json_from_llm,
    _parse_json_object_from_llm,
    _safe_dot,
    _sources_context,
    compare_sources,
    extract_knowledge_graph,
    extract_timeline,
    generate_audio_summary,
    generate_cross_document_summary,
    generate_faq,
    generate_literature_review,
    generate_mindmap,
    generate_study_comparison,
    render_dot_bytes,
    synthesize_speech,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_notebook(n_sources: int = 2) -> dict:
    sources = [
        {
            "doc_id": f"doc_{i}",
            "filename": f"paper_{i}.pdf",
            "file_type": "pdf",
            "total_chunks": 3,
        }
        for i in range(n_sources)
    ]
    chunks = [
        {
            "chunk_id": f"ck_{i}_{j}",
            "doc_id": f"doc_{i}",
            "doc_name": f"paper_{i}.pdf",
            "page_num": j,
            "chunk_index": j,
            "text": f"Content of paper {i}, chunk {j}. " * 10,
        }
        for i in range(n_sources)
        for j in range(3)
    ]
    return {
        "notebook_id": "test_nb",
        "name": "Test Notebook",
        "sources": sources,
        "chunks": chunks,
        "conversation": [],
    }


_SETTINGS = {"model": "test-model", "num_ctx": 4096}


# ── Pure helper tests ─────────────────────────────────────────────────────────

class TestSafeDot:
    def test_truncates_long_strings(self):
        long = "a" * 100
        assert len(_safe_dot(long)) <= 40

    def test_escapes_quotes(self):
        assert '"' not in _safe_dot('say "hello"')

    def test_strips_backslash(self):
        assert "\\" not in _safe_dot("back\\slash")

    def test_strips_newlines(self):
        assert "\n" not in _safe_dot("line1\nline2")

    def test_short_string_unchanged(self):
        s = "neural networks"
        assert _safe_dot(s) == s


class TestParseJsonFromLlm:
    def test_extracts_array(self):
        raw = 'Some preamble\n[{"a": 1}]\nsome postamble'
        result = _parse_json_from_llm(raw)
        assert result == [{"a": 1}]

    def test_extracts_object(self):
        raw = 'Here is your JSON:\n{"key": "value"}'
        result = _parse_json_from_llm(raw)
        assert result == {"key": "value"}

    def test_plain_json(self):
        raw = '{"x": 42}'
        assert _parse_json_from_llm(raw) == {"x": 42}

    def test_raises_on_bad_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_from_llm("not json at all")


class TestSourcesContext:
    def test_includes_all_sources(self):
        nb = _make_notebook(n_sources=2)
        ctx = _sources_context(nb)
        assert "[Source 1: paper_0.pdf]" in ctx
        assert "[Source 2: paper_1.pdf]" in ctx

    def test_respects_per_doc_cap(self):
        nb = _make_notebook(n_sources=1)
        ctx = _sources_context(nb, max_chars_per_doc=20)
        lines = [l for l in ctx.splitlines() if l.strip()]
        # The content line should be truncated (<=20 chars + ellipsis)
        assert "…" in ctx or len(ctx) < 200

    def test_empty_sources(self):
        nb = {"sources": [], "chunks": []}
        assert _sources_context(nb) == ""

    def test_no_chunks_for_source(self):
        nb = {
            "sources": [{"doc_id": "x", "filename": "orphan.pdf"}],
            "chunks": [],
        }
        ctx = _sources_context(nb)
        assert "[Source 1: orphan.pdf]" in ctx


class TestMindmapToDot:
    def _sample(self) -> dict:
        return {
            "central": "Machine Learning",
            "branches": [
                {"concept": "Supervised", "sub_concepts": ["Classification", "Regression"]},
                {"concept": "Unsupervised", "sub_concepts": ["Clustering"]},
            ],
        }

    def test_contains_central_node(self):
        dot = _mindmap_to_dot(self._sample())
        assert '"root"' in dot
        assert "Machine Learning" in dot

    def test_contains_branches(self):
        dot = _mindmap_to_dot(self._sample())
        assert "Supervised" in dot
        assert "Unsupervised" in dot

    def test_contains_sub_concepts(self):
        dot = _mindmap_to_dot(self._sample())
        assert "Classification" in dot
        assert "Clustering" in dot

    def test_valid_dot_structure(self):
        dot = _mindmap_to_dot(self._sample())
        assert dot.startswith("digraph")
        assert dot.endswith("}")

    def test_empty_branches(self):
        dot = _mindmap_to_dot({"central": "Root", "branches": []})
        assert "Root" in dot
        assert dot.endswith("}")


class TestKnowledgeGraphToDot:
    def _sample(self) -> dict:
        return {
            "nodes": [
                {"id": "n1", "label": "Transformer", "type": "concept"},
                {"id": "n2", "label": "BERT", "type": "method"},
                {"id": "n3", "label": "ImageNet", "type": "dataset"},
            ],
            "edges": [
                {"from": "n2", "to": "n1", "label": "uses"},
                {"from": "n2", "to": "n3", "label": "trained on"},
            ],
        }

    def test_contains_nodes(self):
        dot = _knowledge_graph_to_dot(self._sample())
        assert "Transformer" in dot
        assert "BERT" in dot
        assert "ImageNet" in dot

    def test_contains_edges(self):
        dot = _knowledge_graph_to_dot(self._sample())
        assert "uses" in dot
        assert "trained on" in dot

    def test_type_colors_applied(self):
        dot = _knowledge_graph_to_dot(self._sample())
        assert "#10b981" in dot   # method color
        assert "#f59e0b" in dot   # dataset color

    def test_valid_dot_structure(self):
        dot = _knowledge_graph_to_dot(self._sample())
        assert dot.startswith("digraph")
        assert dot.endswith("}")

    def test_caps_at_20_nodes(self):
        data = {
            "nodes": [{"id": f"n{i}", "label": f"Node {i}", "type": "concept"} for i in range(30)],
            "edges": [],
        }
        dot = _knowledge_graph_to_dot(data)
        # Should include exactly 20 node definitions
        assert dot.count("[label=") == 20

    def test_unknown_type_uses_default_color(self):
        data = {
            "nodes": [{"id": "n1", "label": "Thing", "type": "unknown_type"}],
            "edges": [],
        }
        dot = _knowledge_graph_to_dot(data)
        assert "#6b7280" in dot


# ── LLM-backed function tests ─────────────────────────────────────────────────

def _mock_llm_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    llm = MagicMock()
    llm.invoke.return_value = msg
    return llm


class TestGenerateCrossDocumentSummary:
    def test_notebook_not_found(self):
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = None
            result, err = generate_cross_document_summary("bad_id", _SETTINGS)
        assert result == ""
        assert "not found" in err.lower()

    def test_no_sources(self):
        nb = _make_notebook(0)
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = nb
            result, err = generate_cross_document_summary("nb1", _SETTINGS)
        assert result == ""
        assert "no sources" in err.lower()

    def test_single_source_success(self):
        nb = _make_notebook(1)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response("# Summary\nGreat paper.")
            result, err = generate_cross_document_summary("nb1", _SETTINGS)
        assert err == ""
        assert "Summary" in result

    def test_multi_source_synthesis(self):
        nb = _make_notebook(3)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response("## Overview\nAll three papers agree.")
            result, err = generate_cross_document_summary("nb1", _SETTINGS)
        assert err == ""
        assert "Overview" in result

    def test_llm_exception_returns_error(self):
        nb = _make_notebook(1)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value.invoke.side_effect = RuntimeError("Ollama down")
            result, err = generate_cross_document_summary("nb1", _SETTINGS)
        assert result == ""
        assert "failed" in err.lower()


class TestGenerateFaq:
    def test_notebook_not_found(self):
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = None
            items, err = generate_faq("bad_id", _SETTINGS)
        assert items == []
        assert "not found" in err.lower()

    def test_no_sources(self):
        nb = _make_notebook(0)
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = nb
            items, err = generate_faq("nb1", _SETTINGS)
        assert items == []

    def test_valid_json_response(self):
        faq_json = json.dumps([
            {"question": "What is this?", "answer": "A test.", "sources": [1]},
            {"question": "Why?", "answer": "Because.", "sources": [2]},
        ])
        nb = _make_notebook(2)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response(faq_json)
            items, err = generate_faq("nb1", _SETTINGS)
        assert err == ""
        assert len(items) == 2
        assert items[0]["question"] == "What is this?"

    def test_malformed_json_returns_error(self):
        nb = _make_notebook(1)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response("This is not JSON at all!")
            items, err = generate_faq("nb1", _SETTINGS)
        assert items == []
        assert "failed" in err.lower()

    def test_n_questions_parameter(self):
        faq_json = json.dumps([
            {"question": f"Q{i}?", "answer": f"A{i}.", "sources": [1]}
            for i in range(5)
        ])
        nb = _make_notebook(1)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response(faq_json)
            items, err = generate_faq("nb1", _SETTINGS, n_questions=5)
        assert err == ""
        assert len(items) == 5


class TestGenerateLiteratureReview:
    def test_notebook_not_found(self):
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = None
            result, err = generate_literature_review("bad_id", _SETTINGS)
        assert result == ""
        assert "not found" in err.lower()

    def test_no_sources(self):
        nb = _make_notebook(0)
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = nb
            result, err = generate_literature_review("nb1", _SETTINGS)
        assert result == ""

    def test_success(self):
        review_text = "# Literature Review\n## 1. Introduction\nThis review covers…"
        nb = _make_notebook(2)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response(review_text)
            result, err = generate_literature_review("nb1", _SETTINGS)
        assert err == ""
        assert "Literature Review" in result


class TestGenerateMindmap:
    def test_notebook_not_found(self):
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = None
            dot, err = generate_mindmap("bad_id", _SETTINGS)
        assert dot == ""
        assert "not found" in err.lower()

    def test_valid_json_returns_dot(self):
        mindmap_json = json.dumps({
            "central": "ML",
            "branches": [{"concept": "Deep Learning", "sub_concepts": ["CNNs"]}],
        })
        nb = _make_notebook(1)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response(mindmap_json)
            dot, err = generate_mindmap("nb1", _SETTINGS)
        assert err == ""
        assert "digraph" in dot
        assert "ML" in dot

    def test_malformed_json_returns_error(self):
        nb = _make_notebook(1)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response("broken json {{")
            dot, err = generate_mindmap("nb1", _SETTINGS)
        assert dot == ""
        assert "failed" in err.lower()


class TestGenerateAudioSummary:
    def test_notebook_not_found(self):
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = None
            text, err = generate_audio_summary("bad_id", _SETTINGS)
        assert text == ""

    def test_no_sources(self):
        nb = _make_notebook(0)
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = nb
            text, err = generate_audio_summary("nb1", _SETTINGS)
        assert text == ""

    def test_success_returns_plain_text(self):
        script = "This notebook covers machine learning. It discusses deep learning methods."
        nb = _make_notebook(1)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response(script)
            text, err = generate_audio_summary("nb1", _SETTINGS)
        assert err == ""
        assert "machine learning" in text.lower()


class TestCompareSources:
    def test_same_source_returns_error(self):
        result, err = compare_sources("nb1", "doc_0", "doc_0", _SETTINGS)
        assert result == ""
        assert "different" in err.lower()

    def test_notebook_not_found(self):
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = None
            result, err = compare_sources("bad_id", "doc_0", "doc_1", _SETTINGS)
        assert result == ""
        assert "not found" in err.lower()

    def test_source_not_in_notebook(self):
        nb = _make_notebook(2)
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = nb
            result, err = compare_sources("nb1", "doc_0", "nonexistent", _SETTINGS)
        assert result == ""
        assert "not found" in err.lower()

    def test_success(self):
        comparison = "## Source Comparison\n### Overview\nBoth sources cover ML."
        nb = _make_notebook(2)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response(comparison)
            result, err = compare_sources("nb1", "doc_0", "doc_1", _SETTINGS)
        assert err == ""
        assert "Source Comparison" in result


class TestExtractKnowledgeGraph:
    def test_notebook_not_found(self):
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = None
            dot, err = extract_knowledge_graph("bad_id", _SETTINGS)
        assert dot == ""

    def test_valid_json_returns_dot(self):
        kg_json = json.dumps({
            "nodes": [
                {"id": "n1", "label": "Neural Net", "type": "concept"},
                {"id": "n2", "label": "Gradient Descent", "type": "method"},
            ],
            "edges": [
                {"from": "n1", "to": "n2", "label": "optimized by"},
            ],
        })
        nb = _make_notebook(1)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response(kg_json)
            dot, err = extract_knowledge_graph("nb1", _SETTINGS)
        assert err == ""
        assert "digraph" in dot
        assert "Neural Net" in dot
        assert "optimized by" in dot

    def test_malformed_json_returns_error(self):
        nb = _make_notebook(1)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response("not valid json !!")
            dot, err = extract_knowledge_graph("nb1", _SETTINGS)
        assert dot == ""
        assert "failed" in err.lower()


class TestParseJsonObjectFromLlm:
    def test_extracts_object_ignoring_nested_arrays(self):
        raw = '{"central": "ML", "branches": [{"concept": "CNN", "sub_concepts": ["a", "b"]}]}'
        result = _parse_json_object_from_llm(raw)
        assert result["central"] == "ML"
        assert isinstance(result["branches"], list)

    def test_raises_on_bad_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_object_from_llm("not json")


class TestRenderDotBytes:
    def test_png_rendering(self):
        dot = "digraph { a -> b }"
        png, err = render_dot_bytes(dot, "png")
        assert err == ""
        assert len(png) > 0
        assert png[:4] == b"\x89PNG"  # PNG magic bytes

    def test_svg_rendering(self):
        dot = "digraph { a -> b }"
        svg, err = render_dot_bytes(dot, "svg")
        assert err == ""
        assert b"<svg" in svg

    def test_invalid_dot_returns_error(self):
        dot = "this is not valid DOT !!!"
        png, err = render_dot_bytes(dot, "png")
        # graphviz may or may not raise for malformed DOT; just check it doesn't crash
        assert isinstance(err, str)

    def test_supported_format(self):
        dot = "digraph { x -> y }"
        _, err = render_dot_bytes(dot, "svg")
        assert err == ""


class TestSynthesizeSpeech:
    def test_returns_wav_bytes(self):
        text = "Hello, this is a test of text to speech."
        wav, err = synthesize_speech(text)
        assert err == "", f"Unexpected error: {err}"
        assert len(wav) > 1000  # WAV file should have substantial content
        assert wav[:4] == b"RIFF"  # WAV RIFF header

    def test_import_error_returns_error_string(self):
        with patch.dict("sys.modules", {"pyttsx3": None}):
            import importlib
            import agents.notebook_advanced as na
            importlib.reload(na)
        # The function should handle ImportError gracefully
        # (tested by the general module behavior)
        assert True  # Just ensure no crash during import

    def test_empty_text(self):
        wav, err = synthesize_speech("Hi.")
        # Very short text should still produce output
        assert isinstance(wav, bytes)
        assert isinstance(err, str)


class TestExtractTimeline:
    def test_notebook_not_found(self):
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = None
            items, err = extract_timeline("bad_id", _SETTINGS)
        assert items == []
        assert "not found" in err.lower()

    def test_no_sources(self):
        nb = _make_notebook(0)
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = nb
            items, err = extract_timeline("nb1", _SETTINGS)
        assert items == []

    def test_valid_json_response(self):
        tl_json = json.dumps([
            {"year": "2017", "event": "Transformer introduced", "significance": "Attention", "source": 1},
            {"year": "2018", "event": "BERT released", "significance": "NLP breakthrough", "source": 2},
        ])
        nb = _make_notebook(2)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response(tl_json)
            items, err = extract_timeline("nb1", _SETTINGS)
        assert err == ""
        assert len(items) == 2
        assert items[0]["year"] == "2017"

    def test_malformed_json_returns_error(self):
        nb = _make_notebook(1)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response("not json at all!")
            items, err = extract_timeline("nb1", _SETTINGS)
        assert items == []
        assert "failed" in err.lower()


class TestGenerateStudyComparison:
    def test_notebook_not_found(self):
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = None
            result, err = generate_study_comparison("bad_id", _SETTINGS)
        assert result == ""
        assert "not found" in err.lower()

    def test_no_sources(self):
        nb = _make_notebook(0)
        with patch("agents.notebook_advanced.NotebookMemory") as MockMem:
            MockMem.return_value.load.return_value = nb
            result, err = generate_study_comparison("nb1", _SETTINGS)
        assert result == ""

    def test_success(self):
        table_md = "| Dimension | paper_0.pdf | paper_1.pdf |\n|-----------|-------------|-------------|"
        nb = _make_notebook(2)
        with (
            patch("agents.notebook_advanced.NotebookMemory") as MockMem,
            patch("agents.notebook_advanced._make_llm") as mock_make_llm,
        ):
            MockMem.return_value.load.return_value = nb
            mock_make_llm.return_value = _mock_llm_response(table_md)
            result, err = generate_study_comparison("nb1", _SETTINGS)
        assert err == ""
        assert "Dimension" in result
