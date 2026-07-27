"""backend/tests/test_notebook_advanced_service.py
─────────────────────────────────────────────────────
Unit tests for backend/app/services/notebook_advanced_service.py.

Every agents.notebook_advanced.* function is imported at module level into
notebook_advanced_service's own namespace (``from agents.notebook_advanced
import generate_cross_document_summary, ...``), so each is mocked at the
service module's own import site -- the same boundary
test_notebook_pipeline_service.py uses for run_notebook_pipeline. The export
helpers (render_dot_bytes / build_docx / build_pdf / synthesize_speech) do
their own LOCAL imports inside the function body, so they're mocked at their
*defining* module instead.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.schemas.notebook_advanced import (
    AudioSummaryRequest,
    CitationTimelineRequest,
    CompareSourcesRequest,
    CrossDocumentSummaryRequest,
    FaqRequest,
    KnowledgeGraphRequest,
    LiteratureReviewRequest,
    MindmapRequest,
    StudyComparisonRequest,
)
from backend.app.services import notebook_advanced_service as service

# ─────────────────────────────────────────────────────────────────────────────
# build_settings
# ─────────────────────────────────────────────────────────────────────────────

def test_build_settings_omits_unset_fields():
    req = CrossDocumentSummaryRequest(notebook_id="nb1")
    assert service.build_settings(req) == {}


def test_build_settings_includes_provided_fields():
    req = CrossDocumentSummaryRequest(
        notebook_id="nb1",
        model="llama3.1:8b",
        num_ctx=4096,
        temperature_level="creative",
    )
    assert service.build_settings(req) == {
        "model": "llama3.1:8b",
        "num_ctx": 4096,
        "temperature_level": "creative",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simple (notebook_id, settings) -> (value, error) tools
# ─────────────────────────────────────────────────────────────────────────────

_SIMPLE_CASES = [
    pytest.param(
        service.run_cross_document_summary,
        "backend.app.services.notebook_advanced_service.generate_cross_document_summary",
        CrossDocumentSummaryRequest(notebook_id="nb1"),
        "summary",
        "Synthesized summary.",
        id="cross_document_summary",
    ),
    pytest.param(
        service.run_mindmap,
        "backend.app.services.notebook_advanced_service.generate_mindmap",
        MindmapRequest(notebook_id="nb1"),
        "mindmap_dot",
        "digraph mindmap {}",
        id="mindmap",
    ),
    pytest.param(
        service.run_audio_summary,
        "backend.app.services.notebook_advanced_service.generate_audio_summary",
        AudioSummaryRequest(notebook_id="nb1"),
        "audio_script",
        "Spoken script.",
        id="audio_summary",
    ),
    pytest.param(
        service.run_knowledge_graph,
        "backend.app.services.notebook_advanced_service.extract_knowledge_graph",
        KnowledgeGraphRequest(notebook_id="nb1"),
        "knowledge_graph_dot",
        "digraph knowledge_graph {}",
        id="knowledge_graph",
    ),
    pytest.param(
        service.run_study_comparison,
        "backend.app.services.notebook_advanced_service.generate_study_comparison",
        StudyComparisonRequest(notebook_id="nb1"),
        "study_comparison",
        "| Dimension | Source 1 |",
        id="study_comparison",
    ),
]


@pytest.mark.parametrize("run_fn,patch_target,req,field,value", _SIMPLE_CASES)
def test_simple_tool_happy_path(run_fn, patch_target, req, field, value):
    with patch(patch_target, return_value=(value, "")) as mock_fn:
        result = run_fn(req, MagicMock())

    assert result == {"notebook_id": "nb1", field: value}
    mock_fn.assert_called_once_with("nb1", {})


@pytest.mark.parametrize("run_fn,patch_target,req,field,value", _SIMPLE_CASES)
def test_simple_tool_raises_runtime_error_on_failure(run_fn, patch_target, req, field, value):
    with patch(patch_target, return_value=("", "boom")):
        with pytest.raises(RuntimeError, match="boom"):
            run_fn(req, MagicMock())


@pytest.mark.parametrize("run_fn,patch_target,req,field,value", _SIMPLE_CASES)
def test_simple_tool_ticks_stream_callback_once(run_fn, patch_target, req, field, value):
    cb = MagicMock()
    with patch(patch_target, return_value=(value, "")):
        run_fn(req, cb)

    cb.assert_called_once()
    stage, payload = cb.call_args.args
    assert isinstance(stage, str) and stage
    assert payload["progress_pct"] == 50
    assert isinstance(payload["label"], str) and payload["label"]


# ─────────────────────────────────────────────────────────────────────────────
# run_faq (extra n_questions kwarg, list-of-dict result)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_faq_happy_path_uses_default_n_questions():
    req = FaqRequest(notebook_id="nb1")
    faqs = [{"question": "Q1?", "answer": "A1.", "sources": [1]}]
    with patch(
        "backend.app.services.notebook_advanced_service.generate_faq",
        return_value=(faqs, ""),
    ) as mock_fn:
        result = service.run_faq(req, MagicMock())

    assert result == {"notebook_id": "nb1", "faqs": faqs}
    mock_fn.assert_called_once_with("nb1", {}, n_questions=8)


def test_run_faq_forwards_custom_n_questions():
    req = FaqRequest(notebook_id="nb1", n_questions=3)
    with patch(
        "backend.app.services.notebook_advanced_service.generate_faq",
        return_value=([], ""),
    ) as mock_fn:
        service.run_faq(req, MagicMock())

    mock_fn.assert_called_once_with("nb1", {}, n_questions=3)


def test_run_faq_raises_runtime_error_on_failure():
    req = FaqRequest(notebook_id="nb1")
    with patch(
        "backend.app.services.notebook_advanced_service.generate_faq",
        return_value=([], "FAQ generation failed: boom"),
    ):
        with pytest.raises(RuntimeError, match="FAQ generation failed"):
            service.run_faq(req, MagicMock())


# ─────────────────────────────────────────────────────────────────────────────
# run_literature_review (3-tuple return: review, references, error)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_literature_review_happy_path():
    req = LiteratureReviewRequest(notebook_id="nb1")
    references = [{"n": 1, "doc_name": "paper.pdf", "page": 2, "snippet": "...", "doc_id": "d1"}]
    with patch(
        "backend.app.services.notebook_advanced_service.generate_literature_review",
        return_value=("## Literature Review", references, ""),
    ) as mock_fn:
        result = service.run_literature_review(req, MagicMock())

    assert result == {
        "notebook_id": "nb1",
        "review": "## Literature Review",
        "references": references,
    }
    mock_fn.assert_called_once_with("nb1", {})


def test_run_literature_review_raises_runtime_error_on_failure():
    req = LiteratureReviewRequest(notebook_id="nb1")
    with patch(
        "backend.app.services.notebook_advanced_service.generate_literature_review",
        return_value=("", [], "Literature review generation failed: boom"),
    ):
        with pytest.raises(RuntimeError, match="Literature review generation failed"):
            service.run_literature_review(req, MagicMock())


# ─────────────────────────────────────────────────────────────────────────────
# run_compare_sources (extra doc_id_a/doc_id_b args)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_compare_sources_forwards_both_doc_ids():
    req = CompareSourcesRequest(notebook_id="nb1", doc_id_a="docA", doc_id_b="docB")
    with patch(
        "backend.app.services.notebook_advanced_service.compare_sources",
        return_value=("Comparison.", ""),
    ) as mock_fn:
        result = service.run_compare_sources(req, MagicMock())

    assert result == {"notebook_id": "nb1", "comparison": "Comparison."}
    mock_fn.assert_called_once_with("nb1", "docA", "docB", {})


def test_run_compare_sources_raises_runtime_error_on_failure():
    req = CompareSourcesRequest(notebook_id="nb1", doc_id_a="docA", doc_id_b="docB")
    with patch(
        "backend.app.services.notebook_advanced_service.compare_sources",
        return_value=("", "Please select two different sources to compare."),
    ):
        with pytest.raises(RuntimeError, match="two different sources"):
            service.run_compare_sources(req, MagicMock())


# ─────────────────────────────────────────────────────────────────────────────
# run_citation_timeline (extra enrich_with_abstracts arg)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_citation_timeline_forwards_enrich_flag():
    req = CitationTimelineRequest(notebook_id="nb1", enrich_with_abstracts=True)
    timeline = [{"year": "2020", "title": "T", "authors": "A", "gist": "G", "source": 1, "url": ""}]
    with patch(
        "backend.app.services.notebook_advanced_service.extract_citation_timeline",
        return_value=(timeline, ""),
    ) as mock_fn:
        result = service.run_citation_timeline(req, MagicMock())

    assert result == {"notebook_id": "nb1", "timeline": timeline}
    mock_fn.assert_called_once_with("nb1", True, {})


def test_run_citation_timeline_defaults_enrich_flag_to_false():
    req = CitationTimelineRequest(notebook_id="nb1")
    with patch(
        "backend.app.services.notebook_advanced_service.extract_citation_timeline",
        return_value=([], ""),
    ) as mock_fn:
        service.run_citation_timeline(req, MagicMock())

    mock_fn.assert_called_once_with("nb1", False, {})


def test_run_citation_timeline_raises_runtime_error_on_failure():
    req = CitationTimelineRequest(notebook_id="nb1")
    with patch(
        "backend.app.services.notebook_advanced_service.extract_citation_timeline",
        return_value=([], "No references/bibliography section could be found or parsed."),
    ):
        with pytest.raises(RuntimeError, match="No references"):
            service.run_citation_timeline(req, MagicMock())


# ─────────────────────────────────────────────────────────────────────────────
# Exports (sync -- no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def test_build_document_docx_delegates_to_build_docx():
    with patch("tools.export_tools.build_docx", return_value=b"DOCX") as mock_build:
        content = service.build_document_docx("## Review")

    assert content == b"DOCX"
    mock_build.assert_called_once_with("## Review", [])


def test_build_document_pdf_delegates_to_build_pdf():
    with patch("tools.export_tools.build_pdf", return_value=b"PDF") as mock_build:
        content = service.build_document_pdf("## Review")

    assert content == b"PDF"
    mock_build.assert_called_once_with("## Review", [])


def test_build_dot_image_delegates_to_render_dot_bytes():
    with patch(
        "agents.notebook_advanced.render_dot_bytes", return_value=(b"PNGDATA", "")
    ) as mock_render:
        image, error = service.build_dot_image("digraph{}", "png")

    assert image == b"PNGDATA"
    assert error == ""
    mock_render.assert_called_once_with("digraph{}", "png")


def test_build_audio_wav_delegates_to_synthesize_speech():
    with patch(
        "agents.notebook_advanced.synthesize_speech", return_value=(b"WAVDATA", "")
    ) as mock_synth:
        wav, error = service.build_audio_wav("Spoken script.")

    assert wav == b"WAVDATA"
    assert error == ""
    mock_synth.assert_called_once_with("Spoken script.")
