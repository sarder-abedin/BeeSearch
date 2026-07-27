"""backend/app/services/notebook_advanced_service.py
───────────────────────────────────────────────────────
Service layer for Mode 2 Phase C: the 9 standalone Research Notebook
advanced tools (``agents/notebook_advanced.py``).

Unlike Phase B's multi-node pipeline, each of these 9 functions is a single
(or few-call) operation -- so each ``run_*`` below adapts one function's
``(result, error_string)`` contract directly into the job-runner's
``Dict[str, Any]`` return shape, mirroring
``research_assistant_service.py::run_ask``'s simpler single-stage pattern
rather than ``notebook_pipeline_service.py``'s per-node adapter. A non-empty
``error_string`` is raised as a ``RuntimeError`` so the job ends in the same
"error" status / 409-on-export-attempt behaviour as a pipeline failure,
since none of these 9 functions return a partial result alongside an error.

Export helpers mirror ``notebook_pipeline_service.py``'s own: pure
content-in/bytes-out, no LLM call, no notebook lookup -- the router resolves
*which* job field to export (and 404s if it's empty) before calling these.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple

from agents.notebook_advanced import (
    compare_sources,
    extract_citation_timeline,
    extract_knowledge_graph,
    generate_audio_summary,
    generate_cross_document_summary,
    generate_faq,
    generate_literature_review,
    generate_mindmap,
    generate_paper_review,
    generate_study_comparison,
    reviewer_chat,
)
from config.settings import get_settings

from ..schemas.notebook_advanced import (
    AudioSummaryRequest,
    CitationTimelineRequest,
    CompareSourcesRequest,
    CrossDocumentSummaryRequest,
    FaqRequest,
    KnowledgeGraphRequest,
    LiteratureReviewRequest,
    MindmapRequest,
    PaperReviewRequest,
    ReviewChatRequest,
    StudyComparisonRequest,
)

logger = logging.getLogger(__name__)
cfg = get_settings()

StreamCallback = Callable[[str, Dict[str, Any]], None]

# Mirrors every notebook_advanced_service.run_* function's single stage --
# there's no per-node graph to stream, just one "working" tick before the
# LLM call so a poller observes a non-empty stage at least once.
_RUNNING_LABELS: Dict[str, str] = {
    "cross_document_summary": "Generating cross-document summary",
    "faq": "Generating FAQ",
    "literature_review": "Generating literature review",
    "mindmap": "Extracting mind map",
    "audio_summary": "Generating audio summary script",
    "compare_sources": "Comparing sources",
    "knowledge_graph": "Extracting knowledge graph",
    "citation_timeline": "Building citation timeline",
    "study_comparison": "Generating study comparison table",
    "paper_review": "Generating paper review",
    "reviewer_chat": "Generating reviewer response",
}


def build_settings(req: Any) -> Dict[str, Any]:
    """Build the ``settings`` dict every ``notebook_advanced`` function expects.

    Only forwards optional overrides that were actually provided, deferring
    to each function's own ``cfg``/``DEFAULT_TEMPERATURE_LEVEL`` fallback
    otherwise -- the same pattern as
    ``research_assistant_service.py::build_settings``.
    """
    settings: Dict[str, Any] = {}
    if req.model:
        settings["model"] = req.model
    if req.num_ctx is not None:
        settings["num_ctx"] = req.num_ctx
    if req.temperature_level:
        settings["temperature_level"] = req.temperature_level
    return settings


def _tick(stream_callback: StreamCallback, stage: str) -> None:
    stream_callback(stage, {"label": _RUNNING_LABELS[stage], "progress_pct": 50})


# ─────────────────────────────────────────────────────────────────────────────
# Run (one per feature, background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_document_summary(req: CrossDocumentSummaryRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "cross_document_summary")
    summary, error = generate_cross_document_summary(req.notebook_id, build_settings(req))
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "summary": summary}


def run_faq(req: FaqRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "faq")
    faqs, error = generate_faq(req.notebook_id, build_settings(req), n_questions=req.n_questions)
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "faqs": faqs}


def run_literature_review(req: LiteratureReviewRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "literature_review")
    review, references, error = generate_literature_review(req.notebook_id, build_settings(req))
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "review": review, "references": references}


def run_mindmap(req: MindmapRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "mindmap")
    dot, error = generate_mindmap(req.notebook_id, build_settings(req))
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "mindmap_dot": dot}


def run_audio_summary(req: AudioSummaryRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "audio_summary")
    script, error = generate_audio_summary(req.notebook_id, build_settings(req))
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "audio_script": script}


def run_compare_sources(req: CompareSourcesRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "compare_sources")
    comparison, error = compare_sources(req.notebook_id, req.doc_id_a, req.doc_id_b, build_settings(req))
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "comparison": comparison}


def run_knowledge_graph(req: KnowledgeGraphRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "knowledge_graph")
    dot, error = extract_knowledge_graph(req.notebook_id, build_settings(req))
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "knowledge_graph_dot": dot}


def run_citation_timeline(req: CitationTimelineRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "citation_timeline")
    timeline, error = extract_citation_timeline(req.notebook_id, req.enrich_with_abstracts, build_settings(req))
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "timeline": timeline}


def run_study_comparison(req: StudyComparisonRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "study_comparison")
    comparison, error = generate_study_comparison(req.notebook_id, build_settings(req))
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "study_comparison": comparison}


def run_paper_review(req: PaperReviewRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "paper_review")
    review, refs, error = generate_paper_review(req.notebook_id, req.doc_id, build_settings(req))
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "paper_review": review, "paper_review_refs": refs}


def run_reviewer_chat(req: ReviewChatRequest, stream_callback: StreamCallback) -> Dict[str, Any]:
    _tick(stream_callback, "reviewer_chat")
    history = [{"role": item.role, "content": item.content} for item in req.chat_history]
    response, error = reviewer_chat(
        req.notebook_id,
        req.doc_id,
        req.review_text,
        history,
        req.user_message,
        build_settings(req),
    )
    if error:
        raise RuntimeError(error)
    return {"notebook_id": req.notebook_id, "reviewer_chat_response": response}


# ─────────────────────────────────────────────────────────────────────────────
# Exports (sync -- no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def build_document_docx(text: str) -> bytes:
    from tools.export_tools import build_docx
    return build_docx(text, [])


def build_document_pdf(text: str) -> bytes:
    from tools.export_tools import build_pdf
    return build_pdf(text, [])


def build_dot_image(dot: str, fmt: str) -> Tuple[bytes, str]:
    """Render a Graphviz DOT string (mind map or knowledge graph) to PNG/SVG bytes.

    Returns (image_bytes, error_string) -- mirrors
    ``notebook_pipeline_service.py::build_knowledge_graph_image`` directly.
    """
    from agents.notebook_advanced import render_dot_bytes
    return render_dot_bytes(dot, fmt)


def build_audio_wav(text: str) -> Tuple[bytes, str]:
    """Synthesize an audio script to WAV bytes via pyttsx3 (offline TTS).

    Returns (wav_bytes, error_string); gracefully degrades (never raises) if
    pyttsx3/espeak-ng aren't installed -- same contract as build_dot_image's
    graphviz dependency.
    """
    from agents.notebook_advanced import synthesize_speech
    return synthesize_speech(text)
