"""backend/app/routers/notebook_advanced.py
─────────────────────────────────────────────
Mode 2 Phase C (9 standalone Research Notebook advanced tools) over HTTP.

Each ``POST /<feature>`` kicks off its ``agents.notebook_advanced`` function
on a background thread and returns a job id immediately (202 Accepted); the
frontend polls ``GET /jobs/{job_id}`` -- shared across all 9 features, since
they all write through the same job store and the same ``AdvancedResult``
envelope -- then reads the relevant field(s) once ``status == "done"``.

Export endpoints mirror ``main.py::_cmd_notebook_advanced``'s own output
files and the Streamlit advanced tabs' download buttons (``_docx_pdf_buttons``/
``_dot_export_buttons``/the Audio tab's WAV synth) and are all synchronous --
no LLM call, so a job round-trip would just add latency for no benefit.
Plain ``.md``/``.txt`` downloads for FAQ and Citation Timeline are NOT
exposed here: both are already fully available as structured JSON in the
polled job result, and the frontend composes the same Markdown the
Streamlit tabs build inline -- a server round-trip would just re-derive what
the client already has.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

from .. import jobs
from ..schemas.jobs import JobCreated
from ..schemas.notebook_advanced import (
    AdvancedJobStatus,
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
from ..services import notebook_advanced_service as service
from ..services import notebook_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notebook/advanced", tags=["notebook-advanced"])

TextArtifact = Literal["summary", "review", "audio-script", "comparison", "study-comparison"]
DocumentArtifact = Literal["summary", "review", "study-comparison"]
DotArtifact = Literal["mindmap", "knowledge-graph"]

_TEXT_FIELDS: Dict[str, str] = {
    "summary": "summary",
    "audio-script": "audio_script",
    "comparison": "comparison",
    "study-comparison": "study_comparison",
}
_DOT_FIELDS: Dict[str, str] = {
    "mindmap": "mindmap_dot",
    "knowledge-graph": "knowledge_graph_dot",
}


def _get_finished_result(job_id: str) -> Dict[str, Any]:
    """Look up *job_id* and return its ``result`` -- only once the job has
    finished successfully. Shared by every export endpoint, mirrors
    ``notebook_pipeline.py``'s helper of the same name."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status == "error":
        raise HTTPException(status_code=409, detail=f"Job failed: {job.error}")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"Job is not finished yet (status={job.status}).")
    return job.result


def _resolve_text(result: Dict[str, Any], artifact: str) -> str:
    """Resolve one ``TextArtifact``/``DocumentArtifact`` to its plain-text content.

    'review' is the only artifact needing composition: literature review's
    body excludes the References section (kept structured for the live
    snippet-expander UI), so the flattened export re-appends it via
    references_list_to_markdown() -- mirrors
    main.py::_cmd_notebook_advanced's own 'review' branch and the Streamlit
    Literature Review tab's full_md.
    """
    if artifact == "review":
        from agents.notebook_advanced import references_list_to_markdown
        review = result.get("review", "") or ""
        references = result.get("references", []) or []
        if not references:
            return review
        return review.rstrip() + "\n\n" + references_list_to_markdown(references)
    field = _TEXT_FIELDS.get(artifact, "")
    return (result.get(field, "") or "") if field else ""


def _require_notebook(notebook_id: str) -> None:
    if not notebook_service.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")


# ─────────────────────────────────────────────────────────────────────────────
# Run (one trigger endpoint per feature, background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/cross-document-summary", response_model=JobCreated, status_code=202)
def run_cross_document_summary(req: CrossDocumentSummaryRequest) -> JobCreated:
    _require_notebook(req.notebook_id)
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_cross_document_summary(req, cb))
    return JobCreated(job_id=job.id)


@router.post("/faq", response_model=JobCreated, status_code=202)
def run_faq(req: FaqRequest) -> JobCreated:
    _require_notebook(req.notebook_id)
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_faq(req, cb))
    return JobCreated(job_id=job.id)


@router.post("/literature-review", response_model=JobCreated, status_code=202)
def run_literature_review(req: LiteratureReviewRequest) -> JobCreated:
    _require_notebook(req.notebook_id)
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_literature_review(req, cb))
    return JobCreated(job_id=job.id)


@router.post("/mindmap", response_model=JobCreated, status_code=202)
def run_mindmap(req: MindmapRequest) -> JobCreated:
    _require_notebook(req.notebook_id)
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_mindmap(req, cb))
    return JobCreated(job_id=job.id)


@router.post("/audio-summary", response_model=JobCreated, status_code=202)
def run_audio_summary(req: AudioSummaryRequest) -> JobCreated:
    _require_notebook(req.notebook_id)
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_audio_summary(req, cb))
    return JobCreated(job_id=job.id)


@router.post("/compare-sources", response_model=JobCreated, status_code=202)
def run_compare_sources(req: CompareSourcesRequest) -> JobCreated:
    _require_notebook(req.notebook_id)
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_compare_sources(req, cb))
    return JobCreated(job_id=job.id)


@router.post("/knowledge-graph", response_model=JobCreated, status_code=202)
def run_knowledge_graph(req: KnowledgeGraphRequest) -> JobCreated:
    _require_notebook(req.notebook_id)
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_knowledge_graph(req, cb))
    return JobCreated(job_id=job.id)


@router.post("/citation-timeline", response_model=JobCreated, status_code=202)
def run_citation_timeline(req: CitationTimelineRequest) -> JobCreated:
    _require_notebook(req.notebook_id)
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_citation_timeline(req, cb))
    return JobCreated(job_id=job.id)


@router.post("/study-comparison", response_model=JobCreated, status_code=202)
def run_study_comparison(req: StudyComparisonRequest) -> JobCreated:
    _require_notebook(req.notebook_id)
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_study_comparison(req, cb))
    return JobCreated(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=AdvancedJobStatus)
def get_job_status(job_id: str) -> AdvancedJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return AdvancedJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Export: plain text / DOCX / PDF / PNG / SVG / WAV (all sync -- no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/export/text/{artifact}", response_class=PlainTextResponse)
def export_text(job_id: str, artifact: TextArtifact) -> PlainTextResponse:
    result = _get_finished_result(job_id)
    content = _resolve_text(result, artifact)
    if not content:
        raise HTTPException(status_code=404, detail=f"No {artifact} content available for this job.")
    return PlainTextResponse(content=content, media_type="text/markdown")


@router.get("/jobs/{job_id}/export/document/{artifact}/{fmt}")
def export_document(job_id: str, artifact: DocumentArtifact, fmt: Literal["docx", "pdf"]) -> Response:
    result = _get_finished_result(job_id)
    content = _resolve_text(result, artifact)
    if not content:
        raise HTTPException(status_code=404, detail=f"No {artifact} content available for this job.")
    if fmt == "docx":
        body = service.build_document_docx(content)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        body = service.build_document_pdf(content)
        media_type = "application/pdf"
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{artifact}.{fmt}"'},
    )


@router.get("/jobs/{job_id}/export/dot/{artifact}/{fmt}")
def export_dot(job_id: str, artifact: DotArtifact, fmt: Literal["png", "svg"]) -> Response:
    result = _get_finished_result(job_id)
    dot = result.get(_DOT_FIELDS[artifact], "") or ""
    if not dot:
        raise HTTPException(status_code=404, detail=f"No {artifact} available for this job.")
    image, error = service.build_dot_image(dot, fmt)
    if not image:
        raise HTTPException(status_code=503, detail=f"{artifact} rendering failed: {error}")
    media_type = "image/svg+xml" if fmt == "svg" else "image/png"
    return Response(
        content=image,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{artifact}.{fmt}"'},
    )


@router.get("/jobs/{job_id}/export/audio/wav")
def export_audio_wav(job_id: str) -> Response:
    result = _get_finished_result(job_id)
    script = result.get("audio_script", "") or ""
    if not script:
        raise HTTPException(status_code=404, detail="No audio script available for this job.")
    wav, error = service.build_audio_wav(script)
    if not wav:
        raise HTTPException(status_code=503, detail=f"Audio synthesis failed: {error}")
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="audio_summary.wav"'},
    )
