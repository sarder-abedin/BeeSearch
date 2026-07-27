"""backend/app/routers/notebook_pipeline.py
─────────────────────────────────────────────
Mode 2 Phase B (7-agent Research Notebook analysis pipeline) over HTTP.

``POST /run`` kicks off ``agents.notebook_pipeline_graph.run_notebook_pipeline``
on a background thread and returns a job id immediately (202 Accepted); the
frontend polls ``GET /jobs/{job_id}`` for the same per-agent progress the CLI's
``--notebook-pipeline`` prints live, then the full ``PipelineResult`` once
``status == "done"``.

Export endpoints mirror ``main.py::_cmd_notebook_pipeline``'s own output
files (summary/citations/study-guide/podcast as Markdown or plain text,
study guide as DOCX/PDF, knowledge graph as PNG/SVG) and are all synchronous
-- no LLM call, so a job round-trip would just add latency for no benefit.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

from .. import jobs
from ..schemas.jobs import JobCreated
from ..schemas.notebook_pipeline import PipelineJobStatus, PipelineRequest
from ..services import notebook_pipeline_service as service
from ..services import notebook_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notebook/pipeline", tags=["notebook-pipeline"])

_TEXT_ARTIFACTS: Dict[str, str] = {
    "summary": "cross_summary",
    "citations": "citation_report",
    "study-guide": "study_guide",
    "podcast": "podcast_script",
}


def _get_finished_result(job_id: str) -> Dict[str, Any]:
    """Look up *job_id* and return its ``result`` (the pipeline's final state)
    -- only once the job has finished successfully.

    Shared by every export endpoint so each gets the same 404/409 behaviour
    for a missing, unfinished, or failed job rather than re-implementing it.
    """
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status == "error":
        raise HTTPException(status_code=409, detail=f"Pipeline job failed: {job.error}")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"Pipeline job is not finished yet (status={job.status}).")
    return job.result


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline run (background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/run", response_model=JobCreated, status_code=202)
def run_pipeline(req: PipelineRequest) -> JobCreated:
    if not notebook_service.notebook_exists(req.notebook_id):
        raise HTTPException(status_code=404, detail=f"Notebook '{req.notebook_id}' not found.")
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_pipeline(req, cb))
    return JobCreated(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=PipelineJobStatus)
def get_job_status(job_id: str) -> PipelineJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return PipelineJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Export: plain text / DOCX / PDF / PNG / SVG (all sync -- no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/export/text/{artifact}", response_class=PlainTextResponse)
def export_text(job_id: str, artifact: Literal["summary", "citations", "study-guide", "podcast"]) -> PlainTextResponse:
    result = _get_finished_result(job_id)
    field = _TEXT_ARTIFACTS[artifact]
    content = result.get(field, "") or ""
    if not content:
        raise HTTPException(status_code=404, detail=f"No {artifact} content available for this job.")
    return PlainTextResponse(content=content, media_type="text/markdown")


@router.get("/jobs/{job_id}/export/study-guide/{fmt}")
def export_study_guide(job_id: str, fmt: Literal["docx", "pdf"]) -> Response:
    result = _get_finished_result(job_id)
    study_guide = result.get("study_guide", "") or ""
    if not study_guide:
        raise HTTPException(status_code=404, detail="No study guide available for this job.")
    if fmt == "docx":
        content = service.build_study_guide_docx(study_guide)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        content = service.build_study_guide_pdf(study_guide)
        media_type = "application/pdf"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="study_guide.{fmt}"'},
    )


@router.get("/jobs/{job_id}/export/knowledge-graph/{fmt}")
def export_knowledge_graph(job_id: str, fmt: Literal["png", "svg"]) -> Response:
    result = _get_finished_result(job_id)
    dot = result.get("knowledge_graph_dot", "") or ""
    if not dot:
        raise HTTPException(status_code=404, detail="No knowledge graph available for this job.")
    image, error = service.build_knowledge_graph_image(dot, fmt)
    if not image:
        raise HTTPException(status_code=503, detail=f"Knowledge graph rendering failed: {error}")
    media_type = "image/svg+xml" if fmt == "svg" else "image/png"
    return Response(
        content=image,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="knowledge_graph.{fmt}"'},
    )
