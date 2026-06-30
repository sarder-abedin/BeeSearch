"""backend/app/routers/notebook_report.py
─────────────────────────────────────────────
Mode 2 Phase E (Research Report workflow) over HTTP.

``POST /run`` kicks off ``agents.graph.run_research`` on a background thread
and returns a job id immediately (202 Accepted); the frontend polls
``GET /jobs/{job_id}`` for the same per-step progress
``ui/tabs/notebook.py::_tab_research_report`` shows live, then the full
``ReportResult`` once ``status == "done"``.

The Markdown report itself gets no export endpoint -- it is already
returned verbatim as ``ReportResult.report``, so the frontend builds the
download client-side from the in-memory string (mirrors
``ui/helpers.py::render_report``, which also just hands Streamlit the
string it already has, no regeneration step). Citation export (BibTeX/RIS)
is different: ``tools.citation_tools.refs_to_bibtex``/``refs_to_ris`` are
pure Python with no client-side equivalent, so that conversion stays a
server round-trip -- mirrors ``notebook_pipeline.py``'s ``export_text`` +
``_get_finished_result`` pattern exactly, and matches ``ui/helpers.py::
render_citation_downloads``'s two buttons, including reusing
``text/plain`` as the media type for both formats since that is what the
Streamlit buttons already send.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from tools.citation_tools import refs_to_bibtex, refs_to_ris

from .. import jobs
from ..schemas.jobs import JobCreated
from ..schemas.notebook_report import ReportJobStatus, ReportRequest
from ..services import notebook_report_service as service
from ..services import notebook_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notebook/report", tags=["notebook-report"])


def _get_finished_result(job_id: str) -> Dict[str, Any]:
    """Look up *job_id* and return its ``result`` -- only once the job has
    finished successfully.

    Shared by the citation export endpoint so it gets the same 404/409
    behaviour as every other phase's export endpoints.
    """
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status == "error":
        raise HTTPException(status_code=409, detail=f"Report job failed: {job.error}")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"Report job is not finished yet (status={job.status}).")
    return job.result


# ─────────────────────────────────────────────────────────────────────────────
# Report run (background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/run", response_model=JobCreated, status_code=202)
def run_report(req: ReportRequest) -> JobCreated:
    if not notebook_service.notebook_exists(req.notebook_id):
        raise HTTPException(status_code=404, detail=f"Notebook '{req.notebook_id}' not found.")
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_report(req, cb))
    return JobCreated(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=ReportJobStatus)
def get_job_status(job_id: str) -> ReportJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ReportJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Export: BibTeX / RIS (sync -- no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/export/citations/{fmt}", response_class=PlainTextResponse)
def export_citations(job_id: str, fmt: Literal["bibtex", "ris"]) -> PlainTextResponse:
    result = _get_finished_result(job_id)
    references = result.get("references") or []
    content = refs_to_bibtex(references) if fmt == "bibtex" else refs_to_ris(references)
    return PlainTextResponse(content=content, media_type="text/plain")
