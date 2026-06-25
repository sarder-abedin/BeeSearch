"""backend/app/routers/research_assistant.py
────────────────────────────────────────────────
Mode 3 (AI Research Assistant) over HTTP.

``POST /api/research-assistant/ask`` kicks off
``agents.research_assistant.run_research_assistant`` on a background thread
and returns a job id immediately (202 Accepted); the frontend polls
``GET /api/research-assistant/jobs/{job_id}`` for the same
``stream_callback`` progress (searching/reading/answering/done) the CLI and
Streamlit surfaces already render live, then the final result once
``status == "done"``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from .. import jobs
from ..schemas.jobs import JobCreated
from ..schemas.research_assistant import AskJobStatus, AskRequest
from ..services.research_assistant_service import run_ask

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/research-assistant", tags=["research-assistant"])


@router.post("/ask", response_model=JobCreated, status_code=202)
def ask(req: AskRequest) -> JobCreated:
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: run_ask(req, cb))
    return JobCreated(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=AskJobStatus)
def get_job_status(job_id: str) -> AskJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return AskJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )
