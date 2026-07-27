"""backend/app/routers/notebook_explain.py
─────────────────────────────────────────────
Mode 2 Phase D (Explain tab / storyteller pipeline) over HTTP.

``POST /turn`` kicks off one Explain turn (``agents.story_graph.
run_story_turn``) on a background thread and returns a job id immediately
(202 Accepted); the frontend polls ``GET /jobs/{job_id}`` -- the same
in-memory job store every other notebook router shares
(``backend/app/jobs.py``), so job ids never collide across routers.
``GET /{notebook_id}/history`` mirrors ``routers/notebook.py``'s own
history endpoint, reading back whatever Explain turns have been persisted
for this notebook (empty list if Explain has never been used here yet --
no separate "start session" endpoint exists, matching the Streamlit tab's
own auto-create-on-first-message behavior).
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from .. import jobs
from ..schemas.jobs import JobCreated
from ..schemas.notebook_explain import ExplainJobStatus, ExplainRequest, ExplainTurn
from ..services import notebook_explain_service as service
from ..services import notebook_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notebook/explain", tags=["notebook-explain"])


def _require_notebook(notebook_id: str) -> None:
    if not notebook_service.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")


@router.post("/turn", response_model=JobCreated, status_code=202)
def run_turn(req: ExplainRequest) -> JobCreated:
    _require_notebook(req.notebook_id)
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_explain_turn(req, cb))
    return JobCreated(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=ExplainJobStatus)
def get_job_status(job_id: str) -> ExplainJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ExplainJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )


@router.get("/{notebook_id}/history", response_model=List[ExplainTurn])
def get_history(notebook_id: str) -> List[ExplainTurn]:
    _require_notebook(notebook_id)
    return [ExplainTurn(**t) for t in service.get_history(notebook_id)]
