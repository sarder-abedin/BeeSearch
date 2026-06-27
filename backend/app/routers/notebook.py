"""backend/app/routers/notebook.py
───────────────────────────────────
Mode 2 Phase A (Research Notebook core) over HTTP: notebook CRUD, source
upload/removal, conversation history, and chat turns.

Chat follows the same background-job + polling pattern as Mode 1 / Mode 3:
``POST /chat`` kicks off ``agents.notebook_graph.run_notebook_turn`` on a
background thread and returns a job id immediately (202 Accepted); the
frontend polls ``GET /jobs/{job_id}`` for the same ``stream_callback``
progress (retrieve/answer/save/notebook_eval) the Streamlit tab already
renders live, then the final result once ``status == "done"``.

Everything else (notebook CRUD, source upload/removal, history) is a small,
fast, non-LLM operation against ``NotebookMemory``, so those run synchronously.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import jobs
from ..schemas.jobs import JobCreated
from ..schemas.notebook import (
    ChatJobStatus,
    ChatRequest,
    ConversationTurn,
    CreateNotebookRequest,
    DeleteNotebookResult,
    NotebookDetail,
    NotebookSummary,
    RemoveSourceResult,
    RenameNotebookRequest,
    UploadSourceResult,
)
from ..services import notebook_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notebook", tags=["notebook"])


# ─────────────────────────────────────────────────────────────────────────────
# Notebook CRUD
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/notebooks", response_model=NotebookSummary, status_code=201)
def create_notebook(req: CreateNotebookRequest) -> NotebookSummary:
    return notebook_service.create_notebook(req)


@router.get("/notebooks", response_model=List[NotebookSummary])
def list_notebooks() -> List[NotebookSummary]:
    return notebook_service.list_notebooks()


@router.get("/notebooks/{notebook_id}", response_model=NotebookDetail)
def get_notebook(notebook_id: str) -> NotebookDetail:
    detail = notebook_service.get_notebook_detail(notebook_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    return detail


@router.delete("/notebooks/{notebook_id}", response_model=DeleteNotebookResult)
def delete_notebook(notebook_id: str) -> DeleteNotebookResult:
    if not notebook_service.delete_notebook(notebook_id):
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    return DeleteNotebookResult(deleted=True)


@router.post("/notebooks/{notebook_id}/rename", response_model=NotebookSummary)
def rename_notebook(notebook_id: str, req: RenameNotebookRequest) -> NotebookSummary:
    summary = notebook_service.rename_notebook(notebook_id, req.new_name)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    return summary


@router.get("/notebooks/{notebook_id}/history", response_model=List[ConversationTurn])
def get_history(notebook_id: str, max_turns: int = 8) -> List[ConversationTurn]:
    return notebook_service.get_history(notebook_id, max_turns=max_turns)


# ─────────────────────────────────────────────────────────────────────────────
# Source upload / removal
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/notebooks/{notebook_id}/sources", response_model=UploadSourceResult)
async def upload_source(notebook_id: str, file: UploadFile = File(...)) -> UploadSourceResult:
    file_bytes = await file.read()
    try:
        return notebook_service.upload_source(notebook_id, file.filename or "upload", file_bytes)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/notebooks/{notebook_id}/sources/{doc_id}", response_model=RemoveSourceResult)
def remove_source(notebook_id: str, doc_id: str) -> RemoveSourceResult:
    return RemoveSourceResult(removed=notebook_service.remove_source(notebook_id, doc_id))


# ─────────────────────────────────────────────────────────────────────────────
# Chat (background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=JobCreated, status_code=202)
def chat(req: ChatRequest) -> JobCreated:
    if not notebook_service.notebook_exists(req.notebook_id):
        raise HTTPException(status_code=404, detail=f"Notebook '{req.notebook_id}' not found.")
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: notebook_service.run_chat_turn(req, cb))
    return JobCreated(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=ChatJobStatus)
def get_job_status(job_id: str) -> ChatJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ChatJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )
