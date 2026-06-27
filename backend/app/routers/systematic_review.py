"""backend/app/routers/systematic_review.py
─────────────────────────────────────────────
Mode 1 (Systematic Literature Review) over HTTP.

``POST /api/systematic-review/run`` kicks off the 7-node PRISMA pipeline
(``agents.systematic_review_graph.run_systematic_review``) on a background
thread and returns a job id immediately (202 Accepted), mirroring Mode 3's
``research_assistant`` router. The frontend polls
``GET /api/systematic-review/jobs/{job_id}`` for the same
``progress_pct``/``status_detail`` fields the Streamlit tab's live progress
bar already reads, then the full ``SRResult`` once ``status == "done"``.

The 8 Explore-tool deep-dives share one generic trigger
(``POST .../jobs/{job_id}/explore/{tool}``) and one generic poll
(``GET .../tool-jobs/{job_id}``) -- see
``backend/app/services/systematic_review_service.py::run_explore_tool`` --
plus a handful of dedicated *synchronous* endpoints for the parts of the
Streamlit tab that don't call an LLM (Evidence Map, Meta-Analysis seed/pool,
Markdown/DOCX/PDF export, guided templates), where a job round-trip would
just add latency for no benefit. Meta-Analysis "draft from abstracts" and
the plain-language summaries *do* call the LLM, so they go through the same
background-job pattern as the main run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

from .. import jobs
from ..schemas.jobs import JobCreated
from ..schemas.systematic_review import (
    EvidenceMapResponse,
    ExploreTool,
    ExploreToolRequest,
    ExportRequest,
    GrammarCheckRequest,
    GrammarCheckResponse,
    MetaAnalysisDraftRequest,
    MetaAnalysisPoolRequest,
    MetaAnalysisPoolResponse,
    MetaAnalysisSeedResponse,
    PlainLanguageSummaryRequest,
    SRJobStatus,
    SRRequest,
    SRTemplate,
    ToolJobStatus,
)
from ..services import systematic_review_service as service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/systematic-review", tags=["systematic-review"])


def _get_finished_result(job_id: str) -> Dict[str, Any]:
    """Look up *job_id* and return its ``result`` (the SR ``final_state``) --
    only once the job has finished successfully.

    Shared by every endpoint that operates on an already-completed review
    (Explore tools, Evidence Map, Meta-Analysis, exports, plain-language
    summaries) so each gets the same 404/409 behaviour for a missing,
    unfinished, or failed job rather than re-implementing it for each one.
    """
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status == "error":
        raise HTTPException(status_code=409, detail=f"Review job failed: {job.error}")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"Review job is not finished yet (status={job.status}).")
    return job.result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline run
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/run", response_model=JobCreated, status_code=202)
def run(req: SRRequest) -> JobCreated:
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_sr(req, cb))
    return JobCreated(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=SRJobStatus)
def get_job_status(job_id: str) -> SRJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return SRJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Explore tools (generic background-job dispatch, shared by all 8)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/explore/{tool}", response_model=JobCreated, status_code=202)
def trigger_explore_tool(job_id: str, tool: ExploreTool, req: ExploreToolRequest) -> JobCreated:
    final_state = _get_finished_result(job_id)
    tool_job = jobs.create_job()
    jobs.run_in_background(
        tool_job, lambda cb: service.run_explore_tool(tool, final_state, req.options, cb)
    )
    return JobCreated(job_id=tool_job.id)


@router.get("/tool-jobs/{job_id}", response_model=ToolJobStatus)
def get_tool_job_status(job_id: str) -> ToolJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ToolJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Map (sync -- no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/evidence-map", response_model=EvidenceMapResponse)
def get_evidence_map(job_id: str) -> EvidenceMapResponse:
    final_state = _get_finished_result(job_id)
    return EvidenceMapResponse(**service.build_evidence_map(final_state.get("evidence_table", [])))


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Analysis: seed (sync) -> draft (background job, LLM) -> pool (sync)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/meta-analysis/seed", response_model=MetaAnalysisSeedResponse)
def seed_meta_analysis(job_id: str) -> MetaAnalysisSeedResponse:
    final_state = _get_finished_result(job_id)
    rows = service.seed_meta_analysis(final_state.get("evidence_table", []))
    return MetaAnalysisSeedResponse(rows=rows, measure_labels=service.get_measure_labels())


@router.post("/jobs/{job_id}/meta-analysis/draft", response_model=JobCreated, status_code=202)
def draft_meta_analysis(job_id: str, req: MetaAnalysisDraftRequest) -> JobCreated:
    final_state = _get_finished_result(job_id)
    options: Dict[str, Any] = {"model": req.model, "num_ctx": req.num_ctx}
    row_dicts = [r.model_dump() for r in req.rows]
    tool_job = jobs.create_job()
    jobs.run_in_background(
        tool_job,
        lambda cb: {
            "rows": service.draft_meta_analysis_rows(final_state, row_dicts, req.measure, options, cb)
        },
    )
    return JobCreated(job_id=tool_job.id)


@router.post("/meta-analysis/pool", response_model=MetaAnalysisPoolResponse)
def pool_meta_analysis(req: MetaAnalysisPoolRequest) -> MetaAnalysisPoolResponse:
    return MetaAnalysisPoolResponse(**service.pool_meta_analysis(req))


# ─────────────────────────────────────────────────────────────────────────────
# Export: Markdown / DOCX / PDF (all sync -- no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/export/markdown", response_class=PlainTextResponse)
def export_markdown(job_id: str) -> PlainTextResponse:
    final_state = _get_finished_result(job_id)
    md = service.build_markdown(final_state.get("research_question", ""), final_state)
    return PlainTextResponse(content=md, media_type="text/markdown")


@router.post("/jobs/{job_id}/export/docx")
def export_docx(job_id: str, req: ExportRequest) -> Response:
    final_state = _get_finished_result(job_id)
    content = service.build_docx(final_state, req)
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="systematic_review.docx"'},
    )


@router.post("/jobs/{job_id}/export/pdf")
def export_pdf(job_id: str, req: ExportRequest) -> Response:
    final_state = _get_finished_result(job_id)
    content = service.build_pdf(final_state, req)
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="systematic_review.pdf"'},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plain-language summaries (background job -- calls the LLM)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/plain-language-summary", response_model=JobCreated, status_code=202)
def trigger_plain_language_summary(job_id: str, req: PlainLanguageSummaryRequest) -> JobCreated:
    final_state = _get_finished_result(job_id)
    tool_job = jobs.create_job()
    jobs.run_in_background(
        tool_job,
        lambda cb: service.generate_plain_language_summary(final_state, req.format, req.model, req.num_ctx),
    )
    return JobCreated(job_id=tool_job.id)


# ─────────────────────────────────────────────────────────────────────────────
# Guided templates + grammar-check gate (both sync, no job needed)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/templates", response_model=List[SRTemplate])
def list_templates() -> List[Dict[str, Any]]:
    return service.list_templates()


@router.post("/grammar-check", response_model=GrammarCheckResponse)
def grammar_check(req: GrammarCheckRequest) -> GrammarCheckResponse:
    return GrammarCheckResponse(**service.check_grammar(req))
