"""backend/app/routers/paper_graph.py
──────────────────────────────────────
HTTP layer for the paper discovery features.

Feature 1 (Similarity Graph): long-running, uses the standard job pattern.
Feature 2 (Discovery Network): create_collection is sync (fast metadata
fetch); expand is long-running and uses the job pattern.
Building-block endpoints (paper metadata, references, citations, author
papers) are sync — each is a single proxied S2 API call.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from .. import jobs
from ..schemas.jobs import JobCreated
from ..schemas.paper_graph import (
    CollectionResponse,
    CreateCollectionRequest,
    ExpandCollectionRequest,
    ExpandJobStatus,
    GraphDataSchema,
    PaperGraphJobStatus,
    PaperNodeSchema,
    SimilarityGraphRequest,
)
from ..services import paper_graph_service as service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/paper-graph", tags=["paper-graph"])


# ── Feature 1: Similarity Graph (one-shot, background job) ───────────────────

@router.post("/similarity-graph", response_model=JobCreated, status_code=202)
def run_similarity_graph(req: SimilarityGraphRequest) -> JobCreated:
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: service.run_similarity_graph(req, cb))
    return JobCreated(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=PaperGraphJobStatus)
def get_similarity_graph_job(job_id: str) -> PaperGraphJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return PaperGraphJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )


# ── Building blocks (sync — single S2 API call each) ─────────────────────────

@router.get("/papers/{paper_id}", response_model=PaperNodeSchema)
def get_paper(paper_id: str) -> PaperNodeSchema:
    from paper_graph.s2_client import get_client
    node = get_client().get_paper(paper_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id!r} not found.")
    return PaperNodeSchema(
        id=node.id, title=node.title, authors=node.authors, year=node.year,
        venue=node.venue, abstract=node.abstract, citation_count=node.citation_count,
        url=node.url,
    )


@router.get("/papers/{paper_id}/references", response_model=List[str])
def get_references(paper_id: str) -> List[str]:
    from paper_graph.s2_client import get_client
    return get_client().get_references(paper_id)


@router.get("/papers/{paper_id}/citations", response_model=List[str])
def get_citations(paper_id: str) -> List[str]:
    from paper_graph.s2_client import get_client
    return get_client().get_citations(paper_id)


@router.get("/authors/{author_id}/papers", response_model=List[PaperNodeSchema])
def get_author_papers(author_id: str) -> List[PaperNodeSchema]:
    from paper_graph.s2_client import get_client
    nodes = get_client().get_author_papers(author_id)
    return [
        PaperNodeSchema(
            id=n.id, title=n.title, authors=n.authors, year=n.year,
            venue=n.venue, abstract=n.abstract, citation_count=n.citation_count,
            url=n.url,
        )
        for n in nodes
    ]


# ── Feature 2: Discovery Network ─────────────────────────────────────────────

@router.post("/collections", response_model=CollectionResponse, status_code=201)
def create_collection(req: CreateCollectionRequest) -> CollectionResponse:
    try:
        return service.create_collection(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/collections/{collection_id}", response_model=CollectionResponse)
def get_collection(collection_id: str) -> CollectionResponse:
    from paper_graph.collection_store import get_store
    from paper_graph.graph_builder import build_discovery_graph
    from ..services.paper_graph_service import _graph_to_schema

    store = get_store()
    col = store.get(collection_id)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id!r} not found.")
    graph = build_discovery_graph(col.paper_nodes, col.edges)
    return CollectionResponse(
        collection_id=col.id,
        graph=_graph_to_schema(graph.nodes, graph.edges),
    )


@router.post("/collections/{collection_id}/expand", response_model=JobCreated, status_code=202)
def expand_collection(collection_id: str, req: ExpandCollectionRequest) -> JobCreated:
    from paper_graph.collection_store import get_store
    if get_store().get(collection_id) is None:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id!r} not found.")
    job = jobs.create_job()
    jobs.run_in_background(
        job, lambda cb: service.expand_collection(collection_id, req, cb)
    )
    return JobCreated(job_id=job.id)


@router.get("/collections/{collection_id}/jobs/{job_id}", response_model=ExpandJobStatus)
def get_expand_job(collection_id: str, job_id: str) -> ExpandJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ExpandJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )
