"""backend/app/schemas/paper_graph.py
──────────────────────────────────────
Pydantic request/response shapes for the paper discovery features:
  Feature 1 — Similarity Graph (Connected Papers–style, one-shot)
  Feature 2 — Discovery Network (ResearchRabbit–style, incremental)
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .jobs import JobStatusBase


# ── Shared node/edge types ────────────────────────────────────────────────────

class PaperNodeSchema(BaseModel):
    id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None   # None displayed as "Unavailable" in the UI
    citation_count: Optional[int] = None
    url: Optional[str] = None


class GraphEdgeSchema(BaseModel):
    source: str
    target: str
    weight: float
    edge_type: str   # "similarity" | "reference" | "citation" | "recommendation" | "co_author"


class GraphDataSchema(BaseModel):
    nodes: List[PaperNodeSchema] = Field(default_factory=list)
    edges: List[GraphEdgeSchema] = Field(default_factory=list)
    partial: bool = False
    notice: str = ""


# ── Feature 1: Similarity Graph ───────────────────────────────────────────────

class SimilarityGraphRequest(BaseModel):
    """Seed paper may be given as an S2 paperId (preferred) or as a title
    string that the service will resolve via S2 title search."""
    paper_id: str = Field(
        ...,
        description="Semantic Scholar paperId OR a paper title to resolve via search.",
    )
    top_n: int = Field(50, ge=5, le=100, description="Maximum candidates to keep in the graph.")
    bc_weight: float = Field(0.5, ge=0.0, le=1.0, description="Bibliographic-coupling weight.")
    cc_weight: float = Field(0.5, ge=0.0, le=1.0, description="Co-citation weight.")


class PaperGraphJobResult(BaseModel):
    graph: GraphDataSchema = Field(default_factory=GraphDataSchema)


class PaperGraphJobStatus(JobStatusBase):
    result: Optional[PaperGraphJobResult] = None


# ── Feature 2: Discovery Network ──────────────────────────────────────────────

class CreateCollectionRequest(BaseModel):
    """One or more seed papers — IDs or titles, resolved the same way as
    SimilarityGraphRequest.paper_id."""
    seed_paper_ids: List[str] = Field(
        ..., min_length=1, description="Semantic Scholar paperIds or title strings."
    )


ExpandRelationship = Literal["earlier", "later", "similar", "authors"]


class ExpandCollectionRequest(BaseModel):
    node_id: str = Field(..., description="S2 paperId of the node to expand.")
    relationship: ExpandRelationship = Field(
        ...,
        description=(
            "earlier = references of this paper; "
            "later = papers citing this paper; "
            "similar = S2 Recommendations API; "
            "authors = other papers by this paper's authors."
        ),
    )


class CollectionResponse(BaseModel):
    collection_id: str
    graph: GraphDataSchema


class ExpandJobResult(BaseModel):
    collection_id: str
    graph: GraphDataSchema


class ExpandJobStatus(JobStatusBase):
    result: Optional[ExpandJobResult] = None
