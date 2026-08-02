"""paper_graph/graph_builder.py
──────────────────────────────
Assembles the node/edge JSON the frontend consumes from scored candidate lists
and collection state.  No I/O; all inputs are plain Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .s2_client import PaperNode


# ── Wire types ────────────────────────────────────────────────────────────────

@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float        # similarity score or 1.0 for structural edges
    edge_type: str       # "similarity" | "reference" | "citation" | "recommendation" | "co_author"


@dataclass
class GraphData:
    nodes: List[PaperNode]
    edges: List[GraphEdge]
    partial: bool = False   # True when rate-limits cut the candidate pool short
    notice: str = ""        # Human-readable explanation shown in the UI banner


# ── Builders ─────────────────────────────────────────────────────────────────

def build_similarity_graph(
    origin: PaperNode,
    scored_candidates: List[Tuple[str, float]],
    paper_meta: Dict[str, PaperNode],
) -> GraphData:
    """Build the one-shot similarity graph for Feature 1.

    Parameters
    ----------
    origin:            the seed paper node
    scored_candidates: list of (paper_id, score) from similarity.rank_candidates()
    paper_meta:        dict of all PaperNode objects fetched for the candidates
    """
    nodes: List[PaperNode] = [origin]
    edges: List[GraphEdge] = []

    for paper_id, score in scored_candidates:
        node = paper_meta.get(paper_id)
        if node is None:
            continue
        nodes.append(node)
        edges.append(
            GraphEdge(
                source=origin.id,
                target=paper_id,
                weight=round(score, 4),
                edge_type="similarity",
            )
        )

    return GraphData(nodes=nodes, edges=edges)


def build_discovery_graph(
    paper_nodes: Dict[str, PaperNode],
    edges: List[GraphEdge],
    partial: bool = False,
    notice: str = "",
) -> GraphData:
    """Build the incremental discovery graph for Feature 2.

    Parameters
    ----------
    paper_nodes: all PaperNode objects currently in the collection
    edges:       all edges accumulated so far
    """
    return GraphData(
        nodes=list(paper_nodes.values()),
        edges=edges,
        partial=partial,
        notice=notice,
    )
