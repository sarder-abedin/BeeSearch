"""backend/app/services/paper_graph_service.py
───────────────────────────────────────────────
Service layer for the paper discovery features.

Feature 1 — Similarity Graph:
  run_similarity_graph() resolves a seed paper, fetches its references and
  citations, builds a candidate pool, scores each candidate with
  bibliographic coupling + co-citation, and returns the top-N as a graph.

Feature 2 — Discovery Network:
  create_collection() resolves seed papers and creates an in-memory collection.
  expand_collection() fetches a node's neighborhood for a given relationship
  type and merges new papers/edges into the existing collection.

Rate-limit degradation: any S2 API failure is logged and results in a
partial graph (partial=True, notice set) rather than an error response.
Every node/edge shown to the user traces back to an actual API response.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional

from paper_graph.collection_store import get_store
from paper_graph.graph_builder import GraphEdge, build_discovery_graph, build_similarity_graph
from paper_graph.s2_client import PaperNode, get_client
from paper_graph.similarity import rank_candidates

from ..schemas.paper_graph import (
    CollectionResponse,
    CreateCollectionRequest,
    ExpandCollectionRequest,
    GraphDataSchema,
    GraphEdgeSchema,
    PaperNodeSchema,
    SimilarityGraphRequest,
)

logger = logging.getLogger(__name__)

StreamCallback = Callable[[str, Dict[str, Any]], None]

# Maximum candidates fetched per list (refs + citations).
# The free-tier budget is ~100 req/5 min; fetching 200 refs and 200 citations
# for 1 origin plus ~50 batch-fetches for metadata fits comfortably.
_CANDIDATE_CAP = 100  # per list (references / citations)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_schema(node: PaperNode) -> PaperNodeSchema:
    return PaperNodeSchema(
        id=node.id,
        title=node.title,
        authors=node.authors,
        year=node.year,
        venue=node.venue,
        abstract=node.abstract,
        citation_count=node.citation_count,
        url=node.url,
    )


def _graph_to_schema(nodes, edges, partial=False, notice="") -> GraphDataSchema:
    return GraphDataSchema(
        nodes=[_to_schema(n) for n in nodes],
        edges=[
            GraphEdgeSchema(
                source=e.source,
                target=e.target,
                weight=e.weight,
                edge_type=e.edge_type,
            )
            for e in edges
        ],
        partial=partial,
        notice=notice,
    )


# arXiv ID: optional "arXiv:" prefix + YYMM.NNNNN[vN]
_ARXIV_RE = re.compile(r'^(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$', re.IGNORECASE)
# DOI bare or prefixed: 10.XXXX/anything
_DOI_RE = re.compile(r'^(?:DOI:)?(10\.\d{4,}/.+)$', re.IGNORECASE)
# PubMed ID: plain digits or PMID:digits
_PMID_RE = re.compile(r'^(?:PMID:|pmid:)?(\d{6,9})$')


def _extract_id_from_url(url: str) -> Optional[str]:
    """Extract an S2-compatible identifier from a paper page URL.

    Handles:
      - arxiv.org/abs/XXXX.XXXXX  → "arXiv:XXXX.XXXXX"
      - doi.org/10.xxx/yyy        → "DOI:10.xxx/yyy"
      - pubmed.ncbi.nlm.nih.gov/NNNNNNN → "PMID:NNNNNNN"
      - URLs with DOI in path (Springer, Science, Wiley, etc.)
        e.g. link.springer.com/article/10.1007/s00xxx
             science.org/doi/10.1126/science.xxx
             onlinelibrary.wiley.com/doi/10.1111/xxx
    """
    # arXiv URL
    m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', url, re.IGNORECASE)
    if m:
        return f"arXiv:{m.group(1)}"
    # doi.org redirect
    m = re.search(r'(?:dx\.)?doi\.org/(10\.\d{4,}/.+?)(?:[?#\s]|$)', url, re.IGNORECASE)
    if m:
        return f"DOI:{m.group(1).rstrip('/')}"
    # DOI embedded in URL path (Springer, Science, Wiley, F1000Research, …)
    m = re.search(r'/(?:doi/(?:full/|abs/|10\.|)?|article/)(10\.\d{4,}/[^\s?#]+)', url, re.IGNORECASE)
    if m:
        return f"DOI:{m.group(1).rstrip('/')}"
    # PubMed
    m = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d{6,9})', url, re.IGNORECASE)
    if m:
        return f"PMID:{m.group(1)}"
    return None


def _resolve_paper_id(paper_id_or_title: str) -> Optional[PaperNode]:
    """Resolve a paper_id string to a PaperNode.

    Accepts, in priority order:
      - 40-hex-char S2 paperId
      - arXiv ID      (2312.01234 or arXiv:2312.01234)
      - DOI           (10.1109/xxx or DOI:10.xxx/yyy)
      - PubMed ID     (PMID:12345678 or plain 6–9-digit number)
      - Paper page URL from arxiv.org, doi.org, Springer, Science, Wiley,
        PubMed — DOI/arXiv ID extracted from URL then looked up via S2
      - Google Scholar URL — extracts the search query (?q=…) and runs a
        title search on Semantic Scholar
      - Free-text title, with progressive fallbacks:
          full title → first 100 chars → first 8 words
        (S2 BM25 search degrades on long exact titles)

    Note: IEEE Xplore, ScienceDirect, and Springer papers all have DOIs shown
    on their paper pages ("Cite this article" / "Export citation" sections).
    Paste the DOI directly (e.g. 10.1109/TNNLS.2023.xxx) or the full DOI URL.
    """
    client = get_client()
    q = paper_id_or_title.strip()

    # 40-char hex S2 paperId
    if len(q) == 40 and all(c in "0123456789abcdefABCDEF" for c in q):
        node = client.get_paper(q)
        if node:
            return node

    # arXiv ID  (S2 accepts "arXiv:YYMM.NNNNN")
    m = _ARXIV_RE.match(q)
    if m:
        node = client.get_paper(f"arXiv:{m.group(1)}")
        if node:
            return node

    # DOI  (S2 accepts "DOI:10.xxx/yyy")
    m = _DOI_RE.match(q)
    if m:
        node = client.get_paper(f"DOI:{m.group(1)}")
        if node:
            return node

    # PubMed ID  (S2 accepts "PMID:nnnnnnn")
    m = _PMID_RE.match(q)
    if m:
        node = client.get_paper(f"PMID:{m.group(1)}")
        if node:
            return node

    # URL from any major publisher — extract arXiv ID, DOI, or PMID from URL
    if q.startswith(("http://", "https://")):
        if "scholar.google." in q:
            # Google Scholar doesn't embed paper IDs in URLs; extract the search query
            try:
                from urllib.parse import parse_qs, unquote_plus, urlparse
                gs_params = parse_qs(urlparse(q).query)
                gs_title = (gs_params.get("q") or [""])[0]
                if gs_title:
                    results = client.search_paper(unquote_plus(gs_title), limit=3)
                    if results:
                        return results[0]
            except Exception:
                pass
        else:
            extracted = _extract_id_from_url(q)
            if extracted:
                node = client.get_paper(extracted)
                if node:
                    return node

    # Title search — three progressively shorter queries
    results = client.search_paper(q, limit=3)
    if results:
        return results[0]
    if len(q) > 100:
        results = client.search_paper(q[:100], limit=3)
        if results:
            return results[0]
    words = q.split()
    if len(words) > 8:
        results = client.search_paper(" ".join(words[:8]), limit=3)
        if results:
            return results[0]

    return None


# ── Feature 1: Similarity Graph ───────────────────────────────────────────────

def run_similarity_graph(
    req: SimilarityGraphRequest,
    cb: StreamCallback,
) -> Dict[str, Any]:
    """Compute a one-shot similarity graph from a single origin paper.

    Steps:
    1. Resolve origin paper
    2. Fetch origin's references and citations
    3. Build candidate pool (union of refs + citers, capped at _CANDIDATE_CAP each)
    4. Fetch each candidate's own references (needed for bibliographic coupling)
    5. Build citing_index for co-citation
    6. Score and rank with similarity.rank_candidates()
    7. Batch-fetch metadata for the top-N candidates
    8. Return GraphData
    """
    client = get_client()
    partial = False
    notices: List[str] = []

    cb("resolving", {"step": "Resolving seed paper…"})
    origin = _resolve_paper_id(req.paper_id)
    if origin is None:
        raise ValueError(
            f"Could not find '{req.paper_id}' on Semantic Scholar. "
            "Try pasting the Semantic Scholar paper ID directly (the 40-character "
            "hex string from the paper's URL at semanticscholar.org)."
        )

    cb("fetching_refs", {"step": f"Fetching references and citations for '{origin.title[:60]}'…"})
    origin_refs_list = client.get_references(origin.id)[:_CANDIDATE_CAP]
    origin_citers_list = client.get_citations(origin.id)[:_CANDIDATE_CAP]

    # Candidate pool = union of references + citers (excluding origin itself)
    candidate_ids = list(
        {r for r in origin_refs_list + origin_citers_list if r != origin.id}
    )

    if not candidate_ids:
        # No candidates at all — return a graph with only the origin node
        return {
            "graph": _graph_to_schema(
                [origin], [],
                partial=True,
                notice="No related papers found via Semantic Scholar for this paper.",
            ).model_dump()
        }

    cb("scoring", {"step": f"Scoring {len(candidate_ids)} candidates…"})

    # Build refs_index: origin_id → frozenset, candidate_id → frozenset
    refs_index: Dict[str, FrozenSet[str]] = {
        origin.id: frozenset(origin_refs_list)
    }
    for cid in candidate_ids:
        cid_refs = client.get_references(cid)
        refs_index[cid] = frozenset(cid_refs)

    # Build citing_index from the union of all candidates' citation lists
    # (who cites each paper in our pool).  We approximate: the origin's own
    # citers are already in origin_citers_list; for candidates we check
    # intersection of origin_refs_list citers.
    citing_index: Dict[str, FrozenSet[str]] = {
        origin.id: frozenset(origin_citers_list),
    }
    for cid in candidate_ids:
        # Papers that cite cid are expensive to fetch per-candidate under the
        # free rate limit.  Approximate: any candidate that cites origin also
        # co-cites another candidate citing origin = use candidate_ids as proxy.
        # For higher fidelity with a partner key, replace with:
        #   citing_index[cid] = frozenset(client.get_citations(cid))
        citing_index[cid] = frozenset(
            c for c in origin_citers_list if c in refs_index.get(cid, frozenset())
        )

    scored = rank_candidates(
        origin_id=origin.id,
        candidate_ids=candidate_ids,
        refs_index=refs_index,
        citing_index=citing_index,
        top_n=req.top_n,
        bc_weight=req.bc_weight,
        cc_weight=req.cc_weight,
    )

    if len(candidate_ids) > req.top_n * 3:
        partial = True
        notices.append(
            f"Candidate pool capped at {_CANDIDATE_CAP} references and "
            f"{_CANDIDATE_CAP} citations; some related papers may not appear."
        )

    cb("fetching_metadata", {"step": f"Fetching metadata for top {len(scored)} papers…"})
    top_ids = [pid for pid, _ in scored]
    meta_nodes = client.batch_get_papers(top_ids)
    paper_meta = {n.id: n for n in meta_nodes}

    graph = build_similarity_graph(origin, scored, paper_meta)
    graph.partial = partial
    graph.notice = " ".join(notices)

    return {
        "graph": _graph_to_schema(
            graph.nodes, graph.edges, graph.partial, graph.notice
        ).model_dump()
    }


# ── Feature 2: Discovery Network ─────────────────────────────────────────────

def create_collection(req: CreateCollectionRequest) -> CollectionResponse:
    """Resolve seed papers and create a new in-memory collection."""
    seed_nodes: List[PaperNode] = []
    for paper_id_or_title in req.seed_paper_ids:
        node = _resolve_paper_id(paper_id_or_title)
        if node:
            seed_nodes.append(node)
        else:
            logger.warning("Could not resolve seed paper: %r", paper_id_or_title)

    if not seed_nodes:
        raise ValueError(
            "None of the provided seed papers could be found on Semantic Scholar. "
            "Try using Semantic Scholar paper IDs (40-character hex strings from "
            "the paper URLs at semanticscholar.org) instead of titles."
        )

    store = get_store()
    collection = store.create(seed_nodes)

    graph = build_discovery_graph(collection.paper_nodes, collection.edges)
    return CollectionResponse(
        collection_id=collection.id,
        graph=_graph_to_schema(graph.nodes, graph.edges),
    )


def expand_collection(
    collection_id: str,
    req: ExpandCollectionRequest,
    cb: StreamCallback,
) -> Dict[str, Any]:
    """Fetch a node's neighborhood for the given relationship and merge into the collection.

    Relationship types:
      earlier  — references of this paper (earlier work it builds on)
      later    — papers citing this paper (later work building on it)
      similar  — S2 Recommendations API with this paper as a positive example
      authors  — other publications by this paper's authors
    """
    store = get_store()
    client = get_client()

    collection = store.get(collection_id)
    if collection is None:
        raise ValueError(f"Collection {collection_id!r} not found.")

    node_id = req.node_id
    relationship = req.relationship

    cb("expanding", {"step": f"Fetching {relationship} papers for node {node_id[:8]}…"})

    new_nodes: List[PaperNode] = []
    new_edges: List[GraphEdge] = []

    if relationship == "earlier":
        ref_ids = client.get_references(node_id)[:50]
        meta = client.batch_get_papers(ref_ids)
        for n in meta:
            new_nodes.append(n)
            new_edges.append(GraphEdge(source=node_id, target=n.id, weight=1.0, edge_type="reference"))

    elif relationship == "later":
        cit_ids = client.get_citations(node_id)[:50]
        meta = client.batch_get_papers(cit_ids)
        for n in meta:
            new_nodes.append(n)
            new_edges.append(GraphEdge(source=n.id, target=node_id, weight=1.0, edge_type="citation"))

    elif relationship == "similar":
        # Use all current collection IDs as positive examples for richer recommendations
        positive_ids = list(collection.paper_ids)[:100]
        recs = client.get_recommendations(positive_ids)
        for n in recs:
            new_nodes.append(n)
            new_edges.append(GraphEdge(source=node_id, target=n.id, weight=1.0, edge_type="recommendation"))

    elif relationship == "authors":
        # Fetch the focal paper to get its author list, then pull each author's papers
        focal = client.get_paper(node_id)
        if focal:
            # S2 author lookup requires the raw author object with an authorId field
            # We re-fetch with author fields to get authorIds
            try:
                import requests as _req
                from config.settings import get_settings as _cfg
                _s = _cfg()
                headers = {"User-Agent": "BeeSearch/1.0"}
                if _s.semantic_scholar_api_key:
                    headers["x-api-key"] = _s.semantic_scholar_api_key
                r = _req.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/{node_id}",
                    params={"fields": "authors"},
                    headers=headers,
                    timeout=15,
                )
                r.raise_for_status()
                authors_data = r.json().get("authors", [])
                for author in authors_data[:3]:  # cap to 3 authors to stay within rate limit
                    author_id = author.get("authorId")
                    if author_id:
                        author_papers = client.get_author_papers(author_id)
                        for n in author_papers[:15]:
                            if n.id != node_id:
                                new_nodes.append(n)
                                new_edges.append(
                                    GraphEdge(source=node_id, target=n.id, weight=1.0, edge_type="co_author")
                                )
            except Exception as exc:
                logger.warning("Author papers fetch failed: %s", exc)

    cb("merging", {"step": "Merging into collection…"})
    updated = store.add_papers(collection_id, new_nodes, new_edges)
    if updated is None:
        raise ValueError(f"Collection {collection_id!r} disappeared during expand.")

    graph = build_discovery_graph(updated.paper_nodes, updated.edges)
    return {
        "collection_id": collection_id,
        "graph": _graph_to_schema(graph.nodes, graph.edges).model_dump(),
    }
