"""
tools/citation_network.py
─────────────────────────
Ego citation network for a set of included papers (ego-only scope).

Checks edges *between* the included papers themselves: does paper A cite
paper B? Uses Semantic Scholar /paper/search + /paper/{id}/references.

Also tracks citations to papers *outside* the included set
(``external_counts``) so ``find_gap_candidates`` can surface papers that are
frequently cited by the corpus but were not themselves screened in — useful
for spotting gaps in a systematic review's coverage.

Returns a networkx DiGraph and a Pyvis HTML string for Streamlit.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

_S2_BASE = "https://api.semanticscholar.org/graph/v1"


def _headers() -> Dict[str, str]:
    h = {"User-Agent": "BeeSearch/1.0"}
    if cfg.semantic_scholar_api_key:
        h["x-api-key"] = cfg.semantic_scholar_api_key
    return h


def _find_s2_id(title: str) -> Optional[str]:
    """Find Semantic Scholar paper ID by title search."""
    try:
        resp = requests.get(
            f"{_S2_BASE}/paper/search",
            params={"query": title, "limit": 1, "fields": "paperId,title"},
            headers=_headers(),
            timeout=10,
        )
        if resp.status_code == 429:
            time.sleep(3)
            return None
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            found = data[0].get("title", "")
            norm_q = re.sub(r"\W+", "", title.lower())[:30]
            norm_f = re.sub(r"\W+", "", found.lower())[:30]
            if norm_q and norm_f and norm_q[:15] == norm_f[:15]:
                return data[0]["paperId"]
    except Exception as e:
        logger.debug("S2 ID lookup failed for '%s': %s", title[:40], e)
    return None


def _get_references(s2_id: str) -> List[str]:
    """Return S2 paper IDs that this paper cites."""
    try:
        resp = requests.get(
            f"{_S2_BASE}/paper/{s2_id}/references",
            params={"fields": "paperId", "limit": 100},
            headers=_headers(),
            timeout=10,
        )
        if resp.status_code == 429:
            time.sleep(3)
            return []
        resp.raise_for_status()
        return [
            r["citedPaper"]["paperId"]
            for r in resp.json().get("data", [])
            if r.get("citedPaper", {}).get("paperId")
        ]
    except Exception as e:
        logger.debug("S2 references failed for %s: %s", s2_id, e)
        return []


def _get_paper_metadata(
    s2_id: str, fields: str = "title,year,venue,url,externalIds"
) -> Optional[Dict]:
    """Fetch metadata for a single Semantic Scholar paper ID."""
    try:
        resp = requests.get(
            f"{_S2_BASE}/paper/{s2_id}",
            params={"fields": fields},
            headers=_headers(),
            timeout=10,
        )
        if resp.status_code == 429:
            time.sleep(3)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.debug("S2 metadata lookup failed for %s: %s", s2_id, e)
        return None


def get_paper_abstract(title: str) -> Optional[Dict]:
    """
    Look up a paper on Semantic Scholar by title and return its abstract
    and TL;DR summary.

    Used by the Notebook Citation Timeline feature's "enrich with
    abstracts" toggle to turn a bare cited-work title into a one-line gist.

    Returns ``{"title", "year", "abstract", "tldr", "url"}`` or ``None`` if
    the paper couldn't be found.
    """
    s2_id = _find_s2_id(title)
    if not s2_id:
        return None
    meta = _get_paper_metadata(s2_id, fields="title,year,abstract,tldr,url")
    if not meta:
        return None
    tldr = meta.get("tldr") or {}
    return {
        "title": meta.get("title") or title,
        "year": meta.get("year"),
        "abstract": meta.get("abstract") or "",
        "tldr": tldr.get("text") or "",
        "url": meta.get("url") or "",
    }


def build_citation_network(
    papers: List[Dict],
    max_papers: int = 30,
) -> Tuple[object, Dict[str, Dict], Dict[str, int]]:
    """
    Build an ego citation network from a list of included papers.

    Returns (nx.DiGraph, node_metadata, external_counts):
      DiGraph nodes are citation_keys; edges are directed citations (A→B = A cites B)
      node_metadata maps citation_key → {title, year, quality, journal, s2_id}
      external_counts maps S2 paper IDs *outside* the included set to the
      number of included papers that cite them (gap-finder candidates)
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("pip install networkx")

    papers = papers[:max_papers]
    G = nx.DiGraph()
    node_meta: Dict[str, Dict] = {}
    ck_to_s2: Dict[str, str] = {}

    for paper in papers:
        ck = paper.get("citation_key") or paper.get("title", "")[:30]
        title = paper.get("title", "")
        if not title:
            continue

        meta = {
            "title": title,
            "year": paper.get("year"),
            "quality": paper.get("quality", "Medium"),
            "journal": paper.get("journal", ""),
            "s2_id": None,
        }
        node_meta[ck] = meta
        G.add_node(ck, **meta)

        s2_id = _find_s2_id(title)
        if s2_id:
            ck_to_s2[ck] = s2_id
            node_meta[ck]["s2_id"] = s2_id
        time.sleep(0.4)

    s2_to_ck = {v: k for k, v in ck_to_s2.items()}
    external_counts: Dict[str, int] = {}

    for ck_a, s2_a in ck_to_s2.items():
        for ref_id in _get_references(s2_a):
            if ref_id == s2_a:
                continue
            ck_b = s2_to_ck.get(ref_id)
            if ck_b:
                if ck_b != ck_a:
                    G.add_edge(ck_a, ck_b, relation="cites")
            else:
                external_counts[ref_id] = external_counts.get(ref_id, 0) + 1
        time.sleep(0.4)

    logger.info(
        "Citation network: %d nodes, %d edges, %d external papers referenced",
        G.number_of_nodes(), G.number_of_edges(), len(external_counts),
    )
    return G, node_meta, external_counts


def find_gap_candidates(
    external_counts: Dict[str, int],
    min_citations: int = 2,
    max_candidates: int = 8,
) -> List[Dict]:
    """
    Identify papers frequently cited by the included set but not themselves
    included — candidates for a second screening pass.

    Returns a list of dicts sorted by ``cited_by_count`` descending, each:
      {s2_id, title, year, venue, url, cited_by_count}
    """
    candidates = sorted(
        (item for item in external_counts.items() if item[1] >= min_citations),
        key=lambda item: -item[1],
    )[:max_candidates]

    results: List[Dict] = []
    for s2_id, count in candidates:
        meta = _get_paper_metadata(s2_id)
        if not meta:
            continue
        results.append({
            "s2_id": s2_id,
            "title": meta.get("title") or "Unknown title",
            "year": meta.get("year"),
            "venue": meta.get("venue") or "",
            "url": meta.get("url") or "",
            "cited_by_count": count,
        })
        time.sleep(0.4)

    return results


def network_to_pyvis_html(G: object, node_meta: Dict[str, Dict]) -> str:
    """Convert networkx DiGraph to interactive Pyvis HTML string."""
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pip install pyvis")

    net = Network(height="500px", width="100%", directed=True, bgcolor="#0F172A", font_color="white")
    net.barnes_hut(spring_length=120)

    quality_colors = {"High": "#10B981", "Medium": "#F59E0B", "Low": "#EF4444"}

    for node_id, data in node_meta.items():
        color = quality_colors.get(data.get("quality", "Medium"), "#F59E0B")
        label = f"{node_id}\n({data.get('year', '?')})"
        title_text = f"{data.get('title', '')}\n{data.get('journal', '')}"
        net.add_node(node_id, label=label, title=title_text, color=color, size=15)

    for src, dst in G.edges():
        net.add_edge(src, dst, arrows="to", color="#888888")

    return net.generate_html()


def network_stats(G: object) -> Dict[str, object]:
    """Return basic graph statistics."""
    if G.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "most_cited": [],
            "most_citing": [],
            "isolated": 0,
            "isolated_papers": [],
        }
    in_deg = sorted(G.in_degree(), key=lambda x: -x[1])
    out_deg = sorted(G.out_degree(), key=lambda x: -x[1])
    isolated_papers = [n for n in G.nodes() if G.degree(n) == 0]
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "most_cited": [(n, d) for n, d in in_deg[:5] if d > 0],
        "most_citing": [(n, d) for n, d in out_deg[:5] if d > 0],
        "isolated": len(isolated_papers),
        "isolated_papers": isolated_papers,
    }
