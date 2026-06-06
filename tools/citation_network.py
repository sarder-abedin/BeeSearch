"""
tools/citation_network.py
─────────────────────────
Ego citation network for a set of included papers (ego-only scope).

Checks edges *between* the included papers themselves: does paper A cite
paper B? Uses Semantic Scholar /paper/search + /paper/{id}/references.

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
    h = {"User-Agent": "ResearchBuddy/1.0"}
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


def build_citation_network(
    papers: List[Dict],
    max_papers: int = 30,
) -> Tuple[object, Dict[str, Dict]]:
    """
    Build an ego citation network from a list of included papers.

    Returns (nx.DiGraph, node_metadata):
      DiGraph nodes are citation_keys; edges are directed citations (A→B = A cites B)
      node_metadata maps citation_key → {title, year, quality, journal, s2_id}
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

    for ck_a, s2_a in ck_to_s2.items():
        for ref_id in _get_references(s2_a):
            ck_b = s2_to_ck.get(ref_id)
            if ck_b and ck_b != ck_a:
                G.add_edge(ck_a, ck_b, relation="cites")
        time.sleep(0.4)

    logger.info("Citation network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G, node_meta


def network_to_pyvis_html(G: object, node_meta: Dict[str, Dict]) -> str:
    """Convert networkx DiGraph to interactive Pyvis HTML string."""
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pip install pyvis")

    net = Network(height="500px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
    net.barnes_hut(spring_length=120)

    quality_colors = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}

    for node_id, data in node_meta.items():
        color = quality_colors.get(data.get("quality", "Medium"), "#f39c12")
        label = f"{node_id}\n({data.get('year', '?')})"
        title_text = f"{data.get('title', '')}\n{data.get('journal', '')}"
        net.add_node(node_id, label=label, title=title_text, color=color, size=15)

    for src, dst in G.edges():
        net.add_edge(src, dst, arrows="to", color="#888888")

    return net.generate_html()


def network_stats(G: object) -> Dict[str, object]:
    """Return basic graph statistics."""
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "most_cited": [], "most_citing": [], "isolated": 0}
    in_deg = sorted(G.in_degree(), key=lambda x: -x[1])
    out_deg = sorted(G.out_degree(), key=lambda x: -x[1])
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "most_cited": [(n, d) for n, d in in_deg[:5] if d > 0],
        "most_citing": [(n, d) for n, d in out_deg[:5] if d > 0],
        "isolated": sum(1 for n in G.nodes() if G.degree(n) == 0),
    }
