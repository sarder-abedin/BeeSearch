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
    """Build request headers for the Semantic Scholar API, adding the API key if configured."""
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
            # S2's free tier rate-limits aggressively; back off once and give up rather
            # than blocking the whole network build on a single lookup.
            time.sleep(3)
            return None
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            found = data[0].get("title", "")
            # S2 search is fuzzy and can return a near-match instead of the exact paper;
            # comparing only the first 15 normalised chars tolerates punctuation/casing
            # differences while still rejecting an unrelated top hit.
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
            # Abstract kept on node_meta so classify_citation_stances() has text to
            # reason over; not used for rendering (the tooltip uses title/journal).
            "abstract": paper.get("abstract", "") or "",
        }
        node_meta[ck] = meta
        G.add_node(ck, **meta)

        s2_id = _find_s2_id(title)
        if s2_id:
            ck_to_s2[ck] = s2_id
            node_meta[ck]["s2_id"] = s2_id
        # Pace requests below S2's unauthenticated rate limit (~1 req/sec); this loop
        # makes one request per paper, so 0.4s keeps us comfortably under that.
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
                # Not in the included set — track how often the corpus cites it so
                # find_gap_candidates can surface it as a possible screening miss.
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


# ── Smart Citations: stance classification (Supporting / Contrasting / Mentioning) ──

# scite.ai-style citation classification. Each directed citation edge (A cites B)
# is labelled with how A engages with B, inferred from the two papers' abstracts.
_STANCE_VALUES = ("Supporting", "Contrasting", "Mentioning")
_STANCE_EDGE_COLORS = {
    "Supporting": "#10B981",   # green — A's findings agree with / build on B
    "Contrasting": "#EF4444",  # red   — A disputes / contradicts B
    "Mentioning": "#888888",   # gray  — neutral reference, no clear stance (default)
}


def _stance_llm(model_name: str, num_ctx: int):
    """Build a deterministic (temperature 0) ChatOllama client for citation-stance classification.

    Imported lazily so importing this module (e.g. for network_stats in tests)
    doesn't require langchain_ollama. Temperature is pinned to 0.0 because this
    is a grading-style judgement, not generation — the same choice the
    self-reflective RAG grader makes.
    """
    import httpx
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=0.0,
        num_predict=128,
        num_ctx=num_ctx or cfg.num_ctx,
        sync_client_kwargs={"timeout": httpx.Timeout(120.0)},
    )


def _parse_stance(raw: str) -> Dict[str, str]:
    """Parse an LLM stance reply into ``{"stance", "confidence"}``.

    Tolerant of the small models' habits: finds the first of
    Supporting/Contrasting/Mentioning anywhere in the text (case-insensitive)
    and an optional high/medium/low confidence word. Defaults to
    ``Mentioning`` / ``low`` when nothing recognisable is present — the neutral,
    safe assumption, so a parse failure never invents a Supporting/Contrasting
    claim that isn't there.
    """
    text = (raw or "").lower()
    stance = "Mentioning"
    for s in _STANCE_VALUES:
        if s.lower() in text:
            stance = s
            break
    confidence = "low"
    for c in ("high", "medium", "low"):
        if c in text:
            confidence = c
            break
    return {"stance": stance, "confidence": confidence}


def classify_single_citation(citing_meta: Dict, cited_meta: Dict, llm) -> Dict[str, str]:
    """Classify how the citing paper engages with the cited paper, from their abstracts.

    Returns ``{"stance", "confidence"}``. Any LLM/parse error degrades to the
    neutral ``Mentioning`` / ``low`` default rather than raising.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    citing_txt = f"{citing_meta.get('title','')}\n{(citing_meta.get('abstract','') or '')[:700]}"
    cited_txt = f"{cited_meta.get('title','')}\n{(cited_meta.get('abstract','') or '')[:700]}"
    system = (
        "You classify a citation: how does the CITING paper engage with the CITED paper?\n"
        "Answer with exactly one label and a confidence:\n"
        "- Supporting: the citing paper's results agree with, confirm, or build on the cited paper.\n"
        "- Contrasting: the citing paper disputes, contradicts, or reports findings against the cited paper.\n"
        "- Mentioning: a neutral reference (background, methods, definitions) with no clear agreement or disagreement.\n"
        "Reply in the form: <Supporting|Contrasting|Mentioning> (confidence: high|medium|low)."
    )
    human = f"CITING PAPER:\n{citing_txt}\n\nCITED PAPER:\n{cited_txt}"
    try:
        raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
        return _parse_stance(raw)
    except Exception as e:
        logger.debug("citation stance classification failed: %s", e)
        return {"stance": "Mentioning", "confidence": "low"}


def classify_citation_stances(
    G: object,
    node_meta: Dict[str, Dict],
    model_name: str,
    num_ctx: int,
    max_edges: int = 40,
) -> Dict[str, int]:
    """Annotate each citation edge of *G* with a ``stance`` and ``confidence`` in place.

    For every directed edge A→B (up to ``max_edges`` to bound LLM calls), classify
    how A engages with B from the two papers' abstracts and store the result on the
    edge (``G[a][b]["stance"]`` / ``["confidence"]``). Edges beyond the cap, or whose
    endpoints lack abstracts, are left at the neutral ``Mentioning`` default so the
    renderer can colour them consistently.

    Returns a summary count, e.g. ``{"Supporting": 3, "Contrasting": 1,
    "Mentioning": 6, "classified": 4}`` (``classified`` = edges actually sent to the
    LLM). Never raises — a per-edge failure degrades that edge to ``Mentioning``.
    """
    counts = {"Supporting": 0, "Contrasting": 0, "Mentioning": 0, "classified": 0}
    edges = list(G.edges())
    if not edges:
        return counts

    llm = _stance_llm(model_name, num_ctx)
    for i, (a, b) in enumerate(edges):
        if i >= max_edges:
            # Over the cap — leave as neutral so the graph stays internally consistent.
            G[a][b].setdefault("stance", "Mentioning")
            G[a][b].setdefault("confidence", "low")
            counts["Mentioning"] += 1
            continue
        meta_a = node_meta.get(a, {})
        meta_b = node_meta.get(b, {})
        if not (meta_a.get("abstract") or meta_b.get("abstract")):
            G[a][b]["stance"] = "Mentioning"
            G[a][b]["confidence"] = "low"
            counts["Mentioning"] += 1
            continue
        verdict = classify_single_citation(meta_a, meta_b, llm)
        G[a][b]["stance"] = verdict["stance"]
        G[a][b]["confidence"] = verdict["confidence"]
        counts[verdict["stance"]] = counts.get(verdict["stance"], 0) + 1
        counts["classified"] += 1

    logger.info(
        "Citation stances: %d supporting, %d contrasting, %d mentioning (%d classified)",
        counts["Supporting"], counts["Contrasting"], counts["Mentioning"], counts["classified"],
    )
    return counts


def network_to_pyvis_html(G: object, node_meta: Dict[str, Dict]) -> str:
    """Convert networkx DiGraph to interactive Pyvis HTML string.

    If edges carry a ``stance`` attribute (set by ``classify_citation_stances``),
    they are coloured by it — green Supporting, red Contrasting, gray Mentioning —
    and the stance/confidence is shown in the edge tooltip. Edges with no stance
    fall back to neutral gray, so an un-classified network renders exactly as before.
    """
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

    for src, dst, data in G.edges(data=True):
        stance = data.get("stance")
        color = _STANCE_EDGE_COLORS.get(stance, "#888888")
        edge_title = f"{stance} (confidence: {data.get('confidence', '?')})" if stance else "cites"
        net.add_edge(src, dst, arrows="to", color=color, title=edge_title)

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
