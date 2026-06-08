"""
tools/trend_analyzer.py
───────────────────────
Research trend analysis: publication volume and citation velocity by year.

Primary source: CrossRef facet API for field-wide year-bucketed counts.
Supplement:     Semantic Scholar when CrossRef returns sparse results.
Instant layer:  Year distribution of the already-fetched SR corpus (no API).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()


def _crossref_year_counts(query: str, start_year: int, end_year: int) -> Dict[int, int]:
    """Use CrossRef facet API to get publication counts per year for a query."""
    counts: Dict[int, int] = {}
    try:
        resp = requests.get(
            cfg.crossref_base_url,
            params={
                "query": query,
                "rows": 0,
                "facet": "published:*",
                "mailto": cfg.crossref_email,
            },
            timeout=15,
        )
        resp.raise_for_status()
        facets = resp.json().get("message", {}).get("facets", {})
        for year_str, count in (facets.get("published") or {}).get("values", {}).items():
            try:
                year = int(year_str)
                if start_year <= year <= end_year:
                    counts[year] = int(count)
            except (ValueError, TypeError):
                pass
    except Exception as e:
        logger.warning("CrossRef facet query failed: %s", e)
    return dict(sorted(counts.items()))


def _s2_year_counts(query: str) -> Dict[int, int]:
    """Query Semantic Scholar and bucket results by year."""
    counts: Dict[int, int] = {}
    try:
        headers = {"User-Agent": "BeeSearch/1.0"}
        if cfg.semantic_scholar_api_key:
            headers["x-api-key"] = cfg.semantic_scholar_api_key
        resp = requests.get(
            f"{cfg.semantic_scholar_base_url}/paper/search",
            params={"query": query, "limit": 100, "fields": "year"},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        for item in resp.json().get("data", []):
            year = item.get("year")
            if isinstance(year, int) and 2000 <= year <= 2026:
                counts[year] = counts.get(year, 0) + 1
    except Exception as e:
        logger.warning("Semantic Scholar trend query failed: %s", e)
    return dict(sorted(counts.items()))


def _classify_trend(year_counts: Dict[int, int]) -> str:
    if len(year_counts) < 4:
        return "insufficient data"
    years = sorted(year_counts)
    recent = years[-3:]
    earlier = years[-6:-3]
    if not earlier:
        return "insufficient data"
    r_avg = sum(year_counts[y] for y in recent) / len(recent)
    e_avg = sum(year_counts[y] for y in earlier) / len(earlier)
    if e_avg == 0:
        return "growing" if r_avg > 0 else "insufficient data"
    ratio = r_avg / e_avg
    if ratio > 1.2:
        return "growing"
    if ratio < 0.8:
        return "declining"
    return "stable"


def analyze_trends(
    research_question: str,
    search_queries: List[str] = None,
    corpus_papers: List[Dict] = None,
    start_year: int = 2000,
) -> Dict[str, Any]:
    """
    Analyze publication trends for a research area.

    Parameters
    ----------
    research_question : primary query
    search_queries    : additional SR queries (first 2 used as supplements)
    corpus_papers     : papers already fetched in the SR run
    start_year        : earliest year to include (default 2000)

    Returns
    -------
    dict with:
      crossref_by_year  — field-wide year counts from CrossRef
      corpus_by_year    — year distribution of fetched SR corpus
      combined_by_year  — merged/max view
      trend             — "growing" | "declining" | "stable" | "insufficient data"
      peak_year         — year with most publications
      total_field       — total CrossRef count across all years
      query_used        — primary query sent to CrossRef
    """
    end_year = 2026

    cr_counts = _crossref_year_counts(research_question, start_year, end_year)
    time.sleep(0.3)

    s2_counts: Dict[int, int] = {}
    if sum(cr_counts.values()) < 30 and search_queries:
        s2_counts = _s2_year_counts((search_queries or [research_question])[0])
        time.sleep(0.3)

    corpus_by_year: Dict[int, int] = {}
    for p in corpus_papers or []:
        year = p.get("year")
        if isinstance(year, (int, float)):
            y = int(year)
            if start_year <= y <= end_year:
                corpus_by_year[y] = corpus_by_year.get(y, 0) + 1

    combined: Dict[int, int] = {}
    for y in set(cr_counts) | set(s2_counts):
        combined[y] = max(cr_counts.get(y, 0), s2_counts.get(y, 0))
    combined = dict(sorted(combined.items()))

    trend_source = combined if combined else corpus_by_year
    trend = _classify_trend(trend_source)
    peak_year = max(combined, key=combined.get) if combined else None

    return {
        "crossref_by_year": cr_counts,
        "corpus_by_year": dict(sorted(corpus_by_year.items())),
        "combined_by_year": combined,
        "trend": trend,
        "peak_year": peak_year,
        "total_field": sum(combined.values()),
        "query_used": research_question,
    }


def trend_to_chart_data(trend_data: Dict[str, Any]) -> str:
    """Serialize trend data as JSON for Plotly/Streamlit rendering."""
    combined = trend_data.get("combined_by_year", {})
    corpus = trend_data.get("corpus_by_year", {})
    years = sorted(set(list(combined.keys()) + list(corpus.keys())))
    return json.dumps({
        "years": years,
        "field_counts": [combined.get(y, 0) for y in years],
        "corpus_counts": [corpus.get(y, 0) for y in years],
        "trend": trend_data.get("trend", "unknown"),
        "peak_year": trend_data.get("peak_year"),
        "total_field": trend_data.get("total_field", 0),
        "query_used": trend_data.get("query_used", ""),
    })
