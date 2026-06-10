"""
tests/test_citation_network.py
───────────────────────────────
Unit tests for tools/citation_network.py:

  - network_stats(): named isolated_papers (not just a count)
  - build_citation_network(): external_counts for citations to papers
    outside the included set (gap-finder)
  - find_gap_candidates(): filtering, sorting, capping, and metadata
    enrichment of gap candidates

All Semantic Scholar HTTP calls are mocked — no network access required.
"""

from __future__ import annotations

from unittest.mock import patch

import networkx as nx

from tools.citation_network import (
    build_citation_network,
    find_gap_candidates,
    network_stats,
)


# ───────────────────────── network_stats ─────────────────────────


def test_network_stats_empty_graph():
    G = nx.DiGraph()
    stats = network_stats(G)

    assert stats["nodes"] == 0
    assert stats["edges"] == 0
    assert stats["isolated"] == 0
    assert stats["isolated_papers"] == []


def test_network_stats_isolated_papers():
    G = nx.DiGraph()
    G.add_node("smith2020")
    G.add_node("jones2019")
    G.add_node("lee2021")
    G.add_edge("smith2020", "jones2019", relation="cites")

    stats = network_stats(G)

    assert stats["nodes"] == 3
    assert stats["edges"] == 1
    assert stats["isolated"] == 1
    assert stats["isolated_papers"] == ["lee2021"]


def test_network_stats_most_cited_and_most_citing():
    G = nx.DiGraph()
    G.add_edge("a", "b", relation="cites")
    G.add_edge("c", "b", relation="cites")
    G.add_node("d")

    stats = network_stats(G)

    assert stats["most_cited"][0] == ("b", 2)
    assert ("a", 1) in stats["most_citing"]
    assert ("c", 1) in stats["most_citing"]
    assert stats["isolated_papers"] == ["d"]


# ──────────────────── build_citation_network ────────────────────

_PAPERS = [
    {"citation_key": "alpha2020", "title": "Alpha Paper", "year": 2020, "quality": "High"},
    {"citation_key": "beta2021", "title": "Beta Paper", "year": 2021, "quality": "Medium"},
]


def _fake_find_s2_id(title: str):
    return {"Alpha Paper": "S2-ALPHA", "Beta Paper": "S2-BETA"}.get(title)


def _fake_get_references(s2_id: str):
    # Both included papers cite the same external paper; alpha also cites beta.
    if s2_id == "S2-ALPHA":
        return ["S2-BETA", "S2-EXTERNAL"]
    if s2_id == "S2-BETA":
        return ["S2-EXTERNAL"]
    return []


@patch("tools.citation_network.time.sleep", return_value=None)
@patch("tools.citation_network._get_references", side_effect=_fake_get_references)
@patch("tools.citation_network._find_s2_id", side_effect=_fake_find_s2_id)
def test_build_citation_network_external_counts(mock_find, mock_refs, mock_sleep):
    G, node_meta, external_counts = build_citation_network(_PAPERS)

    assert G.number_of_nodes() == 2
    assert G.has_edge("alpha2020", "beta2021")
    assert external_counts == {"S2-EXTERNAL": 2}
    assert node_meta["alpha2020"]["s2_id"] == "S2-ALPHA"


@patch("tools.citation_network.time.sleep", return_value=None)
@patch("tools.citation_network._get_references", return_value=[])
@patch("tools.citation_network._find_s2_id", return_value=None)
def test_build_citation_network_no_s2_matches(mock_find, mock_refs, mock_sleep):
    G, node_meta, external_counts = build_citation_network(_PAPERS)

    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 0
    assert external_counts == {}
    assert all(meta["s2_id"] is None for meta in node_meta.values())


# ───────────────────────── find_gap_candidates ─────────────────────────


@patch("tools.citation_network.time.sleep", return_value=None)
@patch("tools.citation_network._get_paper_metadata")
def test_find_gap_candidates_filters_and_sorts(mock_meta, mock_sleep):
    mock_meta.side_effect = lambda s2_id: {
        "S2-A": {"title": "Paper A", "year": 2018, "venue": "Journal A", "url": "https://example.com/a"},
        "S2-B": {"title": "Paper B", "year": 2019, "venue": "Journal B", "url": "https://example.com/b"},
    }.get(s2_id)

    external_counts = {"S2-A": 3, "S2-B": 2, "S2-C": 1}

    results = find_gap_candidates(external_counts, min_citations=2)

    assert [r["s2_id"] for r in results] == ["S2-A", "S2-B"]
    assert results[0]["cited_by_count"] == 3
    assert results[0]["title"] == "Paper A"
    assert results[1]["cited_by_count"] == 2


@patch("tools.citation_network.time.sleep", return_value=None)
@patch("tools.citation_network._get_paper_metadata")
def test_find_gap_candidates_respects_max_candidates(mock_meta, mock_sleep):
    mock_meta.side_effect = lambda s2_id: {"title": s2_id, "year": 2020, "venue": "", "url": ""}

    external_counts = {f"S2-{i}": 10 - i for i in range(10)}

    results = find_gap_candidates(external_counts, min_citations=1, max_candidates=3)

    assert len(results) == 3
    assert results[0]["s2_id"] == "S2-0"


@patch("tools.citation_network.time.sleep", return_value=None)
@patch("tools.citation_network._get_paper_metadata", return_value=None)
def test_find_gap_candidates_no_metadata_skipped(mock_meta, mock_sleep):
    results = find_gap_candidates({"S2-A": 5}, min_citations=2)

    assert results == []


def test_find_gap_candidates_empty_input():
    assert find_gap_candidates({}) == []
