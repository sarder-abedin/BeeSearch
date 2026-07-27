"""
tests/test_citation_stance.py
─────────────────────────────────
Unit tests for the Smart Citations stance classifier (Phase 3) added to
tools/citation_network.py:

  - _parse_stance(): tolerant parsing of the LLM's stance reply
  - classify_single_citation(): per-edge classification, safe default on error
  - classify_citation_stances(): in-place edge annotation, caps, neutral
    fallbacks, summary counts
  - network_to_pyvis_html(): edges coloured by stance (skipped if pyvis absent)

ChatOllama is mocked — no Ollama server or network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from tools.citation_network import (
    _parse_stance,
    classify_citation_stances,
    classify_single_citation,
)


# ── _parse_stance ────────────────────────────────────────────────────────────

def test_parse_stance_canonical_form():
    """The recommended '<Stance> (confidence: <level>)' form parses exactly."""
    assert _parse_stance("Supporting (confidence: high)") == {"stance": "Supporting", "confidence": "high"}


def test_parse_stance_embedded_in_prose():
    """A label buried in chatty output is still recovered, with its confidence word."""
    assert _parse_stance("I'd say this is Contrasting, medium confidence.") == {
        "stance": "Contrasting", "confidence": "medium"}


def test_parse_stance_unrecognisable_defaults_to_neutral():
    """Garbage defaults to the safe neutral Mentioning/low — never invents a stance."""
    assert _parse_stance("???") == {"stance": "Mentioning", "confidence": "low"}


# ── classify_single_citation ─────────────────────────────────────────────────

def test_classify_single_citation_uses_llm_verdict():
    """The classifier returns the parsed stance from the LLM reply."""
    llm = MagicMock()
    llm.invoke.return_value.content = "Supporting (confidence: high)"
    out = classify_single_citation({"title": "A", "abstract": "x"}, {"title": "B", "abstract": "y"}, llm)
    assert out == {"stance": "Supporting", "confidence": "high"}


def test_classify_single_citation_error_degrades_to_neutral():
    """An LLM exception degrades to the neutral default rather than raising."""
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("model down")
    out = classify_single_citation({"title": "A", "abstract": "x"}, {"title": "B", "abstract": "y"}, llm)
    assert out == {"stance": "Mentioning", "confidence": "low"}


# ── classify_citation_stances ────────────────────────────────────────────────

def test_classify_citation_stances_annotates_edges_in_place():
    """Each classified edge gets a stance/confidence; counts reflect the verdicts."""
    G = nx.DiGraph()
    G.add_edge("a", "b", relation="cites")
    node_meta = {"a": {"title": "A", "abstract": "we confirm B"}, "b": {"title": "B", "abstract": "base"}}
    fake = MagicMock()
    fake.invoke.return_value.content = "Supporting (confidence: high)"
    with patch("tools.citation_network._stance_llm", return_value=fake):
        counts = classify_citation_stances(G, node_meta, "m", 4096)
    assert G["a"]["b"]["stance"] == "Supporting"
    assert counts["Supporting"] == 1 and counts["classified"] == 1


def test_classify_citation_stances_skips_edges_without_abstracts():
    """An edge whose endpoints have no abstract is left neutral and not sent to the LLM."""
    G = nx.DiGraph()
    G.add_edge("x", "y", relation="cites")
    node_meta = {"x": {"title": "X"}, "y": {"title": "Y"}}
    fake = MagicMock()
    fake.invoke.return_value.content = "Supporting (confidence: high)"
    with patch("tools.citation_network._stance_llm", return_value=fake):
        counts = classify_citation_stances(G, node_meta, "m", 4096)
    assert G["x"]["y"]["stance"] == "Mentioning"
    assert counts["classified"] == 0
    fake.invoke.assert_not_called()


def test_classify_citation_stances_respects_max_edges_cap():
    """Edges beyond max_edges are left neutral (not classified) to bound LLM calls."""
    G = nx.DiGraph()
    for i in range(5):
        G.add_edge(f"s{i}", f"t{i}", relation="cites")
    node_meta = {n: {"title": n, "abstract": "text"} for n in G.nodes()}
    fake = MagicMock()
    fake.invoke.return_value.content = "Mentioning (confidence: low)"
    with patch("tools.citation_network._stance_llm", return_value=fake):
        counts = classify_citation_stances(G, node_meta, "m", 4096, max_edges=2)
    assert counts["classified"] == 2
    assert fake.invoke.call_count == 2


def test_classify_citation_stances_empty_graph():
    """An edgeless graph returns zeroed counts with no LLM construction."""
    counts = classify_citation_stances(nx.DiGraph(), {}, "m", 4096)
    assert counts == {"Supporting": 0, "Contrasting": 0, "Mentioning": 0, "classified": 0}


# ── network_to_pyvis_html (rendering) ────────────────────────────────────────

def test_pyvis_html_colours_edges_by_stance():
    """Supporting edges render green; the stance/confidence appears in the edge tooltip."""
    pytest.importorskip("pyvis")
    from tools.citation_network import network_to_pyvis_html
    G = nx.DiGraph()
    G.add_edge("a", "b", relation="cites", stance="Supporting", confidence="high")
    node_meta = {
        "a": {"title": "A", "year": 2020, "quality": "High", "journal": "J"},
        "b": {"title": "B", "year": 2019, "quality": "Medium", "journal": "K"},
    }
    html = network_to_pyvis_html(G, node_meta)
    assert "#10B981" in html  # green supporting edge colour
