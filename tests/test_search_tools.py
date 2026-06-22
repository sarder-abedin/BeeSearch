"""
tests/test_search_tools.py
───────────────────────────
Unit tests for tools/search_tools.py::WebSearcher.

WebSearcher.search() must never raise — callers across the Research Notebook
Chat tab, the Research Report workflow, the Explain tab's online-search
router, and the MCP `web_search` tool all rely on it degrading to an empty
list on any failure, including the "no results" case that newer ddgs/
duckduckgo_search releases raise as an exception rather than returning [].

A fake `duckduckgo_search` module is injected into sys.modules for each test
so these run regardless of whether the real ddgs/duckduckgo_search package is
installed in the environment — no real network access is used.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from tools.search_tools import WebResult, WebSearcher


def _fake_ddgs_module(monkeypatch, *, results=None, side_effect=None):
    """Inject a fake `duckduckgo_search` module so `from duckduckgo_search import DDGS` resolves to it."""
    instance = MagicMock()
    if side_effect is not None:
        instance.text.side_effect = side_effect
    else:
        instance.text.return_value = iter(results or [])
    context_manager = MagicMock()
    context_manager.__enter__.return_value = instance
    context_manager.__exit__.return_value = False

    fake_module = types.ModuleType("duckduckgo_search")
    fake_module.DDGS = MagicMock(return_value=context_manager)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", fake_module)


def test_search_returns_mapped_results_on_success(monkeypatch):
    """A successful ddgs.text() call is mapped into WebResult objects with source='duckduckgo'."""
    raw = [
        {"title": "Ensemble Methods Explained", "href": "https://example.com/a", "body": "An overview..."},
        {"title": "Bagging vs Boosting", "href": "https://example.com/b", "body": "Comparison..."},
    ]
    _fake_ddgs_module(monkeypatch, results=raw)

    results = WebSearcher().search("ensemble methods", max_results=5)

    assert len(results) == 2
    assert all(isinstance(r, WebResult) for r in results)
    assert results[0].title == "Ensemble Methods Explained"
    assert results[0].url == "https://example.com/a"
    assert results[0].source == "duckduckgo"


def test_search_returns_empty_list_when_no_results_raised(monkeypatch):
    """Newer ddgs releases raise on zero results instead of returning []; search() must still return []."""
    _fake_ddgs_module(monkeypatch, side_effect=Exception("No results found."))

    results = WebSearcher().search("a query with no hits", max_results=5)

    assert results == []


def test_search_returns_empty_list_on_network_error(monkeypatch):
    """A genuine network/library failure is also swallowed, not raised, by search()."""
    _fake_ddgs_module(monkeypatch, side_effect=ConnectionError("boom"))

    results = WebSearcher().search("any query", max_results=5)

    assert results == []


def test_search_empty_results_list_is_handled(monkeypatch):
    """A clean empty iterable from ddgs.text() (no exception) also yields []."""
    _fake_ddgs_module(monkeypatch, results=[])

    results = WebSearcher().search("obscure query", max_results=5)

    assert results == []


def test_web_result_default_source_is_duckduckgo():
    """WebResult's default source should reflect the actual backend (DuckDuckGo), not the old Google label."""
    r = WebResult(title="t", url="u", snippet="s")
    assert r.source == "duckduckgo"
