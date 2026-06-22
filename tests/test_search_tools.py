"""
tests/test_search_tools.py
───────────────────────────
Unit tests for tools/search_tools.py::WebSearcher.

WebSearcher.search() must never raise — callers across the Research Notebook
Chat tab, the Research Report workflow, the Explain tab's online-search
router, and the MCP `web_search` tool all rely on it degrading to an empty
list on any failure, including the "no results" case that newer ddgs/
duckduckgo_search releases raise as an exception rather than returning [].

Also covers the research-domain re-ranking helpers (_domain_of,
_matches_any_domain, _research_rank_score) added so DuckDuckGo's generic
relevance/SEO ranking doesn't bury arxiv.org/nature.com/etc. results behind
SEO-optimized summary blogs — see tools/search_tools.py's "Web Searcher"
section comment for the rationale.

A fake `duckduckgo_search` module is injected into sys.modules for each test
so these run regardless of whether the real ddgs/duckduckgo_search package is
installed in the environment — no real network access is used.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from tools.search_tools import (
    WebResult,
    WebSearcher,
    _domain_of,
    _matches_any_domain,
    _research_rank_score,
)


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


# ── _domain_of ────────────────────────────────────────────────────────────────

def test_domain_of_extracts_lowercase_netloc():
    assert _domain_of("https://Arxiv.org/abs/1234") == "arxiv.org"


def test_domain_of_returns_empty_string_for_unparseable_url():
    assert _domain_of("") == ""


# ── _matches_any_domain ───────────────────────────────────────────────────────

def test_matches_any_domain_matches_exact_domain():
    assert _matches_any_domain("arxiv.org", ("arxiv.org",)) is True


def test_matches_any_domain_matches_subdomain():
    assert _matches_any_domain("pubmed.ncbi.nlm.nih.gov", ("ncbi.nlm.nih.gov",)) is True


def test_matches_any_domain_is_dot_boundary_safe():
    """A lookalike domain like fooarxiv.org must not match arxiv.org as a substring."""
    assert _matches_any_domain("fooarxiv.org", ("arxiv.org",)) is False


def test_matches_any_domain_false_when_no_candidates_match():
    assert _matches_any_domain("example.com", ("arxiv.org", "nature.com")) is False


# ── _research_rank_score ──────────────────────────────────────────────────────

def test_research_rank_score_zero_for_research_tld():
    assert _research_rank_score("https://web.mit.edu/paper.pdf") == 0


def test_research_rank_score_zero_for_recognized_research_domain():
    assert _research_rank_score("https://arxiv.org/abs/1234") == 0


def test_research_rank_score_zero_for_research_domain_subdomain():
    assert _research_rank_score("https://pubmed.ncbi.nlm.nih.gov/12345") == 0


def test_research_rank_score_one_for_ordinary_domain():
    assert _research_rank_score("https://example.com/blog-post") == 1


def test_research_rank_score_two_for_low_value_domain():
    assert _research_rank_score("https://www.pinterest.com/pin/123") == 2


def test_research_rank_score_one_for_unparseable_url():
    assert _research_rank_score("") == 1


# ── WebSearcher.search re-ranking and dedup ──────────────────────────────────

def test_search_reranks_research_domains_ahead_of_ordinary_and_low_value(monkeypatch):
    """A mixed batch is reordered so research domains lead and low-value domains trail, ties aside."""
    raw = [
        {"title": "Pinterest board", "href": "https://www.pinterest.com/pin/1", "body": "..."},
        {"title": "Some SEO blog", "href": "https://example.com/blog", "body": "..."},
        {"title": "arXiv preprint", "href": "https://arxiv.org/abs/9999", "body": "..."},
        {"title": "Nature article", "href": "https://www.nature.com/articles/x", "body": "..."},
    ]
    _fake_ddgs_module(monkeypatch, results=raw)

    results = WebSearcher().search("deep learning", max_results=4)

    urls = [r.url for r in results]
    assert urls.index("https://arxiv.org/abs/9999") < urls.index("https://example.com/blog")
    assert urls.index("https://www.nature.com/articles/x") < urls.index("https://example.com/blog")
    assert urls[-1] == "https://www.pinterest.com/pin/1"


def test_search_preserves_original_order_within_same_rank_tier(monkeypatch):
    """Same-tier results keep DuckDuckGo's own relative order (stable sort), not a guessed re-ranking."""
    raw = [
        {"title": "Ordinary result A", "href": "https://example.com/a", "body": "..."},
        {"title": "Ordinary result B", "href": "https://example.org/b", "body": "..."},
    ]
    _fake_ddgs_module(monkeypatch, results=raw)

    results = WebSearcher().search("query", max_results=2)

    assert [r.url for r in results] == ["https://example.com/a", "https://example.org/b"]


def test_search_deduplicates_by_url(monkeypatch):
    raw = [
        {"title": "Same page", "href": "https://example.com/x", "body": "..."},
        {"title": "Same page again", "href": "https://example.com/x", "body": "..."},
    ]
    _fake_ddgs_module(monkeypatch, results=raw)

    results = WebSearcher().search("query", max_results=5)

    assert len(results) == 1


def test_search_deduplicates_by_normalized_title(monkeypatch):
    """Different URLs (e.g. mirrors) carrying the same title are treated as one result."""
    raw = [
        {"title": "Attention Is All You Need", "href": "https://example.com/paper", "body": "..."},
        {"title": "Attention is all you need!", "href": "https://mirror.example.com/paper", "body": "..."},
    ]
    _fake_ddgs_module(monkeypatch, results=raw)

    results = WebSearcher().search("transformers", max_results=5)

    assert len(results) == 1


def test_search_overfetches_and_truncates_to_max_results(monkeypatch):
    """search() asks ddgs.text() for more than max_results so re-ranking has a real pool, then truncates."""
    instance = MagicMock()
    instance.text.return_value = iter([])
    context_manager = MagicMock()
    context_manager.__enter__.return_value = instance
    context_manager.__exit__.return_value = False
    fake_module = types.ModuleType("duckduckgo_search")
    fake_module.DDGS = MagicMock(return_value=context_manager)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", fake_module)

    WebSearcher().search("query", max_results=5)

    instance.text.assert_called_once_with("query", max_results=15)


def test_search_caps_overfetch_at_twenty(monkeypatch):
    instance = MagicMock()
    instance.text.return_value = iter([])
    context_manager = MagicMock()
    context_manager.__enter__.return_value = instance
    context_manager.__exit__.return_value = False
    fake_module = types.ModuleType("duckduckgo_search")
    fake_module.DDGS = MagicMock(return_value=context_manager)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", fake_module)

    WebSearcher().search("query", max_results=50)

    instance.text.assert_called_once_with("query", max_results=20)
