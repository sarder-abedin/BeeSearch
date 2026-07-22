"""
tests/test_citation_context.py
─────────────────────────────────
Unit tests for tools/citation_context.py (Phase 4 — Citation Context).

The sentence-matching core is pure and tested directly; the network fetch
(_fetch_fulltext) is mocked so extract_citation_context's ok / unavailable /
not_found status paths are all exercised without touching the network.
"""

from __future__ import annotations

from unittest.mock import patch

from tools.citation_context import (
    _best_fulltext_url,
    _distinctive_title_tokens,
    _first_author_surname,
    extract_citation_context,
    find_citation_mentions,
    split_sentences,
)


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_first_author_surname_both_orderings():
    """Surname is recovered from both 'Given Family' and 'Family, Given'."""
    assert _first_author_surname(["John Smith"]) == "Smith"
    assert _first_author_surname(["Smith, John"]) == "Smith"
    assert _first_author_surname([]) == ""


def test_distinctive_title_tokens_skips_stopwords():
    """Distinctive tokens skip common words and prefer the longest."""
    toks = _distinctive_title_tokens("A Novel Deep Learning Approach for Protein Folding")
    assert "Protein" in toks or "Folding" in toks
    assert "the" not in [t.lower() for t in toks]


def test_split_sentences_collapses_linebreaks():
    """PDF line-broken text is collapsed and split on sentence boundaries."""
    sents = split_sentences("First sentence.\nStill first? No.\nSecond one here.")
    assert sents[0] == "First sentence."
    assert any("Second one here" in s for s in sents)


# ── find_citation_mentions ───────────────────────────────────────────────────

def test_find_citation_mentions_by_author_surname():
    """A sentence naming the cited first author is returned, matched_on author."""
    text = ("We extend earlier work. Smith et al. (2020) reported a positive effect. "
            "Unrelated content follows.")
    cited = {"authors": ["Jane Smith"], "year": 2020, "title": "Positive Effects"}
    out = find_citation_mentions(text, cited)
    assert len(out) == 1
    assert out[0]["matched_on"] == "author:Smith"
    assert "Smith et al." in out[0]["sentence"]


def test_find_citation_mentions_by_title_token_with_year():
    """With no surname present, a distinctive title token plus the year matches."""
    text = "The Folding dynamics were first characterised in 2019 by other groups."
    cited = {"authors": ["Zhang"], "year": 2019, "title": "Protein Folding Dynamics"}
    out = find_citation_mentions(text, cited)
    assert out and out[0]["matched_on"].startswith("title:")


def test_find_citation_mentions_none_when_absent():
    """No surname/title/year signal in the text → no mentions."""
    cited = {"authors": ["Obscure"], "year": 1990, "title": "Unrelated Topic"}
    assert find_citation_mentions("Nothing relevant in this text.", cited) == []


def test_find_citation_mentions_caps_results():
    """No more than max_contexts sentences are returned."""
    text = " ".join(f"Smith found result {i}." for i in range(10))
    cited = {"authors": ["Smith"], "year": 2020, "title": "X"}
    assert len(find_citation_mentions(text, cited, max_contexts=2)) == 2


# ── _best_fulltext_url ───────────────────────────────────────────────────────

def test_best_fulltext_url_prefers_open_access_pdf():
    """An explicit openAccessPdf url wins over a plain url."""
    paper = {"openAccessPdf": {"url": "https://oa.org/p.pdf"}, "url": "https://doi.org/10.x"}
    assert _best_fulltext_url(paper) == "https://oa.org/p.pdf"


def test_best_fulltext_url_skips_s2_api_and_empty():
    """A Semantic Scholar API url is not usable full text; no url → empty string."""
    assert _best_fulltext_url({"url": "https://api.semanticscholar.org/paper/abc"}) == ""
    assert _best_fulltext_url({}) == ""


# ── extract_citation_context (status paths) ──────────────────────────────────

def test_extract_unavailable_without_url():
    """No fetchable URL → 'unavailable', no contexts, no network."""
    result = extract_citation_context({"title": "A", "authors": ["X"]}, {"title": "B", "authors": ["Y"]})
    assert result["status"] == "unavailable"
    assert result["contexts"] == []


def test_extract_ok_when_text_contains_citation():
    """Fetched full text containing the citation → 'ok' with the matching sentence."""
    citing = {"title": "A", "authors": ["X"], "url": "https://oa.org/a.pdf"}
    cited = {"title": "B", "authors": ["Jane Smith"], "year": 2020}
    with patch("tools.citation_context._fetch_fulltext",
               return_value=("Smith et al. (2020) found a clear effect.", "pdf")):
        result = extract_citation_context(citing, cited)
    assert result["status"] == "ok"
    assert result["contexts"] and "Smith" in result["contexts"][0]["sentence"]


def test_extract_not_found_when_text_lacks_citation():
    """Fetched text without the cited work → 'not_found' (read succeeded, no match)."""
    citing = {"title": "A", "authors": ["X"], "url": "https://oa.org/a.pdf"}
    cited = {"title": "Obscure", "authors": ["Zzyzx"], "year": 1990}
    with patch("tools.citation_context._fetch_fulltext", return_value=("Totally unrelated prose.", "html")):
        result = extract_citation_context(citing, cited)
    assert result["status"] == "not_found"


def test_extract_unavailable_when_fetch_fails():
    """A failed fetch (empty text) → 'unavailable', not a crash."""
    citing = {"title": "A", "authors": ["X"], "url": "https://oa.org/a.pdf"}
    with patch("tools.citation_context._fetch_fulltext", return_value=("", "")):
        result = extract_citation_context(citing, {"title": "B", "authors": ["Y"]})
    assert result["status"] == "unavailable"
