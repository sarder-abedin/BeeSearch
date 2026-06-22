"""
tests/test_research_report_web_queries.py
───────────────────────────────────────────
Unit tests for agents/graph.py::_clean_generated_query.

LLM-generated academic search queries often arrive wrapped in straight or
curly quotes (e.g. '"new object misclassification solutions"'). Left in
place, those quotes turn the query into an exact-phrase search against
arXiv/Google Scholar, which reliably returns 0 results for any specific
multi-word phrase — silently emptying the Research Report's references list
even when web/academic search otherwise works.

Pure stdlib — no network access or heavy deps required.
"""

from __future__ import annotations

import pytest

from agents.graph import _clean_generated_query


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('"new object misclassification solutions"', "new object misclassification solutions"),
        ("'adapting classifiers to novel objects'", "adapting classifiers to novel objects"),
        ("“curly quoted query”", "curly quoted query"),
        ("‘single curly quoted query’", "single curly quoted query"),
        ('3. "numbered and quoted query"', "numbered and quoted query"),
        ("- plain query no quotes", "plain query no quotes"),
        ("plain unquoted query", "plain unquoted query"),
        ("   padded with whitespace   ", "padded with whitespace"),
    ],
)
def test_clean_generated_query_strips_wrapping_quotes_and_bullets(raw, expected):
    """Wrapping quotes (straight/curly/single) and leading bullets/numbering are stripped."""
    assert _clean_generated_query(raw) == expected


def test_clean_generated_query_does_not_strip_internal_quotes():
    """Quotes inside the phrase (not wrapping it) are left alone."""
    assert _clean_generated_query('the "attention is all you need" paper') == (
        'the "attention is all you need" paper'
    )
