"""
tests/test_notebook_web_search_note.py
─────────────────────────────────────────
Unit tests for agents/notebook_nodes.py::_web_search_status_note.

When a notebook has sources, the Chat tab's answer_node falls through to the
main LLM call even if "Auto web search" found nothing or errored — and
without this footnote, that answer is indistinguishable from one where web
search was simply switched off. The note makes the outcome of auto web
search visible directly in the assistant's reply.

Pure stdlib — no network access or heavy deps required.
"""

from __future__ import annotations

from agents.notebook_nodes import _web_search_status_note


def test_no_note_when_web_search_not_involved():
    """Plain retrieval modes (no web search attempted) get no footnote."""
    for mode in ("react", "bm25", "fallback", "empty"):
        assert _web_search_status_note(mode) == ""


def test_no_note_when_web_search_contributed_results():
    """When web search found results, no "nothing found" footnote is appended."""
    assert _web_search_status_note("react+web") == ""
    assert _web_search_status_note("fallback+web") == ""


def test_note_when_web_search_errored():
    """A "+web_error" retrieval mode gets a footnote saying web search failed."""
    note = _web_search_status_note("fallback+web_error")
    assert "failed" in note
    assert "notebook sources" in note


def test_note_when_web_search_found_nothing():
    """A "+web_empty" retrieval mode gets a footnote saying web search found nothing."""
    note = _web_search_status_note("react+web_empty")
    assert "found no additional results" in note
    assert "notebook sources" in note
