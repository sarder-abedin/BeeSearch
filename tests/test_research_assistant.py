"""
tests/test_research_assistant.py
─────────────────────────────────────
Unit tests for agents/research_assistant.py (Phase 1 — AI Research Assistant).

Covers the pure citation-grounding logic (source numbering, code-rebuilt
citations from the [n] markers actually used, the defensive References-stripper)
and the run_research_assistant orchestration for both the grounded and
no-sources-found paths.

ChatOllama and the search backends are mocked — no Ollama server or network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.research_assistant import (
    build_citations,
    build_numbered_sources,
    run_research_assistant,
    _strip_llm_references_section,
)


# ── build_numbered_sources ───────────────────────────────────────────────────

def test_build_numbered_sources_one_namespace_papers_then_web():
    """Academic papers are numbered first, web results after, in one [n] namespace."""
    papers = [{"title": "P1", "authors": ["A Smith"], "year": 2020, "abstract": "finding one"}]
    web = [{"title": "Blog", "url": "http://x", "snippet": "web text"}]
    ctx, smap = build_numbered_sources(papers, web)
    assert "[1] (P1" in ctx and "[2] (Blog" in ctx
    assert smap[1]["kind"] == "academic"
    assert smap[2]["kind"] == "web"


def test_build_numbered_sources_respects_char_budget():
    """A tiny char budget stops adding sources once at least one is included."""
    papers = [
        {"title": "P1", "authors": ["A"], "year": 2020, "abstract": "x" * 200},
        {"title": "P2", "authors": ["B"], "year": 2021, "abstract": "y" * 200},
    ]
    ctx, smap = build_numbered_sources(papers, [], max_chars=100, max_chars_per_source=200)
    assert 1 in smap and 2 not in smap


def test_build_numbered_sources_handles_objects_not_just_dicts():
    """Paper-like objects (attribute access, .to_apa()) work as well as dicts."""
    class P:
        title = "Obj Paper"
        authors = ["Jane Doe"]
        year = 2022
        abstract = "object abstract"
        url = "http://o"
        source = "arxiv"

        def to_apa(self):
            return "Doe, J. (2022). Obj Paper."

    ctx, smap = build_numbered_sources([P()], [])
    assert smap[1]["title"] == "Obj Paper"
    assert smap[1]["apa"].startswith("Doe")


# ── build_citations ──────────────────────────────────────────────────────────

def test_build_citations_only_cited_numbers_in_order():
    """Citations are rebuilt from the [n] actually present, in ascending order."""
    smap = {
        1: {"kind": "academic", "title": "P1"},
        2: {"kind": "academic", "title": "P2"},
        3: {"kind": "web", "title": "W"},
    }
    body = "Claim two [2]. Claim one [1]. The web agrees [3]."
    cites = build_citations(body, smap)
    assert [c["n"] for c in cites] == [1, 2, 3]


def test_build_citations_ignores_hallucinated_numbers():
    """A cited number with no backing source is dropped, not surfaced or crashed on."""
    smap = {1: {"kind": "academic", "title": "P1"}}
    cites = build_citations("Real [1] and invented [99].", smap)
    assert [c["n"] for c in cites] == [1]


def test_build_citations_empty_when_nothing_cited():
    """A general-knowledge answer that cites nothing yields an empty list."""
    assert build_citations("No citations here at all.", {1: {"title": "P1"}}) == []


# ── _strip_llm_references_section ────────────────────────────────────────────

def test_strip_references_section_removes_model_written_list():
    """A self-written References section is cut; the body is preserved."""
    body = "The answer body.\n\n**References**\n[1] Something"
    assert _strip_llm_references_section(body) == "The answer body."


def test_strip_references_section_noop_when_absent():
    """No References heading → body returned unchanged (trailing ws trimmed)."""
    assert _strip_llm_references_section("Just an answer.") == "Just an answer."


# ── run_research_assistant (orchestration) ───────────────────────────────────

def _two_call_llm(answer: str, followups_json: str = "[]"):
    """Mock ChatOllama whose two invoke() calls return the answer, then follow-ups."""
    llm = MagicMock()
    msg1 = MagicMock(); msg1.content = answer
    msg2 = MagicMock(); msg2.content = followups_json
    llm.invoke.side_effect = [msg1, msg2]
    return llm


def test_run_research_assistant_grounded_rebuilds_citations():
    """With sources found, the answer is grounded and citations come from the [n] used."""
    found = {
        "papers": [{"title": "Sleep & Memory", "authors": ["A Smith"], "year": 2020,
                    "abstract": "sleep helps memory", "url": "http://p1", "source": "arxiv"}],
        "web_results": [],
    }
    llm = _two_call_llm("Sleep improves memory [1].", '["What about naps?"]')
    with patch("agents.research_assistant.search_literature", return_value=found), \
         patch("agents.research_assistant.ChatOllama", return_value=llm):
        result = run_research_assistant("Does sleep help memory?", {"model": "m", "num_ctx": 4096})

    assert result["grounded"] is True
    assert result["academic_count"] == 1
    assert [c["n"] for c in result["citations"]] == [1]
    assert result["suggested_questions"] == ["What about naps?"]
    assert "**References**" not in result["answer"]


def test_run_research_assistant_no_sources_is_ungrounded_no_citations():
    """With no sources, the answer is flagged ungrounded and carries no citations."""
    found = {"papers": [], "web_results": []}
    llm = _two_call_llm("Generally, sleep is thought to help memory.", "[]")
    with patch("agents.research_assistant.search_literature", return_value=found), \
         patch("agents.research_assistant.ChatOllama", return_value=llm):
        result = run_research_assistant("Does sleep help memory?", {"model": "m", "num_ctx": 4096})

    assert result["grounded"] is False
    assert result["citations"] == []
    assert result["answer"]


def test_run_research_assistant_stream_callback_errors_never_break_answer():
    """A throwing stream_callback is swallowed — the answer still comes back."""
    found = {"papers": [{"title": "P", "authors": ["X"], "year": 2020, "abstract": "a", "url": "u"}],
             "web_results": []}
    llm = _two_call_llm("Answer [1].", "[]")

    def boom(stage, info):
        raise ValueError("callback exploded")

    with patch("agents.research_assistant.search_literature", return_value=found), \
         patch("agents.research_assistant.ChatOllama", return_value=llm):
        result = run_research_assistant("q", {"model": "m", "num_ctx": 4096}, stream_callback=boom)
    assert result["answer"].startswith("Answer")
