"""
tests/test_literature_review_citations.py
─────────────────────────────────────────────
Unit tests for agents/notebook_advanced.py's Literature Review citation fix.

Bug: generate_literature_review asked the LLM to "attribute claims to source
filenames" but gave it only one citable number per DOCUMENT (via
_sources_context), and let it freely write its own References section. The
model defaulted to academic per-claim bracket citations like [1]..[7] anyway
(habit from training, not grounded in anything the prompt actually provided),
then wrote a References list that didn't match — e.g. citing [1]-[7] inline
but listing only "[1] paper.pdf" in References, with [2]-[7] unaccounted for.

Fix: _build_numbered_excerpts tags every CHUNK (with its real page number) as
a distinct citable excerpt, and _build_references_section rebuilds the
References list from whichever numbers the model actually cited in the body
— rather than trusting the model's own self-written list. The model is told
not to write its own References section; _strip_llm_references_section is a
defensive backstop in case it does anyway.

ChatOllama/NotebookMemory are mocked — no network access or Ollama server
required.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.notebook_advanced import (
    _build_numbered_excerpts,
    _build_references_section,
    _strip_llm_references_section,
    generate_literature_review,
)


def _chunk(doc_id, doc_name, page_num, chunk_index, text):
    return {
        "chunk_id": f"{doc_id}_{chunk_index}",
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page_num": page_num,
        "chunk_index": chunk_index,
        "text": text,
    }


# ── _build_numbered_excerpts ─────────────────────────────────────────────────

def test_build_numbered_excerpts_tags_each_chunk_with_its_own_page():
    """Each chunk gets its own number and real page tag, not one shared per-document number."""
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "paper.pdf"}],
        "chunks": [
            _chunk("d1", "paper.pdf", 1, 0, "First chunk text."),
            _chunk("d1", "paper.pdf", 2, 1, "Second chunk text."),
        ],
    }
    context, excerpts = _build_numbered_excerpts(notebook)
    assert "[1] (source: paper.pdf, p. 1)" in context
    assert "[2] (source: paper.pdf, p. 2)" in context
    assert len(excerpts) == 2
    assert excerpts[0]["page_num"] == 1
    assert excerpts[1]["page_num"] == 2


def test_build_numbered_excerpts_orders_chunks_by_chunk_index():
    """Chunks must be numbered in document order even if stored out of order."""
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "paper.pdf"}],
        "chunks": [
            _chunk("d1", "paper.pdf", 2, 1, "Second chunk text."),
            _chunk("d1", "paper.pdf", 1, 0, "First chunk text."),
        ],
    }
    _, excerpts = _build_numbered_excerpts(notebook)
    assert [e["chunk_index"] for e in excerpts] == [0, 1]


def test_build_numbered_excerpts_respects_per_doc_char_budget():
    """A tiny per-doc char cap must stop pulling in more chunks for that doc."""
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "paper.pdf"}],
        "chunks": [
            _chunk("d1", "paper.pdf", 1, 0, "x" * 50),
            _chunk("d1", "paper.pdf", 2, 1, "y" * 50),
        ],
    }
    _, excerpts = _build_numbered_excerpts(notebook, max_chars_per_doc=50)
    assert len(excerpts) == 1


# ── _strip_llm_references_section ────────────────────────────────────────────

def test_strip_llm_references_section_removes_markdown_heading():
    body = "## 6. Conclusion\nSome text.\n\n## References\n[1] paper.pdf"
    assert _strip_llm_references_section(body) == "## 6. Conclusion\nSome text."


def test_strip_llm_references_section_removes_bold_inline_label():
    """Matches the exact pattern seen in the bug report: a bold inline label, not a heading."""
    body = "## 6. Conclusion\nSome text.\n\n**References**: [1] 2003.04816v1.pdf"
    assert _strip_llm_references_section(body) == "## 6. Conclusion\nSome text."


def test_strip_llm_references_section_is_noop_when_absent():
    body = "## 6. Conclusion\nSome text with no references heading."
    assert _strip_llm_references_section(body) == body


# ── _build_references_section ────────────────────────────────────────────────

def test_build_references_section_lists_every_cited_number():
    """All seven citation numbers a model used must appear — not collapsed into one entry."""
    excerpts = [_chunk("d1", "paper.pdf", n, n - 1, f"chunk {n}") for n in range(1, 8)]
    body = "Findings [1][2][3] and limitations [4][5][6][7] are discussed."
    section = _build_references_section(body, excerpts)
    for n in range(1, 8):
        assert f"[{n}] paper.pdf (p. {n})" in section


def test_build_references_section_falls_back_to_filenames_when_nothing_cited():
    """No inline citations at all — fall back to listing distinct source filenames."""
    excerpts = [
        _chunk("d1", "paper1.pdf", 1, 0, "text"),
        _chunk("d2", "paper2.pdf", 1, 0, "text"),
    ]
    section = _build_references_section("No bracket citations here.", excerpts)
    assert "- paper1.pdf" in section
    assert "- paper2.pdf" in section


def test_build_references_section_ignores_out_of_range_numbers():
    """A hallucinated citation number with no backing excerpt must not crash or appear."""
    excerpts = [_chunk("d1", "paper.pdf", 1, 0, "text")]
    section = _build_references_section("Cites [1] and an invented [99].", excerpts)
    assert "[1] paper.pdf" in section
    assert "[99]" not in section


# ── generate_literature_review (integration) ─────────────────────────────────

def test_generate_literature_review_rebuilds_references_from_actual_citations():
    """
    Reproduces the bug report: the model cites [1]-[3] inline but writes its
    own incomplete References line. The returned review must keep the body,
    drop the model's own References line, and append a correct one covering
    every number actually cited.
    """
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "2003.04816v1.pdf"}],
        "chunks": [
            _chunk("d1", "2003.04816v1.pdf", 1, 0, "Energy efficiency results."),
            _chunk("d1", "2003.04816v1.pdf", 2, 1, "AoI threshold results."),
            _chunk("d1", "2003.04816v1.pdf", 4, 2, "Limitations discussion."),
        ],
    }
    fake_llm_output = (
        "# Literature Review\n## 1. Introduction\nScope [1].\n"
        "## 4. Key Findings & Evidence\nResults improved [2].\n"
        "## 5. Critical Analysis\nGaps remain [3].\n\n"
        "**References**: [1] 2003.04816v1.pdf"
    )
    with patch("agents.notebook_advanced.NotebookMemory") as mock_mem, \
         patch("agents.notebook_advanced._invoke", return_value=fake_llm_output):
        mock_mem.return_value.load.return_value = notebook
        result, err = generate_literature_review("nb1", {"model": "llama3.1:8b"})

    assert err == ""
    assert "Scope [1]" in result
    assert "Gaps remain [3]" in result
    assert result.count("**References**") == 0
    assert "[1] 2003.04816v1.pdf (p. 1)" in result
    assert "[2] 2003.04816v1.pdf (p. 2)" in result
    assert "[3] 2003.04816v1.pdf (p. 4)" in result
