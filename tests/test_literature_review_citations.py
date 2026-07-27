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
a distinct citable excerpt, and _build_references_list rebuilds the
References list from whichever numbers the model actually cited in the body
— rather than trusting the model's own self-written list. The model is told
not to write its own References section; _strip_llm_references_section is a
defensive backstop in case it does anyway.

_build_references_list returns structured {"n", "doc_name", "page", ...}
dicts (raw, 0-based "page") for the live Streamlit snippet-expander UI;
references_list_to_markdown() flattens that into the same "## References"
text block used by export paths. Page numbers stored on chunks are 0-based
internally and only converted to the 1-based "p. N" a user sees in their PDF
at the final display/flatten step (format_page_label) — these tests use
0-based fixtures throughout to exercise that conversion, not bypass it.

ChatOllama/NotebookMemory are mocked — no network access or Ollama server
required.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.notebook_advanced import (
    _build_numbered_excerpts,
    _build_references_list,
    _strip_llm_references_section,
    generate_literature_review,
    references_list_to_markdown,
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
            _chunk("d1", "paper.pdf", 0, 0, "First chunk text."),
            _chunk("d1", "paper.pdf", 1, 1, "Second chunk text."),
        ],
    }
    context, excerpts = _build_numbered_excerpts(notebook)
    assert "[1] (source: paper.pdf, p. 1)" in context
    assert "[2] (source: paper.pdf, p. 2)" in context
    assert len(excerpts) == 2
    assert excerpts[0]["page_num"] == 0
    assert excerpts[1]["page_num"] == 1


def test_build_numbered_excerpts_orders_chunks_by_chunk_index():
    """Chunks must be numbered in document order even if stored out of order."""
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "paper.pdf"}],
        "chunks": [
            _chunk("d1", "paper.pdf", 1, 1, "Second chunk text."),
            _chunk("d1", "paper.pdf", 0, 0, "First chunk text."),
        ],
    }
    _, excerpts = _build_numbered_excerpts(notebook)
    assert [e["chunk_index"] for e in excerpts] == [0, 1]


def test_build_numbered_excerpts_respects_per_doc_char_budget():
    """A tiny per-doc char cap must stop pulling in more chunks for that doc."""
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "paper.pdf"}],
        "chunks": [
            _chunk("d1", "paper.pdf", 0, 0, "x" * 50),
            _chunk("d1", "paper.pdf", 1, 1, "y" * 50),
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


# ── _build_references_list / references_list_to_markdown ───────────────────

def test_build_references_list_lists_every_cited_number():
    """All seven citation numbers a model used must appear — not collapsed into one entry."""
    excerpts = [_chunk("d1", "paper.pdf", n, n, f"chunk {n}") for n in range(7)]
    body = "Findings [1][2][3] and limitations [4][5][6][7] are discussed."
    refs = _build_references_list(body, excerpts)

    assert [r["n"] for r in refs] == [1, 2, 3, 4, 5, 6, 7]
    for n in range(1, 8):
        ref = next(r for r in refs if r["n"] == n)
        assert ref["doc_name"] == "paper.pdf"
        assert ref["page"] == n - 1  # raw 0-based page carried from the excerpt

    markdown = references_list_to_markdown(refs)
    for n in range(1, 8):
        assert f"[{n}] paper.pdf (p. {n})" in markdown


def test_build_references_list_falls_back_to_filenames_when_nothing_cited():
    """No inline citations at all — fall back to listing distinct source filenames."""
    excerpts = [
        _chunk("d1", "paper1.pdf", 0, 0, "text"),
        _chunk("d2", "paper2.pdf", 0, 0, "text"),
    ]
    refs = _build_references_list("No bracket citations here.", excerpts)
    assert [r["doc_name"] for r in refs] == ["paper1.pdf", "paper2.pdf"]
    assert all(r["n"] is None for r in refs)

    markdown = references_list_to_markdown(refs)
    assert "- paper1.pdf" in markdown
    assert "- paper2.pdf" in markdown


def test_build_references_list_ignores_out_of_range_numbers():
    """A hallucinated citation number with no backing excerpt must not crash or appear."""
    excerpts = [_chunk("d1", "paper.pdf", 0, 0, "text")]
    refs = _build_references_list("Cites [1] and an invented [99].", excerpts)
    assert [r["n"] for r in refs] == [1]
    assert refs[0]["doc_name"] == "paper.pdf"

    markdown = references_list_to_markdown(refs)
    assert "[1] paper.pdf" in markdown
    assert "[99]" not in markdown


# ── generate_literature_review (integration) ─────────────────────────────────

def test_generate_literature_review_rebuilds_references_from_actual_citations():
    """
    Reproduces the bug report: the model cites [1]-[3] inline but writes its
    own incomplete References line. The returned body must keep the body,
    drop the model's own References line, and the references list must cover
    every number actually cited — not just the one the model wrote itself.
    """
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "2003.04816v1.pdf"}],
        "chunks": [
            _chunk("d1", "2003.04816v1.pdf", 0, 0, "Energy efficiency results."),
            _chunk("d1", "2003.04816v1.pdf", 1, 1, "AoI threshold results."),
            _chunk("d1", "2003.04816v1.pdf", 3, 2, "Limitations discussion."),
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
        body, references, err = generate_literature_review("nb1", {"model": "llama3.1:8b"})

    assert err == ""
    assert "Scope [1]" in body
    assert "Gaps remain [3]" in body
    assert body.count("**References**") == 0

    markdown = references_list_to_markdown(references)
    assert "[1] 2003.04816v1.pdf (p. 1)" in markdown
    assert "[2] 2003.04816v1.pdf (p. 2)" in markdown
    assert "[3] 2003.04816v1.pdf (p. 4)" in markdown
