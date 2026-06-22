"""
tests/test_explain_citation_grounding.py
──────────────────────────────────────────
Unit tests for agents/story_nodes.py's Explain-tab citation grounding.

Same bug class as Literature Review's (see test_literature_review_citations.py),
on the Explain tab: the storyteller was told to "quote short passages" from
document_context with no numbered tags to ground a citation in, and — when
online search ran — was told to write its own References list with no
code-level check that it matched what it actually cited.

Fix: build_numbered_doc_context tags each chunk with its real page number the
same way notebook_advanced.py's _build_numbered_excerpts does for Literature
Review, baking the tags directly into the string persisted as
document_context so no story-session schema change is needed.
_parse_doc_excerpts recovers that mapping on every later turn; _build_references_section
rebuilds an accurate references list from whichever [n]/[Source N] numbers the
model actually cited — covering both document excerpts and online search
results in one unified, code-generated list. _strip_llm_references_section is
the same defensive backstop used for Literature Review.

ChatOllama is mocked — no network access or Ollama server required.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.story_nodes import (
    _build_references_section,
    _parse_doc_excerpts,
    _strip_llm_references_section,
    build_numbered_doc_context,
    storyteller_node,
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


# ── build_numbered_doc_context ───────────────────────────────────────────────

def test_build_numbered_doc_context_tags_each_chunk_with_its_own_page():
    """Each chunk gets its own number and real page tag, not an untagged blob."""
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "paper.pdf"}],
        "chunks": [
            _chunk("d1", "paper.pdf", 1, 0, "First chunk text."),
            _chunk("d1", "paper.pdf", 2, 1, "Second chunk text."),
        ],
    }
    context = build_numbered_doc_context(notebook)
    assert "[1] (source: paper.pdf, p. 1)" in context
    assert "[2] (source: paper.pdf, p. 2)" in context


def test_build_numbered_doc_context_orders_chunks_by_chunk_index():
    """Chunks must be numbered in document order even if stored out of order."""
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "paper.pdf"}],
        "chunks": [
            _chunk("d1", "paper.pdf", 2, 1, "Second chunk text."),
            _chunk("d1", "paper.pdf", 1, 0, "First chunk text."),
        ],
    }
    context = build_numbered_doc_context(notebook)
    assert context.index("[1]") < context.index("[2]")
    assert "First chunk text." in context.split("[2]")[0]


def test_build_numbered_doc_context_respects_char_budget():
    """A tiny char cap must stop pulling in more chunks."""
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "paper.pdf"}],
        "chunks": [
            _chunk("d1", "paper.pdf", 1, 0, "x" * 50),
            _chunk("d1", "paper.pdf", 2, 1, "y" * 50),
        ],
    }
    context = build_numbered_doc_context(notebook, max_chars=50, max_chars_per_chunk=50)
    assert "[2]" not in context


# ── _parse_doc_excerpts ──────────────────────────────────────────────────────

def test_parse_doc_excerpts_recovers_mapping_from_tagged_string():
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "paper.pdf"}],
        "chunks": [_chunk("d1", "paper.pdf", 3, 0, "Some text.")],
    }
    context = build_numbered_doc_context(notebook)
    assert _parse_doc_excerpts(context) == {1: {"doc_name": "paper.pdf", "page_num": 3}}


def test_parse_doc_excerpts_returns_empty_for_untagged_legacy_context():
    """Sessions created before this fix have no [n] (source: ...) tags — must not crash."""
    assert _parse_doc_excerpts("Some old plain-text snippet.\n\n---\nAnother snippet.") == {}


# ── _strip_llm_references_section ────────────────────────────────────────────

def test_strip_llm_references_section_removes_bold_inline_label():
    body = "Some explanation.\n\n**References**: [1] paper.pdf"
    assert _strip_llm_references_section(body) == "Some explanation."


def test_strip_llm_references_section_is_noop_when_absent():
    body = "Some explanation with no references heading."
    assert _strip_llm_references_section(body) == body


# ── _build_references_section ────────────────────────────────────────────────

def test_build_references_section_covers_both_doc_and_online_numbering():
    """One unified list rebuilt from both citation namespaces actually used."""
    doc_excerpts = {1: {"doc_name": "paper.pdf", "page_num": 2}}
    online_results = [{"title": "Some Blog Post", "url": "https://example.com"}]
    body = "Documents say X [1]. The web adds Y [Source 1]."
    section = _build_references_section(body, doc_excerpts, online_results)
    assert "[1] paper.pdf (p. 2)" in section
    assert "[Source 1] Some Blog Post — https://example.com" in section


def test_build_references_section_empty_when_nothing_cited():
    """Unlike Literature Review, a conversational turn may legitimately cite nothing."""
    assert _build_references_section("General background, no citations here.", {}, []) == ""


def test_build_references_section_ignores_out_of_range_numbers():
    """A hallucinated citation number with no backing excerpt must not crash or appear."""
    doc_excerpts = {1: {"doc_name": "paper.pdf", "page_num": 1}}
    section = _build_references_section("Cites [1] and an invented [99].", doc_excerpts, [])
    assert "[1] paper.pdf" in section
    assert "[99]" not in section


# ── storyteller_node (integration) ───────────────────────────────────────────

def test_storyteller_node_rebuilds_references_from_actual_citations():
    """
    The model cites [1]/[2] inline but writes its own incomplete References
    line. The returned response must keep the body, drop the model's own
    References line, and append a correct one covering every number actually
    cited.
    """
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "2003.04816v1.pdf"}],
        "chunks": [
            _chunk("d1", "2003.04816v1.pdf", 1, 0, "Energy efficiency results."),
            _chunk("d1", "2003.04816v1.pdf", 2, 1, "AoI threshold results."),
        ],
    }
    doc_context = build_numbered_doc_context(notebook)
    fake_llm_output = (
        "Findings show improvement [1] and a threshold effect [2].\n\n"
        "**References**: [1] 2003.04816v1.pdf\n\n"
        '{"suggested_questions": ["Q1?", "Q2?", "Q3?"]}'
    )
    state = {
        "user_message": "What did the paper find?",
        "document_context": doc_context,
        "explanation_style": "simple",
        "explanation_level": "intermediate",
    }
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = fake_llm_output
        result = storyteller_node(state)

    response = result["assistant_response"]
    assert "improvement [1]" in response
    assert "threshold effect [2]" in response
    assert response.count("**References**:") == 0
    assert "[1] 2003.04816v1.pdf (p. 1)" in response
    assert "[2] 2003.04816v1.pdf (p. 2)" in response


def test_storyteller_node_omits_references_when_nothing_cited():
    """A general-knowledge turn with no document context must not get a spurious References section."""
    state = {
        "user_message": "What is machine learning in general?",
        "document_context": "",
        "explanation_style": "simple",
        "explanation_level": "novice",
    }
    fake_llm_output = (
        "Machine learning is a field of AI focused on learning from data.\n\n"
        '{"suggested_questions": ["Q1?", "Q2?", "Q3?"]}'
    )
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = fake_llm_output
        result = storyteller_node(state)

    assert "References" not in result["assistant_response"]
