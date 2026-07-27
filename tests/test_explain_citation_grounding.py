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
_parse_doc_excerpts recovers that mapping (plus the excerpt's snippet text) on
every later turn; _build_citations_list rebuilds an accurate, structured
citations list from whichever [n]/[Source N] numbers the model actually
cited — covering both document excerpts and online search results in one
unified, code-generated list consumed by ui/tabs/notebook.py's
snippet-expander UI (_render_citations), rather than text baked into the
response body. _strip_llm_references_section is the same defensive backstop
used for Literature Review, in case the model writes its own References
section anyway despite being told not to.

page_num is 0-based internally and only becomes the 1-based "p. N" a user
sees in their PDF at the final display step (format_page_label) — these
tests use 0-based fixtures throughout, matching test_literature_review_
citations.py, to exercise that conversion rather than bypass it.

ChatOllama is mocked — no network access or Ollama server required.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.story_nodes import (
    _build_citations_list,
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
            _chunk("d1", "paper.pdf", 0, 0, "First chunk text."),
            _chunk("d1", "paper.pdf", 1, 1, "Second chunk text."),
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
            _chunk("d1", "paper.pdf", 1, 1, "Second chunk text."),
            _chunk("d1", "paper.pdf", 0, 0, "First chunk text."),
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
            _chunk("d1", "paper.pdf", 0, 0, "x" * 50),
            _chunk("d1", "paper.pdf", 1, 1, "y" * 50),
        ],
    }
    context = build_numbered_doc_context(notebook, max_chars=50, max_chars_per_chunk=50)
    assert "[2]" not in context


# ── _parse_doc_excerpts ──────────────────────────────────────────────────────

def test_parse_doc_excerpts_recovers_mapping_from_tagged_string():
    """page_num must come back raw/0-based, undoing the +1 baked into the display tag."""
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "paper.pdf"}],
        "chunks": [_chunk("d1", "paper.pdf", 3, 0, "Some text.")],
    }
    context = build_numbered_doc_context(notebook)
    assert _parse_doc_excerpts(context) == {
        1: {"doc_name": "paper.pdf", "page_num": 3, "snippet": "Some text."}
    }


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


# ── _build_citations_list ────────────────────────────────────────────────────

def test_build_citations_list_covers_both_doc_and_online_numbering():
    """One unified, structured list rebuilt from both citation namespaces actually used."""
    doc_excerpts = {1: {"doc_name": "paper.pdf", "page_num": 2, "snippet": "Excerpt text."}}
    online_results = [{"title": "Some Blog Post", "url": "https://example.com", "snippet": "Blog snippet."}]
    body = "Documents say X [1]. The web adds Y [Source 1]."
    citations = _build_citations_list(body, doc_excerpts, online_results)

    assert citations == [
        {"n": 1, "doc_name": "paper.pdf", "page": 2, "snippet": "Excerpt text."},
        {"n": "Source 1", "doc_name": "Some Blog Post", "snippet": "Blog snippet.", "url": "https://example.com"},
    ]


def test_build_citations_list_empty_when_nothing_cited():
    """Unlike Literature Review, a conversational turn may legitimately cite nothing."""
    assert _build_citations_list("General background, no citations here.", {}, []) == []


def test_build_citations_list_ignores_out_of_range_numbers():
    """A hallucinated citation number with no backing excerpt must not crash or appear."""
    doc_excerpts = {1: {"doc_name": "paper.pdf", "page_num": 0, "snippet": "text"}}
    citations = _build_citations_list("Cites [1] and an invented [99].", doc_excerpts, [])
    assert citations == [{"n": 1, "doc_name": "paper.pdf", "page": 0, "snippet": "text"}]


# ── storyteller_node (integration) ───────────────────────────────────────────

def test_storyteller_node_rebuilds_citations_from_actual_citations():
    """
    The model cites [1]/[2] inline but writes its own incomplete References
    line. The returned response must keep the body and drop the model's own
    References line; result["citations"] must cover every number actually
    cited — not just the one the model wrote itself.
    """
    notebook = {
        "sources": [{"doc_id": "d1", "filename": "2003.04816v1.pdf"}],
        "chunks": [
            _chunk("d1", "2003.04816v1.pdf", 0, 0, "Energy efficiency results."),
            _chunk("d1", "2003.04816v1.pdf", 1, 1, "AoI threshold results."),
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

    assert result["citations"] == [
        {"n": 1, "doc_name": "2003.04816v1.pdf", "page": 0, "snippet": "Energy efficiency results."},
        {"n": 2, "doc_name": "2003.04816v1.pdf", "page": 1, "snippet": "AoI threshold results."},
    ]


def test_storyteller_node_omits_citations_when_nothing_cited():
    """A general-knowledge turn with no document context must get no citations at all."""
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
    assert result["citations"] == []
