"""
tests/test_text_parsing.py
───────────────────────────
Unit tests for tools/text_parsing.py:

  - extract_suggested_questions(): pulls a trailing "suggested_questions"
    JSON array off LLM output, tolerating markdown-bolded/italicized keys,
    smart quotes, trailing commas, and several JSON-shape variants.

  - extract_references_section(): isolates the bibliography at the end of
    an academic paper, for the Notebook Citation Timeline feature.

Pure stdlib — no network access or heavy deps required.
"""

from __future__ import annotations

from tools.text_parsing import extract_references_section, extract_suggested_questions


def test_full_json_object():
    raw = (
        "Here is the explanation.\n\n"
        '{"suggested_questions": ["Q1?", "Q2?", "Q3?"]}'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Here is the explanation."
    assert questions == ["Q1?", "Q2?", "Q3?"]


def test_quoted_key_without_braces():
    raw = (
        "Here is the explanation.\n\n"
        '"suggested_questions": ["Q1?", "Q2?", "Q3?"]'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Here is the explanation."
    assert questions == ["Q1?", "Q2?", "Q3?"]


def test_bare_key_with_colon():
    raw = (
        "Here is the explanation.\n\n"
        'suggested_questions: ["Q1?", "Q2?", "Q3?"]'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Here is the explanation."
    assert questions == ["Q1?", "Q2?", "Q3?"]


def test_bare_key_with_equals():
    raw = (
        "Here is the explanation.\n\n"
        'suggested_questions = ["Q1?", "Q2?", "Q3?"]'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Here is the explanation."
    assert questions == ["Q1?", "Q2?", "Q3?"]


def test_markdown_bold_key_reported_bug():
    """Reproduces the reported bug: the LLM bolds the key with **...**.

    Streamlit renders the asterisks away, so the user sees a plain
    `suggested_questions: [...]` block with no clickable buttons. The
    fix normalizes the bolded key before pattern-matching.
    """
    raw = (
        "A UAV base station balances coverage and energy use.\n\n"
        '**suggested_questions**: [ "How might a UAV-BS detect an unexpected '
        'energy drain?", "What strategies could be implemented to optimize '
        'UAV-BS trajectories when facing sudden energy drains?", "Could '
        "machine learning algorithms improve a UAV-BS's ability to adapt to "
        'unexpected obstacles?" ]'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "A UAV base station balances coverage and energy use."
    assert questions == [
        "How might a UAV-BS detect an unexpected energy drain?",
        "What strategies could be implemented to optimize UAV-BS trajectories when facing sudden energy drains?",
        "Could machine learning algorithms improve a UAV-BS's ability to adapt to unexpected obstacles?",
    ]


def test_markdown_italic_and_underscore_keys():
    for opener, closer in [("*", "*"), ("_", "_"), ("__", "__")]:
        raw = (
            "Body text.\n\n"
            f'{opener}suggested_questions{closer}: ["Q1?", "Q2?"]'
        )
        body, questions = extract_suggested_questions(raw)
        assert body == "Body text.", (opener, closer)
        assert questions == ["Q1?", "Q2?"], (opener, closer)


def test_smart_quotes_are_normalized():
    raw = (
        "Body text.\n\n"
        "suggested_questions: [“What is X?”, “What is Y?”]"
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Body text."
    assert questions == ["What is X?", "What is Y?"]


def test_trailing_comma_is_tolerated():
    raw = (
        "Body text.\n\n"
        'suggested_questions: ["Q1?", "Q2?", "Q3?",]'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Body text."
    assert questions == ["Q1?", "Q2?", "Q3?"]


def test_max_questions_caps_results():
    raw = 'suggested_questions: ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]'
    _, questions = extract_suggested_questions(raw, max_questions=2)
    assert questions == ["Q1?", "Q2?"]


def test_no_match_returns_raw_unchanged():
    raw = "Just a plain explanation with no trailing JSON block."
    body, questions = extract_suggested_questions(raw)
    assert body == raw
    assert questions == []


# ── extract_references_section() ─────────────────────────────────────────

def test_references_heading_extracts_trailing_section():
    text = (
        "Title of the Paper\n\n"
        "Abstract\nThis paper studies X.\n\n"
        "1. Introduction\nSome introduction text. " + ("filler " * 50) + "\n\n"
        "References\n"
        "[1] Smith, J. (2020). A great paper. Journal of Things.\n"
        "[2] Doe, A. (2019). Another paper. Conference of Stuff."
    )
    section = extract_references_section(text)
    assert section.startswith("[1] Smith, J. (2020)")
    assert "[2] Doe, A. (2019)" in section
    assert "Introduction" not in section


def test_bibliography_and_works_cited_headings_recognized():
    for heading in ["Bibliography", "Works Cited", "Literature Cited", "REFERENCES"]:
        text = ("Body text. " * 50) + f"\n\n{heading}\n[1] Author (2021). Paper title."
        section = extract_references_section(text)
        assert section == "[1] Author (2021). Paper title.", heading


def test_early_mention_is_ignored_in_favor_of_later_heading():
    text = (
        "Table of Contents\n"
        "1. Introduction\n"
        "2. References\n"
        "3. Appendix\n\n"
        + ("Body text discussing prior work. " * 30) + "\n\n"
        "References\n"
        "[1] Real Citation (2022). The actual bibliography entry."
    )
    section = extract_references_section(text)
    assert section == "[1] Real Citation (2022). The actual bibliography entry."


def test_no_heading_returns_empty_string():
    text = "Just a regular document body with no bibliography heading at all. " * 10
    assert extract_references_section(text) == ""


def test_heading_with_trailing_colon():
    text = ("Body text. " * 50) + "\n\nReferences:\n[1] Author (2023). Title."
    section = extract_references_section(text)
    assert section == "[1] Author (2023). Title."
