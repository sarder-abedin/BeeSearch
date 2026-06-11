"""
tools/text_parsing.py
──────────────────────
Helpers for cleaning up raw LLM text output and source documents.

extract_suggested_questions() pulls the trailing "suggested_questions" JSON
array that Notebook Chat and Explain prompts ask the LLM to append, tolerating
the various formats and minor JSON quirks local models tend to produce.

extract_references_section() locates the bibliography at the end of an
academic paper, for the Notebook "Citation Timeline" feature.
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple

_SQ_PATTERNS = [
    # Full JSON object  {"suggested_questions": [...]}
    re.compile(r'\{[^{}]*"suggested_questions"\s*:\s*(\[.*?\])\s*\}', re.DOTALL),
    # Quoted key without outer braces  "suggested_questions": [...]
    re.compile(r'"suggested_questions"\s*:\s*(\[.*?\])', re.DOTALL),
    # Bare key  suggested_questions: [...] or suggested_questions = [...]
    re.compile(r'suggested_questions\s*[:=]\s*(\[.*?\])', re.DOTALL | re.IGNORECASE),
]

# LLMs often bold/italicize the key (e.g. **suggested_questions**:), which would
# otherwise prevent every pattern above from matching.
_BOLD_KEY = re.compile(r'[*_]{1,2}(suggested_questions)[*_]{1,2}', re.IGNORECASE)

# Trailing comma before the closing bracket — common LLM JSON mistake.
_TRAILING_COMMA = re.compile(r',\s*\]')

# Curly/smart quotes used as string delimiters instead of straight ASCII quotes.
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def extract_suggested_questions(raw: str, max_questions: int = 3) -> Tuple[str, List[str]]:
    """
    Split a trailing ``suggested_questions`` JSON array off LLM output.

    Returns ``(body, questions)``:
      - ``body``: ``raw`` with the matched block (and anything after it) removed
      - ``questions``: up to ``max_questions`` strings, or ``[]`` if no block
        could be parsed (in which case ``body == raw``, unchanged)
    """
    # Normalize a markdown-bolded/italicized key so the patterns below can match it.
    text = _BOLD_KEY.sub(r"\1", raw)

    for pat in _SQ_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        block = m.group(1).translate(_SMART_QUOTES)
        block = _TRAILING_COMMA.sub("]", block)
        try:
            candidates = json.loads(block)
        except Exception:
            continue
        if isinstance(candidates, list) and candidates:
            questions = [str(q) for q in candidates if q][:max_questions]
            return text[:m.start()].strip(), questions

    return raw, []


# Bibliography sections always start with one of these headings, alone on
# their own line (optionally followed by a colon). Requiring end-of-line
# avoids false positives like "...the actual bibliography entry follows".
_REFERENCES_HEADING = re.compile(
    r'\b(references|bibliography|works\s+cited|literature\s+cited)\b[ \t]*:?[ \t]*(?=\n|$)',
    re.IGNORECASE,
)


def extract_references_section(text: str) -> str:
    """
    Return the text following the last "References"/"Bibliography"/etc.
    heading in ``text``, or ``""`` if no such heading is found.

    Used by the Notebook Citation Timeline feature to isolate a source
    document's bibliography from its body text. Heading matches in the
    first 40% of the document (e.g. a table-of-contents entry or an
    in-text mention) are ignored unless nothing later qualifies.
    """
    matches = list(_REFERENCES_HEADING.finditer(text))
    if not matches:
        return ""
    cutoff = len(text) * 0.4
    candidates = [m for m in matches if m.start() >= cutoff] or matches
    return text[candidates[-1].end():].strip()
