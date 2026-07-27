"""
tools/web_loader.py
────────────────────
Fetch a web page and turn it into a ProcessedDocument so it can be ingested
into a Research Notebook exactly like an uploaded file.

Reuses tools.call_analyzer.fetch_call_text for the actual HTTP fetch + HTML
cleaning (it already strips nav/script/style, decodes entities, handles
timeouts and friendly HTTP-error messages), then runs the cleaned text through
the standard DocumentProcessor so chunking/citation metadata is identical to
uploaded documents.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _title_from_url(url: str) -> str:
    """Derive a short, human-readable source name from a URL."""
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    path = parsed.path.rstrip("/").split("/")[-1]
    path = re.sub(r"\.(html?|php|aspx?)$", "", path, flags=re.IGNORECASE)
    path = re.sub(r"[-_]+", " ", path).strip()
    if path:
        return f"{host} — {path}"[:80]
    return host[:80] or url[:80]


def load_url_as_document(
    url: str,
    processor,
    timeout: int = 20,
) -> Tuple[Optional[object], str]:
    """
    Fetch `url`, clean it, and return (ProcessedDocument, error).

    On success error == "". On failure the document is None and error holds a
    user-friendly message.

    Parameters
    ----------
    processor : a tools.document_tools.DocumentProcessor instance
    """
    from tools.call_analyzer import fetch_call_text

    text, err = fetch_call_text(url, timeout=timeout)
    if err:
        return None, err
    if not text.strip():
        return None, "The page contained no extractable text."

    name = _title_from_url(url)
    try:
        doc = processor.process_raw_text(text, name=name)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Failed to process web page %s: %s", url, exc)
        return None, f"Could not process page content: {exc}"

    return doc, ""
