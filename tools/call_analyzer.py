"""
tools/call_analyzer.py
───────────────────────
Fetch a web page over HTTP and reduce it to clean, readable plain text.

`fetch_call_text` is the low-level fetch+clean primitive that
tools.web_loader.load_url_as_document relies on to turn an arbitrary URL
into notebook-ingestible text: it issues the HTTP request, turns
timeouts/HTTP failures into friendly messages, and strips
script/style/nav/footer markup so only the page's readable content
remains (matching what uploaded documents go through).
"""

from __future__ import annotations

import logging
import re
from html import unescape
from html.parser import HTMLParser
from typing import List, Tuple

import requests

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; ResearchBuddy/1.0; +https://github.com/sarder-abedin/ResearchBuddy) "
    "research-notebook-fetcher"
)

# Tags whose entire contents (including nested tags) should be discarded outright.
_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "noscript", "svg", "form", "aside", "iframe"}
# Tags that introduce a line break so block structure survives as plain text.
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "section", "article", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6",
}


class _TextExtractor(HTMLParser):
    """Reduces an HTML document to readable text, dropping markup/chrome."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        raw = unescape("".join(self._chunks))
        lines = (re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines())
        return "\n".join(ln for ln in lines if ln)


def fetch_call_text(url: str, timeout: int = 20) -> Tuple[str, str]:
    """
    Fetch `url` and return (cleaned_text, error).

    On success, error == "" and cleaned_text holds the page's readable
    content with scripts/styles/navigation stripped and HTML entities
    decoded. On failure, cleaned_text == "" and error holds a short,
    user-friendly message safe to show directly in the UI.
    """
    try:
        resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return "", f"The page took too long to respond (>{timeout}s)."
    except requests.exceptions.ConnectionError:
        return "", "Could not connect to that URL — check the address and your network connection."
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        return "", f"The page returned an error (HTTP {status})."
    except requests.exceptions.RequestException as exc:
        return "", f"Could not fetch the page: {exc}"

    extractor = _TextExtractor()
    try:
        extractor.feed(resp.text)
        extractor.close()
    except Exception as exc:
        logger.warning("HTML parsing failed for %s: %s", url, exc)
        return "", "Could not parse the page's HTML content."

    return extractor.text(), ""
