"""tools/call_analyzer.py
Fetch a web page URL and extract its text content.
Used by web_loader.py to ingest web pages as notebook documents.
"""
from __future__ import annotations

import html as _html
import json
import logging
import re
from html.parser import HTMLParser
from typing import Dict, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_MAX_CHARS = 14_000


class _StripHTML(HTMLParser):
    _SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside"}

    def __init__(self):
        super().__init__()
        self._buf: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self._SKIP and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth:
            s = data.strip()
            if s:
                self._buf.append(s)

    def text(self) -> str:
        raw = " ".join(self._buf)
        raw = _html.unescape(raw)
        return re.sub(r"\s{2,}", " ", raw).strip()


def fetch_call_text(url: str, timeout: int = 20) -> Tuple[str, str]:
    """Fetch a web page; return (text, error). error='' on success."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return "", "URL must start with http:// or https://"

        resp = requests.get(url, headers=_FETCH_HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        ct = resp.headers.get("Content-Type", "").lower()
        if "html" in ct or "text" in ct:
            parser = _StripHTML()
            parser.feed(resp.text)
            text = parser.text()
        else:
            text = resp.text

        if len(text) < 120:
            return "", (
                "The page returned very little text — it may require JavaScript "
                "or a login to access."
            )

        return text[:_MAX_CHARS], ""

    except requests.exceptions.Timeout:
        return "", f"Request timed out after {timeout}s."
    except requests.exceptions.SSLError:
        return "", "SSL certificate error."
    except requests.exceptions.HTTPError as exc:
        if exc.response is None:
            return "", f"HTTP request failed: {exc}"
        code = exc.response.status_code
        if code == 403:
            return "", "Access denied (HTTP 403)."
        if code == 404:
            return "", "Page not found (HTTP 404)."
        return "", f"HTTP {code}: {exc.response.reason}"
    except Exception as exc:
        return "", f"Could not fetch URL: {exc}"


def extract_call_requirements(page_text: str, llm) -> dict:
    """Extract structured requirements from a page via LLM (used by web_loader)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "You are a research requirements analyst. Extract structured requirements "
        "from the given page text. Return ONLY valid JSON."
    )
    human = f"PAGE TEXT:\n{page_text}\n\nReturn a JSON dict with: title, summary, key_points (list), requirements (list)."

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = resp.content.strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        idx = raw.find("{")
        if idx >= 0:
            import json as _json
            decoder = _json.JSONDecoder()
            result, _ = decoder.raw_decode(raw, idx)
            if isinstance(result, dict):
                return result
    except Exception as exc:
        logger.warning("extract_call_requirements failed: %s", exc)

    return {}


def format_call_context_block(analysis: dict) -> str:
    """Format extracted page analysis for injection into LLM prompts."""
    if not analysis:
        return ""
    lines = ["\nPAGE CONTEXT:"]
    if analysis.get("title"):
        lines.append(f"  Title: {analysis['title']}")
    if analysis.get("summary"):
        lines.append(f"  Summary: {analysis['summary']}")
    if analysis.get("key_points"):
        lines.append("  Key Points:")
        for pt in analysis["key_points"]:
            lines.append(f"    - {pt}")
    return "\n".join(lines)
