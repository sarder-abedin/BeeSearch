"""tools/call_analyzer.py
──────────────────────────
Fetch a project call / RFP URL and extract structured requirements via LLM.

Used by Mode 4 (Proposal Writer) to pre-analyze a funding call before
generating the proposal, so every section targets the actual call criteria.
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

_MAX_CHARS = 14_000   # chars passed to LLM — large enough for full call text


# ── HTML text extractor ────────────────────────────────────────────────────────

class _StripHTML(HTMLParser):
    """Strip HTML tags, skip nav/script/style, decode entities."""

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


# ── Public: fetch call page ────────────────────────────────────────────────────

def fetch_call_text(url: str, timeout: int = 20) -> Tuple[str, str]:
    """Fetch a project call page; return (text, error).  error='' on success."""
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
                "or a login to access. Try copying the call text manually."
            )

        return text[:_MAX_CHARS], ""

    except requests.exceptions.Timeout:
        return "", f"Request timed out after {timeout}s. The server did not respond."
    except requests.exceptions.SSLError:
        return "", "SSL certificate error. Try the http:// version of the URL if available."
    except requests.exceptions.HTTPError as exc:
        if exc.response is None:
            return "", f"HTTP request failed: {exc}"
        code = exc.response.status_code
        reason = exc.response.reason
        if code == 403:
            return "", "Access denied (HTTP 403). The site may block automated requests."
        if code == 404:
            return "", "Page not found (HTTP 404). Check the URL."
        return "", f"HTTP {code}: {reason}"
    except Exception as exc:
        return "", f"Could not fetch URL: {exc}"


# ── Public: LLM extraction ─────────────────────────────────────────────────────

def extract_call_requirements(page_text: str, llm) -> dict:
    """
    Send page text to the LLM and extract structured call requirements.
    Returns a dict; all fields optional (may be null / empty list).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "You are a research funding expert. Extract structured requirements from a "
        "project call / funding opportunity. Return ONLY valid JSON — no prose."
    )

    human = f"""PROJECT CALL TEXT:
{page_text}

Extract every piece of information relevant to writing a successful proposal.
Return EXACTLY this JSON structure (use null for missing string fields, [] for missing lists):

{{
  "call_title": "Full name of the call or RFP",
  "funder": "Funding agency or organisation name",
  "deadline": "Application deadline (exact date if available)",
  "budget_range": "Typical / maximum budget per project",
  "duration": "Expected project duration",
  "eligibility": ["Who can apply — institutions, countries, career stage, etc."],
  "required_sections": ["Sections or documents the proposal MUST include"],
  "evaluation_criteria": ["Criteria used to score / rank proposals"],
  "key_themes": ["Research topics, challenges, or themes the call targets"],
  "restrictions": ["What is explicitly excluded or NOT funded"],
  "important_notes": ["Deadlines, formatting rules, page limits, co-funding requirements, etc."],
  "clarifying_questions": [
    {{
      "question": "Question the researcher should answer to write a targeted proposal?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "recommended": "Option A"
    }}
  ],
  "recommendations": [
    "2–3 concrete writing tips specific to this call's evaluation criteria"
  ]
}}

Rules for clarifying_questions:
- Generate 2–4 questions
- Each question must have 3–4 concrete answer options
- "recommended" must exactly match one of the options — it is the most likely \
  best choice for a typical researcher applying to this call

Return ONLY the JSON object."""

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = resp.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        # Use JSONDecoder.raw_decode to find the FIRST valid JSON object,
        # avoiding greedy regex that collects everything between first { and last }.
        idx = raw.find("{")
        if idx >= 0:
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(raw, idx)
            if isinstance(result, dict):
                return result
        logger.warning("LLM call analysis returned no valid JSON object")
    except Exception as exc:
        logger.warning("Call requirement extraction failed: %s", exc)

    return {}


# ── Public: render helpers ─────────────────────────────────────────────────────

def format_call_context_block(analysis: dict) -> str:
    """
    Return a formatted block for injection into LLM section prompts.
    Empty string if analysis is empty or None.
    """
    if not analysis:
        return ""

    lines = ["\nPROJECT CALL REQUIREMENTS (must be reflected in the proposal):"]

    meta_parts = []
    if analysis.get("call_title"):
        meta_parts.append(f"Call: {analysis['call_title']}")
    if analysis.get("funder"):
        meta_parts.append(f"Funder: {analysis['funder']}")
    if analysis.get("deadline"):
        meta_parts.append(f"Deadline: {analysis['deadline']}")
    if analysis.get("budget_range"):
        meta_parts.append(f"Budget: {analysis['budget_range']}")
    if analysis.get("duration"):
        meta_parts.append(f"Duration: {analysis['duration']}")
    if meta_parts:
        lines.append("  " + " | ".join(meta_parts))

    def _fmt_list(label: str, items: list) -> None:
        if items:
            lines.append(f"  {label}:")
            for item in items:
                lines.append(f"    - {item}")

    _fmt_list("Required Sections", analysis.get("required_sections", []))
    _fmt_list("Evaluation Criteria", analysis.get("evaluation_criteria", []))
    _fmt_list("Key Themes", analysis.get("key_themes", []))
    _fmt_list("Restrictions", analysis.get("restrictions", []))
    _fmt_list("Important Notes", analysis.get("important_notes", []))

    return "\n".join(lines)
