"""
tools/citation_context.py
─────────────────────────────
Citation context extraction (scite.ai-style "snippet" view), best-effort and
open-access only.

Given a citing paper and a cited paper, this tries to surface the actual
sentence(s) in the citing paper's full text that reference the cited work —
the "where exactly does A cite B, and what does it say there?" view that an
abstract alone can't provide.

It is deliberately *descoped* and fails safe, matching the rest of BeeSearch:
full text is only attempted when the citing paper exposes a fetchable URL
(ideally the Semantic Scholar ``openAccessPdf`` link the search layer already
prefers). When no open-access text is available, the fetch fails, the PDF
parser isn't installed, or the cited work simply isn't detected in the text,
the function returns a structured "unavailable"/"not_found" result rather than
raising — so callers can show "no context available" instead of an error.

The sentence-matching core (``find_citation_mentions``) is pure and unit-tested
without any network access; only ``_fetch_fulltext`` touches the network.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Words too common to identify a cited paper by — skipped when picking a
# "distinctive" title token to search the citing text for.
_TITLE_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "with", "using",
    "via", "from", "by", "study", "analysis", "approach", "method", "methods",
    "model", "models", "review", "novel", "based", "toward", "towards", "case",
    "paper", "new", "data", "system", "systems", "learning", "deep", "neural",
}


def _first_author_surname(authors: List[str]) -> str:
    """Return the first author's surname, or "" if unavailable.

    Handles both "Given Family" and "Family, Given" orderings — the search
    backends are not consistent about which they return.
    """
    if not authors:
        return ""
    first = (authors[0] or "").strip()
    if not first:
        return ""
    if "," in first:
        return first.split(",")[0].strip()
    return first.split()[-1].strip()


def _distinctive_title_tokens(title: str, max_tokens: int = 3) -> List[str]:
    """Pick the longest non-stopword tokens from a title to search the citing text for."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", title or "")
    cand = [w for w in words if w.lower() not in _TITLE_STOPWORDS]
    cand.sort(key=len, reverse=True)
    seen: set = set()
    out: List[str] = []
    for w in cand:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            out.append(w)
        if len(out) >= max_tokens:
            break
    return out


def split_sentences(text: str) -> List[str]:
    """Split *text* into sentences with a lightweight regex (no NLP dependency).

    Splits on sentence-final punctuation followed by whitespace. Good enough for
    locating a citing sentence in academic prose; collapses internal whitespace
    so a sentence broken across PDF lines reads as one.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text)
    return [p.strip() for p in parts if p.strip()]


def find_citation_mentions(
    text: str,
    cited_paper: Dict[str, Any],
    max_contexts: int = 3,
) -> List[Dict[str, str]]:
    """Find sentences in *text* that plausibly cite ``cited_paper``.

    A sentence matches if it contains the cited first-author surname, or a
    distinctive title token (optionally corroborated by the publication year).
    Returns up to ``max_contexts`` dicts ``{"sentence", "matched_on"}`` in order
    of appearance; ``[]`` when nothing matches. Pure function — no network.
    """
    surname = _first_author_surname(cited_paper.get("authors", []) or [])
    title_tokens = _distinctive_title_tokens(cited_paper.get("title", "") or "")
    year = str(cited_paper.get("year") or "")

    surname_l = surname.lower()
    tokens_l = [t.lower() for t in title_tokens]

    results: List[Dict[str, str]] = []
    for sentence in split_sentences(text):
        s_l = sentence.lower()
        matched_on: Optional[str] = None
        if surname_l and surname_l in s_l:
            matched_on = f"author:{surname}"
        else:
            hit = next((t for t in tokens_l if t in s_l), None)
            if hit and (not year or year in sentence):
                matched_on = f"title:{hit}"
        if matched_on:
            results.append({"sentence": sentence, "matched_on": matched_on})
        if len(results) >= max_contexts:
            break
    return results


def _best_fulltext_url(paper: Dict[str, Any]) -> str:
    """Return a URL worth attempting full-text extraction on, or "" if none.

    Prefers an explicit open-access PDF URL, then the paper's main URL. Skips
    pure DOI landing pages and Semantic Scholar API URLs, which are HTML
    redirects rather than retrievable full text.
    """
    for key in ("open_access_pdf", "openAccessPdf", "pdf_url", "url"):
        val = paper.get(key)
        if isinstance(val, dict):
            val = val.get("url", "")
        if not val or not isinstance(val, str):
            continue
        if val.startswith("http") and "api.semanticscholar.org" not in val:
            return val
    return ""


def _fetch_fulltext(url: str, timeout: int = 20) -> tuple[str, str]:
    """Fetch *url* and return ``(text, kind)`` where kind is "pdf"/"html", or ``("", "")``.

    PDFs are parsed with pdfplumber (capped at the first 20 pages to bound time
    and memory); HTML has its tags stripped. Any failure — network error,
    missing pdfplumber, unreadable content — returns ``("", "")`` so the caller
    degrades to "context unavailable" rather than raising.
    """
    try:
        import requests
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "BeeSearch/1.0"})
        resp.raise_for_status()
    except Exception as e:
        logger.debug("citation context fetch failed for %s: %s", url[:60], e)
        return "", ""

    content_type = resp.headers.get("Content-Type", "").lower()
    is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")

    if is_pdf:
        try:
            import io

            import pdfplumber
            text_parts: List[str] = []
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                for page in pdf.pages[:20]:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts), "pdf"
        except Exception as e:
            logger.debug("citation context PDF parse failed for %s: %s", url[:60], e)
            return "", ""

    # HTML — strip tags. Drop script/style bodies first so their contents don't
    # leak into the extracted prose.
    try:
        html = resp.text
        html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text, "html"
    except Exception as e:
        logger.debug("citation context HTML parse failed for %s: %s", url[:60], e)
        return "", ""


def extract_citation_context(
    citing_paper: Dict[str, Any],
    cited_paper: Dict[str, Any],
    timeout: int = 20,
    max_contexts: int = 3,
) -> Dict[str, Any]:
    """Best-effort: surface the sentence(s) where ``citing_paper`` cites ``cited_paper``.

    Returns a status dict — never raises:
      - ``{"status": "ok", "source_url", "kind", "contexts": [...]}`` on success
      - ``{"status": "unavailable", "reason", "contexts": []}`` when there's no
        open-access full text to read or it couldn't be fetched/parsed
      - ``{"status": "not_found", "reason", "contexts": []}`` when the full text
        was read but the cited work wasn't detected in it

    Each context is ``{"sentence", "matched_on"}``. Only the citing paper's full
    text is fetched, and only when it exposes a usable open-access URL.
    """
    url = _best_fulltext_url(citing_paper)
    if not url:
        return {
            "status": "unavailable",
            "reason": "No open-access full-text URL available for the citing paper.",
            "contexts": [],
        }

    text, kind = _fetch_fulltext(url, timeout=timeout)
    if not text:
        return {
            "status": "unavailable",
            "reason": "Could not fetch or parse the citing paper's full text.",
            "contexts": [],
            "source_url": url,
        }

    contexts = find_citation_mentions(text, cited_paper, max_contexts=max_contexts)
    if not contexts:
        return {
            "status": "not_found",
            "reason": "The cited paper was not detected in the citing paper's full text.",
            "contexts": [],
            "source_url": url,
        }

    return {"status": "ok", "source_url": url, "kind": kind, "contexts": contexts}
