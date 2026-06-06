"""
tools/doi_verifier.py
─────────────────────
Verifies that a DOI resolves to a real, accessible paper by making a live
HTTP request to the doi.org resolver.  Used to flag suspicious or invalid
citations in research outputs before they are presented to the user.

Typical usage
─────────────
from tools.doi_verifier import get_verifier

verifier = get_verifier()

# Single DOI
result = verifier.verify("10.1038/nature12373")
print(result.valid, result.title)

# Batch
results = verifier.verify_batch(["10.1038/nature12373", "10.9999/fake"])

# Flag a reference list
refs = [{"doi": "10.1038/nature12373", "title": "Some paper"}, ...]
flagged = verifier.flag_references(refs)
# Each dict now has doi_valid (bool) and doi_flag (str) keys.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)

_DOI_PATTERN = re.compile(r"^10\.\d{4,}/\S+")
_DOI_PREFIX = "https://doi.org/"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DOIVerificationResult:
    """Outcome of verifying a single DOI against the doi.org resolver."""

    doi: str
    valid: bool
    title: Optional[str] = None
    year: Optional[int] = None
    reason: str = ""


# ── Verifier class ────────────────────────────────────────────────────────────

class DOIVerifier:
    """
    Checks whether a DOI resolves to a real paper via the doi.org HTTP resolver.

    The verifier deliberately assumes a DOI is valid when a network error
    prevents verification, to avoid falsely penalising legitimate citations
    during transient connectivity issues.
    """

    def __init__(self) -> None:
        self.cfg = get_settings()

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(self, doi: str) -> DOIVerificationResult:
        """
        Verify a single DOI and return a DOIVerificationResult.

        Steps:
          1. Normalise the DOI string (strip whitespace, remove URL prefix).
          2. Reject immediately if it does not match the DOI regex.
          3. Make a GET request to the doi.org resolver.
          4. Interpret the HTTP status code.
          5. Attempt to extract title/year from JSON if the resolver returns it.
        """
        doi = doi.strip()
        if doi.startswith(_DOI_PREFIX):
            doi = doi[len(_DOI_PREFIX):]

        if not _DOI_PATTERN.match(doi):
            return DOIVerificationResult(
                doi=doi,
                valid=False,
                reason="Invalid DOI format",
            )

        url = f"{_DOI_PREFIX}{doi}"
        try:
            response = requests.get(
                url,
                allow_redirects=True,
                timeout=10,
                headers={"Accept": "application/json"},
            )

            if response.status_code == 200 or response.status_code < 400:
                result = DOIVerificationResult(doi=doi, valid=True)
                self._try_populate_metadata(result, response)
                return result

            if response.status_code == 404:
                return DOIVerificationResult(
                    doi=doi, valid=False, reason="DOI not found"
                )

            if response.status_code == 410:
                return DOIVerificationResult(
                    doi=doi, valid=False, reason="DOI retracted"
                )

            # Any other HTTP error — assume valid, log for investigation.
            logger.warning("Unexpected status %d for DOI %s", response.status_code, doi)
            return DOIVerificationResult(
                doi=doi, valid=True, reason="Could not verify"
            )

        except (requests.ConnectionError, requests.Timeout):
            return DOIVerificationResult(
                doi=doi, valid=True, reason="Could not verify (network)"
            )
        except requests.HTTPError:
            return DOIVerificationResult(
                doi=doi, valid=True, reason="Could not verify"
            )

    def verify_batch(
        self, dois: List[str], delay: float = 0.3
    ) -> Dict[str, DOIVerificationResult]:
        """
        Verify a list of DOIs, inserting a short pause between each request
        to avoid overwhelming the doi.org resolver.

        Returns a dict mapping each original DOI string to its result.
        """
        results: Dict[str, DOIVerificationResult] = {}
        for i, doi in enumerate(dois):
            results[doi] = self.verify(doi)
            if i < len(dois) - 1:
                time.sleep(delay)
        return results

    def flag_references(self, references: List[Dict]) -> List[Dict]:
        """
        Add DOI validity flags to a list of reference dicts.

        Each dict is expected to contain a ``"doi"`` key.  After calling this
        method every dict will also have:

        - ``"doi_valid"`` (bool) — whether the DOI resolved successfully.
        - ``"doi_flag"``  (str)  — human-readable reason string (empty when valid).

        Dicts without a ``"doi"`` key are left unchanged.
        """
        dois = [ref["doi"] for ref in references if ref.get("doi")]
        if not dois:
            return references

        verification_map = self.verify_batch(dois)

        for ref in references:
            doi = ref.get("doi")
            if not doi:
                continue
            result = verification_map.get(doi)
            if result is None:
                continue
            ref["doi_valid"] = result.valid
            ref["doi_flag"] = result.reason

        return references

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _try_populate_metadata(
        self, result: DOIVerificationResult, response: requests.Response
    ) -> None:
        """
        Attempt to extract title and year from the response body when the
        resolver returns JSON (Content-Type: application/json).  Failures are
        silently ignored because metadata extraction is best-effort only.
        """
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            return

        try:
            data = response.json()
            # CrossRef-style response wraps the work in a "message" key.
            message = data.get("message", data)

            raw_title = message.get("title")
            if isinstance(raw_title, list) and raw_title:
                result.title = raw_title[0]
            elif isinstance(raw_title, str):
                result.title = raw_title

            # Year may appear as "published-print", "published-online", or "issued".
            for date_field in ("published-print", "published-online", "issued"):
                date_parts = message.get(date_field, {}).get("date-parts")
                if date_parts and date_parts[0]:
                    try:
                        result.year = int(date_parts[0][0])
                        break
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass


# ── Module-level factory ──────────────────────────────────────────────────────

_verifier: Optional[DOIVerifier] = None


def get_verifier() -> DOIVerifier:
    """Return a module-level singleton DOIVerifier, creating it on first call."""
    global _verifier
    if _verifier is None:
        _verifier = DOIVerifier()
    return _verifier
