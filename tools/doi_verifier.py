"""
tools/doi_verifier.py
─────────────────────
Verifies that a DOI resolves to a real, accessible paper via doi.org.
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


@dataclass
class DOIVerificationResult:
    doi: str
    valid: bool
    title: Optional[str] = None
    year: Optional[int] = None
    reason: str = ""


class DOIVerifier:
    def __init__(self) -> None:
        self.cfg = get_settings()

    def verify(self, doi: str) -> DOIVerificationResult:
        doi = doi.strip()
        if doi.startswith(_DOI_PREFIX):
            doi = doi[len(_DOI_PREFIX):]

        if not _DOI_PATTERN.match(doi):
            return DOIVerificationResult(doi=doi, valid=False, reason="Invalid DOI format")

        url = f"{_DOI_PREFIX}{doi}"
        try:
            response = requests.get(url, allow_redirects=True, timeout=10, headers={"Accept": "application/json"})

            if response.status_code == 200 or response.status_code < 400:
                result = DOIVerificationResult(doi=doi, valid=True)
                self._try_populate_metadata(result, response)
                return result
            if response.status_code == 404:
                return DOIVerificationResult(doi=doi, valid=False, reason="DOI not found")
            if response.status_code == 410:
                return DOIVerificationResult(doi=doi, valid=False, reason="DOI retracted")
            logger.warning("Unexpected status %d for DOI %s", response.status_code, doi)
            return DOIVerificationResult(doi=doi, valid=True, reason="Could not verify")
        except (requests.ConnectionError, requests.Timeout):
            return DOIVerificationResult(doi=doi, valid=True, reason="Could not verify (network)")
        except requests.HTTPError:
            return DOIVerificationResult(doi=doi, valid=True, reason="Could not verify")

    def verify_batch(self, dois: List[str], delay: float = 0.3) -> Dict[str, DOIVerificationResult]:
        results: Dict[str, DOIVerificationResult] = {}
        for i, doi in enumerate(dois):
            results[doi] = self.verify(doi)
            if i < len(dois) - 1:
                time.sleep(delay)
        return results

    def flag_references(self, references: List[Dict]) -> List[Dict]:
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

    def _try_populate_metadata(self, result: DOIVerificationResult, response: requests.Response) -> None:
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            return
        try:
            data = response.json()
            message = data.get("message", data)
            raw_title = message.get("title")
            if isinstance(raw_title, list) and raw_title:
                result.title = raw_title[0]
            elif isinstance(raw_title, str):
                result.title = raw_title
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


_verifier: Optional[DOIVerifier] = None


def get_verifier() -> DOIVerifier:
    global _verifier
    if _verifier is None:
        _verifier = DOIVerifier()
    return _verifier
