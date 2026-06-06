"""
tools/preprint_tracker.py
─────────────────────────
Detects arXiv preprints in a corpus and links them to published versions
via CrossRef title search. Also flags retracted papers.

Statuses:
  "journal"    — sourced directly from a journal with a DOI
  "published"  — arXiv preprint matched to a published journal version
  "preprint"   — arXiv paper; no published version found
  "retracted"  — CrossRef retraction notice detected
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional

import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()


def _crossref_by_title(title: str) -> Optional[Dict]:
    """Search CrossRef for an item matching the title. Returns the best item or None."""
    try:
        resp = requests.get(
            cfg.crossref_base_url,
            params={
                "query.title": title,
                "rows": 3,
                "mailto": cfg.crossref_email,
                "select": "DOI,title,container-title,published,update-policy,relation",
            },
            timeout=10,
        )
        resp.raise_for_status()
        for item in resp.json().get("message", {}).get("items", []):
            item_title = (item.get("title") or [""])[0]
            norm_q = re.sub(r"\W+", "", title.lower())[:30]
            norm_i = re.sub(r"\W+", "", item_title.lower())[:30]
            if norm_q and norm_i and norm_q[:15] == norm_i[:15]:
                return item
    except Exception as e:
        logger.debug("CrossRef title search failed for '%s': %s", title[:40], e)
    return None


def _is_retracted(item: Dict) -> bool:
    if "retract" in (item.get("update-policy") or "").lower():
        return True
    if (item.get("relation") or {}).get("is-retracted-by"):
        return True
    return False


def track_preprints(papers: List[Dict]) -> List[Dict]:
    """
    Check each paper's publication status and return a tracking list.

    Each entry:
      paper            — original paper dict
      preprint_status  — "journal" | "published" | "preprint" | "retracted"
      published_doi    — DOI of the confirmed published version (if found)
      published_venue  — Journal name (if found)
      note             — human-readable explanation
    """
    results = []

    for paper in papers:
        source = paper.get("source", "")
        existing_doi = paper.get("doi")
        title = paper.get("title", "")

        # Non-arXiv papers with a DOI are already confirmed journal papers
        if source != "arxiv" and existing_doi:
            results.append({
                "paper": paper,
                "preprint_status": "journal",
                "published_doi": existing_doi,
                "published_venue": paper.get("journal", ""),
                "note": "Sourced directly from a journal database with DOI.",
            })
            continue

        item = _crossref_by_title(title)
        time.sleep(0.25)

        if item:
            doi = item.get("DOI", "")
            venue = (item.get("container-title") or [""])[0]
            if _is_retracted(item):
                results.append({
                    "paper": paper,
                    "preprint_status": "retracted",
                    "published_doi": doi,
                    "published_venue": venue,
                    "note": f"RETRACTED — CrossRef flagged a retraction notice. DOI: {doi}",
                })
            elif source == "arxiv":
                results.append({
                    "paper": paper,
                    "preprint_status": "published",
                    "published_doi": doi,
                    "published_venue": venue,
                    "note": f"arXiv preprint — published version found in {venue or 'a journal'} (DOI: {doi}).",
                })
            else:
                results.append({
                    "paper": paper,
                    "preprint_status": "journal",
                    "published_doi": doi,
                    "published_venue": venue,
                    "note": f"Published in {venue or 'a journal'} (DOI: {doi}).",
                })
        else:
            if source == "arxiv":
                results.append({
                    "paper": paper,
                    "preprint_status": "preprint",
                    "published_doi": None,
                    "published_venue": None,
                    "note": "arXiv preprint — no matching published version found. Verify before citing.",
                })
            else:
                results.append({
                    "paper": paper,
                    "preprint_status": "journal",
                    "published_doi": existing_doi,
                    "published_venue": paper.get("journal", ""),
                    "note": "Journal paper (CrossRef lookup inconclusive).",
                })

    return results


def preprint_summary(results: List[Dict]) -> Dict[str, int]:
    """Count papers by preprint_status."""
    out: Dict[str, int] = {"journal": 0, "published": 0, "preprint": 0, "retracted": 0}
    for r in results:
        status = r.get("preprint_status", "journal")
        out[status] = out.get(status, 0) + 1
    return out
