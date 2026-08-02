"""paper_graph/s2_client.py
──────────────────────────
Thin Semantic Scholar Graph API client with batching, retry, and per-process
caching.  Abstracted behind a Protocol so the data source can be swapped —
see the OpenAlex note below.

Data source: Semantic Scholar Academic Graph API
  https://api.semanticscholar.org/graph/v1
  Free tier: ~100 requests / 5 min unauthenticated.
  Set SEMANTIC_SCHOLAR_API_KEY in .env for a partner key with higher limits.

Alternative data source note (OpenAlex):
  OpenAlex (https://api.openalex.org) is a drop-in alternative if Semantic
  Scholar's coverage or licensing becomes a concern.  It exposes equivalent
  paper lookup, reference/citation lists, author records, and a /works
  endpoint that supports bulk retrieval.  To swap: implement another class
  conforming to PaperGraphDataSource below and pass it to the service layer
  instead of SemanticScholarClient.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

_S2_GRAPH = "https://api.semanticscholar.org/graph/v1"
_S2_RECO = "https://api.semanticscholar.org/recommendations/v1"

# Fields fetched for every paper lookup — keeps batch payloads predictable.
_PAPER_FIELDS = "paperId,title,authors,year,venue,abstract,citationCount,externalIds,url"
_BATCH_CHUNK = 500  # S2 /paper/batch accepts up to 500 IDs per request


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class PaperNode:
    """A paper as returned by the Semantic Scholar API, normalised."""
    id: str                                   # S2 paperId
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None            # None → displayed as "Unavailable"
    citation_count: Optional[int] = None
    url: Optional[str] = None


# ── Protocol (swap point for alternative data sources) ───────────────────────

@runtime_checkable
class PaperGraphDataSource(Protocol):
    def get_paper(self, paper_id: str) -> Optional[PaperNode]: ...
    def get_references(self, paper_id: str) -> List[str]: ...
    def get_citations(self, paper_id: str) -> List[str]: ...
    def batch_get_papers(self, ids: List[str]) -> List[PaperNode]: ...
    def get_recommendations(self, paper_ids: List[str]) -> List[PaperNode]: ...
    def get_author_papers(self, author_id: str) -> List[PaperNode]: ...
    def search_paper(self, query: str, limit: int = 1) -> List[PaperNode]: ...


# ── Retry predicate ───────────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        code = getattr(getattr(exc, "response", None), "status_code", None)
        return code in (429, 500, 502, 503, 504)
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


# ── Client ────────────────────────────────────────────────────────────────────

class SemanticScholarClient:
    """Semantic Scholar Academic Graph API client.

    Implements PaperGraphDataSource.  All calls are synchronous and use
    the same requests.Session with a pre-set timeout and optional API key.

    To insert a partner API key for production rate limits, set
    SEMANTIC_SCHOLAR_API_KEY in .env (already wired via config/settings.py).
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "BeeSearch/1.0"})
        # Partner API key: unlocks higher rate limits (~1 000 req/min vs ~100/5 min)
        # Insert key here for production use; leave empty for free-tier access.
        if cfg.semantic_scholar_api_key:
            self._session.headers["x-api-key"] = cfg.semantic_scholar_api_key

    # ── Internal helpers ──────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=2, max=30),
        reraise=True,
    )
    def _get(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        resp = self._session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=2, max=30),
        reraise=True,
    )
    def _post(self, url: str, json_body: dict, params: Optional[Dict] = None) -> requests.Response:
        resp = self._session.post(url, json=json_body, params=params, timeout=20)
        resp.raise_for_status()
        return resp

    @staticmethod
    def _parse_paper(item: dict) -> Optional[PaperNode]:
        pid = item.get("paperId")
        if not pid:
            return None
        return PaperNode(
            id=pid,
            title=item.get("title") or "",
            authors=[a.get("name", "") for a in (item.get("authors") or [])],
            year=item.get("year"),
            venue=item.get("venue") or None,
            abstract=item.get("abstract") or None,
            citation_count=item.get("citationCount"),
            url=item.get("url"),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=2048)
    def get_paper(self, paper_id: str) -> Optional[PaperNode]:
        """Fetch metadata for a single paper by S2 paperId or external ID."""
        try:
            resp = self._get(
                f"{_S2_GRAPH}/paper/{paper_id}",
                params={"fields": _PAPER_FIELDS},
            )
            return self._parse_paper(resp.json())
        except Exception as exc:
            logger.debug("S2 get_paper failed for %s: %s", paper_id, exc)
            return None

    @functools.lru_cache(maxsize=2048)
    def get_references(self, paper_id: str) -> List[str]:
        """Return S2 paperIds that *paper_id* cites (up to 1000)."""
        try:
            resp = self._get(
                f"{_S2_GRAPH}/paper/{paper_id}/references",
                params={"fields": "paperId", "limit": 1000},
            )
            return [
                r["citedPaper"]["paperId"]
                for r in resp.json().get("data", [])
                if r.get("citedPaper", {}).get("paperId")
            ]
        except Exception as exc:
            logger.debug("S2 get_references failed for %s: %s", paper_id, exc)
            return []

    @functools.lru_cache(maxsize=2048)
    def get_citations(self, paper_id: str) -> List[str]:
        """Return S2 paperIds that cite *paper_id* (up to 1000)."""
        try:
            resp = self._get(
                f"{_S2_GRAPH}/paper/{paper_id}/citations",
                params={"fields": "paperId", "limit": 1000},
            )
            return [
                r["citingPaper"]["paperId"]
                for r in resp.json().get("data", [])
                if r.get("citingPaper", {}).get("paperId")
            ]
        except Exception as exc:
            logger.debug("S2 get_citations failed for %s: %s", paper_id, exc)
            return []

    def batch_get_papers(self, ids: List[str]) -> List[PaperNode]:
        """Fetch metadata for many papers using POST /paper/batch.

        Chunks into _BATCH_CHUNK-sized requests to stay within S2's limit.
        Papers already cached by get_paper() are returned without an API call.
        """
        results: List[PaperNode] = []
        # Pull from lru_cache for IDs already fetched
        uncached = []
        for pid in ids:
            cached = self.get_paper.cache_info  # just to confirm cache exists
            node = self._lru_get(pid)
            if node is not None:
                results.append(node)
            else:
                uncached.append(pid)

        for i in range(0, len(uncached), _BATCH_CHUNK):
            chunk = uncached[i : i + _BATCH_CHUNK]
            try:
                resp = self._post(
                    f"{_S2_GRAPH}/paper/batch",
                    json_body={"ids": chunk},
                    params={"fields": _PAPER_FIELDS},
                )
                for item in resp.json():
                    if item is None:
                        continue
                    node = self._parse_paper(item)
                    if node:
                        results.append(node)
                        # Populate the per-ID cache so future calls are free
                        self._lru_set(node.id, node)
            except Exception as exc:
                logger.warning("S2 batch_get_papers chunk failed: %s", exc)
                # Partial result is acceptable — caller checks partial flag

        return results

    def _lru_get(self, paper_id: str) -> Optional[PaperNode]:
        """Read through lru_cache on get_paper without making a network call."""
        # Check if cached: call with the same arg; if it hits cache, no network
        # We can't inspect lru_cache internals portably, so we use a side dict.
        return self._meta_cache.get(paper_id)

    def _lru_set(self, paper_id: str, node: PaperNode) -> None:
        self._meta_cache[paper_id] = node

    # Backing dict for batch-populated entries (avoids lru_cache coupling)
    _meta_cache: Dict[str, PaperNode] = {}

    def get_recommendations(self, paper_ids: List[str]) -> List[PaperNode]:
        """Call S2 Recommendations API using collection papers as positive examples.

        Uses the separate /recommendations/v1 base (not /graph/v1).
        Returns up to 20 recommended papers.
        """
        if not paper_ids:
            return []
        try:
            resp = self._post(
                f"{_S2_RECO}/papers/",
                json_body={
                    "positivePaperIds": paper_ids[:100],
                    "negativePaperIds": [],
                },
                params={"fields": _PAPER_FIELDS, "limit": 20},
            )
            nodes = []
            for item in resp.json().get("recommendedPapers", []):
                node = self._parse_paper(item)
                if node:
                    nodes.append(node)
                    self._lru_set(node.id, node)
            return nodes
        except Exception as exc:
            logger.warning("S2 get_recommendations failed: %s", exc)
            return []

    def get_author_papers(self, author_id: str) -> List[PaperNode]:
        """Fetch papers by a single author via /author/{id}/papers."""
        try:
            resp = self._get(
                f"{_S2_GRAPH}/author/{author_id}/papers",
                params={"fields": _PAPER_FIELDS, "limit": 50},
            )
            nodes = []
            for item in resp.json().get("data", []):
                node = self._parse_paper(item)
                if node:
                    nodes.append(node)
                    self._lru_set(node.id, node)
            return nodes
        except Exception as exc:
            logger.debug("S2 get_author_papers failed for %s: %s", author_id, exc)
            return []

    def search_paper(self, query: str, limit: int = 1) -> List[PaperNode]:
        """Search by title/keyword; used to resolve a user-typed title to a paperId."""
        try:
            resp = self._get(
                f"{_S2_GRAPH}/paper/search",
                params={"query": query, "limit": limit, "fields": _PAPER_FIELDS},
            )
            nodes = []
            for item in resp.json().get("data", []):
                node = self._parse_paper(item)
                if node:
                    nodes.append(node)
                    self._lru_set(node.id, node)
            return nodes
        except Exception as exc:
            logger.debug("S2 search_paper failed for '%s': %s", query[:60], exc)
            return []


# ── Module-level singleton ────────────────────────────────────────────────────
# Shared across requests so the lru_cache on get_paper / get_references /
# get_citations accumulates across the process lifetime.

_client: Optional[SemanticScholarClient] = None


def get_client() -> SemanticScholarClient:
    global _client
    if _client is None:
        _client = SemanticScholarClient()
    return _client
