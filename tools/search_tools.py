"""
tools/search_tools.py
─────────────────────
Two search back-ends that power the agent's external knowledge:

  1. AcademicSearcher  — queries arXiv + Semantic Scholar + CrossRef
                         → returns peer-reviewed, citable papers

  2. WebSearcher       — queries the web via DuckDuckGo (ddgs), no API key needed
                         → useful when academic coverage is thin

TUTORIAL NOTE
─────────────
All three academic APIs are FREE and require no API key for basic usage:

  • arXiv         https://arxiv.org/help/api
  • Semantic Scholar https://api.semanticscholar.org  (optional key for ↑ rate)
  • CrossRef      https://api.crossref.org            (free, use polite pool)

For scientific integrity the agent always prefers arXiv + Semantic Scholar
results over general web results when citations are needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import quote_plus

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config.settings import get_settings

logger = logging.getLogger(__name__)

cfg = get_settings()


def _is_retryable(exc: BaseException) -> bool:
    """
    Decide whether @retry should retry *exc*.

    Retries only transient failures — rate-limiting (429) and server-side
    errors (5xx) — plus connection/timeout errors. Other HTTP errors (e.g.
    404, 400) are treated as permanent and propagate immediately.
    """
    if isinstance(exc, requests.exceptions.HTTPError):
        code = getattr(getattr(exc, "response", None), "status_code", None)
        return code in (429, 500, 502, 503, 504)
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


# ── Shared Data Class ─────────────────────────────────────────────────────────

@dataclass
class Paper:
    """
    A single academic reference, normalised from any source.
    The `citation_key` field is the APA-like in-text citation marker.
    """
    title: str
    authors: List[str]
    year: Optional[int]
    abstract: str
    url: str
    doi: Optional[str] = None
    journal: Optional[str] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    source: str = "unknown"           # "arxiv" | "semantic_scholar" | "crossref"
    citation_key: str = ""            # e.g. "Smith et al., 2022"
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Auto-derive citation_key (e.g. "Smith et al., 2022") from authors/year if not supplied."""
        if not self.citation_key:
            first_author = self.authors[0].split()[-1] if self.authors else "Unknown"
            et_al = " et al." if len(self.authors) > 1 else ""
            yr = str(self.year) if self.year else "n.d."
            self.citation_key = f"{first_author}{et_al}, {yr}"

    def to_apa(self) -> str:
        """Format this paper as an APA 7th-edition reference string."""
        authors_str = "; ".join(self.authors[:6])
        if len(self.authors) > 6:
            authors_str += " et al."
        yr = str(self.year) if self.year else "n.d."
        venue = self.journal or self.venue or "Preprint"
        doi_str = f" https://doi.org/{self.doi}" if self.doi else f" {self.url}"
        return f"{authors_str} ({yr}). {self.title}. *{venue}*.{doi_str}"


@dataclass
class WebResult:
    """A single result from a general web search."""
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"


# ── arXiv Searcher ────────────────────────────────────────────────────────────

class ArxivSearcher:
    """
    Searches arXiv via its open REST API.
    Returns Paper objects with proper metadata including arXiv IDs.

    arXiv is a preprint server — papers are not always peer-reviewed,
    but they are citable and often represent cutting-edge research.
    """

    BASE_URL = "https://export.arxiv.org/api/query"

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
    def search(self, query: str, max_results: int = 8) -> List[Paper]:
        """Search arXiv and return structured Paper objects."""
        try:
            import arxiv
        except ImportError:
            raise ImportError("pip install arxiv")

        client = arxiv.Client(num_retries=3, delay_seconds=2)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: List[Paper] = []
        for result in client.results(search):
            papers.append(
                Paper(
                    title=result.title,
                    authors=[str(a) for a in result.authors],
                    year=result.published.year if result.published else None,
                    abstract=result.summary[:800],
                    url=result.entry_id,
                    doi=result.doi,
                    journal="arXiv",
                    venue="arXiv",
                    source="arxiv",
                    tags=result.categories,
                )
            )

        logger.info("arXiv: found %d papers for query '%s'", len(papers), query[:60])
        return papers


# ── Semantic Scholar Searcher ─────────────────────────────────────────────────

class SemanticScholarSearcher:
    """
    Queries the Semantic Scholar Graph API for peer-reviewed publications.

    Unlike arXiv, Semantic Scholar indexes published journal and conference
    papers, so citation counts and peer-review status are more reliable.
    """

    FIELDS = "title,authors,year,abstract,externalIds,venue,citationCount,url,openAccessPdf"

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
    def search(self, query: str, limit: int = 8) -> List[Paper]:
        """Search Semantic Scholar and return structured Paper objects."""
        headers = {"User-Agent": "AgenticResearchAssistant/1.0"}
        if cfg.semantic_scholar_api_key:
            headers["x-api-key"] = cfg.semantic_scholar_api_key

        params = {
            "query": query,
            "limit": limit,
            "fields": self.FIELDS,
        }

        resp = requests.get(
            f"{cfg.semantic_scholar_base_url}/paper/search",
            params=params,
            headers=headers,
            timeout=15,
        )

        resp.raise_for_status()
        data = resp.json().get("data", [])

        papers: List[Paper] = []
        for item in data:
            authors = [a.get("name", "") for a in item.get("authors", [])]
            ext_ids = item.get("externalIds") or {}
            # Only store real DOIs — arXiv IDs are NOT DOIs and produce broken doi.org links
            doi = ext_ids.get("DOI")
            arxiv_id = ext_ids.get("ArXiv")
            pdf_url = (item.get("openAccessPdf") or {}).get("url", "")
            # Prefer open-access PDF, then paper URL, then arXiv URL, then S2 fallback
            url = (
                pdf_url
                or item.get("url")
                or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")
                or f"https://api.semanticscholar.org/paper/{item.get('paperId','')}"
            )

            papers.append(
                Paper(
                    title=item.get("title", "Untitled"),
                    authors=authors,
                    year=item.get("year"),
                    abstract=(item.get("abstract") or "")[:800],
                    url=url,
                    doi=doi,
                    journal=item.get("venue"),
                    venue=item.get("venue"),
                    citation_count=item.get("citationCount"),
                    source="semantic_scholar",
                )
            )

        logger.info(
            "Semantic Scholar: found %d papers for query '%s'", len(papers), query[:60]
        )
        return papers


# ── CrossRef DOI Resolver ─────────────────────────────────────────────────────

class CrossRefResolver:
    """
    Resolves DOIs and searches CrossRef for published article metadata.
    CrossRef is the authoritative source for DOI → metadata mapping.
    """

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
    def resolve_doi(self, doi: str) -> Optional[Paper]:
        """Fetch full bibliographic metadata for a known DOI."""
        url = f"{cfg.crossref_base_url}/{doi}"
        params = {"mailto": cfg.crossref_email}

        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        msg = resp.json().get("message", {})
        return self._parse_crossref(msg)

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
    def search(self, query: str, rows: int = 5) -> List[Paper]:
        """Free-text search across CrossRef's 140 M+ paper index."""
        params = {
            "query": query,
            "rows": rows,
            "mailto": cfg.crossref_email,
            "select": "title,author,published,DOI,URL,abstract,container-title,is-referenced-by-count",
        }

        resp = requests.get(cfg.crossref_base_url, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        return [p for item in items if (p := self._parse_crossref(item)) is not None]

    def _parse_crossref(self, msg: dict) -> Optional[Paper]:
        """Convert one raw CrossRef "message" object into a Paper, or None if it has no title."""
        title_list = msg.get("title", [])
        if not title_list:
            return None

        authors_raw = msg.get("author", [])
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in authors_raw
        ]

        pub_date = msg.get("published", {})
        date_parts = pub_date.get("date-parts", [[None]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None

        doi = msg.get("DOI", "")
        url = msg.get("URL") or (f"https://doi.org/{doi}" if doi else "")
        journal = (msg.get("container-title") or [None])[0]

        abstract_raw = msg.get("abstract", "")
        # CrossRef abstracts may contain JATS XML tags — strip them
        import re
        abstract = re.sub(r"<[^>]+>", "", abstract_raw)[:800]

        return Paper(
            title=title_list[0],
            authors=authors or ["Unknown"],
            year=year,
            abstract=abstract,
            url=url,
            doi=doi,
            journal=journal,
            citation_count=msg.get("is-referenced-by-count"),
            source="crossref",
        )


# ── Aggregated Academic Searcher ──────────────────────────────────────────────

# ── Google Scholar Searcher ───────────────────────────────────────────────────

class GoogleScholarSearcher:
    """
    Searches Google Scholar via the `scholarly` library (no API key required).

    scholarly scrapes Google Scholar HTML — it may be rate-limited or blocked
    by Google after many requests. Handled gracefully with a silent fallback.
    """

    def search(self, query: str, max_results: int = 6) -> List[Paper]:
        """Search Google Scholar for `query`, returning up to `max_results` papers (`[]` if blocked/unavailable)."""
        try:
            from scholarly import scholarly as _sch
        except ImportError:
            logger.debug("scholarly not installed — Google Scholar skipped")
            return []

        papers: List[Paper] = []
        try:
            search_gen = _sch.search_pubs(query)
            for i, result in enumerate(search_gen):
                if i >= max_results:
                    break
                try:
                    bib = result.get("bib", {})
                    title = bib.get("title", "")
                    if not title:
                        continue

                    authors_raw = bib.get("author", "")
                    if isinstance(authors_raw, list):
                        authors = authors_raw
                    else:
                        authors = [a.strip() for a in re.split(r" and |,", str(authors_raw)) if a.strip()]

                    year_raw = bib.get("pub_year") or bib.get("year")
                    year = int(year_raw) if year_raw else None
                    abstract = (bib.get("abstract") or "")[:800]
                    venue = bib.get("venue") or bib.get("journal") or bib.get("booktitle") or ""
                    url = result.get("pub_url") or result.get("eprint_url") or ""
                    citation_count = result.get("num_citations")

                    papers.append(Paper(
                        title=title,
                        authors=authors or ["Unknown"],
                        year=year,
                        abstract=abstract,
                        url=url,
                        doi=None,
                        journal=venue,
                        venue=venue,
                        citation_count=citation_count,
                        source="google_scholar",
                    ))
                except Exception as inner_e:
                    logger.debug("Parsing Google Scholar result failed: %s", inner_e)
        except Exception as e:
            logger.warning("Google Scholar search failed (rate-limited or blocked): %s", e)

        logger.info("Google Scholar: found %d papers for query '%s'", len(papers), query[:60])
        return papers


# ── Aggregated Academic Searcher ──────────────────────────────────────────────

class AcademicSearcher:
    """
    Orchestrates Google Scholar + arXiv + Semantic Scholar searches.

    Google Scholar is searched first (primary per configuration).
    Results are deduplicated and ranked by citation count.
    """

    def __init__(self):
        """Construct the underlying per-source searchers (Google Scholar, arXiv, Semantic Scholar, CrossRef)."""
        self.google_scholar = GoogleScholarSearcher()
        self.arxiv = ArxivSearcher()
        self.semantic = SemanticScholarSearcher()
        self.crossref = CrossRefResolver()

    def search(
        self,
        query: str,
        max_per_source: int = 6,
        include_crossref: bool = False,
    ) -> List[Paper]:
        """
        Run parallel searches across academic databases and deduplicate.

        Returns a combined, deduplicated list ranked by citation count.
        """
        papers: List[Paper] = []

        # Google Scholar (primary — searched first)
        try:
            papers.extend(self.google_scholar.search(query, max_results=max_per_source))
        except Exception as e:
            logger.warning("Google Scholar search failed: %s", e)

        # arXiv
        try:
            papers.extend(self.arxiv.search(query, max_results=max_per_source))
        except Exception as e:
            logger.warning("arXiv search failed: %s", e)

        # Semantic Scholar
        try:
            papers.extend(self.semantic.search(query, limit=max_per_source))
        except Exception as e:
            logger.warning("Semantic Scholar search failed: %s", e)

        # CrossRef (optional, slower)
        if include_crossref:
            try:
                papers.extend(self.crossref.search(query, rows=max_per_source // 2))
            except Exception as e:
                logger.warning("CrossRef search failed: %s", e)

        # Deduplicate by normalised title
        seen_titles: set[str] = set()
        unique: List[Paper] = []
        for p in papers:
            key = re.sub(r"\W+", "", p.title.lower())[:60]
            if key and key not in seen_titles:
                seen_titles.add(key)
                unique.append(p)

        # Sort: peer-reviewed first, then by citation count descending.
        # Google Scholar and Semantic Scholar results are preferred over arXiv
        # preprints. Papers with citation_count=None are ranked by year
        # (newest first) rather than treated as 0 — they may be cutting-edge.
        unique.sort(
            key=lambda p: (
                0 if p.source in ("semantic_scholar", "google_scholar") else 1,
                0 if p.citation_count is not None else 1,
                -(p.citation_count or 0),
                -(p.year or 0),
            )
        )

        logger.info("AcademicSearcher: %d unique papers for '%s'", len(unique), query[:60])
        return unique

    def resolve_doi(self, doi: str) -> Optional[Paper]:
        """Look up full metadata for a known DOI via CrossRef."""
        return self.crossref.resolve_doi(doi)


import re  # noqa: E402  (used in AcademicSearcher.search)


# ── Web Searcher ──────────────────────────────────────────────────────────────

# Domains that reliably carry peer-reviewed, preprint, or other primary
# research content. DuckDuckGo ranks for general relevance/SEO signals, which
# regularly surfaces SEO-optimized summary blogs and content farms ahead of
# the primary sources a researcher actually wants — these lists let
# WebSearcher re-rank toward research-grade domains without hiding anything
# DuckDuckGo returned.
_RESEARCH_TLDS = (".edu", ".gov", ".ac.uk", ".ac.jp", ".edu.au")
_RESEARCH_DOMAINS = (
    "arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com",
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com",
    "semanticscholar.org", "openreview.net", "aclanthology.org",
    "proceedings.neurips.cc", "openaccess.thecvf.com", "dl.acm.org", "acm.org",
    "ieee.org", "ieeexplore.ieee.org",
    "nature.com", "sciencedirect.com", "springer.com", "springeropen.com",
    "wiley.com", "onlinelibrary.wiley.com", "jstor.org", "tandfonline.com",
    "sagepub.com", "cambridge.org", "oup.com", "academic.oup.com",
    "plos.org", "frontiersin.org", "mdpi.com", "researchgate.net",
    "cell.com", "thelancet.com", "nejm.org", "bmj.com",
)

# Kept deliberately tiny — a result is only ever ranked last here, never
# dropped, so a borderline-but-relevant hit is never hidden from the user.
_LOW_VALUE_DOMAINS = ("pinterest.com", "quora.com")


def _domain_of(url: str) -> str:
    """Extract the lowercase netloc from a URL, or "" if unparseable."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _matches_any_domain(domain: str, candidates: tuple) -> bool:
    """True if *domain* equals or is a subdomain of one of *candidates* (dot-boundary safe — "fooarxiv.org" must not match "arxiv.org")."""
    return any(domain == d or domain.endswith("." + d) for d in candidates)


def _research_rank_score(url: str) -> int:
    """Lower sorts first: 0 = recognized research domain, 1 = ordinary, 2 = low-value."""
    domain = _domain_of(url)
    if not domain:
        return 1
    if domain.endswith(_RESEARCH_TLDS) or _matches_any_domain(domain, _RESEARCH_DOMAINS):
        return 0
    if _matches_any_domain(domain, _LOW_VALUE_DOMAINS):
        return 2
    return 1


class WebSearcher:
    """Search the web using DuckDuckGo (ddgs), re-ranked toward research-grade domains. No API key required."""

    def search(self, query: str, max_results: int = 5) -> List[WebResult]:
        """
        Run a DuckDuckGo text search, then deduplicate and re-rank for research use.

        Over-fetches (up to 3x max_results, capped at 20) so the research-domain
        re-rank has a real pool to work with beyond DuckDuckGo's top N, then
        applies a stable sort by _research_rank_score — ties keep DuckDuckGo's
        own relative order rather than guessing at topical relevance ourselves.
        Returns an empty list (logged, not raised) on any failure.
        """
        try:
            try:
                from duckduckgo_search import DDGS  # duckduckgo-search package (pip name)
            except ImportError:
                from ddgs import DDGS  # alternate package name in some installs
            fetch_n = min(max(max_results * 3, max_results), 20)
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=fetch_n))

            seen_urls: set = set()
            seen_titles: set = set()
            deduped = []
            for r in raw:
                url = r.get("href", "")
                title_key = re.sub(r"\W+", "", r.get("title", "").lower())[:60]
                if not url or url in seen_urls or (title_key and title_key in seen_titles):
                    continue
                seen_urls.add(url)
                if title_key:
                    seen_titles.add(title_key)
                deduped.append(r)
            deduped.sort(key=lambda r: _research_rank_score(r.get("href", "")))

            results = [
                WebResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=(r.get("body") or "")[:500],
                    source="duckduckgo",
                )
                for r in deduped[:max_results]
            ]
            logger.info("WebSearcher: found %d result(s) for query '%s'", len(results), query[:60])
            return results
        except Exception as e:
            # Some ddgs/duckduckgo_search versions raise on "no results" instead of
            # returning an empty list — that's a normal outcome, not a failure, so
            # it's logged at info rather than warning level.
            if "no results" in str(e).lower():
                logger.info("WebSearcher: 0 results for query '%s'", query[:60])
            else:
                logger.warning("WebSearcher: web search failed for query '%s': %s", query[:60], e)
            return []


# ── Module-level convenience functions ───────────────────────────────────────

def search_semantic_scholar(query: str, max_results: int = 8) -> List[Paper]:
    """Search Semantic Scholar and return a list of Paper objects."""
    return SemanticScholarSearcher().search(query, limit=max_results)


def search_arxiv(query: str, max_results: int = 8) -> List[Paper]:
    """Search arXiv and return a list of Paper objects."""
    return ArxivSearcher().search(query, max_results=max_results)
