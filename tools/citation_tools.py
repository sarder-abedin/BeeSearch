"""
tools/citation_tools.py
────────────────────────
BibTeX and RIS citation export helpers.

Pure stdlib — no new dependencies required.

Usage
─────
from tools.citation_tools import refs_to_bibtex, refs_to_ris

# refs is a list of dicts like the ones built by reference_compilation_node:
# {title, authors, year, journal, doi, url, citation_key, source, ...}

bib_text = refs_to_bibtex(references)   # → .bib file content
ris_text = refs_to_ris(references)       # → .ris file content

TUTORIAL NOTE
─────────────
BibTeX and RIS are the two universal reference formats understood by every
reference manager (Zotero, Mendeley, EndNote, JabRef, etc.). Providing these
lets researchers integrate AI-generated references into their existing workflow
without manual re-entry.

BibTeX key format: <first-author-last-name><year>[a/b/c…]
  • smith2023       — single paper by Smith in 2023
  • smith2023a/b    — two papers by Smith in 2023 (collision handling)
"""

from __future__ import annotations

import re
from typing import Dict, List


# ── Key generation ─────────────────────────────────────────────────────────────

def _make_bibtex_key(ref: Dict, existing_keys: set | None = None) -> str:
    """
    Generate a BibTeX key from the reference dict.

    Pattern: <lastname><year>[suffix]
    The suffix (a, b, c…) is added only when a collision is detected.
    `existing_keys` is updated in-place when provided.
    """
    authors = ref.get("authors") or []
    year    = str(ref.get("year") or "nd")

    if authors:
        first_author = authors[0]
        # Take the first token (last name in "Last First" or "Last, F." format)
        last_name = re.sub(r"[^a-zA-Z]", "", first_author.split()[0]).lower()
        if not last_name:
            last_name = "anon"
    else:
        last_name = "anon"

    base_key = f"{last_name}{year}"

    if existing_keys is None:
        return base_key

    # Collision handling: append a, b, c…
    if base_key not in existing_keys:
        existing_keys.add(base_key)
        return base_key

    for suffix in "abcdefghijklmnopqrstuvwxyz":
        candidate = f"{base_key}{suffix}"
        if candidate not in existing_keys:
            existing_keys.add(candidate)
            return candidate

    # Extreme edge case: all 26 suffixes used — fall back to title hash
    fallback = f"{base_key}_{abs(hash(ref.get('title',''))):04x}"
    existing_keys.add(fallback)
    return fallback


# ── BibTeX ─────────────────────────────────────────────────────────────────────

def _escape_bibtex(text: str) -> str:
    """Escape characters that break BibTeX parsers."""
    return (
        str(text)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def ref_to_bibtex(ref: Dict, key: str | None = None) -> str:
    """
    Convert a single reference dict to a BibTeX @article{} entry.

    Uses @article for peer-reviewed/CrossRef sources and @misc for
    arXiv preprints (since they may not have a journal name).
    """
    if key is None:
        key = _make_bibtex_key(ref)

    # Entry type
    source = ref.get("source", "")
    if source == "arxiv":
        entry_type = "misc"
    else:
        entry_type = "article"

    # Authors: BibTeX uses "Last, First and Last, First" format
    authors = ref.get("authors") or []
    bibtex_authors = " and ".join(_escape_bibtex(a) for a in authors)

    fields: list[str] = []

    if bibtex_authors:
        fields.append(f"  author    = {{{bibtex_authors}}}")

    title = _escape_bibtex(ref.get("title") or "Untitled")
    fields.append(f"  title     = {{{title}}}")

    year = ref.get("year")
    if year:
        fields.append(f"  year      = {{{year}}}")

    journal = ref.get("journal")
    if journal and entry_type == "article":
        fields.append(f"  journal   = {{{_escape_bibtex(journal)}}}")

    doi = ref.get("doi")
    if doi:
        fields.append(f"  doi       = {{{doi}}}")

    url = ref.get("url")
    if url:
        fields.append(f"  url       = {{{url}}}")

    if source == "arxiv":
        fields.append(f"  note      = {{arXiv preprint}}")
    elif source == "semantic_scholar":
        fields.append(f"  note      = {{Semantic Scholar}}")

    body = ",\n".join(fields)
    return f"@{entry_type}{{{key},\n{body}\n}}"


def refs_to_bibtex(references: List[Dict]) -> str:
    """
    Convert a list of reference dicts to a complete .bib file string.

    Deduplicates BibTeX keys using the a/b/c suffix scheme.
    """
    if not references:
        return "% No references\n"

    used_keys: set = set()
    entries: list[str] = []

    for ref in references:
        key = _make_bibtex_key(ref, existing_keys=used_keys)
        entries.append(ref_to_bibtex(ref, key=key))

    header = (
        "% BibTeX references exported by Agentic Research Assistant\n"
        f"% {len(entries)} reference(s)\n\n"
    )
    return header + "\n\n".join(entries) + "\n"


# ── RIS ────────────────────────────────────────────────────────────────────────

def ref_to_ris(ref: Dict) -> str:
    """
    Convert a single reference dict to a RIS format block.

    RIS is a tagged text format:
      TY  - <type>      (JOUR = journal, RPRT = report/preprint)
      AU  - <author>    (one line per author)
      TI  - <title>
      PY  - <year>
      JO  - <journal>
      DO  - <DOI>
      UR  - <URL>
      ER  -              (end of record — mandatory)
    """
    source = ref.get("source", "")
    ris_type = "RPRT" if source == "arxiv" else "JOUR"

    lines: list[str] = [f"TY  - {ris_type}"]

    for author in (ref.get("authors") or []):
        lines.append(f"AU  - {author}")

    title = ref.get("title") or "Untitled"
    lines.append(f"TI  - {title}")

    year = ref.get("year")
    if year:
        lines.append(f"PY  - {year}")

    journal = ref.get("journal")
    if journal:
        lines.append(f"JO  - {journal}")

    doi = ref.get("doi")
    if doi:
        lines.append(f"DO  - {doi}")

    url = ref.get("url")
    if url:
        lines.append(f"UR  - {url}")

    abstract = ref.get("abstract_snippet")
    if abstract:
        lines.append(f"AB  - {abstract}")

    lines.append("ER  - ")
    return "\n".join(lines)


def refs_to_ris(references: List[Dict]) -> str:
    """
    Convert a list of reference dicts to a complete .ris file string.
    """
    if not references:
        return ""

    blocks = [ref_to_ris(r) for r in references]
    return "\n\n".join(blocks) + "\n"
