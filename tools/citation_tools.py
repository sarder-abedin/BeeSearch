"""
tools/citation_tools.py
────────────────────────
BibTeX and RIS citation export helpers.

Pure stdlib — no new dependencies required.

Usage
─────
from tools.citation_tools import refs_to_bibtex, refs_to_ris

bib_text = refs_to_bibtex(references)   # → .bib file content
ris_text = refs_to_ris(references)       # → .ris file content
"""

from __future__ import annotations

import re
from typing import Dict, List


def _make_bibtex_key(ref: Dict, existing_keys: set | None = None) -> str:
    authors = ref.get("authors") or []
    year    = str(ref.get("year") or "nd")

    if authors:
        first_author = authors[0]
        last_name = re.sub(r"[^a-zA-Z]", "", first_author.split()[0]).lower()
        if not last_name:
            last_name = "anon"
    else:
        last_name = "anon"

    base_key = f"{last_name}{year}"

    if existing_keys is None:
        return base_key

    if base_key not in existing_keys:
        existing_keys.add(base_key)
        return base_key

    for suffix in "abcdefghijklmnopqrstuvwxyz":
        candidate = f"{base_key}{suffix}"
        if candidate not in existing_keys:
            existing_keys.add(candidate)
            return candidate

    fallback = f"{base_key}_{abs(hash(ref.get('title',''))):04x}"
    existing_keys.add(fallback)
    return fallback


def _escape_bibtex(text: str) -> str:
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
    if key is None:
        key = _make_bibtex_key(ref)

    source = ref.get("source", "")
    entry_type = "misc" if source == "arxiv" else "article"

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
    if not references:
        return "% No references\n"

    used_keys: set = set()
    entries: list[str] = []

    for ref in references:
        key = _make_bibtex_key(ref, existing_keys=used_keys)
        entries.append(ref_to_bibtex(ref, key=key))

    header = (
        "% BibTeX references exported by ResearchBuddy\n"
        f"% {len(entries)} reference(s)\n\n"
    )
    return header + "\n\n".join(entries) + "\n"


def ref_to_ris(ref: Dict) -> str:
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
    if not references:
        return ""
    blocks = [ref_to_ris(r) for r in references]
    return "\n\n".join(blocks) + "\n"
