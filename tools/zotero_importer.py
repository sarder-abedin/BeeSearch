"""tools/zotero_importer.py — BibTeX/.bib file parser and notebook importer"""
from __future__ import annotations
import logging, re
from typing import Any, Dict, List, Tuple
logger = logging.getLogger(__name__)

def parse_bibtex(bibtex_content: str) -> List[Dict[str, Any]]:
    entries = []
    pattern = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,\s*(.*?)\n\s*\}", re.DOTALL | re.MULTILINE)
    for match in pattern.finditer(bibtex_content):
        entry: Dict[str, Any] = {"type": match.group(1).lower(), "key": match.group(2).strip()}
        fields_text = match.group(3)
        for fm in re.finditer(r"(\w+)\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", fields_text):
            name = fm.group(1).lower()
            value = re.sub(r"\{([^}]*)\}", r"\1", fm.group(2).strip())
            entry[name] = re.sub(r"\\[a-zA-Z]+\s*", "", value).strip()
        for fm in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', fields_text):
            name = fm.group(1).lower()
            if name not in entry:
                entry[name] = fm.group(2).strip()
        entries.append(entry)
    return entries

def import_bibtex_to_notebook(bibtex_content: str, notebook_id: str, settings: dict) -> Tuple[int, List[str]]:
    from agents.notebook_memory import NotebookMemory
    from tools.document_tools import DocumentProcessor, ProcessedDocument
    from config.settings import get_settings as _cfg
    from tools.hybrid_store import _stores as _hybrid_stores
    _c = _cfg()
    processor = DocumentProcessor(chunk_size=settings.get("chunk_size", _c.chunk_size), overlap=settings.get("chunk_overlap", _c.chunk_overlap))
    try:
        entries = parse_bibtex(bibtex_content)
    except Exception as e:
        return 0, [f"Failed to parse BibTeX: {e}"]
    if not entries:
        return 0, ["No BibTeX entries found."]
    memory = NotebookMemory()
    added, errors = 0, []
    for entry in entries:
        try:
            key = entry.get("key", "unknown")
            title = entry.get("title", "Untitled")
            content = "\n".join(filter(None, [f"Title: {title}", f"Authors: {entry.get('author','')}", f"Year: {entry.get('year','')}", f"Journal: {entry.get('journal', entry.get('booktitle',''))}", f"DOI: {entry.get('doi','')}", f"Abstract:\n{entry.get('abstract','')}" if entry.get('abstract') else "", f"Notes:\n{entry.get('note','')}" if entry.get('note') else ""]))
            metadata = {"source": "bibtex", "citation_key": key, "title": title, "authors": entry.get("author",""), "year": entry.get("year",""), "doi": entry.get("doi","")}
            chunks = processor.chunk_text(content, metadata=metadata)
            processed = ProcessedDocument(content=content, filename=f"{key}_{title[:30].replace(' ','_')}.bib", file_type="bib", chunks=chunks, metadata=metadata)
            if memory.add_source(notebook_id, processed, source_type="bibtex"):
                _hybrid_stores.pop(f"notebook_{notebook_id}", None)
                added += 1
        except Exception as e:
            errors.append(f"Failed to import '{entry.get('key','?')}': {e}")
    return added, errors
