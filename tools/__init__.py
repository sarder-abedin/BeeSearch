"""
tools/__init__.py
─────────────────
Lazy re-exports for the tools package.

Uses module-level __getattr__ so that importing any submodule
(e.g. `from tools.citation_tools import refs_to_bibtex`) does NOT
trigger the full import chain (tenacity, faiss, chromadb, etc.).
Heavy deps are only loaded when the specific name is first accessed.
"""

from __future__ import annotations

import importlib
from typing import Any

# Map of public name → (module path, attribute name)
_EXPORTS: dict[str, tuple[str, str]] = {
    "DocumentProcessor":       ("tools.document_tools",  "DocumentProcessor"),
    "AcademicSearcher":        ("tools.search_tools",    "AcademicSearcher"),
    "WebSearcher":             ("tools.search_tools",    "WebSearcher"),
    "refs_to_bibtex":          ("tools.citation_tools",  "refs_to_bibtex"),
    "refs_to_ris":             ("tools.citation_tools",  "refs_to_ris"),
    "OllamaEmbedder":          ("tools.embeddings",      "OllamaEmbedder"),
    "HybridStore":             ("tools.hybrid_store",    "HybridStore"),
    "get_or_create_store":     ("tools.hybrid_store",    "get_or_create_store"),
    "generate_clarifying_questions": ("tools.clarifier", "generate_clarifying_questions"),
    "DOIVerifier":             ("tools.doi_verifier",    "DOIVerifier"),
    "get_verifier":            ("tools.doi_verifier",    "get_verifier"),
    "fetch_call_text":         ("tools.call_analyzer",   "fetch_call_text"),
    "extract_call_requirements": ("tools.call_analyzer", "extract_call_requirements"),
    "format_call_context_block": ("tools.call_analyzer", "format_call_context_block"),
    "safe_shutdown":           ("tools.shutdown",        "safe_shutdown"),
    "free_port":               ("tools.shutdown",        "free_port"),
    "flush_chromadb":          ("tools.shutdown",        "flush_chromadb"),
    "is_port_in_use":          ("tools.shutdown",        "is_port_in_use"),
    "install_signal_handlers": ("tools.shutdown",        "install_signal_handlers"),
    "load_url_as_document":    ("tools.web_loader",      "load_url_as_document"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_path, attr = _EXPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        # Cache in module dict so subsequent accesses are O(1)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
