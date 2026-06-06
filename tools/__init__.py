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
    # ── New analysis tools ────────────────────────────────────────────
    "generate_prisma_mermaid":  ("tools.prisma_diagram",        "generate_prisma_mermaid"),
    "generate_prisma_dot":      ("tools.prisma_diagram",        "generate_prisma_dot"),
    "run_sensitivity_analysis": ("tools.sensitivity_analysis",  "run_sensitivity_analysis"),
    "build_sensitivity_scenarios": ("tools.sensitivity_analysis", "build_sensitivity_scenarios"),
    "save_monitor_state":       ("tools.literature_monitor",    "save_monitor_state"),
    "load_monitor_state":       ("tools.literature_monitor",    "load_monitor_state"),
    "find_new_papers":          ("tools.literature_monitor",    "find_new_papers"),
    "list_monitors":            ("tools.literature_monitor",    "list_monitors"),
    "delete_monitor":           ("tools.literature_monitor",    "delete_monitor"),
    "monitor_id_from_question": ("tools.literature_monitor",    "monitor_id_from_question"),
    "generate_preregistration": ("tools.preregistration",       "generate_preregistration"),
    "generate_prisma_checklist": ("tools.preregistration",      "generate_prisma_checklist"),
    "extract_structured_row":   ("tools.extraction_table",      "extract_structured_row"),
    "build_extraction_table":   ("tools.extraction_table",      "build_extraction_table"),
    "extraction_table_to_csv":  ("tools.extraction_table",      "extraction_table_to_csv"),
    "extraction_table_to_markdown": ("tools.extraction_table",  "extraction_table_to_markdown"),
    "map_research_gaps":        ("tools.research_gaps",         "map_research_gaps"),
    "generate_hypotheses":      ("tools.hypothesis_generator",  "generate_hypotheses"),
    "parse_bibtex":             ("tools.zotero_importer",       "parse_bibtex"),
    "import_bibtex_to_notebook": ("tools.zotero_importer",      "import_bibtex_to_notebook"),
    "sr_to_notebook":           ("tools.bridge",                "sr_to_notebook"),
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
