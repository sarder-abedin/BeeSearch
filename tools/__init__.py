"""
tools/__init__.py
──────────────────
BeeSearch lazy re-exports for the `tools` package.

`tools/` holds the standalone utility functions used by the `agents/`
pipelines and the UI (search, document processing, citation export,
PRISMA reporting, etc.). Rather than importing every submodule eagerly
(which would pull in heavy deps like faiss/chromadb/langchain_ollama as
soon as `import tools` runs anywhere), `_EXPORTS` maps each public name to
its `(module_path, attr)` location and `__getattr__` resolves and caches it
on first access. New public tool functions should be added to `_EXPORTS`
rather than imported at module scope here.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # ── Document processing ──────────────────────────────────────────────────
    "DocumentProcessor":       ("tools.document_tools",  "DocumentProcessor"),
    # ── Search ───────────────────────────────────────────────────────────────
    "AcademicSearcher":        ("tools.search_tools",    "AcademicSearcher"),
    "GoogleScholarSearcher":   ("tools.search_tools",    "GoogleScholarSearcher"),
    "WebSearcher":             ("tools.search_tools",    "WebSearcher"),
    # ── SR: Literature Discovery ─────────────────────────────────────────────
    "screen_abstracts":        ("tools.abstract_screener",  "screen_abstracts"),
    "screener_summary":        ("tools.abstract_screener",  "screener_summary"),
    "build_citation_network":  ("tools.citation_network",   "build_citation_network"),
    "network_to_pyvis_html":   ("tools.citation_network",   "network_to_pyvis_html"),
    "network_stats":           ("tools.citation_network",   "network_stats"),
    "find_gap_candidates":     ("tools.citation_network",   "find_gap_candidates"),
    "get_paper_abstract":      ("tools.citation_network",   "get_paper_abstract"),
    "classify_citation_stances": ("tools.citation_network", "classify_citation_stances"),
    "extract_citation_context": ("tools.citation_context",  "extract_citation_context"),
    "find_citation_mentions":  ("tools.citation_context",   "find_citation_mentions"),
    "track_preprints":         ("tools.preprint_tracker",   "track_preprints"),
    "preprint_summary":        ("tools.preprint_tracker",   "preprint_summary"),
    # ── SR: PRISMA Report ────────────────────────────────────────────────────
    "generate_prisma_docx":    ("tools.prisma_report",      "generate_prisma_docx"),
    "generate_prisma_pdf":     ("tools.prisma_report",      "generate_prisma_pdf"),
    "build_sr_markdown":       ("tools.prisma_report",      "build_sr_markdown"),
    # ── SR: Plain Language Summaries ─────────────────────────────────────────
    "generate_patient_summary":  ("tools.plain_language",   "generate_patient_summary"),
    "generate_policy_brief":     ("tools.plain_language",   "generate_policy_brief"),
    "generate_press_release":    ("tools.plain_language",   "generate_press_release"),
    "generate_all_summaries":    ("tools.plain_language",   "generate_all_summaries"),
    # ── SR: Discovery & Trends ───────────────────────────────────────────────
    "analyze_trends":          ("tools.trend_analyzer",     "analyze_trends"),
    "trend_to_chart_data":     ("tools.trend_analyzer",     "trend_to_chart_data"),
    "build_evidence_map_data": ("tools.evidence_map",       "build_evidence_map_data"),
    "evidence_map_to_plotly_html": ("tools.evidence_map",   "evidence_map_to_plotly_html"),
    "evidence_map_to_png":     ("tools.evidence_map",       "evidence_map_to_png"),
    "detect_concept_drift":    ("tools.concept_drift",      "detect_concept_drift"),
    # ── SR: Meta-Analysis ────────────────────────────────────────────────────
    "run_meta_analysis":       ("tools.meta_analysis",      "run_meta_analysis"),
    "extract_effect_size_row": ("tools.meta_analysis",      "extract_effect_size_row"),
    "meta_analysis_to_forest_plotly": ("tools.meta_analysis", "meta_analysis_to_forest_plotly"),
    "meta_analysis_to_forest_png":    ("tools.meta_analysis", "meta_analysis_to_forest_png"),
    "MEASURE_LABELS":          ("tools.meta_analysis",      "MEASURE_LABELS"),
    "seed_meta_rows":          ("tools.meta_analysis",      "seed_meta_rows"),
    # ── SR: Sensitivity Analysis ─────────────────────────────────────────────
    "run_sensitivity_analysis":   ("tools.sensitivity_analysis", "run_sensitivity_analysis"),
    "build_sensitivity_scenarios": ("tools.sensitivity_analysis", "build_sensitivity_scenarios"),
    # ── SR: Literature Monitor ───────────────────────────────────────────────
    "save_monitor_state":      ("tools.literature_monitor", "save_monitor_state"),
    "load_monitor_state":      ("tools.literature_monitor", "load_monitor_state"),
    "list_monitors":           ("tools.literature_monitor", "list_monitors"),
    "delete_monitor":          ("tools.literature_monitor", "delete_monitor"),
    "find_new_papers":         ("tools.literature_monitor", "find_new_papers"),
    "monitor_id_from_question": ("tools.literature_monitor", "monitor_id_from_question"),
    # ── Shared ───────────────────────────────────────────────────────────────
    "refs_to_bibtex":          ("tools.citation_tools",  "refs_to_bibtex"),
    "refs_to_ris":             ("tools.citation_tools",  "refs_to_ris"),
    "OllamaEmbedder":          ("tools.embeddings",      "OllamaEmbedder"),
    "HybridStore":             ("tools.hybrid_store",    "HybridStore"),
    "get_or_create_store":     ("tools.hybrid_store",    "get_or_create_store"),
    "generate_clarifying_questions": ("tools.clarifier", "generate_clarifying_questions"),
    "safe_shutdown":           ("tools.shutdown",        "safe_shutdown"),
    "free_port":               ("tools.shutdown",        "free_port"),
    "flush_chromadb":          ("tools.shutdown",        "flush_chromadb"),
    "load_url_as_document":    ("tools.web_loader",      "load_url_as_document"),
    # ── Writing style ────────────────────────────────────────────────────────
    "ANTI_AI_TELL_INSTRUCTION":          ("tools.writing_style", "ANTI_AI_TELL_INSTRUCTION"),
    "ANTI_AI_TELL_NARRATIVE_INSTRUCTION": ("tools.writing_style", "ANTI_AI_TELL_NARRATIVE_INSTRUCTION"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve `tools.<name>` lazily via the `_EXPORTS` table (PEP 562).

    On first access, imports the owning submodule, fetches the attribute,
    and caches it in this module's globals so subsequent lookups skip
    `__getattr__` entirely (plain attribute access from then on).

    Raises:
        AttributeError: if `name` is not a key in `_EXPORTS`.
    """
    if name in _EXPORTS:
        module_path, attr = _EXPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
