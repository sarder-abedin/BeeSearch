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
    "GoogleScholarSearcher":   ("tools.search_tools",    "GoogleScholarSearcher"),
    "WebSearcher":             ("tools.search_tools",    "WebSearcher"),
    # ── Literature Discovery ─────────────────────────────────────────────────
    "screen_abstracts":        ("tools.abstract_screener",  "screen_abstracts"),
    "screener_summary":        ("tools.abstract_screener",  "screener_summary"),
    "build_citation_network":  ("tools.citation_network",   "build_citation_network"),
    "network_to_pyvis_html":   ("tools.citation_network",   "network_to_pyvis_html"),
    "network_stats":           ("tools.citation_network",   "network_stats"),
    "track_preprints":         ("tools.preprint_tracker",   "track_preprints"),
    "preprint_summary":        ("tools.preprint_tracker",   "preprint_summary"),
    # ── PRISMA Report ────────────────────────────────────────────────────────
    "generate_prisma_docx":    ("tools.prisma_report",      "generate_prisma_docx"),
    "generate_prisma_pdf":     ("tools.prisma_report",      "generate_prisma_pdf"),
    # ── Plain Language Summaries ─────────────────────────────────────────────
    "generate_patient_summary":  ("tools.plain_language",   "generate_patient_summary"),
    "generate_policy_brief":     ("tools.plain_language",   "generate_policy_brief"),
    "generate_press_release":    ("tools.plain_language",   "generate_press_release"),
    "generate_all_summaries":    ("tools.plain_language",   "generate_all_summaries"),
    # ── Discovery & Trends ───────────────────────────────────────────────────
    "analyze_trends":          ("tools.trend_analyzer",     "analyze_trends"),
    "trend_to_chart_data":     ("tools.trend_analyzer",     "trend_to_chart_data"),
    "build_evidence_map_data": ("tools.evidence_map",       "build_evidence_map_data"),
    "evidence_map_to_plotly_html": ("tools.evidence_map",   "evidence_map_to_plotly_html"),
    "evidence_map_to_png":     ("tools.evidence_map",       "evidence_map_to_png"),
    "detect_concept_drift":    ("tools.concept_drift",      "detect_concept_drift"),
    "refs_to_bibtex":          ("tools.citation_tools",  "refs_to_bibtex"),
    "refs_to_ris":             ("tools.citation_tools",  "refs_to_ris"),
    "OllamaEmbedder":          ("tools.embeddings",      "OllamaEmbedder"),
    "HybridStore":             ("tools.hybrid_store",    "HybridStore"),
    "get_or_create_store":     ("tools.hybrid_store",    "get_or_create_store"),
    "analyse_writing_style":   ("tools.style_profiler",  "analyse_writing_style"),
    "generate_clarifying_questions": ("tools.clarifier", "generate_clarifying_questions"),
    "recommend_mode":          ("tools.cli_recommender", "recommend_mode"),
    "recommend_post_research": ("tools.cli_recommender", "recommend_post_research"),
    "recommend_proposal_pre":  ("tools.cli_recommender", "recommend_proposal_pre"),
    "recommend_proposal_post": ("tools.cli_recommender", "recommend_proposal_post"),
    "recommend_story_turn":    ("tools.cli_recommender", "recommend_story_turn"),
    "recommend_wisdom_turn":   ("tools.cli_recommender", "recommend_wisdom_turn"),
    "recommend_startup":       ("tools.cli_recommender", "recommend_startup"),
    "print_recommendations":   ("tools.cli_recommender", "print_recommendations"),
    "DOIVerifier":             ("tools.doi_verifier",    "DOIVerifier"),
    "get_verifier":            ("tools.doi_verifier",    "get_verifier"),
    "get_template":            ("tools.funding_templates", "get_template"),
    "get_word_count_targets":  ("tools.funding_templates", "get_word_count_targets"),
    "AGENCY_NAMES":            ("tools.funding_templates", "AGENCY_NAMES"),
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
