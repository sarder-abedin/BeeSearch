"""backend/app/services/research_assistant_service.py
─────────────────────────────────────────────────────────
Service layer for Mode 3 (AI Research Assistant): builds the settings dict
the same way the CLI (``main.py::_cmd_ask``) and Streamlit sidebar
(``ui/sidebar.py::render_sidebar``) already do, then calls
``agents.research_assistant.run_research_assistant`` unmodified.

Optional ``AskRequest`` overrides (model/num_ctx/temperature_level) are only
added to the settings dict when provided, so omitting them defers to
``run_research_assistant``'s own fallback chain (``cfg.ollama_model``,
``cfg.num_ctx``, ``DEFAULT_TEMPERATURE_LEVEL``) instead of re-implementing
those defaults here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from agents.research_assistant import run_research_assistant

from ..schemas.research_assistant import AskRequest


def build_settings(req: AskRequest) -> Dict[str, Any]:
    settings: Dict[str, Any] = {"include_crossref": req.include_crossref}
    if req.model:
        settings["model"] = req.model
    if req.num_ctx is not None:
        settings["num_ctx"] = req.num_ctx
    if req.temperature_level:
        settings["temperature_level"] = req.temperature_level
    return settings


def run_ask(
    req: AskRequest, stream_callback: Callable[[str, Dict[str, Any]], None]
) -> Dict[str, Any]:
    """Run the Mode 3 pipeline for one request; returns the raw result dict."""
    settings = build_settings(req)
    return run_research_assistant(
        req.question.strip(),
        settings,
        stream_callback=stream_callback,
        include_web=req.include_web,
    )
