"""
tools/grammar_check.py
──────────────────────
Lightweight LLM-backed grammar/spelling/punctuation correction for short
user-entered queries (research questions, search goals, chat messages…).
"""

from __future__ import annotations

import logging
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

_SYSTEM_PROMPT = (
    "You are a careful copy-editor. Fix ONLY grammar, spelling, and punctuation "
    "mistakes in the user's text. Preserve their meaning, tone, technical terms, "
    "field-specific jargon, formatting, and line breaks exactly — do not rephrase, "
    "expand, shorten, or otherwise improve the writing style beyond correcting "
    "outright mistakes. If the text already has no errors, return it completely "
    "unchanged. Reply with ONLY the corrected text — no preamble, no explanation, "
    "no surrounding quotes."
)


def _make_llm(model_name: str, num_ctx: int) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=0.1,
        num_predict=1024,
        num_ctx=min(num_ctx or cfg.num_ctx, 8192),
        sync_client_kwargs={"timeout": httpx.Timeout(90.0)},
    )


def check_and_fix_grammar(text: str, *, model_name: str = "", num_ctx: int = 8192) -> Dict[str, object]:
    """Run a quick LLM grammar/spelling/punctuation pass over `text`.

    Returns ``{"original": str, "corrected": str, "changed": bool}``.
    On any LLM error, ``corrected`` falls back to ``original`` and ``changed`` is False.
    """
    original = text.strip()
    if not original:
        return {"original": original, "corrected": original, "changed": False}

    try:
        llm = _make_llm(model_name, num_ctx)
        corrected = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=original),
        ]).content.strip().strip('"')
    except Exception as e:
        logger.warning(f"Grammar check failed, keeping original text: {e}")
        corrected = original

    return {
        "original": original,
        "corrected": corrected,
        "changed": corrected != original,
    }
