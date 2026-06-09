"""
tools/grammar_check.py
──────────────────────
Lightweight LLM-backed spelling/punctuation correction for short
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
    "You are a minimal proofreader. Your ONLY task is to fix clear misspellings "
    "and broken punctuation. You must follow these rules without exception:\n"
    "1. Do NOT rephrase, reorder, or restructure any sentence.\n"
    "2. Do NOT add or remove words — only correct words that are clearly misspelled.\n"
    "3. Never change technical terms, proper nouns, acronyms, or field-specific "
    "jargon — even if they look unusual.\n"
    "4. Preserve the original sentence structure, phrasing, and meaning exactly.\n"
    "5. If the text contains no clear spelling or punctuation errors, return it "
    "VERBATIM — not a single character changed.\n"
    "6. Reply with ONLY the corrected text. No preamble, no explanation, no quotes."
)


def _make_llm(model_name: str, num_ctx: int) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=0.0,
        num_predict=512,
        num_ctx=min(num_ctx or cfg.num_ctx, 4096),
        sync_client_kwargs={"timeout": httpx.Timeout(60.0)},
    )


def _is_conservative_fix(original: str, corrected: str, max_change_ratio: float = 0.25) -> bool:
    """Return True if the correction looks like a genuine minor fix, not a rewrite.

    Rejects the suggestion when:
    - word count changed by more than 20 %
    - more than 25 % of word positions differ
    """
    orig_words = original.lower().split()
    corr_words = corrected.lower().split()
    if not orig_words:
        return True
    # Reject if length changed substantially
    if abs(len(orig_words) - len(corr_words)) / len(orig_words) > 0.20:
        return False
    # Count positional word differences
    min_len = min(len(orig_words), len(corr_words))
    changes = sum(1 for a, b in zip(orig_words[:min_len], corr_words[:min_len]) if a != b)
    return (changes / len(orig_words)) <= max_change_ratio


def check_and_fix_grammar(
    text: str,
    *,
    model_name: str = "",
    num_ctx: int = 8192,
    context_hint: str = "",
) -> Dict[str, object]:
    """Run a conservative LLM spelling/punctuation pass over ``text``.

    Parameters
    ----------
    text:         The user-entered text to check.
    model_name:   Ollama model to use (defaults to configured model).
    num_ctx:      Context window size (capped at 4096 for this fast task).
    context_hint: Short description of what the text is (e.g. "systematic
                  review research question") injected so the model avoids
                  mis-correcting domain vocabulary.

    Returns ``{"original": str, "corrected": str, "changed": bool}``.
    On any error or detected rewrite, returns the original unchanged.
    """
    original = text.strip()
    if not original:
        return {"original": original, "corrected": original, "changed": False}

    user_content = original
    if context_hint:
        user_content = f"[This is a {context_hint}]\n\n{original}"

    try:
        llm = _make_llm(model_name, num_ctx)
        corrected = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]).content.strip().strip('"').strip("'")
    except Exception as e:
        logger.warning(f"Grammar check failed, keeping original: {e}")
        return {"original": original, "corrected": original, "changed": False}

    # Strip any context prefix the model may have echoed back
    if context_hint and corrected.startswith(f"[This is a {context_hint}]"):
        corrected = corrected[len(f"[This is a {context_hint}]"):].strip()

    # Reject suggestions that look like rewrites rather than minor fixes
    if not _is_conservative_fix(original, corrected):
        logger.warning(
            "Grammar check returned a likely rewrite (too many word changes) — "
            "keeping original text."
        )
        return {"original": original, "corrected": original, "changed": False}

    return {
        "original": original,
        "corrected": corrected,
        "changed": corrected != original,
    }
