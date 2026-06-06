"""
agents/grammar_nodes.py
────────────────────────
Five nodes for the Grammar Proofreading Mode (Mode 6).

Pipeline
────────
  text_loader → grammar_analysis → polish → style_advisor → (grammar_eval in eval_nodes.py)

Design notes
────────────
• temperature=0.1 for grammar_analysis (precision), 0.2 for polish (conservative
  rewrite — fluent but not creative), 0.3 for style_advisor (richer suggestions)
• The polish node uses context-specific system prompts from _STYLE_PROMPTS so the
  rewrite is shaped by the chosen writing context, not just labelled with it
• Revision path: non-zero refinement_round injects the previous polished text + user
  feedback into the polish node's human prompt
• Fallback: every node catches all exceptions and degrades gracefully (empty outputs,
  warnings in errors[]) — the graph never raises
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.grammar_state import GrammarState
from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

# ── Context-specific system prompts for the polish node ───────────────────────

_STYLE_PROMPTS: Dict[str, str] = {
    "academic": (
        "You are a professional academic editor specialising in journal papers, theses, "
        "and research reports. Rewrite the provided text so it is clear, precise, and "
        "suitable for peer review. Apply these rules strictly:\n"
        "- Use formal third-person voice (avoid 'I', 'we', 'you' where possible)\n"
        "- Passive voice is acceptable and often preferred in methods sections\n"
        "- Use precise technical language; preserve domain-specific terminology unchanged\n"
        "- Eliminate contractions (don't → do not, it's → it is)\n"
        "- Use appropriate academic hedging (may, suggests, indicates, appears to)\n"
        "- Ensure subject-verb agreement and consistent tense throughout\n"
        "- Keep citations, figure references, and technical notation exactly as given\n"
        "- Improve sentence clarity and logical flow between paragraphs\n"
        "- Preserve the full length and all content of the original text. Do not omit or condense any section — rewrite every part."
    ),
    "professional_email": (
        "You are a professional business communication editor. Rewrite the provided text "
        "so it is clear, concise, and effective as a professional email, memo, or cover "
        "letter. Apply these rules:\n"
        "- Open with a clear statement of purpose; close with a specific call to action\n"
        "- Use professional but warm tone — respectful without being stiff\n"
        "- Apply correct salutations (Dear Mr/Ms/Dr [Name]) and sign-offs (Kind regards, "
        "Yours sincerely) if present\n"
        "- Eliminate jargon unless the audience clearly shares it\n"
        "- Prefer active voice and short sentences\n"
        "- Remove redundancy — every sentence must earn its place\n"
        "- Fix grammar, spelling, and punctuation errors\n"
        "- Preserve the sender's intent and factual content exactly\n"
        "- Preserve the full length and all content of the original text. Do not omit or condense any section — rewrite every part."
    ),
    "formal": (
        "You are a professional editor for formal documents including legal correspondence, "
        "official letters, policy documents, and reports. Rewrite the provided text to "
        "meet the highest standards of formal English. Apply these rules:\n"
        "- Use strict formal register throughout; no contractions, colloquialisms, or "
        "informal phrasing\n"
        "- Prefer elevated, precise vocabulary (utilise, endeavour, subsequent to)\n"
        "- Maintain authoritative, neutral tone — no emotional language\n"
        "- Ensure well-structured paragraphs with clear topic sentences\n"
        "- Fix all grammar, spelling, and punctuation errors\n"
        "- Preserve all proper nouns, section references, and legal/technical terms exactly\n"
        "- Preserve the full length and all content of the original text. Do not omit or condense any section — rewrite every part."
    ),
    "informal": (
        "You are a friendly writing coach helping with personal writing, blog posts, and "
        "social content. Revise the provided text to be clear and natural while preserving "
        "the author's voice and personality. Apply these rules:\n"
        "- Correct clear grammatical errors and spelling mistakes\n"
        "- Fix punctuation issues that impede readability\n"
        "- Contractions (don't, it's, we're) and conversational phrasing are fine\n"
        "- Colloquialisms and casual expressions are acceptable — preserve the author's "
        "personality and tone\n"
        "- Focus on clarity and natural flow, not formal correctness\n"
        "- Do NOT over-formalise — keep the friendly, approachable voice\n"
        "- Preserve the author's unique expressions unless they are genuinely confusing\n"
        "- Preserve the full length and all content of the original text. Do not omit or condense any section — rewrite every part."
    ),
}

_FALLBACK_STYLE = "formal"


# ── LLM factory ──────────────────────────────────────────────────────────────

def _llm(state: GrammarState, temperature: float = 0.2, num_predict: int = 4096) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=state.get("model_name") or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=state.get("num_ctx") or cfg.num_ctx,
        sync_client_kwargs={"timeout": httpx.Timeout(180.0)},
    )


def _call(llm: ChatOllama, system: str, human: str) -> str:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return response.content.strip()


# ── Node 1: text_loader ───────────────────────────────────────────────────────

def text_loader_node(state: GrammarState) -> Dict[str, Any]:
    """Load and characterise the input text; emit a warning for very long texts."""
    logger.info("[Grammar] Text Loader")

    raw = (state.get("raw_text") or "").strip()
    words = raw.split()
    word_count = len(words)
    sentence_count = raw.count(".") + raw.count("!") + raw.count("?")

    errors: List[str] = list(state.get("errors") or [])

    # Informational warning if text may strain the context window
    num_ctx = state.get("num_ctx") or cfg.num_ctx
    estimated_chars = num_ctx * 4 * 0.6  # rough chars-per-token × 60% headroom
    if len(raw) > estimated_chars:
        msg = (
            f"Text is very long ({word_count:,} words). "
            "Processing capacity depends on the model's context window. "
            "Results may be truncated for extremely long inputs."
        )
        errors.append(msg)
        logger.warning("[Grammar] %s", msg)

    return {
        "raw_text": raw,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "errors": errors,
        "current_step": "text_loader",
        "completed_steps": list(state.get("completed_steps") or []) + ["text_loader"],
        "progress_pct": 10,
        "status_detail": f"{word_count:,} words · {sentence_count} sentences",
    }


# ── Node 2: grammar_analysis ──────────────────────────────────────────────────

def grammar_analysis_node(state: GrammarState) -> Dict[str, Any]:
    """Identify grammar, spelling, and punctuation issues as a JSON array."""
    logger.info("[Grammar] Grammar Analysis")

    raw = state.get("raw_text") or ""
    if not raw:
        return {
            "issues_found": [],
            "current_step": "grammar_analysis",
            "completed_steps": list(state.get("completed_steps") or []) + ["grammar_analysis"],
            "progress_pct": 35,
            "status_detail": "No text to analyse",
        }

    focus_areas: List[str] = state.get("focus_areas") or []
    focus_str = ", ".join(focus_areas) if focus_areas else "grammar, spelling, punctuation, style, clarity"
    style_level = state.get("style_level") or "professional_email"

    system = (
        "You are an expert English proofreader. Analyse the provided text and return ONLY "
        "a JSON array of issues found. Each element must have these exact keys: "
        '"type" (grammar|spelling|punctuation|style|clarity), '
        '"original" (the problematic phrase, max 15 words), '
        '"suggestion" (the corrected version), '
        '"explanation" (one sentence explaining the issue), '
        '"severity" (error|warning|suggestion). '
        "Return [] if no issues are found. No preamble, no markdown fences."
    )

    human = (
        f"Writing context: {style_level}\n"
        f"Focus areas: {focus_str}\n\n"
        f"TEXT TO ANALYSE:\n{raw}"
    )

    issues: List[Dict[str, Any]] = []
    errors: List[str] = list(state.get("errors") or [])

    try:
        model = _llm(state, temperature=0.1, num_predict=2048)
        raw_response = _call(model, system, human)
        match = re.search(r"\[.*?\]", raw_response, re.DOTALL)
        if match:
            issues = json.loads(match.group(0))
            if not isinstance(issues, list):
                issues = []
        else:
            logger.warning("[Grammar] grammar_analysis: no JSON array in response")
            errors.append("Grammar analysis produced no structured output; issues list is empty.")
    except Exception as exc:
        logger.warning("[Grammar] grammar_analysis failed: %s", exc)
        errors.append(f"Grammar analysis error: {exc}")

    return {
        "issues_found": issues,
        "errors": errors,
        "current_step": "grammar_analysis",
        "completed_steps": list(state.get("completed_steps") or []) + ["grammar_analysis"],
        "progress_pct": 35,
        "status_detail": f"Found {len(issues)} {'issue' if len(issues) == 1 else 'issues'}",
    }


# ── Node 3: polish ────────────────────────────────────────────────────────────

def polish_node(state: GrammarState) -> Dict[str, Any]:
    """
    Primary output node — rewrite the text for clarity, fluency, and correctness
    suited to the requested writing context.

    Uses context-specific system prompts from _STYLE_PROMPTS. On revision rounds
    (refinement_round > 0), incorporates user feedback into the rewrite.
    """
    logger.info("[Grammar] Polish")

    raw = state.get("raw_text") or ""
    style_level = (state.get("style_level") or "professional_email").lower()
    issues: List[Dict] = state.get("issues_found") or []
    refinement_round: int = state.get("refinement_round") or 0
    feedback: str = (state.get("feedback") or "").strip()
    errors: List[str] = list(state.get("errors") or [])

    system = _STYLE_PROMPTS.get(style_level, _STYLE_PROMPTS[_FALLBACK_STYLE])
    system += (
        "\n\nAfter the rewritten text, output exactly the line:\n"
        "---CHANGES---\n"
        "Then list the key changes as a Markdown bullet list (what was changed and why). "
        "Keep each bullet concise."
    )

    # Build issue hints
    issues_hint = ""
    if issues:
        lines = []
        for i, iss in enumerate(issues[:20], 1):
            lines.append(
                f"{i}. [{iss.get('type','?')}] \"{iss.get('original','')}\" "
                f"→ \"{iss.get('suggestion','')}\" ({iss.get('explanation','')})"
            )
        issues_hint = "\n\nDetected issues to address:\n" + "\n".join(lines)

    if refinement_round > 0 and feedback:
        # Revision path: include previous polished text and feedback
        previous = state.get("polished_text") or raw
        human = (
            f"PREVIOUS VERSION:\n{previous}\n\n"
            f"USER FEEDBACK:\n{feedback}\n\n"
            f"ORIGINAL TEXT:\n{raw}"
            f"{issues_hint}\n\n"
            "Revise the original text taking the above feedback into account while "
            f"maintaining the {style_level} writing context."
        )
    else:
        human = f"TEXT TO REWRITE:\n{raw}{issues_hint}"

    polished_text = raw  # fallback: return original unchanged
    change_summary = ""

    try:
        model = _llm(state, temperature=0.2, num_predict=4096)
        response = _call(model, system, human)

        if "---CHANGES---" in response:
            parts = response.split("---CHANGES---", 1)
            polished_text = parts[0].strip()
            change_summary = parts[1].strip()
        else:
            polished_text = response
            change_summary = ""
    except Exception as exc:
        logger.warning("[Grammar] polish_node failed: %s", exc)
        errors.append(f"Polishing error: {exc}. Original text returned unchanged.")

    # Update feedback history if this is a revision
    feedback_history: List[Dict] = list(state.get("feedback_history") or [])
    if refinement_round > 0 and feedback:
        feedback_history.append({
            "round": refinement_round,
            "feedback": feedback,
            "previous_polished": state.get("polished_text") or raw,
        })

    return {
        "polished_text": polished_text,
        "change_summary": change_summary,
        "feedback_history": feedback_history,
        "errors": errors,
        "current_step": "polish",
        "completed_steps": list(state.get("completed_steps") or []) + ["polish"],
        "progress_pct": 65,
        "status_detail": f"Rewriting for {style_level} context" + (f" (revision {refinement_round})" if refinement_round > 0 else ""),
    }


# ── Node 4: style_advisor ─────────────────────────────────────────────────────

def style_advisor_node(state: GrammarState) -> Dict[str, Any]:
    """
    Generate style improvement suggestions.

    Skips gracefully if neither 'style' nor 'clarity' is in focus_areas
    (when focus_areas is non-empty and contains neither). When focus_areas
    is empty (= all areas), style advice is always generated.
    """
    logger.info("[Grammar] Style Advisor")

    focus_areas: List[str] = state.get("focus_areas") or []
    # If specific areas were selected and style/clarity are not among them, skip
    if focus_areas and not any(f in focus_areas for f in ("style", "clarity")):
        return {
            "style_suggestions": [],
            "current_step": "style_advisor",
            "completed_steps": list(state.get("completed_steps") or []) + ["style_advisor"],
            "progress_pct": 85,
            "status_detail": "Skipped — style/clarity not in selected focus areas",
        }

    polished = state.get("polished_text") or state.get("raw_text") or ""
    style_level = state.get("style_level") or "professional_email"
    errors: List[str] = list(state.get("errors") or [])

    if not polished:
        return {
            "style_suggestions": [],
            "current_step": "style_advisor",
            "completed_steps": list(state.get("completed_steps") or []) + ["style_advisor"],
            "progress_pct": 85,
            "status_detail": "Skipped — no text to analyse",
        }

    system = (
        "You are a professional writing coach. Analyse the provided text and return ONLY "
        "a JSON array of style improvement suggestions. Each element must have these exact "
        'keys: "category" (clarity|conciseness|tone|vocabulary|structure), '
        '"suggestion" (specific actionable advice, max 30 words), '
        '"rationale" (one sentence explaining why this would improve the text). '
        f"The writing context is: {style_level}. "
        "Return 8-10 suggestions. Return [] if the writing is already strong. "
        "No preamble, no markdown fences."
    )

    human = f"TEXT TO REVIEW:\n{polished[:3000]}"

    suggestions: List[Dict[str, Any]] = []

    try:
        model = _llm(state, temperature=0.3, num_predict=2048)
        raw_response = _call(model, system, human)
        match = re.search(r"\[.*?\]", raw_response, re.DOTALL)
        if match:
            suggestions = json.loads(match.group(0))
            if not isinstance(suggestions, list):
                suggestions = []
    except Exception as exc:
        logger.warning("[Grammar] style_advisor failed: %s", exc)
        errors.append(f"Style advisor error: {exc}")

    return {
        "style_suggestions": suggestions,
        "errors": errors,
        "current_step": "style_advisor",
        "completed_steps": list(state.get("completed_steps") or []) + ["style_advisor"],
        "progress_pct": 85,
        "status_detail": f"{len(suggestions)} style {'suggestion' if len(suggestions) == 1 else 'suggestions'}",
    }
