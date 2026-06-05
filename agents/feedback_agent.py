"""
agents/feedback_agent.py
─────────────────────────
Lightweight feedback refinement for all research modes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

MAX_FEEDBACK_ROUNDS = 3

_MODE_LABELS: Dict[str, str] = {
    "literature_search": "literature search report",
    "wisdom": "wisdom insight",
    "systematic_review": "systematic review",
    "notebook_pipeline": "research notebook pipeline output",
    "proposal": "research proposal",
    "grammar_proofreading": "proofread and polished text",
}


def refine_with_feedback(
    original_output: str,
    feedback: str,
    *,
    context: str = "",
    mode: str = "research",
    model_name: str = "llama3.1:8b",
    num_ctx: int = 8192,
) -> str:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    mode_label = _MODE_LABELS.get(mode, "research output")
    llm = ChatOllama(model=model_name, temperature=0.4, num_ctx=num_ctx, num_predict=4096)

    system = (
        f"You are a research assistant refining a {mode_label} based on user feedback. "
        "Apply the feedback precisely. Preserve all citations, reference numbers, and factual "
        "content unless the feedback explicitly asks to change them. Maintain the same overall "
        "structure and Markdown formatting unless instructed otherwise. "
        "Return only the refined output — no preamble, no 'Here is the refined version' prefix."
    )

    context_block = f"\n\nCONTEXT (for reference only, do not modify):\n{context[:2000]}" if context.strip() else ""
    human = (
        f"ORIGINAL OUTPUT:\n{original_output[:6000]}"
        f"{context_block}\n\n"
        f"USER FEEDBACK:\n{feedback}\n\n"
        "Return the refined output now:"
    )

    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return response.content.strip()
    except Exception as exc:
        logger.warning("Feedback refinement LLM call failed: %s", exc)
        return original_output


def make_feedback_entry(round_num: int, feedback: str, previous_output: str) -> Dict[str, Any]:
    """Build a feedback history dict entry."""
    from datetime import datetime, timezone
    return {
        "round": round_num,
        "feedback": feedback,
        "previous_output": previous_output[:3000],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
