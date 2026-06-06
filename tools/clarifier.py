"""
tools/clarifier.py
───────────────────
Socratic clarification engine.

Generates 2–3 focused clarifying questions tailored to the user's goal and
research mode, using a single fast LLM call. Falls back to hardcoded
per-mode questions if the LLM call fails.

Every question is "select" type: 3–4 preset options plus an implicit
"Other (please specify)" fallback rendered by the UI. Each question also
carries a "recommended" field — the agent's suggested best-default answer.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a research requirements analyst. Given a user's goal and the research \
mode they are using, generate EXACTLY 2 or 3 focused clarifying questions to \
better understand what they need.

Rules:
- Questions must be directly relevant to the SPECIFIC goal — not generic
- Do NOT ask about things already stated in the goal
- Every question MUST have 3–4 concrete answer options (type "select")
- Every question MUST include a "recommended" field — the option you suggest \
  as the best default, based on the goal (must exactly match one of the options)
- Focus on the highest-value unknowns: audience, scope, depth, output format, \
  constraints, prior knowledge
- Return ONLY valid JSON (an array). No prose before or after.

Output format — an array of 2 or 3 objects:
[
  {
    "key": "snake_case_key",
    "question": "Question text ending with ?",
    "type": "select",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "recommended": "Option B"
  }
]"""

_MODE_CONTEXT: Dict[str, str] = {
    "document": (
        "The user uploaded one or more documents (papers, reports, notes) and "
        "wants an analysis. Clarify: analysis depth, focus area, intended "
        "audience, comparison vs summary, specific questions to answer."
    ),
    "search": (
        "The user wants to search academic literature (arXiv, Semantic Scholar). "
        "Clarify: date range, discipline focus, intended use (background / gap "
        "analysis / citation gathering), audience."
    ),
    "hybrid": (
        "The user has documents AND wants academic literature searched. "
        "Clarify: whether to prioritise uploaded docs or the broader literature, "
        "analysis depth, audience, key comparison dimensions."
    ),
    "proposal": (
        "The user wants to write a research grant proposal. "
        "Clarify: intended funder / audience, discipline, proposal length / "
        "format, whether they have preliminary results."
    ),
    "story": (
        "The user wants to learn a research concept interactively. "
        "Clarify: their background level, what specifically confuses them, how "
        "they intend to apply this understanding."
    ),
    "wisdom": (
        "The user is seeking evidence-based wisdom about a life, health, or "
        "professional question. Clarify: whether this is personal or professional, "
        "what they have already tried, how urgent the decision is."
    ),
}


def generate_clarifying_questions(
    goal: str,
    mode: str,
    model_name: str,
    ollama_base_url: str,
    num_ctx: int = 4096,
) -> List[Dict[str, Any]]:
    """Return 2–3 Socratic clarifying questions tailored to goal and mode.

    Falls back to hardcoded per-mode questions if the LLM call fails.
    """
    mode_ctx = _MODE_CONTEXT.get(mode, "The user wants to conduct research.")
    human = (
        f"MODE: {mode}\n"
        f"MODE CONTEXT: {mode_ctx}\n\n"
        f"USER GOAL:\n{goal[:800]}"
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": human},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": num_ctx, "num_predict": 512},
    }
    try:
        resp = httpx.post(
            f"{ollama_base_url}/api/chat",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"].strip()
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            questions = _validate_questions(json.loads(match.group()))
            if questions:
                return questions[:3]
    except Exception as exc:
        logger.warning("Clarifier LLM call failed: %s", exc)
    return _fallback_questions(mode)


def _validate_questions(raw: Any) -> List[Dict[str, Any]]:
    """Sanitise LLM-generated questions.

    Ensures every question is "select" type with non-empty options and a
    "recommended" field that exactly matches one of those options.
    """
    if not isinstance(raw, list):
        return []
    valid = []
    for q in raw:
        if not isinstance(q, dict):
            continue
        if not q.get("key") or not q.get("question"):
            continue

        q = dict(q)  # copy before mutating
        q["type"] = "select"

        if not q.get("options"):
            q["options"] = ["Yes", "No", "Partially", "Not applicable"]

        options = q["options"]
        recommended = q.get("recommended", "")
        if not recommended or recommended not in options:
            q["recommended"] = options[0]

        valid.append(q)
    return valid


def _fallback_questions(mode: str) -> List[Dict[str, Any]]:
    """Per-mode fallback questions used when the LLM call fails."""
    fallbacks: Dict[str, List[Dict[str, Any]]] = {
        "document": [
            {
                "key": "audience",
                "question": "Who is the intended audience for this analysis?",
                "type": "select",
                "options": [
                    "Academic researchers",
                    "Industry professionals",
                    "Students",
                    "General public",
                ],
                "recommended": "Academic researchers",
            },
            {
                "key": "focus",
                "question": "What should the analysis prioritise?",
                "type": "select",
                "options": [
                    "Key findings and conclusions",
                    "Methodology critique",
                    "Comparison across documents",
                    "Identify gaps and future work",
                ],
                "recommended": "Key findings and conclusions",
            },
        ],
        "search": [
            {
                "key": "recency",
                "question": "How recent should the literature be?",
                "type": "select",
                "options": [
                    "Last 2 years",
                    "Last 5 years",
                    "Last 10 years",
                    "No restriction",
                ],
                "recommended": "Last 5 years",
            },
            {
                "key": "purpose",
                "question": "What will you use these results for?",
                "type": "select",
                "options": [
                    "Background review",
                    "Gap analysis",
                    "Citation gathering",
                    "State-of-the-art survey",
                ],
                "recommended": "Background review",
            },
        ],
        "hybrid": [
            {
                "key": "emphasis",
                "question": (
                    "Should the report emphasise your uploaded documents "
                    "or the broader literature?"
                ),
                "type": "select",
                "options": [
                    "My documents primarily",
                    "Academic literature primarily",
                    "Equal balance",
                ],
                "recommended": "Equal balance",
            },
            {
                "key": "audience",
                "question": "Who is the intended reader?",
                "type": "select",
                "options": [
                    "Academic peers",
                    "Industry professionals",
                    "Supervisors / funders",
                    "General audience",
                ],
                "recommended": "Academic peers",
            },
        ],
        "proposal": [
            {
                "key": "funder",
                "question": "Who is the intended funder or audience?",
                "type": "select",
                "options": [
                    "EU Horizon Europe",
                    "National Science Foundation (NSF)",
                    "UK Research and Innovation (UKRI)",
                    "Industry / company funding",
                ],
                "recommended": "EU Horizon Europe",
            },
            {
                "key": "length",
                "question": "What is the target proposal format?",
                "type": "select",
                "options": [
                    "Short grant (3–5 pages)",
                    "Standard grant (10–15 pages)",
                    "Fellowship application",
                    "Internal research proposal",
                ],
                "recommended": "Standard grant (10–15 pages)",
            },
        ],
        "story": [
            {
                "key": "level",
                "question": "What is your background level on this topic?",
                "type": "select",
                "options": [
                    "Complete beginner",
                    "Some familiarity",
                    "Intermediate",
                    "Advanced — want a fresh perspective",
                ],
                "recommended": "Some familiarity",
            },
            {
                "key": "learning_goal",
                "question": "What do you hope to achieve after this session?",
                "type": "select",
                "options": [
                    "Build conceptual understanding",
                    "Apply in my own research",
                    "Prepare to explain to others",
                    "Satisfy intellectual curiosity",
                ],
                "recommended": "Build conceptual understanding",
            },
        ],
        "wisdom": [
            {
                "key": "context",
                "question": "Is this primarily a personal or professional situation?",
                "type": "select",
                "options": [
                    "Personal / health",
                    "Professional / career",
                    "Relationships / social",
                    "Academic / learning",
                ],
                "recommended": "Personal / health",
            },
            {
                "key": "tried",
                "question": "What have you already tried or considered?",
                "type": "select",
                "options": [
                    "Nothing yet — seeking guidance",
                    "Done basic online research",
                    "Consulted colleagues or friends",
                    "Tried multiple approaches without success",
                ],
                "recommended": "Nothing yet — seeking guidance",
            },
        ],
    }
    return fallbacks.get(
        mode,
        [
            {
                "key": "audience",
                "question": "Who is the intended audience for this output?",
                "type": "select",
                "options": ["Academic / research", "Professional / industry", "General public"],
                "recommended": "Academic / research",
            },
            {
                "key": "depth",
                "question": "How detailed should the output be?",
                "type": "select",
                "options": ["Overview / summary", "Comprehensive / in-depth"],
                "recommended": "Overview / summary",
            },
        ],
    )
