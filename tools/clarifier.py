"""
tools/clarifier.py
───────────────────
Socratic clarification engine.

Generates 2–3 focused clarifying questions tailored to the user's goal and
research mode, using a single fast LLM call. Falls back to hardcoded
per-mode questions if the LLM call fails.
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
    "systematic_review": (
        "The user wants to conduct a PRISMA-style systematic literature review. "
        "Clarify: research question focus, inclusion/exclusion criteria, "
        "intended audience, synthesis depth."
    ),
    "notebook": (
        "The user uploaded documents and wants to explore them interactively. "
        "Clarify: analysis depth, focus area, intended audience, "
        "comparison vs summary, specific questions to answer."
    ),
}


def generate_clarifying_questions(
    goal: str,
    mode: str,
    model_name: str,
    ollama_base_url: str,
    num_ctx: int = 4096,
) -> List[Dict[str, Any]]:
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
    if not isinstance(raw, list):
        return []
    valid = []
    for q in raw:
        if not isinstance(q, dict):
            continue
        if not q.get("key") or not q.get("question"):
            continue
        q = dict(q)
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
    fallbacks: Dict[str, List[Dict[str, Any]]] = {
        "systematic_review": [
            {
                "key": "audience",
                "question": "Who is the intended audience for this systematic review?",
                "type": "select",
                "options": ["Academic researchers", "Clinical practitioners", "Policy makers", "Students"],
                "recommended": "Academic researchers",
            },
            {
                "key": "scope",
                "question": "How broad should the literature search be?",
                "type": "select",
                "options": ["Narrow — specific subfield", "Moderate — field-wide", "Broad — cross-disciplinary"],
                "recommended": "Moderate — field-wide",
            },
        ],
        "notebook": [
            {
                "key": "audience",
                "question": "Who is the intended audience for this analysis?",
                "type": "select",
                "options": ["Academic researchers", "Industry professionals", "Students", "General public"],
                "recommended": "Academic researchers",
            },
            {
                "key": "focus",
                "question": "What should the analysis prioritise?",
                "type": "select",
                "options": ["Key findings and conclusions", "Methodology critique", "Comparison across documents", "Identify gaps and future work"],
                "recommended": "Key findings and conclusions",
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
