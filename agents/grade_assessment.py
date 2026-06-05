"""agents/grade_assessment.py — GRADE evidence grading"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()


def _llm(model_name: str, num_ctx: int) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=0.1,
        num_predict=1024,
        num_ctx=num_ctx or cfg.num_ctx,
        sync_client_kwargs={"timeout": httpx.Timeout(300.0)},
    )


def _call(llm: ChatOllama, system: str, human: str) -> str:
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content.strip()


GRADE_DOMAINS = [
    "risk_of_bias",
    "inconsistency",
    "indirectness",
    "imprecision",
    "publication_bias",
]


def grade_evidence_body(
    evidence_table: List[Dict[str, Any]],
    research_question: str,
    rob_table: List[Dict[str, Any]],
    model_name: str,
    num_ctx: int,
) -> Dict[str, Any]:
    """
    Apply GRADE framework to the body of evidence.
    RCTs start at High; observational studies start at Low.
    """
    if not evidence_table:
        return {}

    llm = _llm(model_name, num_ctx)
    has_rcts = any(
        any(kw in e.get("study_design", "").lower()
            for kw in ["rct", "randomis", "randomiz", "trial"])
        for e in evidence_table
    )
    start_level = "High" if has_rcts else "Low"
    n = len(evidence_table)

    evidence_summary = "\n".join(
        f"[{e.get('citation_key','')}] {e.get('study_design','')} — "
        f"Quality: {e.get('quality','')} — Finding: {e.get('key_finding','')[:120]}"
        for e in evidence_table[:15]
    )
    rob_summary = "\n".join(
        f"[{r.get('citation_key','')}] {r.get('tool','')} overall: {r.get('overall','')}"
        for r in rob_table[:15]
    ) if rob_table else ""

    raw = _call(
        llm,
        f"""You are applying the GRADE framework to rate the overall quality of evidence.
Starting level: {start_level}  |  Studies: {n}  |  Research question: {research_question}

For each domain rate: "no concern", "-1", or "-2" (downgrade levels).
Return ONLY valid JSON:
{{
  "starting_level": "{start_level}",
  "domains": {{"risk_of_bias": "...", "inconsistency": "...", "indirectness": "...", "imprecision": "...", "publication_bias": "..."}},
  "overall_grade": "High|Moderate|Low|Very low",
  "summary": "3-4 sentences",
  "certainty_statement": "Based on [N] studies, we have [grade] certainty that..."
}}""",
        f"Evidence:\n{evidence_summary}\n\nRisk of bias:\n{rob_summary}",
    )

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group(0)) if match else {}
    except Exception:
        result = {
            "starting_level": start_level,
            "domains": {d: "no concern" for d in GRADE_DOMAINS},
            "overall_grade": start_level,
            "summary": "GRADE assessment could not be completed automatically.",
            "certainty_statement": f"Based on {n} studies, certainty could not be determined.",
        }
    return result
