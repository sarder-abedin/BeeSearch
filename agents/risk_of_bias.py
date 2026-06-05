"""agents/risk_of_bias.py — Cochrane RoB 2 / ROBINS-I risk of bias assessment"""
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


def _llm(model_name: str, num_ctx: int, temperature: float = 0.1) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=512,
        num_ctx=num_ctx or cfg.num_ctx,
        sync_client_kwargs={"timeout": httpx.Timeout(300.0)},
    )


def _call(llm: ChatOllama, system: str, human: str) -> str:
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content.strip()


ROB2_DOMAINS = [
    "randomisation_process",
    "deviations_from_interventions",
    "missing_outcome_data",
    "measurement_of_outcome",
    "selection_of_reported_result",
]

ROBINS_DOMAINS = [
    "bias_due_to_confounding",
    "bias_in_selection_of_participants",
    "bias_in_classification_of_interventions",
    "bias_due_to_deviations",
    "bias_due_to_missing_data",
    "bias_in_measurement_of_outcomes",
    "bias_in_selection_of_reported_result",
]


def assess_risk_of_bias(
    paper: Dict[str, Any],
    model_name: str,
    num_ctx: int,
) -> Dict[str, Any]:
    """
    Assess RoB 2 for RCTs and ROBINS-I for observational studies.
    Returns dict with tool, domains, overall, and justification.
    """
    study_design = paper.get("study_design", "Unknown").lower()
    is_rct = any(kw in study_design for kw in ["rct", "randomis", "randomiz", "trial"])

    llm = _llm(model_name, num_ctx)
    tool = "RoB 2" if is_rct else "ROBINS-I"
    domains = ROB2_DOMAINS if is_rct else ROBINS_DOMAINS

    domain_json = json.dumps({d: "Low|Some concerns|High" for d in domains})

    raw = _call(
        llm,
        f"""You are a systematic review methodologist applying {tool} risk of bias assessment.
Rate each domain as exactly one of: Low, Some concerns, or High.
Use the paper's title, abstract, study design, and findings for your assessment.
Return ONLY valid JSON matching this schema:
{domain_json}
Also add keys: "overall": "Low|Some concerns|High", "justification": "2-3 sentences"
""",
        f"Title: {paper.get('title','')}\nDesign: {paper.get('study_design','')}\n"
        f"Sample size: {paper.get('sample_size','')}\nFinding: {paper.get('key_finding','')}\n"
        f"Abstract: {paper.get('abstract','')[:600]}",
    )

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group(0)) if match else {}
    except Exception:
        result = {d: "Some concerns" for d in domains}
        result["overall"] = "Some concerns"
        result["justification"] = "Automated assessment — insufficient data in abstract."

    result["tool"] = tool
    result["citation_key"] = paper.get("citation_key", "")
    result["title"] = paper.get("title", "")[:80]
    return result


def assess_rob_batch(
    evidence_table: List[Dict[str, Any]],
    model_name: str,
    num_ctx: int,
) -> List[Dict[str, Any]]:
    """Assess risk of bias for all papers in evidence_table."""
    results = []
    for paper in evidence_table:
        try:
            rob = assess_risk_of_bias(paper, model_name, num_ctx)
            results.append(rob)
        except Exception as e:
            logger.warning("RoB assessment failed for '%s': %s", paper.get("title", "")[:40], e)
            results.append({
                "citation_key": paper.get("citation_key", ""),
                "title": paper.get("title", "")[:80],
                "tool": "Unknown",
                "overall": "Some concerns",
                "justification": f"Assessment failed: {e}",
            })
    return results
