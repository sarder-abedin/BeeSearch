"""agents/contradiction_detector.py — Contradiction detection across included papers"""
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


def _call(llm, system: str, human: str) -> str:
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content.strip()


def detect_contradictions(
    evidence_table: List[Dict[str, Any]],
    research_question: str,
    model_name: str,
    num_ctx: int,
) -> List[Dict[str, Any]]:
    """
    Detect contradictions and conflicting findings across included papers.
    Returns a list of contradiction groups with consensus scores.
    """
    if len(evidence_table) < 2:
        return []

    llm = _llm(model_name, num_ctx)
    evidence_text = "\n".join(
        f"[{e.get('citation_key','')}] {e.get('title','')[:60]} "
        f"({e.get('year','n.d.')}) — {e.get('key_finding','')}"
        for e in evidence_table[:20]
    )

    raw = _call(
        llm,
        f"""You are a systematic review expert identifying contradictions in evidence.
Research question: {research_question}

Identify 2-5 areas where the included papers have opposing or conflicting findings.
consensus_score: 0=complete disagreement, 100=full consensus.

Return ONLY valid JSON array:
[
  {{
    "claim": "short description of the contested claim",
    "position_a": {{"description": "what position A claims", "papers": ["key1"]}},
    "position_b": {{"description": "what position B claims", "papers": ["key2"]}},
    "consensus_score": 50,
    "explanation": "2-3 sentences explaining the contradiction"
  }}
]
Return [] if no meaningful contradictions exist.""",
        f"Evidence:\n{evidence_text}",
    )

    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        results = json.loads(match.group(0)) if match else []
        if not isinstance(results, list):
            results = []
    except Exception as e:
        logger.warning("Contradiction detection JSON parse failed: %s", e)
        results = []
    return results
