"""tools/research_gaps.py — Research gap mapper"""
from __future__ import annotations
import json, logging, re
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from config.settings import get_settings
logger = logging.getLogger(__name__)
cfg = get_settings()

def _llm(model_name: str, num_ctx: int) -> ChatOllama:
    import httpx
    return ChatOllama(model=model_name or cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0.2, num_predict=1024, num_ctx=num_ctx or cfg.num_ctx, sync_client_kwargs={"timeout": httpx.Timeout(300.0)})

def map_research_gaps(evidence_table: List[Dict[str, Any]], research_question: str, existing_gaps: List[str], model_name: str, num_ctx: int) -> Dict[str, Any]:
    llm = _llm(model_name, num_ctx)
    evidence_summary = "\n".join(f"[{e.get('citation_key','')}] {e.get('study_design','')} — {e.get('key_finding','')[:120]}" for e in evidence_table[:15])
    existing_text = "\n".join(f"- {g}" for g in existing_gaps[:10]) or "(none)"
    raw = llm.invoke([SystemMessage(content=f"""You are a research gap analyst.
Research question: {research_question}
Identify gaps across: population_gaps, methodology_gaps, outcome_gaps, context_gaps, temporal_gaps.
Each gap: {{"gap": "...", "priority": "High|Medium|Low", "rationale": "..."}}
Return ONLY valid JSON:
{{"population_gaps": [...], "methodology_gaps": [...], "outcome_gaps": [...], "context_gaps": [...], "temporal_gaps": [...], "priority_gaps": ["top 3 strings"], "gap_map_summary": "2-3 sentences"}}"""), HumanMessage(content=f"Evidence:\n{evidence_summary}\n\nExisting gaps:\n{existing_text}")]).content.strip()
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group(0)) if match else {}
    except Exception as e:
        logger.warning("Gap mapping failed: %s", e)
        result = {"priority_gaps": existing_gaps[:3], "gap_map_summary": "Gap mapping could not be completed."}
    return result
