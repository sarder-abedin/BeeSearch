"""tools/hypothesis_generator.py — Testable hypothesis generator from research gaps"""
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
    return ChatOllama(model=model_name or cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0.3, num_predict=1024, num_ctx=num_ctx or cfg.num_ctx, sync_client_kwargs={"timeout": httpx.Timeout(300.0)})

def generate_hypotheses(research_gaps: List[str], research_question: str, evidence_summary: str, model_name: str, num_ctx: int, n_hypotheses: int = 5) -> List[Dict[str, Any]]:
    llm = _llm(model_name, num_ctx)
    gaps_text = "\n".join(f"{i+1}. {g}" for i, g in enumerate(research_gaps[:10]))
    raw = llm.invoke([SystemMessage(content=f"""Generate {n_hypotheses} PICO-structured testable hypotheses from research gaps.
Research question: {research_question}
Return ONLY valid JSON array:
[{{"hypothesis": "If [population] receives [intervention], then [outcome] compared to [comparator]", "gap_addressed": "...", "rationale": "...", "suggested_design": "RCT|Cohort|...", "independent_variable": "...", "dependent_variable": "...", "feasibility": "High|Medium|Low", "feasibility_note": "..."}}]"""), HumanMessage(content=f"Gaps:\n{gaps_text}\n\nContext:\n{evidence_summary[:600]}")]).content.strip()
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        result = json.loads(match.group(0)) if match else []
        if not isinstance(result, list):
            result = []
    except Exception as e:
        logger.warning("Hypothesis generation failed: %s", e)
        result = []
    return result[:n_hypotheses]
