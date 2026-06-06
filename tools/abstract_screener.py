"""
tools/abstract_screener.py
──────────────────────────
LLM-based abstract screener for systematic reviews.

For each paper, the LLM scores the title+abstract against the review's
include/exclude criteria and assigns a 0-100 relevance score with a
verdict (include / uncertain / exclude) and a one-sentence rationale.

Usage
-----
    from tools.abstract_screener import screen_abstracts

    scores = screen_abstracts(
        papers=raw_papers,
        inclusion_criteria=["RCT", "human participants"],
        exclusion_criteria=["animal studies"],
        research_question="What is the effect of …",
        model_name="llama3.1:8b",
        num_ctx=32768,
    )
    # Each entry: {"paper": {...}, "score": 87, "verdict": "include", "rationale": "..."}
"""

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


def _make_llm(model_name: str, num_ctx: int) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=0.1,
        num_predict=256,
        num_ctx=num_ctx,
        sync_client_kwargs={"timeout": httpx.Timeout(120.0)},
    )


def _score_one(
    llm: ChatOllama,
    paper: Dict[str, Any],
    research_question: str,
    inclusion_criteria: List[str],
    exclusion_criteria: List[str],
) -> Dict[str, Any]:
    """Score a single paper. Returns a result dict with score, verdict, rationale."""
    inc_block = "; ".join(inclusion_criteria) if inclusion_criteria else "Relevant to the research question"
    exc_block = "; ".join(exclusion_criteria) if exclusion_criteria else "Clearly off-topic"
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")[:600]

    sys_prompt = (
        f"You are a systematic review screener. Score this paper's relevance.\n\n"
        f"Research question: {research_question}\n"
        f"Inclusion criteria: {inc_block}\n"
        f"Exclusion criteria: {exc_block}\n\n"
        "Return ONLY valid JSON:\n"
        '{{"score": <0-100>, "verdict": "include"|"uncertain"|"exclude", "rationale": "one sentence"}}\n\n'
        "Score guide: 80-100=clearly include, 50-79=uncertain (needs full-text), 0-49=exclude"
    )
    try:
        response = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Title: {title}\nAbstract: {abstract}"),
        ])
        raw = response.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group(0)) if match else {}
        score = max(0, min(100, int(result.get("score", 50))))
        verdict = result.get("verdict", "uncertain")
        rationale = result.get("rationale", "")
    except Exception as e:
        logger.warning("Abstract screening failed for '%s': %s", title[:50], e)
        score, verdict, rationale = 50, "uncertain", "Screening failed — review manually"

    return {"paper": paper, "score": score, "verdict": verdict, "rationale": rationale}


def screen_abstracts(
    papers: List[Dict[str, Any]],
    research_question: str,
    inclusion_criteria: List[str] = None,
    exclusion_criteria: List[str] = None,
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
) -> List[Dict[str, Any]]:
    """
    Score every paper in `papers` against the review criteria.

    Returns results sorted by score descending. Each entry:
      paper     — original paper dict
      score     — 0-100 relevance score
      verdict   — "include" | "uncertain" | "exclude"
      rationale — one-sentence explanation
    """
    if not papers:
        return []
    llm = _make_llm(model_name, num_ctx)
    results = [
        _score_one(llm, p, research_question, inclusion_criteria or [], exclusion_criteria or [])
        for p in papers
    ]
    results.sort(key=lambda x: -x["score"])
    return results


def screener_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return verdict counts and mean score from screen_abstracts output."""
    verdicts = [r["verdict"] for r in results]
    return {
        "total": len(results),
        "include": verdicts.count("include"),
        "uncertain": verdicts.count("uncertain"),
        "exclude": verdicts.count("exclude"),
        "mean_score": round(sum(r["score"] for r in results) / len(results), 1) if results else 0,
    }
