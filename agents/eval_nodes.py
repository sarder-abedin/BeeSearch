"""
agents/eval_nodes.py
─────────────────────
Self-evaluation nodes for Research Notebook and Systematic Review modes.

Each node makes a single micro LLM call (temperature=0.1, num_predict=300)
to rate output quality. Failures are logged and silently ignored — eval is
always non-blocking.

Dimensions
──────────
  Notebook Q&A     : answer_grounding (1–5), citation_accuracy (1–5), relevance (1–5)
  Notebook Pipeline: summary_quality (1–5), citation_coverage (1–5), study_guide_quality (1–5)
  All modes: overall (1–5), summary (one sentence)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

_EVAL_SYSTEM = (
    "You are a strict quality evaluator. Respond ONLY with a JSON object. "
    "No preamble, no explanation, no markdown fences."
)


def _eval_llm(model_name: str, num_ctx: int) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=0.1,
        num_predict=300,
        num_ctx=min(num_ctx or cfg.num_ctx, 8192),
        sync_client_kwargs={"timeout": httpx.Timeout(120.0)},
    )


def _run_eval(llm: ChatOllama, human: str) -> Dict[str, Any]:
    """Call the LLM for eval; return parsed dict or {} on any failure."""
    try:
        resp = llm.invoke([SystemMessage(content=_EVAL_SYSTEM), HumanMessage(content=human)])
        raw = resp.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        logger.warning("Eval: no JSON object in response — raw tail: %s", raw[-80:])
    except Exception as e:
        logger.warning("Eval LLM call failed: %s", e)
    return {}


# ── Research Notebook Q&A eval ─────────────────────────────────────────────────

def notebook_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a Research Notebook Q&A response on grounding, citation accuracy, relevance."""
    logger.info("[Eval] Notebook Q&A mode")
    response = state.get("assistant_response", "")
    if not response:
        return {"eval_result": {}}

    result: Dict[str, Any] = {}
    try:
        llm = _eval_llm(state.get("model_name", ""), state.get("num_ctx", 0))
        citations = state.get("citations", [])
        chunks = state.get("retrieved_chunks", [])
        human = f"""Rate this grounded Q&A response from a Research Notebook. Use integers 1–5 (5 = excellent).

QUESTION: {state.get('user_message', '')[:200]}

RESPONSE EXCERPT: {response[:600]}

NUMBER OF CITED SOURCES: {len(citations)}

NUMBER OF RETRIEVED CHUNKS USED: {len(chunks)}

Return this JSON exactly:
{{
  "answer_grounding": <1-5>,
  "citation_accuracy": <1-5>,
  "relevance": <1-5>,
  "overall": <1-5>,
  "summary": "<one sentence assessment>"
}}"""
        result = _run_eval(llm, human)
    except Exception as e:
        logger.warning("Notebook eval node error: %s", e)

    try:
        from tools.ragchecker_eval import check_faithfulness, is_available as ragchecker_available
        if ragchecker_available():
            query = state.get("user_message", "")
            chunks = state.get("retrieved_chunks", [])
            if query and response and chunks:
                retrieved_chunks: list = []
                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, dict):
                        text = chunk.get("text") or chunk.get("content") or chunk.get("page_content") or ""
                        doc_id = chunk.get("doc_id") or chunk.get("source") or f"chunk_{i}"
                    else:
                        text = str(chunk)
                        doc_id = f"chunk_{i}"
                    if text:
                        retrieved_chunks.append({"text": text, "doc_id": doc_id})
                if retrieved_chunks:
                    faith_result = check_faithfulness(
                        query=query,
                        response_text=response,
                        retrieved_chunks=retrieved_chunks,
                        model_name=state.get("model_name") or cfg.ollama_model,
                        ollama_base_url=cfg.ollama_base_url,
                    )
                    result["ragchecker_faithfulness"] = {
                        "faithfulness_score": faith_result.get("faithfulness_score"),
                        "checked_claims": faith_result.get("checked_claims", 0),
                        "supported_claims": faith_result.get("supported_claims", 0),
                        "skipped": faith_result.get("skipped", True),
                    }
    except Exception as e:
        logger.warning("Notebook eval RAGchecker faithfulness check failed: %s", e)

    return {
        "eval_result": result,
        "current_step": "notebook_eval",
        "completed_steps": state.get("completed_steps", []) + ["notebook_eval"],
    }


# ── Research Notebook Pipeline eval ───────────────────────────────────────────

def notebook_pipeline_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the 7-agent pipeline output on summary quality, citation coverage, study guide."""
    logger.info("[Eval] Notebook pipeline mode")
    cross_summary = state.get("cross_summary", "")
    if not cross_summary:
        return {"eval_result": {}}

    result: Dict[str, Any] = {}
    try:
        llm = _eval_llm(state.get("settings", {}).get("model", ""), 0)
        citations = state.get("verified_citations", [])
        per_doc = state.get("per_doc_summaries", {})
        human = f"""Rate this Research Notebook pipeline output. Use integers 1–5 (5 = excellent).

NUMBER OF SOURCE DOCUMENTS: {state.get('doc_count', 0)}

CROSS-DOCUMENT SUMMARY EXCERPT: {cross_summary[:500]}

NUMBER OF PER-DOCUMENT SUMMARIES: {len(per_doc)}

NUMBER OF VERIFIED CITATIONS: {len(citations)}

STUDY GUIDE GENERATED: {'Yes' if state.get('study_guide') else 'No'}

Return this JSON exactly:
{{
  "summary_quality": <1-5>,
  "citation_coverage": <1-5>,
  "study_guide_quality": <1-5>,
  "overall": <1-5>,
  "summary": "<one sentence assessment>"
}}"""
        result = _run_eval(llm, human)
    except Exception as e:
        logger.warning("Notebook pipeline eval node error: %s", e)

    return {
        "eval_result": result,
        "current_step": "notebook_pipeline_eval",
        "completed_steps": state.get("completed_steps", []) + ["notebook_pipeline_eval"],
    }
