"""
agents/eval_nodes.py
─────────────────────
Self-evaluation nodes for all modes (Research, Proposal, Story, Wisdom, Grammar).

Each node makes a single micro LLM call (temperature=0.1, num_predict=300)
to rate output quality along mode-specific dimensions. Failures are logged
and silently ignored — eval is always non-blocking.

Dimensions
──────────
  Research : goal_alignment (1–5), evidence_quality (1–5), clarity (1–5)
  Proposal : goal_alignment (1–5), objectives_quality (1–5), methodology_soundness (1–5)
  Story    : clarity (1–5), style_adherence (1–5)
  Wisdom   : evidence_grounding (1–5), confidence_calibration (1–5), actionability (1–5)
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


# ── Research eval ──────────────────────────────────────────────────────────────

def research_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the research report on goal_alignment, evidence_quality, clarity."""
    logger.info("[Eval] Research mode")
    result: Dict[str, Any] = {}
    try:
        llm = _eval_llm(state.get("model_name", ""), state.get("num_ctx", 0))
        findings_preview = str(state.get("key_findings", [])[:3])[:300]
        human = f"""Rate this research output. Use integers 1–5 (5 = excellent).

RESEARCH GOAL: {state.get('goal', '')[:200]}

REPORT EXCERPT (first 600 chars): {state.get('report', '')[:600]}

KEY FINDINGS SAMPLE: {findings_preview}

NUMBER OF CITED SOURCES: {len(state.get('references', []))}

Return this JSON exactly:
{{
  "goal_alignment": <1-5>,
  "evidence_quality": <1-5>,
  "clarity": <1-5>,
  "overall": <1-5>,
  "summary": "<one sentence assessment>"
}}"""
        result = _run_eval(llm, human)
    except Exception as e:
        logger.warning("Research eval node error: %s", e)

    # Add faithfulness check if ragchecker is available
    try:
        from tools.ragchecker_eval import check_faithfulness, is_available as ragchecker_available
        if ragchecker_available():
            doc_context = state.get("doc_context", "")
            report = state.get("report", "")
            goal = state.get("goal", "")
            if doc_context and report:
                # Build retrieved_chunks from doc_context string or references list
                retrieved_chunks: list = []
                references = state.get("references", [])
                if references:
                    for i, ref in enumerate(references):
                        text = ref.get("abstract_snippet") or ref.get("title") or ""
                        if text:
                            retrieved_chunks.append({
                                "text": text,
                                "doc_id": ref.get("url") or ref.get("doi") or f"ref_{i}",
                            })
                if not retrieved_chunks and doc_context:
                    # Fall back to treating doc_context as a single chunk
                    retrieved_chunks = [{"text": doc_context[:3000], "doc_id": "doc_context"}]
                if retrieved_chunks:
                    faith_result = check_faithfulness(
                        query=goal,
                        response_text=report,
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
        logger.warning("Research eval RAGchecker faithfulness check failed: %s", e)

    return {
        "eval_result": result,
        "current_step": "research_eval",
        "completed_steps": state.get("completed_steps", []) + ["research_eval"],
    }


# ── Proposal eval ──────────────────────────────────────────────────────────────

def proposal_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the proposal on goal_alignment, objectives_quality, methodology_soundness."""
    logger.info("[Eval] Proposal mode")
    result: Dict[str, Any] = {}
    try:
        llm = _eval_llm(state.get("model_name", ""), state.get("num_ctx", 0))
        human = f"""Rate this research proposal. Use integers 1–5 (5 = excellent).

GOAL: {state.get('goal', '')[:200]}

TITLE: {state.get('title', '')[:100]}

OBJECTIVES: {str(state.get('objectives', []))[:300]}

METHODOLOGY EXCERPT: {state.get('methodology', '')[:400]}

ABSTRACT: {state.get('abstract', '')[:300]}

Return this JSON exactly:
{{
  "goal_alignment": <1-5>,
  "objectives_quality": <1-5>,
  "methodology_soundness": <1-5>,
  "overall": <1-5>,
  "summary": "<one sentence assessment>"
}}"""
        result = _run_eval(llm, human)
    except Exception as e:
        logger.warning("Proposal eval node error: %s", e)

    return {
        "eval_result": result,
        "current_step": "proposal_eval",
        "completed_steps": state.get("completed_steps", []) + ["proposal_eval"],
    }


# ── Story eval ─────────────────────────────────────────────────────────────────

def story_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a storyteller response on clarity and style_adherence."""
    logger.info("[Eval] Story mode")
    result: Dict[str, Any] = {}
    try:
        llm = _eval_llm(state.get("model_name", ""), state.get("num_ctx", 0))
        human = f"""Rate this science communication response. Use integers 1–5 (5 = excellent).

TOPIC: {state.get('topic', '')[:100]}

STYLE REQUESTED: {state.get('explanation_style', 'simple')}

RESPONSE EXCERPT: {state.get('assistant_response', '')[:600]}

Return this JSON exactly:
{{
  "clarity": <1-5>,
  "style_adherence": <1-5>,
  "overall": <1-5>,
  "summary": "<one sentence assessment>"
}}"""
        result = _run_eval(llm, human)
    except Exception as e:
        logger.warning("Story eval node error: %s", e)

    return {
        "eval_result": result,
        "current_step": "story_eval",
        "completed_steps": state.get("completed_steps", []) + ["story_eval"],
    }


# ── Wisdom eval ────────────────────────────────────────────────────────────────

def wisdom_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate wisdom synthesis on evidence_grounding, confidence_calibration, actionability.

    Skips evaluation silently when no wisdom has been generated yet (clarification turns).
    """
    logger.info("[Eval] Wisdom mode")
    deep = state.get("deep_understanding", "")
    if not deep:
        return {"eval_result": {}}

    result: Dict[str, Any] = {}
    try:
        llm = _eval_llm(state.get("model_name", ""), state.get("num_ctx", 0))
        human = f"""Rate this wisdom synthesis. Use integers 1–5 (5 = excellent).

TOPIC: {state.get('topic', '')[:100]}

SCIENTIFIC EXPLANATION EXCERPT: {deep[:500]}

ACTIONABLE TAKEAWAYS: {str(state.get('actionable_takeaways', []))[:300]}

CONFIDENCE LEVEL: {state.get('overall_confidence', 'Medium')}

NUMBER OF PAPERS USED: {len(state.get('academic_papers', []))}

Return this JSON exactly:
{{
  "evidence_grounding": <1-5>,
  "confidence_calibration": <1-5>,
  "actionability": <1-5>,
  "overall": <1-5>,
  "summary": "<one sentence assessment>"
}}"""
        result = _run_eval(llm, human)
    except Exception as e:
        logger.warning("Wisdom eval node error: %s", e)

    # Add faithfulness check if ragchecker is available
    try:
        from tools.ragchecker_eval import check_faithfulness, is_available as ragchecker_available
        if ragchecker_available():
            topic = state.get("topic", "")
            # wisdom_synthesis holds the full response; fall back to deep_understanding
            response_text = state.get("wisdom_synthesis", "") or deep
            academic_papers = state.get("academic_papers", [])
            knowledge_sources = state.get("knowledge_sources", [])
            if response_text:
                retrieved_chunks: list = []
                # Prefer academic_papers for chunk text
                for i, paper in enumerate(academic_papers):
                    text = (
                        paper.get("abstract_snippet")
                        or paper.get("abstract")
                        or paper.get("title")
                        or ""
                    )
                    if text:
                        retrieved_chunks.append({
                            "text": text,
                            "doc_id": paper.get("url") or paper.get("doi") or f"paper_{i}",
                        })
                # Also include knowledge_sources if present
                for i, ks in enumerate(knowledge_sources):
                    text = ks.get("text") or ks.get("content") or ks.get("summary") or ""
                    if text:
                        retrieved_chunks.append({
                            "text": text[:1000],
                            "doc_id": ks.get("doc_id") or f"ks_{i}",
                        })
                if retrieved_chunks:
                    faith_result = check_faithfulness(
                        query=topic,
                        response_text=response_text,
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
        logger.warning("Wisdom eval RAGchecker faithfulness check failed: %s", e)

    return {
        "eval_result": result,
        "current_step": "wisdom_eval",
        "completed_steps": state.get("completed_steps", []) + ["wisdom_eval"],
    }


# ── ProposalGPT eval ───────────────────────────────────────────────────────────

def proposal_gpt_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate ProposalGPT output on compliance, reviewer score, and proposal completeness."""
    logger.info("[Eval] ProposalGPT mode")
    result: Dict[str, Any] = {}
    try:
        llm = _eval_llm(state.get("model_name", ""), state.get("num_ctx", 0))
        compliance = state.get("compliance_score", 0)
        reviewer = state.get("overall_score", 0.0)
        sections_filled = sum(
            1 for k in ("executive_summary", "introduction", "research_problem",
                        "methodology", "work_plan", "budget_narrative")
            if state.get(k, "").strip()
        )
        human = f"""Rate this research proposal generated by an AI pipeline. Use integers 1–5 (5 = excellent).

FUNDING AGENCY: {state.get('funding_agency', 'Generic')}

COMPLIANCE SCORE (0–100): {compliance}

OVERALL REVIEWER SCORE (1–5): {reviewer:.1f}

SECTIONS FILLED: {sections_filled} / 6 key sections

ABSTRACT EXCERPT: {state.get('executive_summary', '')[:300]}

Return this JSON exactly:
{{
  "compliance_quality": <1-5>,
  "reviewer_score_quality": <1-5>,
  "proposal_completeness": <1-5>,
  "overall": <1-5>,
  "summary": "<one sentence assessment>"
}}"""
        result = _run_eval(llm, human)
    except Exception as e:
        logger.warning("ProposalGPT eval node error: %s", e)

    return {
        "eval_result": result,
        "current_step": "proposal_gpt_eval",
        "completed_steps": state.get("completed_steps", []) + ["proposal_gpt_eval"],
    }


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

    # Add faithfulness check if ragchecker is available
    try:
        from tools.ragchecker_eval import check_faithfulness, is_available as ragchecker_available
        if ragchecker_available():
            query = state.get("user_message", "")
            chunks = state.get("retrieved_chunks", [])
            if query and response and chunks:
                # Normalise chunk format: accept list of dicts or list of strings
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


# ── Grammar Proofreading eval ─────────────────────────────────────────────────

def grammar_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the polished text on polish_quality, context_fit, error_coverage, fluency."""
    logger.info("[Eval] Grammar Proofreading mode")
    result: Dict[str, Any] = {}

    polished = state.get("polished_text", "")
    if not polished:
        return {
            "eval_result": {},
            "current_step": "grammar_eval",
            "completed_steps": state.get("completed_steps", []) + ["grammar_eval"],
            "progress_pct": 100,
        }

    try:
        llm = _eval_llm(state.get("model_name", ""), state.get("num_ctx", 0))
        issues_count = len(state.get("issues_found", []))
        style_level = state.get("style_level", "professional_email")
        human = f"""Rate this grammar proofreading output. Use integers 1–5 (5 = excellent).

WRITING CONTEXT: {style_level}

ORIGINAL TEXT EXCERPT (first 300 chars): {(state.get('raw_text') or '')[:300]}

POLISHED TEXT EXCERPT (first 400 chars): {polished[:400]}

ISSUES DETECTED: {issues_count}

Return this JSON exactly:
{{
  "polish_quality": <1-5>,
  "context_fit": <1-5>,
  "error_coverage": <1-5>,
  "fluency": <1-5>,
  "overall": <1-5>,
  "summary": "<one sentence assessment>"
}}"""
        result = _run_eval(llm, human)
    except Exception as e:
        logger.warning("Grammar eval node error: %s", e)

    return {
        "eval_result": result,
        "current_step": "grammar_eval",
        "completed_steps": state.get("completed_steps", []) + ["grammar_eval"],
        "progress_pct": 100,
    }
