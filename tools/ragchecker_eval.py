"""
tools/ragchecker_eval.py
─────────────────────────
Lightweight, fully-local RAGChecker-style faithfulness evaluation.

Mirrors the core methodology of RAGChecker
(https://github.com/amazon-science/RAGChecker) — claim-level grounding
checks — but runs entirely through the app's existing Ollama backend, with
no extra heavyweight dependencies:

  1. Extract a handful of atomic factual claims from the response.
  2. Ask the model whether each claim is directly supported by the
     retrieved context.

faithfulness_score = supported_claims / checked_claims
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

_MAX_CLAIMS = 8
_MAX_CHUNKS = 8
_MAX_CONTEXT_CHARS = 6000

_SKIPPED: Dict[str, Any] = {
    "faithfulness_score": None,
    "checked_claims": 0,
    "supported_claims": 0,
    "skipped": True,
}

_CLAIM_SYSTEM = (
    "You extract atomic factual claims from a piece of text. "
    "Respond ONLY with a JSON object. No preamble, no explanation, no markdown fences."
)

_VERIFY_SYSTEM = (
    "You verify whether claims are supported by source passages. "
    "Respond ONLY with a JSON object. No preamble, no explanation, no markdown fences."
)


def is_available() -> bool:
    """Faithfulness checking runs locally via the configured Ollama model,
    so it is available whenever the rest of the app can run."""
    return True


def _make_llm(model_name: str, ollama_base_url: str) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=ollama_base_url or cfg.ollama_base_url,
        temperature=0.0,
        num_predict=512,
        num_ctx=min(cfg.num_ctx, 8192),
        sync_client_kwargs={"timeout": httpx.Timeout(120.0)},
    )


def _extract_json(raw: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _extract_claims(llm: ChatOllama, query: str, response_text: str) -> List[str]:
    human = (
        f"QUESTION: {query[:300]}\n\n"
        f"RESPONSE TO CHECK:\n{response_text[:4000]}\n\n"
        f"Extract up to {_MAX_CLAIMS} distinct, atomic factual claims made in the response "
        "above. Each claim must be a short, self-contained statement of fact (not an "
        "opinion, question, or meta-commentary about sources). Skip greetings and "
        "section headings.\n\n"
        f'Return JSON exactly: {{"claims": ["claim 1", "claim 2", ...]}} '
        f"(0 to {_MAX_CLAIMS} claims)."
    )
    try:
        resp = llm.invoke([SystemMessage(content=_CLAIM_SYSTEM), HumanMessage(content=human)])
        claims = _extract_json(resp.content.strip()).get("claims", [])
        return [str(c).strip() for c in claims if str(c).strip()][:_MAX_CLAIMS]
    except Exception as e:
        logger.debug("RAGchecker claim extraction failed: %s", e)
        return []


def _verify_claims(llm: ChatOllama, claims: List[str], context: str) -> List[bool]:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    human = (
        f"SOURCE PASSAGES:\n{context}\n\n"
        f"CLAIMS TO VERIFY:\n{numbered}\n\n"
        "For each numbered claim, decide whether it is directly supported by the source "
        "passages above. Mark it true only if the passages confirm the claim; mark it "
        "false if the passages contradict it, don't mention it, or only partially "
        "support it.\n\n"
        f'Return JSON exactly: {{"verdicts": [true or false, ...]}} with exactly '
        f"{len(claims)} values, in the same order as the claims."
    )
    verdicts: List[bool] = []
    try:
        resp = llm.invoke([SystemMessage(content=_VERIFY_SYSTEM), HumanMessage(content=human)])
        raw_verdicts = _extract_json(resp.content.strip()).get("verdicts", [])
        verdicts = [bool(v) for v in raw_verdicts]
    except Exception as e:
        logger.debug("RAGchecker claim verification failed: %s", e)

    if len(verdicts) < len(claims):
        verdicts += [False] * (len(claims) - len(verdicts))
    return verdicts[:len(claims)]


def check_faithfulness(
    query: str,
    response_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str = "",
    ollama_base_url: str = "",
) -> Dict[str, Any]:
    """Score how many atomic claims in ``response_text`` are backed by ``retrieved_chunks``.

    Returns a dict with ``faithfulness_score`` (0-1, or None if skipped),
    ``checked_claims``, ``supported_claims``, and ``skipped``.
    """
    if not response_text or not retrieved_chunks:
        return dict(_SKIPPED)

    try:
        llm = _make_llm(model_name, ollama_base_url)

        claims = _extract_claims(llm, query, response_text)
        if not claims:
            return dict(_SKIPPED)

        context_parts: List[str] = []
        total_len = 0
        for i, chunk in enumerate(retrieved_chunks[:_MAX_CHUNKS]):
            doc_id = chunk.get("doc_id") or f"chunk_{i}"
            text = str(chunk.get("text", ""))[:1000]
            if not text:
                continue
            part = f"[{doc_id}] {text}"
            if total_len + len(part) > _MAX_CONTEXT_CHARS:
                break
            context_parts.append(part)
            total_len += len(part)

        if not context_parts:
            return dict(_SKIPPED)

        verdicts = _verify_claims(llm, claims, "\n\n".join(context_parts))

        checked = len(claims)
        supported = sum(1 for v in verdicts if v)
        return {
            "faithfulness_score": supported / checked if checked else None,
            "checked_claims": checked,
            "supported_claims": supported,
            "skipped": False,
        }
    except Exception as e:
        logger.warning("RAGchecker faithfulness check error: %s", e)
        return dict(_SKIPPED)
