"""
tools/ragchecker_eval.py
─────────────────────────
RAGchecker-based faithfulness evaluation.

Provides check_faithfulness() which runs RAGchecker's claim-level faithfulness
metric using a local Ollama model as both extractor and checker.

This module is entirely optional — all public functions are safe to call even
when ragchecker is not installed. is_available() can be used to gate usage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Availability check ─────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if ragchecker can be imported without errors."""
    try:
        import ragchecker  # noqa: F401
        return True
    except Exception:
        return False


# ── Main faithfulness checker ──────────────────────────────────────────────────

def check_faithfulness(
    query: str,
    response_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str = "llama3.1:8b",
    ollama_base_url: str = "http://localhost:11434",
) -> Dict[str, Any]:
    """Run RAGchecker faithfulness evaluation using Ollama as the checker model.

    Parameters
    ----------
    query : str
        The user's original query / research goal.
    response_text : str
        The generated response whose claims will be checked.
    retrieved_chunks : list of dict
        Each dict must have "text" (str) and "doc_id" (str) keys.
    model_name : str
        Ollama model name (e.g. "llama3.1:8b").
    ollama_base_url : str
        Base URL for the Ollama API (e.g. "http://localhost:11434").

    Returns
    -------
    dict with keys:
        faithfulness_score  : float (0.0–1.0) or None when skipped
        checked_claims      : int
        supported_claims    : int
        unsupported_claims  : int
        skipped             : bool
        error               : str or None
    """
    _skipped_result: Dict[str, Any] = {
        "faithfulness_score": None,
        "checked_claims": 0,
        "supported_claims": 0,
        "unsupported_claims": 0,
        "skipped": True,
        "error": None,
    }

    if not retrieved_chunks or not response_text or not query:
        _skipped_result["error"] = "Missing query, response_text, or retrieved_chunks"
        return _skipped_result

    try:
        from ragchecker import RAGResults, RAGChecker
        from ragchecker.metrics import faithfulness
    except ImportError as exc:
        result = dict(_skipped_result)
        result["error"] = f"ragchecker not installed: {exc}"
        return result
    except Exception as exc:
        result = dict(_skipped_result)
        result["error"] = f"ragchecker import error: {exc}"
        return result

    try:
        # Build the contexts list — RAGchecker expects a list of strings per item
        contexts = [
            chunk.get("text", "")
            for chunk in retrieved_chunks
            if chunk.get("text", "").strip()
        ]

        if not contexts:
            result = dict(_skipped_result)
            result["error"] = "No non-empty chunk texts found in retrieved_chunks"
            return result

        # Build a single RAGResults entry
        rag_entry = {
            "query": query,
            "gt_answer": "",          # ground-truth not required for faithfulness
            "response": response_text,
            "retrieved_context": [{"text": c} for c in contexts],
        }

        rag_results = RAGResults.from_data(
            inputs=[rag_entry],
            gt_field="gt_answer",
        )

        # Configure the checker to use Ollama via litellm's ollama/ prefix
        ollama_model = f"ollama/{model_name}"
        # Strip trailing slash from base URL for clean path construction
        api_base = ollama_base_url.rstrip("/")

        checker = RAGChecker(
            extractor_name=ollama_model,
            checker_name=ollama_model,
            extractor_api_base=api_base,
            checker_api_base=api_base,
            batch_size_extractor=1,
            batch_size_checker=1,
        )

        checker.evaluate(rag_results, metrics=[faithfulness])

        # Extract per-item faithfulness data from the results
        scores = []
        total_checked = 0
        total_supported = 0

        for item in rag_results.results:
            # RAGchecker stores per-claim info on each RAGResult item
            item_faithfulness = getattr(item, "faithfulness", None)
            if item_faithfulness is not None:
                scores.append(float(item_faithfulness))

            # Count claims from checker annotations if available
            response_claims = getattr(item, "response_claims", None) or []
            for claim_obj in response_claims:
                total_checked += 1
                # claim_obj.faithfulness_label is True/False/None
                lbl = getattr(claim_obj, "faithfulness_label", None)
                if lbl is True:
                    total_supported += 1

        if not scores:
            # Fall back to the aggregate metric from rag_results
            agg = getattr(rag_results, "metrics", {})
            agg_faith = agg.get("faithfulness", None)
            if agg_faith is not None:
                scores = [float(agg_faith)]

        if not scores:
            result = dict(_skipped_result)
            result["error"] = "RAGchecker returned no faithfulness scores"
            return result

        avg_score = sum(scores) / len(scores)
        unsupported = total_checked - total_supported

        return {
            "faithfulness_score": round(avg_score, 4),
            "checked_claims": total_checked,
            "supported_claims": total_supported,
            "unsupported_claims": max(0, unsupported),
            "skipped": False,
            "error": None,
        }

    except Exception as exc:
        logger.warning("RAGchecker faithfulness check failed: %s", exc, exc_info=True)
        result = dict(_skipped_result)
        result["error"] = str(exc)
        return result
