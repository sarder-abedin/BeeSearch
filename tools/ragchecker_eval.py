"""
tools/ragchecker_eval.py
─────────────────────────
RAGchecker-based faithfulness evaluation (optional — safe to call even when not installed).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import ragchecker  # noqa: F401
        return True
    except Exception:
        return False


def check_faithfulness(
    query: str,
    response_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str = "llama3.1:8b",
    ollama_base_url: str = "http://localhost:11434",
) -> Dict[str, Any]:
    _skipped: Dict[str, Any] = {
        "faithfulness_score": None,
        "checked_claims": 0,
        "supported_claims": 0,
        "unsupported_claims": 0,
        "skipped": True,
        "error": None,
    }

    if not retrieved_chunks or not response_text or not query:
        _skipped["error"] = "Missing query, response_text, or retrieved_chunks"
        return _skipped

    try:
        from ragchecker import RAGResults, RAGChecker
        from ragchecker.metrics import faithfulness
    except ImportError as exc:
        result = dict(_skipped)
        result["error"] = f"ragchecker not installed: {exc}"
        return result
    except Exception as exc:
        result = dict(_skipped)
        result["error"] = f"ragchecker import error: {exc}"
        return result

    try:
        contexts = [c.get("text", "") for c in retrieved_chunks if c.get("text", "").strip()]
        if not contexts:
            result = dict(_skipped)
            result["error"] = "No non-empty chunk texts"
            return result

        rag_entry = {
            "query": query,
            "gt_answer": "",
            "response": response_text,
            "retrieved_context": [{"text": c} for c in contexts],
        }
        rag_results = RAGResults.from_data(inputs=[rag_entry], gt_field="gt_answer")

        ollama_model = f"ollama/{model_name}"
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

        scores = []
        total_checked = 0
        total_supported = 0
        for item in rag_results.results:
            item_faithfulness = getattr(item, "faithfulness", None)
            if item_faithfulness is not None:
                scores.append(float(item_faithfulness))
            for claim_obj in (getattr(item, "response_claims", None) or []):
                total_checked += 1
                if getattr(claim_obj, "faithfulness_label", None) is True:
                    total_supported += 1

        if not scores:
            agg_faith = getattr(rag_results, "metrics", {}).get("faithfulness", None)
            if agg_faith is not None:
                scores = [float(agg_faith)]

        if not scores:
            result = dict(_skipped)
            result["error"] = "RAGchecker returned no faithfulness scores"
            return result

        return {
            "faithfulness_score": round(sum(scores) / len(scores), 4),
            "checked_claims": total_checked,
            "supported_claims": total_supported,
            "unsupported_claims": max(0, total_checked - total_supported),
            "skipped": False,
            "error": None,
        }
    except Exception as exc:
        logger.warning("RAGchecker faithfulness check failed: %s", exc, exc_info=True)
        result = dict(_skipped)
        result["error"] = str(exc)
        return result
