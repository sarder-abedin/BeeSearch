"""
agents/state.py
───────────────
State factory for the Research Report workflow (agents/graph.py).
"""

from __future__ import annotations


def create_initial_state(
    goal: str,
    uploaded_docs: list,
    mode: str,
    include_web_search: bool,
    model_name: str,
    num_ctx: int,
    embed_model: str,
) -> dict:
    """Return the initial state dict for a Research Report run.

    Parameters
    ----------
    goal:              The user's research question or goal.
    uploaded_docs:     List of ProcessedDocument objects from the notebook.
    mode:              "document" (notebook only), "hybrid" (notebook + academic),
                       or "search" (academic only — no notebook sources).
    include_web_search: Also query the web via DuckDuckGo when True.
    model_name:        Ollama model identifier.
    num_ctx:           LLM context window size.
    embed_model:       Embedding model identifier (reserved for future use).
    """
    return {
        # inputs
        "goal": goal,
        "uploaded_docs": uploaded_docs,
        "mode": mode,
        "include_web_search": include_web_search,
        "model_name": model_name,
        "num_ctx": num_ctx,
        "embed_model": embed_model,
        # intermediate (populated by graph steps)
        "search_queries": [],
        "academic_papers": [],
        "web_results": [],
        # outputs
        "report": "",
        "key_findings": [],
        "references": [],
        "eval_result": {},
        "errors": [],
        "progress_pct": 0,
    }
