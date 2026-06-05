"""agents/systematic_review_graph.py — LangGraph for Mode 7 Systematic Review"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph

from agents.systematic_review_nodes import (
    evidence_extraction_node,
    literature_search_node,
    query_generation_node,
    screening_node,
    sr_eval_node,
    synthesis_node,
)
from agents.systematic_review_state import SystematicReviewState

logger = logging.getLogger(__name__)


def build_systematic_review_graph() -> StateGraph:
    graph = StateGraph(SystematicReviewState)
    graph.add_node("query_generation", query_generation_node)
    graph.add_node("literature_search", literature_search_node)
    graph.add_node("screening", screening_node)
    graph.add_node("evidence_extraction", evidence_extraction_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("sr_eval", sr_eval_node)

    graph.add_edge(START, "query_generation")
    graph.add_edge("query_generation", "literature_search")
    graph.add_edge("literature_search", "screening")
    graph.add_edge("screening", "evidence_extraction")
    graph.add_edge("evidence_extraction", "synthesis")
    graph.add_edge("synthesis", "sr_eval")
    graph.add_edge("sr_eval", END)

    return graph.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_systematic_review_graph()
    return _graph


def run_systematic_review(
    initial_state: SystematicReviewState,
    stream_callback: Optional[Callable[[str, Dict], None]] = None,
) -> SystematicReviewState:
    """Run the full systematic review graph and return the final state."""
    graph = _get_graph()
    final_state = dict(initial_state)

    for chunk in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            if isinstance(node_output, dict):
                final_state.update(node_output)
            if stream_callback:
                try:
                    stream_callback(node_name, final_state)
                except Exception as e:
                    logger.warning("Stream callback error at node '%s': %s", node_name, e)

    return final_state
