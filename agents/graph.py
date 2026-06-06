"""
agents/graph.py
───────────────
Assembles the LangGraph StateGraph from individual node functions.

TUTORIAL NOTE — How LangGraph works
─────────────────────────────────────
LangGraph models an agent as a directed graph:

  • Nodes  = functions that transform state
  • Edges  = transitions between nodes
  • State  = a TypedDict shared across all nodes

A ConditionalEdge inspects the state after a node completes and
routes to different next nodes based on the result. This is how we
implement the "should we run web search?" decision.

          START
            │
            ▼
  [document_ingestion]
            │
            ▼
   [query_generation]
            │
            ▼
   [academic_search]
            │
            ▼
      [web_search] ◄─── conditional (only if include_web_search=True)
            │
            ▼
  [document_analysis]
            │
            ▼
 [reference_compilation]
            │
            ▼
  [report_generation]
            │
           END
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from agents.eval_nodes import research_eval_node
from agents.nodes import (
    academic_search_node,
    document_analysis_node,
    document_ingestion_node,
    query_generation_node,
    reference_compilation_node,
    report_generation_node,
    web_search_node,
)
from agents.state import ResearchState

logger = logging.getLogger(__name__)


def _route_after_academic_search(state: ResearchState) -> str:
    """
    Conditional routing: if web search is requested, go there first.
    Otherwise jump straight to document analysis.
    """
    if state.get("include_web_search", False):
        return "web_search"
    return "document_analysis"


def build_research_graph() -> StateGraph:
    """
    Construct and compile the full research agent graph.

    Returns a compiled LangGraph that can be invoked with a ResearchState.
    """
    graph = StateGraph(ResearchState)

    # ── Register nodes ─────────────────────────────────────────
    graph.add_node("document_ingestion", document_ingestion_node)
    graph.add_node("query_generation", query_generation_node)
    graph.add_node("academic_search", academic_search_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("document_analysis", document_analysis_node)
    graph.add_node("reference_compilation", reference_compilation_node)
    graph.add_node("report_generation", report_generation_node)
    graph.add_node("research_eval", research_eval_node)

    # ── Linear edges ──────────────────────────────────────────
    graph.add_edge(START, "document_ingestion")
    graph.add_edge("document_ingestion", "query_generation")
    graph.add_edge("query_generation", "academic_search")

    # ── Conditional edge after academic search ────────────────
    graph.add_conditional_edges(
        "academic_search",
        _route_after_academic_search,
        {
            "web_search": "web_search",
            "document_analysis": "document_analysis",
        },
    )

    graph.add_edge("web_search", "document_analysis")
    graph.add_edge("document_analysis", "reference_compilation")
    graph.add_edge("reference_compilation", "report_generation")
    graph.add_edge("report_generation", "research_eval")
    graph.add_edge("research_eval", END)

    return graph.compile()


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_research(
    initial_state: ResearchState,
    stream_callback=None,
) -> ResearchState:
    """
    Run the full research workflow and return the final state.

    Parameters
    ----------
    initial_state   : created by agents.state.create_initial_state()
    stream_callback : optional callable(step_name, partial_state) for live UI updates

    TUTORIAL NOTE
    ─────────────
    LangGraph supports both .invoke() (blocking) and .stream() (event-driven).
    We use .stream() here so we can call the callback after each node,
    enabling real-time progress updates in the Streamlit UI.
    """
    app = build_research_graph()
    final_state = dict(initial_state)

    for step_output in app.stream(initial_state, config={"recursion_limit": 25}):
        # step_output = {"node_name": {partial_state_updates}}
        for node_name, partial in step_output.items():
            final_state.update(partial)
            step_label = node_name.replace("_", " ").title()
            pct = final_state.get("progress_pct", 0)
            logger.info("✓ %s (%d%%)", step_label, pct)

            if stream_callback:
                try:
                    stream_callback(node_name, final_state)
                except Exception as e:
                    logger.warning("Stream callback error: %s", e)

    return ResearchState(**final_state)
