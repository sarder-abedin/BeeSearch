"""
agents/wisdom_graph.py
───────────────────────
Assembles the LangGraph StateGraph for Wisdom Mode (Mode 6).

Graph topology
──────────────
  START → context_loader → [route_from_context]
      ├─ "wisdom_followup" → wisdom_followup → memory_saver → END
      └─ "clarification"  → clarification → [route_after_clarification]
              ├─ "knowledge_search" → knowledge_search → wisdom_synthesis
              │                    → wisdom_validator → memory_saver → END
              └─ "memory_saver"    → memory_saver → END   (still asking Qs)

Single-turn invocation
──────────────────────
One call to `run_wisdom_turn()` = one user message → one assistant response.
The graph is compiled fresh each invocation (negligible overhead).
State continuity lives in WisdomMemory (JSON files), not in the graph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from agents.eval_nodes import wisdom_eval_node
from agents.wisdom_nodes import (
    clarification_node,
    context_loader_node,
    knowledge_search_node,
    memory_saver_node,
    route_after_clarification,
    route_from_context,
    wisdom_followup_node,
    wisdom_synthesis_node,
    wisdom_validator_node,
)
from agents.wisdom_state import WisdomState

logger = logging.getLogger(__name__)


def build_wisdom_graph() -> StateGraph:
    """Construct and compile the Wisdom Mode LangGraph."""
    graph = StateGraph(WisdomState)

    graph.add_node("context_loader",    context_loader_node)
    graph.add_node("clarification",     clarification_node)
    graph.add_node("knowledge_search",  knowledge_search_node)
    graph.add_node("wisdom_synthesis",  wisdom_synthesis_node)
    graph.add_node("wisdom_validator",  wisdom_validator_node)
    graph.add_node("wisdom_followup",   wisdom_followup_node)
    graph.add_node("memory_saver",      memory_saver_node)
    graph.add_node("wisdom_eval",       wisdom_eval_node)

    # Entry point
    graph.add_edge(START, "context_loader")

    # After context loader: route based on session phase
    graph.add_conditional_edges(
        "context_loader",
        route_from_context,
        {
            "wisdom_followup": "wisdom_followup",
            "clarification":   "clarification",
        },
    )

    # After clarification: ask another question OR proceed to generate
    graph.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "knowledge_search": "knowledge_search",
            "memory_saver":     "memory_saver",
        },
    )

    # Wisdom generation pipeline (linear)
    graph.add_edge("knowledge_search", "wisdom_synthesis")
    graph.add_edge("wisdom_synthesis", "wisdom_validator")
    graph.add_edge("wisdom_validator", "memory_saver")

    # Follow-up path
    graph.add_edge("wisdom_followup", "memory_saver")

    # Eval + final sink
    graph.add_edge("memory_saver", "wisdom_eval")
    graph.add_edge("wisdom_eval",  END)

    return graph.compile()


def run_wisdom_turn(
    initial_state: WisdomState,
    stream_callback=None,
) -> WisdomState:
    """
    Execute one Wisdom Mode turn and return the final state.

    Parameters
    ----------
    initial_state   : Created by wisdom_state.create_wisdom_state()
    stream_callback : Optional callable(node_name, partial_state) for progress

    Returns
    -------
    Final WisdomState — check `phase` and `assistant_response`.
    When `phase == "done"` and `deep_understanding` is non-empty, the full
    wisdom output is available in the state.
    """
    app = build_wisdom_graph()
    final_state = dict(initial_state)

    for step_output in app.stream(initial_state, config={"recursion_limit": 15}):
        for node_name, partial in step_output.items():
            final_state.update(partial)
            pct = final_state.get("progress_pct", 0)
            label = node_name.replace("_", " ").title()
            logger.info("✓ %s (%d%%)", label, pct)

            if stream_callback:
                try:
                    stream_callback(node_name, final_state)
                except Exception as e:
                    logger.warning("Stream callback error: %s", e)

    return WisdomState(**final_state)
