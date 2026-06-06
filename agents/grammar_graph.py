"""
agents/grammar_graph.py
────────────────────────
Assembles the LangGraph StateGraph for Grammar Proofreading Mode (Mode 6).

Graph topology (linear)
──────────────────────
  START → text_loader → grammar_analysis → polish → style_advisor → grammar_eval → END

Single-turn invocation
──────────────────────
One call to `run_grammar_check()` = one proofreading run (or one revision round).
The graph is compiled fresh each invocation (negligible overhead).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from agents.eval_nodes import grammar_eval_node
from agents.grammar_nodes import (
    grammar_analysis_node,
    polish_node,
    style_advisor_node,
    text_loader_node,
)
from agents.grammar_state import GrammarState

logger = logging.getLogger(__name__)


def build_grammar_graph() -> StateGraph:
    """Construct and compile the Grammar Proofreading LangGraph."""
    graph = StateGraph(GrammarState)

    graph.add_node("text_loader",      text_loader_node)
    graph.add_node("grammar_analysis", grammar_analysis_node)
    graph.add_node("polish",           polish_node)
    graph.add_node("style_advisor",    style_advisor_node)
    graph.add_node("grammar_eval",     grammar_eval_node)

    graph.add_edge(START,              "text_loader")
    graph.add_edge("text_loader",      "grammar_analysis")
    graph.add_edge("grammar_analysis", "polish")
    graph.add_edge("polish",           "style_advisor")
    graph.add_edge("style_advisor",    "grammar_eval")
    graph.add_edge("grammar_eval",     END)

    return graph.compile()


def run_grammar_check(
    initial_state: GrammarState,
    stream_callback=None,
) -> GrammarState:
    """
    Execute one Grammar Proofreading run and return the final state.

    Parameters
    ----------
    initial_state   : Created by grammar_state.create_grammar_state()
    stream_callback : Optional callable(node_name, partial_state) for progress updates

    Returns
    -------
    Final GrammarState — check `polished_text` for the primary output.
    """
    app = build_grammar_graph()
    final_state: Dict[str, Any] = dict(initial_state)

    for step_output in app.stream(initial_state, config={"recursion_limit": 10}):
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

    return GrammarState(**final_state)
