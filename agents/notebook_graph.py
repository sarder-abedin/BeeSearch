"""
agents/notebook_graph.py
Assembles the LangGraph StateGraph for the Research Notebook.

Graph: START → retrieve → answer → save → notebook_eval → END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agents.eval_nodes import notebook_eval_node
from agents.notebook_nodes import answer_node, retrieve_node, save_node
from agents.notebook_state import NotebookState

logger = logging.getLogger(__name__)


def build_notebook_graph() -> StateGraph:
    graph = StateGraph(NotebookState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("save", save_node)
    graph.add_node("notebook_eval", notebook_eval_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "save")
    graph.add_edge("save", "notebook_eval")
    graph.add_edge("notebook_eval", END)

    return graph.compile()


def run_notebook_turn(initial_state: NotebookState, stream_callback=None) -> NotebookState:
    app = build_notebook_graph()
    final_state = dict(initial_state)

    for step_output in app.stream(initial_state, config={"recursion_limit": 10}):
        for node_name, partial in step_output.items():
            final_state.update(partial)
            pct = final_state.get("progress_pct", 0)
            logger.info("✓ %s (%d%%)", node_name.replace("_", " ").title(), pct)
            if stream_callback:
                try:
                    stream_callback(node_name, final_state)
                except Exception as e:
                    logger.warning("Stream callback error: %s", e)

    return NotebookState(**final_state)
