"""
agents/story_graph.py
──────────────────────
Assembles the LangGraph StateGraph for the Research Partner (Storytelling).

Graph structure (linear — one path only)
─────────────────────────────────────────

  START
    │
    ▼
  [context_loader]   ← load conversation history + doc context from memory
    │
    ▼
  [source_router]    ← LLM scores doc coverage (0-10); fetches online results if < 6
    │
    ▼
  [storyteller]      ← generate explanation + suggested questions via LLM
    │
    ▼
  [memory_saver]     ← persist user + assistant turns back to JSON file
    │
   END

Single-turn invocation model
─────────────────────────────
One call to `run_story_turn()` = one user message → one assistant response.
The graph is compiled fresh on each invocation (negligible overhead).
Conversation continuity lives in StorytellerMemory (JSON files), not in
the graph state — consistent with how `run_proposal()` works.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from agents.eval_nodes import story_eval_node
from agents.story_nodes import context_loader_node, memory_saver_node, source_router_node, storyteller_node
from agents.story_state import StoryState

logger = logging.getLogger(__name__)


def build_story_graph() -> StateGraph:
    """Construct and compile the Research Partner graph."""
    graph = StateGraph(StoryState)

    graph.add_node("context_loader", context_loader_node)
    graph.add_node("source_router",  source_router_node)
    graph.add_node("storyteller",    storyteller_node)
    graph.add_node("memory_saver",   memory_saver_node)
    graph.add_node("story_eval",     story_eval_node)

    graph.add_edge(START,            "context_loader")
    graph.add_edge("context_loader", "source_router")
    graph.add_edge("source_router",  "storyteller")
    graph.add_edge("storyteller",    "memory_saver")
    graph.add_edge("memory_saver",   "story_eval")
    graph.add_edge("story_eval",     END)

    return graph.compile()


def run_story_turn(
    initial_state: StoryState,
    stream_callback=None,
) -> StoryState:
    """
    Execute one conversational turn and return the final state.

    Parameters
    ----------
    initial_state   : Created by story_state.create_story_state()
    stream_callback : Optional callable(node_name, partial_state) for progress

    Returns
    -------
    Final StoryState with assistant_response and suggested_questions populated.
    """
    app = build_story_graph()
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

    return StoryState(**final_state)
