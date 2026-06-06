"""
agents/notebook_pipeline_graph.py
────────────────────────────────────
Assembles the 7-agent LangGraph pipeline for Mode 8 Research Notebook.

Graph structure (linear — all agents communicate through shared state)
────────────────────────────────────────────────────────────────────────

  START
    │
    ▼
  [ingest]              Agent 1 — Document Ingestion
    │
    ▼
  [summarize]           Agent 2 — Summarization
    │
    ▼
  [retrieve]            Agent 3 — Retrieval (Hybrid RAG)
    │
    ▼
  [verify_citations]    Agent 4 — Citation Verification
    │
    ▼
  [build_kg]            Agent 5 — Knowledge Graph Construction
    │
    ▼
  [generate_study_guide] Agent 6 — Study Guide Generation
    │
    ▼
  [generate_podcast]    Agent 7 — Podcast Script Generation
    │
   END

Invocation model
────────────────
One call to `run_notebook_pipeline()` runs all 7 agents in sequence
and returns the final `NotebookPipelineState` with all outputs populated.
Use the `stream_callback` parameter for live progress updates in the UI.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Any, Optional

from langgraph.graph import END, START, StateGraph

from agents.notebook_pipeline_state import NotebookPipelineState, create_pipeline_state
from agents.eval_nodes import notebook_pipeline_eval_node
from agents.notebook_pipeline_nodes import (
    ingestion_node,
    summarization_node,
    retrieval_node,
    citation_verification_node,
    knowledge_graph_node,
    study_guide_node,
    podcast_script_node,
)

logger = logging.getLogger(__name__)


def build_notebook_pipeline():
    """Construct and compile the 7-agent Research Notebook pipeline."""
    graph = StateGraph(NotebookPipelineState)

    graph.add_node("ingest", ingestion_node)
    graph.add_node("summarize", summarization_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("verify_citations", citation_verification_node)
    graph.add_node("build_kg", knowledge_graph_node)
    graph.add_node("generate_study_guide", study_guide_node)
    graph.add_node("generate_podcast", podcast_script_node)
    graph.add_node("notebook_pipeline_eval", notebook_pipeline_eval_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "summarize")
    graph.add_edge("summarize", "retrieve")
    graph.add_edge("retrieve", "verify_citations")
    graph.add_edge("verify_citations", "build_kg")
    graph.add_edge("build_kg", "generate_study_guide")
    graph.add_edge("generate_study_guide", "generate_podcast")
    graph.add_edge("generate_podcast", "notebook_pipeline_eval")
    graph.add_edge("notebook_pipeline_eval", END)

    return graph.compile()


def run_notebook_pipeline(
    initial_state: NotebookPipelineState,
    stream_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> NotebookPipelineState:
    """
    Execute the full 7-agent pipeline and return the final state.

    Parameters
    ----------
    initial_state   : created by create_pipeline_state()
    stream_callback : optional callable(node_name, partial_state) called after
                      each agent completes — use for live progress updates in UI

    Returns
    -------
    Final NotebookPipelineState with all seven agents' outputs populated.
    """
    app = build_notebook_pipeline()
    final_state: Dict[str, Any] = dict(initial_state)

    _AGENT_LABELS = {
        "ingest": "Agent 1 — Document Ingestion",
        "summarize": "Agent 2 — Summarization",
        "retrieve": "Agent 3 — Retrieval",
        "verify_citations": "Agent 4 — Citation Verification",
        "build_kg": "Agent 5 — Knowledge Graph",
        "generate_study_guide": "Agent 6 — Study Guide",
        "generate_podcast": "Agent 7 — Podcast Script",
    }

    for step_output in app.stream(initial_state, config={"recursion_limit": 20}):
        for node_name, partial in step_output.items():
            final_state.update(partial)
            pct = final_state.get("progress_pct", 0)
            label = _AGENT_LABELS.get(node_name, node_name)
            logger.info("✓ %s (%d%%)", label, pct)
            if stream_callback:
                try:
                    stream_callback(node_name, final_state)
                except Exception as exc:
                    logger.warning("Pipeline stream callback error: %s", exc)

    return NotebookPipelineState(**final_state)
