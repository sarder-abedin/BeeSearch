"""
agents/proposal_gpt_graph.py
─────────────────────────────
LangGraph pipeline for ProposalGPT.

  START
    │
  [1] funding_call_analyzer   → Call objectives, criteria, keywords
    │
  [2] research_planner        → Win strategy, SWOT, reviewer perspective
    │
  [3] literature_review_agent → Literature review, state of art, gaps
    │
  [4] proposal_writer         → 14 core proposal sections
    │
  [5] impact_agent            → Impact, dissemination, ethics, sustainability
    │
  [6] budget_agent            → Personnel / equipment / travel / indirect
    │
  [7] compliance_agent        → Compliance score, keyword coverage
    │
  [8] reviewer_agent          → 5 virtual reviewer scores + report
    │
  [9] improvement_agent       → Rewrite weak sections
    │
   END
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from langgraph.graph import END, START, StateGraph

from agents.proposal_gpt_state import ProposalGPTState
from agents.proposal_gpt_nodes import (
    funding_call_analyzer_node,
    research_planner_node,
    literature_review_agent_node,
    proposal_writer_node,
    impact_agent_node,
    budget_agent_node,
    compliance_agent_node,
    reviewer_agent_node,
    improvement_agent_node,
)
from agents.eval_nodes import proposal_gpt_eval_node

logger = logging.getLogger(__name__)


def build_proposal_gpt_pipeline() -> StateGraph:
    """Build and compile the ProposalGPT LangGraph pipeline."""
    graph = StateGraph(ProposalGPTState)

    graph.add_node("funding_call_analyzer",   funding_call_analyzer_node)
    graph.add_node("research_planner",         research_planner_node)
    graph.add_node("literature_review_agent",  literature_review_agent_node)
    graph.add_node("proposal_writer",          proposal_writer_node)
    graph.add_node("impact_agent",             impact_agent_node)
    graph.add_node("budget_agent",             budget_agent_node)
    graph.add_node("compliance_agent",         compliance_agent_node)
    graph.add_node("reviewer_agent",           reviewer_agent_node)
    graph.add_node("improvement_agent",        improvement_agent_node)
    graph.add_node("proposal_gpt_eval",        proposal_gpt_eval_node)

    graph.add_edge(START,                      "funding_call_analyzer")
    graph.add_edge("funding_call_analyzer",    "research_planner")
    graph.add_edge("research_planner",         "literature_review_agent")
    graph.add_edge("literature_review_agent",  "proposal_writer")
    graph.add_edge("proposal_writer",          "impact_agent")
    graph.add_edge("impact_agent",             "budget_agent")
    graph.add_edge("budget_agent",             "compliance_agent")
    graph.add_edge("compliance_agent",         "reviewer_agent")
    graph.add_edge("reviewer_agent",           "improvement_agent")
    graph.add_edge("improvement_agent",        "proposal_gpt_eval")
    graph.add_edge("proposal_gpt_eval",        END)

    return graph.compile()


def run_proposal_gpt(
    initial_state: ProposalGPTState,
    stream_callback: Optional[Callable[[str, dict], None]] = None,
) -> ProposalGPTState:
    """
    Run the full ProposalGPT pipeline.

    Args:
        initial_state: Created by create_proposal_gpt_state()
        stream_callback: Optional (node_name, partial_state) → None; called
                         after each agent completes for progress reporting.

    Returns:
        Final ProposalGPTState with all sections populated.
    """
    app = build_proposal_gpt_pipeline()
    final_state = dict(initial_state)

    try:
        for step_output in app.stream(initial_state, config={"recursion_limit": 25}):
            for node_name, partial in step_output.items():
                final_state.update(partial)
                if stream_callback:
                    try:
                        stream_callback(node_name, dict(final_state))
                    except Exception as cb_exc:
                        logger.warning("Stream callback error: %s", cb_exc)
    except Exception as exc:
        logger.error("ProposalGPT pipeline failed: %s", exc)
        errors = list(final_state.get("errors", []))
        errors.append(f"Pipeline error: {exc}")
        final_state["errors"] = errors

    return ProposalGPTState(**final_state)
