"""
agents/state.py
───────────────
Defines the shared state object that flows through every node of the
LangGraph agent workflow.

TUTORIAL NOTE — What is LangGraph state?
─────────────────────────────────────────
LangGraph is a graph-based orchestration framework where each "node" is a
Python function that receives the current state, does some work, and
returns a dict of state updates.

The graph is essentially a directed state machine:

  START
    │
    ▼
  [document_ingestion]──►[query_generation]──►[academic_search]
                                                      │
                                                      ▼
                                              [web_search] (optional)
                                                      │
                                                      ▼
                                          [document_analysis_rag]
                                                      │
                                                      ▼
                                          [reference_compilation]
                                                      │
                                                      ▼
                                            [report_generation]
                                                      │
                                                    END

Every node reads from and writes to this ResearchState TypedDict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from tools.document_tools import ProcessedDocument
from tools.search_tools import Paper, WebResult


class ResearchState(TypedDict, total=False):
    # ── User Inputs ───────────────────────────────────────────
    goal: str                              # e.g. "Summarise the climate impacts"
    mode: str                              # "document" | "search" | "hybrid"
    uploaded_docs: List[ProcessedDocument] # Raw parsed documents
    include_web_search: bool               # Whether to supplement with web results

    # ── Intermediate Results ──────────────────────────────────
    search_queries: List[str]              # LLM-generated sub-queries
    doc_context: List[Dict[str, Any]]      # Top-k RAG chunks from vector store
    academic_papers: List[Paper]           # Papers from arXiv + Semantic Scholar
    web_results: List[WebResult]           # DuckDuckGo results (optional)

    # ── Synthesised Outputs ───────────────────────────────────
    references: List[Dict[str, Any]]       # Compiled, deduplicated bibliography
    key_findings: List[str]                # Bullet-point findings from the LLM
    analysis: str                          # Detailed analysis text
    report: str                            # Final formatted report (Markdown)

    # ── Multi-document outputs ────────────────────────────────
    per_doc_analyses: Dict[str, str]       # {filename: focused analysis text}
    multi_doc_synthesis: str               # Cross-document synthesis paragraph

    # ── Workflow Control ──────────────────────────────────────
    current_step: str                      # Name of the running node
    completed_steps: List[str]             # Audit trail of finished nodes
    errors: List[str]                      # Non-fatal errors accumulated
    progress_pct: int                      # 0–100 for UI progress bars

    # ── Hybrid RAG ────────────────────────────────────────────
    embed_model: str                       # Ollama embedding model (e.g. "nomic-embed-text")
    retrieved_chunks: List[Dict[str, Any]] # Chunks retrieved by hybrid search (for UI)
    rag_fallback: bool                     # True if vectorless fallback was used

    # ── Writing Style Profile ─────────────────────────────────
    style_profile: Optional[Dict[str, Any]]  # Loaded style profile dict (or None)

    # ── Socratic Clarifications ───────────────────────────────
    clarifications: Dict[str, str]           # {question_key: answer} from pre-run form

    # ── Quality Evaluation ────────────────────────────────────
    eval_result: Dict[str, Any]            # {goal_alignment, evidence_quality, clarity, overall, summary}
    feedback_history: List[Dict[str, Any]]   # [{round, feedback, previous_output, timestamp}]
    refinement_round: int                    # 0 = original, 1–3 = refined
    rag_reflection_info: List[Dict[str, Any]]  # one entry per query: {query, cycles, total_retrieved, ...}

    # ── Metadata ──────────────────────────────────────────────
    session_id: str                        # Unique ID for this run
    model_name: str                        # Ollama model used
    num_ctx: int                           # LLM context window (tokens)


def create_initial_state(
    goal: str,
    uploaded_docs: Optional[List[ProcessedDocument]] = None,
    mode: str = "hybrid",
    include_web_search: bool = False,
    session_id: str = "",
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
    embed_model: str = "nomic-embed-text",
    style_profile: Optional[Dict] = None,
    clarifications: Optional[Dict[str, str]] = None,
) -> ResearchState:
    """
    Factory function to create a clean starting state.

    Parameters
    ----------
    goal             : The user's research question or instruction
    uploaded_docs    : Pre-processed documents (may be empty for search-only mode)
    mode             : "document" (RAG only) | "search" (web/academic only)
                       | "hybrid" (documents + external search)
    include_web_search: Whether to add DuckDuckGo results
    """
    import uuid

    return ResearchState(
        goal=goal,
        mode=mode,
        uploaded_docs=uploaded_docs or [],
        include_web_search=include_web_search,
        search_queries=[],
        doc_context=[],
        academic_papers=[],
        web_results=[],
        references=[],
        key_findings=[],
        analysis="",
        report="",
        per_doc_analyses={},
        multi_doc_synthesis="",
        embed_model=embed_model,
        retrieved_chunks=[],
        rag_fallback=False,
        current_step="start",
        completed_steps=[],
        errors=[],
        progress_pct=0,
        session_id=session_id or str(uuid.uuid4())[:8],
        model_name=model_name,
        num_ctx=num_ctx,
        style_profile=style_profile,
        clarifications=clarifications or {},
        eval_result={},
        feedback_history=[],
        refinement_round=0,
        rag_reflection_info=[],
    )
