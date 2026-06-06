"""
agents/self_reflective_rag.py
──────────────────────────────
Corrective/Adaptive Self-Reflective RAG for all 5 research modes.

After every retrieval call, a single batched LLM call grades all retrieved
items at once. Irrelevant items are filtered out; if too few pass, a
reformulated query fires a second retrieval cycle (chunk-based modes only).

Grading contexts
────────────────
  Modes 1 & 5 (document chunks)    → grade_chunks()
  Modes 2, 3, 4 (academic papers)  → grade_papers()

Guarantees
──────────
  - Never raises; all failures fall back to returning the original items.
  - Max 2 reflection cycles for chunk-based retrieval.
  - Min 3 relevant items before skipping re-query.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()


def _reflection_llm(model_name: str, num_ctx: int, temperature: float, num_predict: int):
    """Private LLM factory — mirrors eval_nodes._eval_llm but caps ctx at 4096."""
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=min(num_ctx or cfg.num_ctx, 4096),
        sync_client_kwargs={"timeout": httpx.Timeout(60.0)},
    )


def grade_chunks(
    chunks: List[Dict[str, Any]],
    query: str,
    model_name: str = "",
    num_ctx: int = 4096,
) -> List[bool]:
    """
    Grade document chunks for relevance to the query in a single batched LLM call.

    Returns a list of bool (one per chunk). On any failure or length mismatch,
    returns all True (treat everything as relevant — safe fallback).
    """
    if not chunks:
        return []

    try:
        llm = _reflection_llm(model_name, num_ctx, temperature=0.0, num_predict=100)
        numbered = "\n".join(
            f"[{i+1}] {c.get('text', '')[:400]}" for i, c in enumerate(chunks)
        )
        human = (
            f"QUERY: {query}\n\n"
            f"CHUNKS:\n{numbered}\n\n"
            'Return ONLY a JSON object: {"grades": [true, false, ...]} — '
            "one bool per chunk, true = relevant to the query."
        )
        system = (
            "You are a relevance grader. Respond ONLY with a JSON object. "
            "No preamble, no explanation, no markdown fences."
        )
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = response.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return [True] * len(chunks)
        data = json.loads(match.group(0))
        grades = [bool(g) for g in data["grades"]]
        if len(grades) != len(chunks):
            return [True] * len(chunks)
        return grades
    except Exception as exc:
        logger.warning("grade_chunks failed (%s) — treating all as relevant", exc)
        return [True] * len(chunks)


def grade_papers(
    papers: List[Dict[str, Any]],
    query: str,
    model_name: str = "",
    num_ctx: int = 4096,
) -> List[bool]:
    """
    Grade academic paper dicts for relevance to the query in a single batched LLM call.
    Papers must have at least 'title' and 'abstract' keys.

    Returns a list of bool (one per paper). On any failure or length mismatch,
    returns all True (treat everything as relevant — safe fallback).
    """
    if not papers:
        return []

    try:
        llm = _reflection_llm(model_name, num_ctx, temperature=0.0, num_predict=100)
        numbered = "\n".join(
            f"[{i+1}] Title: {p.get('title', '')}\nAbstract: {(p.get('abstract', '') or '')[:300]}"
            for i, p in enumerate(papers)
        )
        human = (
            f"QUERY: {query}\n\n"
            f"PAPERS:\n{numbered}\n\n"
            'Return ONLY a JSON object: {"grades": [true, false, ...]} — '
            "one bool per paper, true = relevant to the query."
        )
        system = (
            "You are a relevance grader. Respond ONLY with a JSON object. "
            "No preamble, no explanation, no markdown fences."
        )
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = response.content.strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return [True] * len(papers)
        data = json.loads(match.group(0))
        grades = [bool(g) for g in data["grades"]]
        if len(grades) != len(papers):
            return [True] * len(papers)
        return grades
    except Exception as exc:
        logger.warning("grade_papers failed (%s) — treating all as relevant", exc)
        return [True] * len(papers)


def rewrite_query(
    original_query: str,
    model_name: str = "",
    num_ctx: int = 4096,
) -> str:
    """
    Reformulate a query that returned too few relevant chunks.
    Returns the first non-empty line of the LLM response, or original_query on any failure.
    """
    try:
        llm = _reflection_llm(model_name, num_ctx, temperature=0.3, num_predict=100)
        system = (
            "You are a search query optimisation expert. "
            "Rewrite the given query to improve academic document retrieval. "
            "Return only the rewritten query — no explanation, no quotes."
        )
        human = f"Rewrite this query for better document retrieval:\n{original_query}"
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
        return lines[0] if lines else original_query
    except Exception as exc:
        logger.warning("rewrite_query failed (%s) — using original", exc)
        return original_query


def self_reflective_retrieve(
    store,
    query: str,
    top_k: int,
    model_name: str = "",
    num_ctx: int = 4096,
    max_cycles: int = 2,
    min_relevant: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Orchestrate self-reflective retrieval for chunk-based modes (1 and 5).

    Returns (filtered_chunks, metadata) where metadata contains:
      cycles, total_retrieved, total_relevant, rewritten_queries, grading_skipped

    Never raises — any failure returns the original retrieved chunks.
    """
    metadata: Dict[str, Any] = {
        "cycles": 0,
        "total_retrieved": 0,
        "total_relevant": 0,
        "rewritten_queries": [],
        "grading_skipped": False,
    }
    cycle1_chunks: List[Dict[str, Any]] = []

    # ── Cycle 1 ──────────────────────────────────────────────────────────────
    try:
        cycle1_chunks = store.search_hybrid(query, k=top_k)
    except Exception as exc:
        logger.warning("self_reflective_retrieve: search_hybrid failed (%s)", exc)
        return [], metadata

    metadata["cycles"] = 1
    metadata["total_retrieved"] = len(cycle1_chunks)

    if not cycle1_chunks:
        return [], metadata

    grades = grade_chunks(cycle1_chunks, query, model_name=model_name, num_ctx=num_ctx)

    # Silent LLM failure detection: all grades True on a non-trivial list
    if all(grades) and len(grades) > 1:
        logger.debug("[SelfReflect] All grades True — assuming grading skipped")
        metadata["grading_skipped"] = True
        metadata["total_relevant"] = len(cycle1_chunks)
        return cycle1_chunks, metadata

    relevant = [c for c, g in zip(cycle1_chunks, grades) if g]
    logger.info(
        "[SelfReflect] cycle=1 query=%r retrieved=%d new, %d pass grading, %d total relevant",
        query[:60], len(cycle1_chunks), len(relevant), len(relevant),
    )

    if len(relevant) >= min_relevant or max_cycles < 2:
        if not relevant:
            relevant = cycle1_chunks  # safety: never return empty
        metadata["total_relevant"] = len(relevant)
        return relevant, metadata

    # ── Cycle 2 ──────────────────────────────────────────────────────────────
    metadata["cycles"] = 2
    rewritten = rewrite_query(query, model_name=model_name, num_ctx=num_ctx)
    metadata["rewritten_queries"].append(rewritten)

    cycle2_chunks: List[Dict[str, Any]] = []
    try:
        cycle2_chunks = store.search_hybrid(rewritten, k=top_k)
    except Exception as exc:
        logger.warning("self_reflective_retrieve: cycle-2 search_hybrid failed (%s)", exc)

    # Deduplicate new chunks against cycle 1 by chunk_id
    seen_ids = {c.get("chunk_id") for c in cycle1_chunks}
    new_chunks = [c for c in cycle2_chunks if c.get("chunk_id") not in seen_ids]
    metadata["total_retrieved"] += len(new_chunks)

    if new_chunks:
        new_grades = grade_chunks(new_chunks, query, model_name=model_name, num_ctx=num_ctx)
        new_relevant = [c for c, g in zip(new_chunks, new_grades) if g]
    else:
        new_relevant = []

    all_relevant = (relevant + new_relevant)[:top_k]

    logger.info(
        "[SelfReflect] cycle=2 rewritten=%r new_retrieved=%d new_relevant=%d total_relevant=%d",
        rewritten[:60], len(new_chunks), len(new_relevant), len(all_relevant),
    )

    if not all_relevant:
        all_relevant = cycle1_chunks  # safety: never return empty
    metadata["total_relevant"] = len(all_relevant)
    return all_relevant, metadata


# ── ReAct retrieval loop (Research Notebook) ──────────────────────────────────

def _react_reason_retrieve(
    original_query: str,
    current_query: str,
    relevant_chunks: List[Dict[str, Any]],
    iteration: int,
    model_name: str,
    num_ctx: int,
) -> Dict[str, Any]:
    """
    Reason step for the notebook ReAct loop.

    Given the current pool of relevant chunks, the LLM decides whether the
    context is sufficient to answer the question or whether another search
    with a different query would add meaningful new material.

    Returns {"action": "done"|"search", "query": str, "reasoning": str}.
    """
    try:
        llm = _reflection_llm(model_name, num_ctx, temperature=0.1, num_predict=150)
        snippets = "\n".join(
            f"[{i+1}] {c.get('text', '')[:200].strip()}"
            for i, c in enumerate(relevant_chunks[:5])
        ) or "None found yet."
        system = (
            "You are a retrieval planning agent. Analyse the retrieved chunks and decide "
            "whether they provide enough context to answer the user's question, or whether "
            "a follow-up search with a different query would surface meaningfully new material.\n"
            'Respond ONLY with JSON: {"action": "done" | "search", '
            '"query": "<focused follow-up query if action=search, else empty string>", '
            '"reasoning": "<one concise sentence>"}'
        )
        human = (
            f"ORIGINAL QUESTION: {original_query}\n"
            f"LAST QUERY USED: {current_query}\n\n"
            f"RELEVANT CHUNKS FOUND SO FAR ({len(relevant_chunks)}):\n{snippets}\n\n"
            "Should we search again with a different angle, or is this sufficient?"
        )
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = re.sub(r"```[a-z]*\n?", "", resp.content.strip()).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        action = data.get("action", "done")
        new_query = (data.get("query") or "").strip() or original_query
        reasoning = data.get("reasoning", "")
        logger.info(
            "[ReAct-Retrieve] iter=%d action=%s reasoning=%r",
            iteration, action, reasoning[:80],
        )
        return {"action": action, "query": new_query, "reasoning": reasoning}
    except Exception as exc:
        logger.warning("_react_reason_retrieve failed (%s) — stopping loop", exc)
        return {"action": "done", "query": original_query, "reasoning": "fallback"}


def react_retrieve(
    store,
    query: str,
    top_k: int,
    model_name: str = "",
    num_ctx: int = 4096,
    max_iterations: int = 3,
    min_relevant: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    ReAct retrieval loop for chunk-based stores (Research Notebook).

    Replaces the fixed 2-cycle self_reflective_retrieve with a principled
    Reason → Act → Observe loop:

      Reason  — LLM evaluates current chunks and decides: "enough" or "search again"
      Act     — hybrid store search with (possibly new) query
      Observe — grade retrieved chunks; accumulate newly-relevant ones

    Exits when min_relevant chunks are accumulated, the LLM says "done",
    or max_iterations is reached.

    Returns (relevant_chunks, metadata). Never raises.
    """
    metadata: Dict[str, Any] = {
        "mode": "react",
        "cycles": 0,
        "iterations": [],
        "total_retrieved": 0,
        "total_relevant": 0,
        "rewritten_queries": [],
        "grading_skipped": False,
    }

    all_relevant: List[Dict[str, Any]] = []
    seen_ids: set = set()
    last_chunks: List[Dict[str, Any]] = []
    current_query = query

    for i in range(max_iterations):
        # ── ACT ──────────────────────────────────────────────────────────────
        try:
            chunks = store.search_hybrid(current_query, k=top_k)
        except Exception as exc:
            logger.warning("[ReAct-Retrieve] search_hybrid failed iter=%d: %s", i + 1, exc)
            break

        last_chunks = chunks or last_chunks

        # ── OBSERVE: grade ────────────────────────────────────────────────────
        if chunks:
            grades = grade_chunks(chunks, query, model_name=model_name, num_ctx=num_ctx)
            # Silent-failure heuristic: if all grades are True on a non-trivial list,
            # grading likely failed — treat as all-relevant but flag it.
            if all(grades) and len(grades) > 1:
                metadata["grading_skipped"] = True
                new_relevant = [c for c in chunks if c.get("chunk_id") not in seen_ids]
            else:
                new_relevant = [
                    c for c, g in zip(chunks, grades)
                    if g and c.get("chunk_id") not in seen_ids
                ]
        else:
            new_relevant = []

        for c in new_relevant:
            seen_ids.add(c.get("chunk_id"))
            all_relevant.append(c)

        iter_info = {
            "iteration": i + 1,
            "query": current_query,
            "retrieved": len(chunks),
            "new_relevant": len(new_relevant),
            "cumulative_relevant": len(all_relevant),
        }
        metadata["iterations"].append(iter_info)
        metadata["total_retrieved"] += len(chunks)
        metadata["cycles"] = i + 1

        logger.info(
            "[ReAct-Retrieve] iter=%d query=%r retrieved=%d new_rel=%d total_rel=%d",
            i + 1, current_query[:60], len(chunks), len(new_relevant), len(all_relevant),
        )

        if len(all_relevant) >= min_relevant:
            break

        if i == max_iterations - 1:
            break  # no reason step on last iteration

        # ── REASON ───────────────────────────────────────────────────────────
        decision = _react_reason_retrieve(
            original_query=query,
            current_query=current_query,
            relevant_chunks=all_relevant,
            iteration=i + 1,
            model_name=model_name,
            num_ctx=num_ctx,
        )

        if decision["action"] == "done":
            break

        next_query = decision["query"]
        if next_query and next_query != current_query:
            metadata["rewritten_queries"].append(next_query)
            current_query = next_query

    result = all_relevant[:top_k] if all_relevant else last_chunks[:top_k]
    metadata["total_relevant"] = len(all_relevant)
    return result, metadata


# ── ReAct paper search loop (Literature Search) ────────────────────────────────

def _react_reason_search(
    goal: str,
    searched_queries: List[str],
    relevant_paper_dicts: List[Dict[str, Any]],
    iteration: int,
    model_name: str,
    num_ctx: int,
) -> Dict[str, Any]:
    """
    Reason step for the literature search ReAct loop.

    Given the research goal, which queries have been tried, and which papers
    are relevant so far, the LLM identifies a gap in coverage and proposes a
    targeted follow-up query — or declares coverage sufficient.

    Returns {"action": "done"|"search", "query": str, "reasoning": str}.
    """
    try:
        llm = _reflection_llm(model_name, num_ctx, temperature=0.2, num_predict=150)
        paper_titles = "\n".join(
            f"[{i+1}] {p.get('title', 'untitled')[:100]}"
            for i, p in enumerate(relevant_paper_dicts[:8])
        ) or "None found yet."
        queries_str = " | ".join(q[:50] for q in searched_queries[-4:])
        system = (
            "You are a systematic literature search strategist. Evaluate whether the "
            "papers found so far give adequate coverage of the research goal, or whether "
            "there is a specific gap that a new, focused query would fill.\n"
            'Respond ONLY with JSON: {"action": "done" | "search", '
            '"query": "<new search query targeting the identified gap, or empty string>", '
            '"reasoning": "<one concise sentence naming the gap>"}'
        )
        human = (
            f"RESEARCH GOAL: {goal}\n\n"
            f"QUERIES ALREADY TRIED: {queries_str}\n\n"
            f"RELEVANT PAPERS FOUND ({len(relevant_paper_dicts)}):\n{paper_titles}\n\n"
            "Is coverage sufficient, or is there a gap worth an additional search?"
        )
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = re.sub(r"```[a-z]*\n?", "", resp.content.strip()).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        action = data.get("action", "done")
        new_query = (data.get("query") or "").strip()
        reasoning = data.get("reasoning", "")
        logger.info(
            "[ReAct-Lit] iter=%d action=%s reasoning=%r",
            iteration, action, reasoning[:80],
        )
        return {"action": action, "query": new_query, "reasoning": reasoning}
    except Exception as exc:
        logger.warning("_react_reason_search failed (%s) — stopping loop", exc)
        return {"action": "done", "query": "", "reasoning": "fallback"}


def react_paper_search(
    searcher,
    initial_queries: List[str],
    goal: str,
    model_name: str = "",
    num_ctx: int = 4096,
    max_per_source: int = 6,
    max_iterations: int = 3,
    min_papers: int = 5,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    ReAct loop for academic paper retrieval (Literature Search, Mode 1).

    Phase 1 — batch: run all initial queries from query_generation_node and
    collect papers (same as the existing academic_search_node).

    Phase 2 — ReAct gap-filling: grade collected papers for relevance to the
    research goal; if fewer than min_papers pass, the LLM reasons about what
    topic angle is missing and fires a targeted gap-filling search. Repeats
    up to max_iterations times.

    Only newly-found papers are graded in each gap-filling round — papers
    already graded in a previous round are not re-evaluated.

    Returns (all_papers, metadata). Never raises.
    """
    import re as _re  # already imported at module level; local alias for clarity

    metadata: Dict[str, Any] = {
        "mode": "react",
        "iterations": [],
        "total_retrieved": 0,
        "total_relevant": 0,
        "gap_queries": [],
    }

    all_papers: List[Any] = []
    seen_titles: set = set()
    searched_queries: List[str] = list(initial_queries)

    # Track grading progress so we only grade newly-added papers each round.
    graded_up_to: int = 0
    relevant_paper_dicts: List[Dict[str, Any]] = []

    def _add_papers(papers) -> int:
        """Deduplicate-insert papers; return count of newly added."""
        added = 0
        for p in papers:
            key = _re.sub(r"\W+", "", (p.title or "").lower())[:60]
            if key and key not in seen_titles:
                seen_titles.add(key)
                all_papers.append(p)
                added += 1
        return added

    # ── Phase 1: run all initial queries ─────────────────────────────────────
    for q in initial_queries:
        try:
            papers = searcher.search(q, max_per_source=max_per_source)
            _add_papers(papers)
        except Exception as exc:
            logger.warning("[ReAct-Lit] initial search failed for %r: %s", q[:40], exc)

    metadata["total_retrieved"] = len(all_papers)
    logger.info("[ReAct-Lit] Phase 1: %d unique papers from %d queries.", len(all_papers), len(initial_queries))

    # ── Phase 2: grade + gap-fill ─────────────────────────────────────────────
    for i in range(max_iterations):
        # Grade only papers not yet evaluated.
        ungraded = all_papers[graded_up_to:]
        if ungraded:
            ungraded_dicts = [
                {"title": p.title or "", "abstract": p.abstract or ""}
                for p in ungraded
            ]
            new_grades = grade_papers(ungraded_dicts, goal, model_name=model_name, num_ctx=num_ctx)
            for pd, g in zip(ungraded_dicts, new_grades):
                if g:
                    relevant_paper_dicts.append(pd)
            graded_up_to = len(all_papers)

        iter_info = {
            "iteration": i + 1,
            "total_papers": len(all_papers),
            "relevant_papers": len(relevant_paper_dicts),
        }
        metadata["iterations"].append(iter_info)
        metadata["total_relevant"] = len(relevant_paper_dicts)

        logger.info(
            "[ReAct-Lit] iter=%d total=%d relevant=%d",
            i + 1, len(all_papers), len(relevant_paper_dicts),
        )

        if len(relevant_paper_dicts) >= min_papers:
            break

        if i == max_iterations - 1:
            break

        # ── REASON: identify gap ──────────────────────────────────────────────
        decision = _react_reason_search(
            goal=goal,
            searched_queries=searched_queries,
            relevant_paper_dicts=relevant_paper_dicts,
            iteration=i + 1,
            model_name=model_name,
            num_ctx=num_ctx,
        )

        if decision["action"] == "done" or not decision["query"]:
            break

        gap_query = decision["query"]
        metadata["gap_queries"].append(gap_query)
        searched_queries.append(gap_query)

        # ── ACT: gap-filling search ───────────────────────────────────────────
        try:
            new_papers = searcher.search(gap_query, max_per_source=max_per_source)
            added = _add_papers(new_papers)
            metadata["total_retrieved"] += added
            logger.info(
                "[ReAct-Lit] gap query %r → %d new papers",
                gap_query[:60], added,
            )
        except Exception as exc:
            logger.warning("[ReAct-Lit] gap search failed for %r: %s", gap_query[:40], exc)

    return all_papers, metadata

