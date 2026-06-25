"""
ui/tabs/research_assistant.py
─────────────────────────────────
Tab container for Mode 3 — AI Research Assistant.

A single-screen, NotebookLM-free entry point: the user types a free-form
research question and gets a literature-grounded answer with inline,
code-rebuilt citations (Elicit / Perplexity / Consensus style). Unlike the
Systematic Review tab there is no PICO/criteria workflow, and unlike the
Research Notebook there is nothing to upload — it searches published
literature directly via `agents.research_assistant.run_research_assistant`.

The assistant is stateless, so this module is the only place its results live,
cached in `st.session_state["ra_last_result"]` so they survive the reruns that
widgets (e.g. the follow-up-question buttons) trigger.
"""

from __future__ import annotations

import logging

import streamlit as st

from agents.research_assistant import run_research_assistant

logger = logging.getLogger(__name__)


def _render_citation(c: dict) -> None:
    """Render one rebuilt citation as an expander with its snippet, APA string, and link."""
    kind = c.get("kind", "academic")
    badge = "🌐 web" if kind == "web" else "📄 paper"
    year = c.get("year")
    title = c.get("title", "Untitled")
    header = f"[{c.get('n')}] {badge} — {title[:80]}" + (f" ({year})" if year else "")
    with st.expander(header):
        if c.get("snippet"):
            st.markdown(f"> {c['snippet']}")
        if c.get("apa"):
            st.caption(c["apa"])
        if c.get("url"):
            st.markdown(f"[Open source]({c['url']})")


def _run_and_store(question: str, settings: dict, include_web: bool) -> None:
    """Run the assistant for `question` with a progress status line and cache the result."""
    status = st.status("Searching published literature…", expanded=True)

    def _cb(stage: str, info: dict) -> None:
        """Translate the assistant's stage callbacks into human-readable status updates."""
        msgs = {
            "searching": "Searching Google Scholar · arXiv · Semantic Scholar"
                         + (" · web" if include_web else "") + "…",
            "reading": f"Reading {info.get('academic_count', 0)} paper(s)"
                       f" and {info.get('web_count', 0)} web result(s)…",
            "answering": "Composing a grounded answer…"
                         if info.get("grounded") else "No sources found — answering from general knowledge…",
            "done": "Done.",
        }
        if stage in msgs:
            status.update(label=msgs[stage])

    try:
        result = run_research_assistant(question, settings, stream_callback=_cb, include_web=include_web)
        st.session_state["ra_last_result"] = result
        status.update(label="Done.", state="complete")
    except Exception as e:
        logger.exception("Research Assistant failed")
        status.update(label=f"Failed: {e}", state="error")
        st.session_state["ra_last_result"] = None


def tab_research_assistant(settings: dict) -> None:
    """Mode 3 — AI Research Assistant. Top-level entry point for this mode.

    Renders a question box + options, drives `run_research_assistant`, and shows
    the grounded answer, a rebuilt citations list, and suggested follow-ups
    (each a button that re-asks). Reads/writes `st.session_state`:
      - `ra_last_result` — the last assistant result dict, persisted across reruns.
      - `ra_question` — the question text widget's backing state; also the target
        the follow-up buttons write into before re-running.
      - `ra_pending` — a follow-up question queued by a button click, consumed on
        the next run so the click both fills the box and triggers a search.
    """
    st.header("Mode 3 — AI Research Assistant")
    st.markdown(
        "Ask a free-form research question and get an answer grounded in **published literature** "
        "with inline citations — no documents to upload, no PRISMA workflow. BeeSearch searches "
        "Google Scholar, arXiv, Semantic Scholar (and the web), reads what it finds, and cites its "
        "sources. Best for orienting questions; use **Mode 1** for an exhaustive systematic review."
    )
    st.divider()

    # A follow-up button may have queued a question on the previous run.
    pending = st.session_state.pop("ra_pending", None)
    if pending:
        st.session_state["ra_question"] = pending

    question = st.text_area(
        "Research question",
        height=90,
        placeholder="e.g. Does intermittent fasting improve insulin sensitivity in adults?",
        key="ra_question",
    )
    col_opts, col_run = st.columns([3, 1])
    with col_opts:
        include_web = st.checkbox(
            "Also search the web (DuckDuckGo)", value=True, key="ra_include_web",
            help="Supplement academic sources with general web results, cited the same way.",
        )
    with col_run:
        ask = st.button("Ask", key="ra_ask", type="primary", use_container_width=True)

    if (ask or pending) and question.strip():
        _run_and_store(question.strip(), settings, include_web)
    elif ask and not question.strip():
        st.warning("Please enter a research question.")

    result = st.session_state.get("ra_last_result")
    if not result:
        return

    st.divider()
    if not result.get("grounded"):
        st.warning(
            "No published sources could be retrieved for this question — the answer below is from "
            "general model knowledge and should be verified against primary literature."
        )

    st.markdown(result.get("answer", "*No answer generated.*"))

    citations = result.get("citations", [])
    if citations:
        st.divider()
        st.subheader(f"Citations ({len(citations)})")
        for c in citations:
            _render_citation(c)

    sources = result.get("sources", [])
    n_cited = len(citations)
    st.caption(
        f"Searched {result.get('academic_count', 0)} paper(s) and {result.get('web_count', 0)} "
        f"web result(s); {len(sources)} used as context, {n_cited} cited in the answer."
    )

    followups = result.get("suggested_questions", [])
    if followups:
        st.divider()
        st.markdown("**Follow-up questions:**")
        for i, fq in enumerate(followups):
            if st.button(fq, key=f"ra_followup_{i}"):
                st.session_state["ra_pending"] = fq
                st.rerun()
