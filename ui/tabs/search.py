"""ui/tabs/search.py — Mode 2: Academic Literature Search"""

from __future__ import annotations

import streamlit as st

from ui.helpers import render_clarification_form


def tab_search(settings: dict) -> tuple[str, list, str, dict]:
    st.header("Mode 2 — Academic Literature Search")
    st.markdown(
        """
No documents needed. Provide a research topic and the agent will:
1. Decompose it into focused sub-queries
2. Search **arXiv** and **Semantic Scholar** for relevant papers
3. Synthesise findings with **properly formatted APA citations**

Best for: exploring a new field, getting a literature overview, or finding
references for a paper you are writing.
"""
    )

    goal = st.text_area(
        "Research Topic / Question",
        placeholder="e.g. What are the current approaches to protein structure prediction "
                    "using deep learning? What is the state of quantum error correction?",
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info("**Sources:** arXiv, Semantic Scholar" + (", CrossRef" if settings["include_crossref"] else ""))
    with col2:
        if settings["include_web"]:
            st.info("**Web search:** Google (FastAPI service)")

    clarifications = render_clarification_form("search", goal, settings)

    with_sr = st.checkbox(
        "Run Systematic Review (PRISMA) on this goal",
        key="sr_search",
        help="After the literature search, run a full PRISMA-style systematic review on the same goal.",
    )

    return goal, [], "search", clarifications, with_sr
