"""projects/mode1_search.py — Mode 1: Literature Search."""
from __future__ import annotations
import streamlit as st


def run(settings: dict) -> None:
    from projects._research_runner import has_active_run, resume_active_run, run_research_workflow, show_stored_result
    from ui.tabs.search import tab_search

    st.header("Literature Search")

    if has_active_run("mode1"):
        resume_active_run("mode1", settings)
        return

    if show_stored_result("mode1", settings):
        return

    goal, uploaded_files, mode, clarifications, with_sr = tab_search(settings)
    if st.button("Run Search", key="run_mode1", type="primary", use_container_width=True):
        if not goal:
            st.warning("Please enter a research goal before running.")
        else:
            run_research_workflow(
                goal, uploaded_files, mode, clarifications, with_sr,
                settings, "outputs/memory/mode1", "mode1",
            )
