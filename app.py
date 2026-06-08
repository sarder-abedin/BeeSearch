"""
app.py — BeeSearch Streamlit entry point
─────────────────────────────────────────────
Two modes:
  Mode 4 — PRISMA Systematic Literature Review
  Mode 5 — Research Notebook (NotebookLM-style grounded Q&A)

Run:  streamlit run app.py
"""
from __future__ import annotations
import logging
import streamlit as st

st.set_page_config(
    page_title="BeeSearch",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.settings import get_settings
from ui.sidebar import render_sidebar
from ui.theme import apply_theme

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
cfg = get_settings()

apply_theme()

_PROJECT_MODULES = {
    "mode1": "projects.mode1_systematic_review",
    "mode2": "projects.mode2_notebook",
}


def main() -> None:
    settings = render_sidebar()

    # ── Restore active_project from URL if session was reset ──────────────────
    if "active_project" not in st.session_state:
        qp_mode = st.query_params.get("mode")
        if qp_mode and qp_mode in _PROJECT_MODULES:
            st.session_state["active_project"] = qp_mode

    active_project = st.session_state.get("active_project")

    if active_project:
        st.query_params["mode"] = active_project

        from projects import PROJECT_REGISTRY
        info = PROJECT_REGISTRY.get(active_project, {})
        col_back, col_title = st.columns([1, 6])
        with col_back:
            if st.button("← All Modes", key="back_to_landing", help="Return to mode selection"):
                st.session_state.pop("active_project", None)
                st.query_params.clear()
                st.rerun()
        with col_title:
            st.markdown(f"**{info.get('name', active_project)}**")

        module_path = _PROJECT_MODULES.get(active_project)
        if not module_path:
            st.error(f"Unknown project: {active_project}")
            return

        import importlib
        mod = importlib.import_module(module_path)
        mod.run(settings)

    else:
        st.query_params.clear()
        from ui.landing import render_landing
        render_landing()


if __name__ == "__main__":
    main()
