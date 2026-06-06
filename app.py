"""
app.py — Streamlit entry point
────────────────────────────────
Landing page dispatches to one of 5 sub-project modes.
Only the selected mode's code is imported (lazy loading).

Run:  streamlit run app.py
"""
from __future__ import annotations
import logging
import streamlit as st

st.set_page_config(
    page_title="Agentic Research Assistant",
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
    "mode1": "projects.mode1_search",
    "mode2": "projects.mode2_proposal",
    "mode3": "projects.mode3_wisdom",
    "mode4": "projects.mode4_systematic_review",
    "mode5": "projects.mode5_notebook",
    "mode6": "projects.mode6_grammar",
}


def main() -> None:
    settings = render_sidebar()

    # ── Restore a saved research session from sidebar click ───────────────────
    load_sid = st.session_state.pop("load_research_session_id", None)
    if load_sid:
        try:
            from agents.memory import ResearchMemory
            from ui.helpers import render_references, render_report, render_key_findings
            sess = ResearchMemory().load(load_sid)
            if sess:
                with st.expander(
                    f"Restored Session — {sess.get('goal','')[:80]}",
                    expanded=True,
                ):
                    st.caption(
                        f"Mode: {sess.get('mode','')} · Model: {sess.get('model_name','')} "
                        f"· Saved: {sess.get('created_at','')[:10]}"
                    )
                    render_key_findings(sess.get("key_findings", []))
                    st.divider()
                    t_rep, t_ref = st.tabs(["Report", "References"])
                    with t_rep:
                        render_report(sess.get("report", ""), load_sid)
                    with t_ref:
                        render_references(sess.get("references", []), key_suffix=f"_restore_{load_sid}")
        except Exception as exc:
            st.error(f"Could not restore session: {exc}")
            logger.warning("Session restore failed: %s", exc)

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

        st.divider()
        with st.expander("Manage Writing Style Profiles"):
            from ui.tabs.style_profiles import tab_style_profiles
            tab_style_profiles(settings)


if __name__ == "__main__":
    main()
