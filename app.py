"""
app.py — BeeSearch Streamlit entry point
─────────────────────────────────────────────
Three modes:
  Mode 1 — PRISMA Systematic Literature Review
  Mode 2 — Research Notebook (NotebookLM-style grounded Q&A)
  Mode 3 — AI Research Assistant (free-form, literature-grounded Q&A)

Run:  streamlit run app.py

This module is Streamlit's script entry point: it runs top-to-bottom on every
rerun (page load, widget interaction, etc.), so module-level statements here
(page config, theme, browser-launch thread) execute on every rerun guarded by
session/env-var checks. Mode dispatch lives in `main()`, which reads
`st.session_state["active_project"]` (mirrored to `st.query_params["mode"]`
so a refresh doesn't lose the active mode) and hands off to
`projects.mode1_systematic_review` / `projects.mode2_notebook` via
`projects.PROJECT_REGISTRY`.
"""
from __future__ import annotations
import logging
import os
import threading
import time
import webbrowser
from pathlib import Path
import streamlit as st

# Open the default browser once when the server first starts.
# Guards:
#   - _BEESEARCH_BROWSER_OPENED env var  → only fires once per process
#   - STREAMLIT_SERVER_HEADLESS=true      → skipped inside Docker containers
if not os.environ.get("_BEESEARCH_BROWSER_OPENED"):
    os.environ["_BEESEARCH_BROWSER_OPENED"] = "1"
    _headless = os.environ.get("STREAMLIT_SERVER_HEADLESS", "false").lower() in ("true", "1")
    if not _headless:
        _port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
        _url = f"http://localhost:{_port}"

        def _open_browser() -> None:
            """Wait for the Streamlit server to be ready, then open it in the default browser."""
            time.sleep(1.5)
            try:
                webbrowser.open(_url)
            except Exception:
                pass

        threading.Thread(target=_open_browser, daemon=True).start()

_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"
_logo = str(_LOGO_PATH) if _LOGO_PATH.exists() else "BeeSearch"

st.set_page_config(
    page_title="BeeSearch",
    page_icon=_logo,
    layout="wide",
    initial_sidebar_state="expanded",
)
if _LOGO_PATH.exists():
    st.logo(str(_LOGO_PATH), size="large")

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
    "mode3": "projects.mode3_research_assistant",
}


def main() -> None:
    """Render the sidebar, then either dispatch to the active project's `run(settings)` or show the landing page.

    Restores `active_project` from the `mode` URL query param if the
    Streamlit session state was reset (e.g. browser refresh), so deep links
    into a mode survive a reload.
    """
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
