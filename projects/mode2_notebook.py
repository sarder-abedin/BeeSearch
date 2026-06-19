"""
projects/mode2_notebook.py
────────────────────────────
Mode 2: Research Notebook.

Thin dispatch shim called by `app.py::main()` (via `PROJECT_REGISTRY`/
`_PROJECT_MODULES` lookup on `st.session_state["active_project"] == "mode2"`)
and by `main.py --notebook` for the CLI entry point. Delegates all rendering
to `ui/tabs/notebook.py::tab_notebook`, the large tab container covering
Chat, Sources, Summary, FAQ, Literature Review, Mind Map, Knowledge Graph,
Citation Timeline, Study Comparison, Pipeline, Research Report, and Explain.
"""
from __future__ import annotations


def run(settings: dict) -> None:
    """Render Mode 2's Streamlit UI using the sidebar-collected `settings` dict.

    Imports `tab_notebook` lazily so selecting Mode 1 from the landing page
    doesn't pull in Mode 2's (heavier) dependencies.
    """
    from ui.tabs.notebook import tab_notebook
    tab_notebook(settings)
