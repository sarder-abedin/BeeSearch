"""
projects/mode3_research_assistant.py
──────────────────────────────────────
Mode 3: AI Research Assistant.

Thin dispatch shim called by `app.py::main()` (via `_PROJECT_MODULES` lookup on
`st.session_state["active_project"] == "mode3"`). Delegates all rendering to
`ui/tabs/research_assistant.py::tab_research_assistant`, the free-form,
literature-grounded question-answering screen.
"""
from __future__ import annotations


def run(settings: dict) -> None:
    """Render Mode 3's Streamlit UI using the sidebar-collected `settings` dict.

    Imports `tab_research_assistant` lazily so selecting another mode from the
    landing page doesn't pull in this mode's search/LLM dependencies.
    """
    from ui.tabs.research_assistant import tab_research_assistant
    tab_research_assistant(settings)
