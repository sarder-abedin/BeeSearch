"""
projects/mode1_systematic_review.py
─────────────────────────────────────
Mode 1: Systematic Literature Review.

Thin dispatch shim called by `app.py::main()` (via `PROJECT_REGISTRY`/
`_PROJECT_MODULES` lookup on `st.session_state["active_project"] == "mode1"`)
and by `main.py --systematic-review` for the CLI entry point. Delegates all
rendering to `ui/tabs/systematic_review.py::tab_systematic_review`, which
drives the PRISMA pipeline (`agents/systematic_review_*.py`).
"""
from __future__ import annotations


def run(settings: dict) -> None:
    """Render Mode 1's Streamlit UI using the sidebar-collected `settings` dict.

    Imports `tab_systematic_review` lazily so selecting Mode 2 from the
    landing page doesn't pull in Mode 1's dependencies.
    """
    from ui.tabs.systematic_review import tab_systematic_review
    tab_systematic_review(settings)
