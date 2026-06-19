"""
projects/__init__.py
─────────────────────
BeeSearch project registry.

`PROJECT_REGISTRY` maps the short project id used in `st.session_state["active_project"]`
and the `?mode=` query param (see `app.py::main()`) to display metadata. `app.py`
looks up `PROJECT_REGISTRY[active_project]["name"]` for the header label, then
separately resolves the id to a `projects.mode{1,2}_*` module via its own
`_PROJECT_MODULES` dict and calls that module's `run(settings)`.
"""

# Keys must match the project ids used in app.py's `_PROJECT_MODULES` dict and
# in `ui/landing.py`'s `_PROJECTS` list — they are not derived from each other.
PROJECT_REGISTRY = {
    "mode1": {"name": "Systematic Literature Review"},
    "mode2": {"name": "Research Notebook"},
}
