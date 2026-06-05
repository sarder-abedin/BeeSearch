"""projects/mode2_notebook.py — Mode 2: Research Notebook."""
from __future__ import annotations


def run(settings: dict) -> None:
    from ui.tabs.notebook import tab_notebook
    tab_notebook(settings)
