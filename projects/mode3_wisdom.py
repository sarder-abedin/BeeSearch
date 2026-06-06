"""projects/mode3_wisdom.py — Mode 3: Wisdom Mode."""
from __future__ import annotations


def run(settings: dict) -> None:
    from ui.tabs.wisdom import tab_wisdom
    tab_wisdom(settings)
