"""projects/mode1_systematic_review.py — Mode 1: Systematic Literature Review."""
from __future__ import annotations


def run(settings: dict) -> None:
    from ui.tabs.systematic_review import tab_systematic_review
    tab_systematic_review(settings)
