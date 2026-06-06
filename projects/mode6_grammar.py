"""projects/mode6_grammar.py — Mode 6: English Grammar Proofreading."""
from __future__ import annotations


def run(settings: dict) -> None:
    from ui.tabs.grammar_proofreading import tab_grammar_proofreading
    tab_grammar_proofreading(settings)
