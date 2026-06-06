"""projects/mode2_proposal.py — Mode 2: ProposalGPT (AI-assisted proposal writing)."""
from __future__ import annotations


def run(settings: dict) -> None:
    from ui.tabs.proposal_gpt import tab_proposal_gpt
    tab_proposal_gpt(settings)
