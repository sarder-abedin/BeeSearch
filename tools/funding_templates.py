"""
tools/funding_templates.py
──────────────────────────
Hardcoded templates for European funding agencies used in Proposal Mode.

Each template captures the agency-specific conventions that writers need to
follow: budget norms, typical duration, consortium requirements, key sections,
approximate word-count targets per section, tone guidance, and any mandatory
content requirements.

Usage
─────
from tools.funding_templates import get_template, get_word_count_targets, AGENCY_NAMES

template = get_template("Horizon Europe")
targets  = get_word_count_targets("Vinnova", overrides={"methodology": 800})
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ── Agency templates ──────────────────────────────────────────────────────────

FUNDING_TEMPLATES: Dict[str, dict] = {
    "Horizon Europe": {
        "agency": "Horizon Europe",
        "region": "EU",
        "typical_budget_range": "€500K – €3M",
        "typical_duration_months": 36,
        "typical_trl_range": "TRL 3–7",
        "consortium_guidance": "Minimum 3 independent legal entities from 3 different EU member states or associated countries",
        "key_sections": ["Abstract", "Excellence", "Impact", "Implementation", "Ethics"],
        "word_count_targets": {
            "introduction": 600,
            "literature_review": 900,
            "methodology": 900,
            "expected_outcomes": 500,
            "abstract": 250,
        },
        "tone_notes": "Emphasise European added value, cross-border collaboration, and societal impact. Use Key Performance Indicators (KPIs). Address the UN Sustainable Development Goals if applicable.",
        "required_sections_note": "Must include: State of the art, Innovation beyond state of the art, Open Science practices.",
    },
    "Vinnova": {
        "agency": "Vinnova",
        "region": "Sweden",
        "typical_budget_range": "SEK 500K – 5M",
        "typical_duration_months": 24,
        "typical_trl_range": "TRL 2–6",
        "consortium_guidance": "Collaboration between academia and industry is strongly encouraged. At least one Swedish organisation required.",
        "key_sections": ["Background", "Objectives", "Activities", "Expected Effects", "Team"],
        "word_count_targets": {
            "introduction": 500,
            "literature_review": 700,
            "methodology": 700,
            "expected_outcomes": 400,
            "abstract": 200,
        },
        "tone_notes": "Emphasise Swedish competitiveness, sustainable development, and clear path to commercialisation or societal application. Concrete measurable effects are expected.",
        "required_sections_note": "Expected effects must address innovation potential, sustainable development, and Swedish research/innovation environment.",
    },
    "SSF (Swedish Foundation for Strategic Research)": {
        "agency": "SSF",
        "region": "Sweden",
        "typical_budget_range": "SEK 10M – 30M",
        "typical_duration_months": 60,
        "typical_trl_range": "TRL 1–4",
        "consortium_guidance": "Multi-institution research consortia preferred. Strong academic track record required.",
        "key_sections": ["Research Plan", "Significance", "Team Competence", "Resources", "Management"],
        "word_count_targets": {
            "introduction": 700,
            "literature_review": 1000,
            "methodology": 1000,
            "expected_outcomes": 600,
            "abstract": 300,
        },
        "tone_notes": "Focus on scientific excellence and strategic importance to Sweden. Demonstrate world-class research team. Long-term research programme perspective.",
        "required_sections_note": "Research must have clear strategic importance for Sweden's future competitiveness in research and industry.",
    },
    "VR (Swedish Research Council)": {
        "agency": "VR",
        "region": "Sweden",
        "typical_budget_range": "SEK 2M – 10M",
        "typical_duration_months": 48,
        "typical_trl_range": "TRL 1–4",
        "consortium_guidance": "Individual researcher grants or small research groups. Principal Investigator must be employed at a Swedish university or research institute.",
        "key_sections": ["Research Question", "State of the Art", "Theory/Approach", "Implementation", "Scientific Renewal"],
        "word_count_targets": {
            "introduction": 600,
            "literature_review": 800,
            "methodology": 800,
            "expected_outcomes": 400,
            "abstract": 200,
        },
        "tone_notes": "Emphasise scientific novelty and methodological rigor. Show clear departure from current state of the art. Ground-breaking or high-risk research is valued.",
        "required_sections_note": "Must demonstrate scientific renewal — explain what is fundamentally new compared to existing research.",
    },
    "None (General)": {
        "agency": "None",
        "region": "General",
        "typical_budget_range": "Unspecified",
        "typical_duration_months": 36,
        "typical_trl_range": "Any",
        "consortium_guidance": "",
        "key_sections": ["Introduction", "Literature Review", "Methodology", "Expected Outcomes"],
        "word_count_targets": {
            "introduction": 600,
            "literature_review": 900,
            "methodology": 800,
            "expected_outcomes": 500,
            "abstract": 250,
        },
        "tone_notes": "Write a clear, well-structured research proposal with appropriate academic language.",
        "required_sections_note": "",
    },
}

AGENCY_NAMES: List[str] = list(FUNDING_TEMPLATES.keys())


# ── Public helpers ────────────────────────────────────────────────────────────

def get_template(agency_name: str) -> dict:
    """Return the template for the given agency, falling back to 'None (General)'."""
    return FUNDING_TEMPLATES.get(agency_name, FUNDING_TEMPLATES["None (General)"])


def get_word_count_targets(agency_name: str, overrides: dict = None) -> dict:
    """Return word count targets for agency, with optional per-section overrides."""
    targets = dict(get_template(agency_name)["word_count_targets"])
    if overrides:
        for section, count in overrides.items():
            if isinstance(count, int) and count > 0:
                targets[section] = count
    return targets
