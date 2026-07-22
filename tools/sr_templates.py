"""
tools/sr_templates.py
──────────────────────
Guided-template presets for Mode 1 (Systematic Literature Review): starting
points for a research question + inclusion/exclusion criteria, keyed by a
short slug.

Pure data + a tiny lookup helper — no Streamlit, no I/O. Originally a literal
list inlined in `ui/tabs/systematic_review.py` (the Streamlit "guided
template" picker); extracted here so the FastAPI backend can expose the same
presets (`GET /api/systematic-review/templates`) without importing that
(Streamlit-dependent) module. `ui.tabs.systematic_review.SR_TEMPLATES` is kept
as a thin alias for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SR_TEMPLATES: List[Dict[str, Any]] = [
    {
        "key": "clinical_rct",
        "label": "Clinical RCT review",
        "description": "Randomised controlled trials evaluating a clinical intervention in human participants.",
        "research_question": "What is the effect of [intervention] on [outcome] in [population]?",
        "inclusion": [
            "Randomised controlled trials (RCTs)",
            "Human participants",
            "Peer-reviewed, published 2010–present",
            "Reports the outcome of interest with quantitative results",
        ],
        "exclusion": [
            "Animal or in-vitro studies",
            "Case reports, editorials, conference abstracts only",
            "No control/comparison group",
            "Non-English publications",
        ],
        "note": "Pairs well with **Statistical Meta-Analysis** (pool effect sizes across trials) "
                "and **Preprint Status** (flags retracted or unpublished trials).",
    },
    {
        "key": "cs_survey",
        "label": "CS literature survey",
        "description": "Computer-science / engineering survey of methods, systems or benchmarks.",
        "research_question": "What approaches have been proposed for [task/problem], and how do they compare on [metric]?",
        "inclusion": [
            "Peer-reviewed papers or well-cited preprints (arXiv)",
            "Proposes, benchmarks, or surveys a method for the stated task",
            "Published within the last 10 years",
            "Reports quantitative results or a clear architectural contribution",
        ],
        "exclusion": [
            "Position papers / opinion pieces with no technical contribution",
            "Duplicate or superseded preprint versions",
            "Workshop posters with no accompanying results",
        ],
        "note": "Pairs well with **Citation Network**, **Concept Drift Tracker** and "
                "**Research Trend Forecaster** — CS moves fast, so track what's rising.",
    },
    {
        "key": "qual_synthesis",
        "label": "Qualitative evidence synthesis",
        "description": "Thematic synthesis of qualitative studies (interviews, ethnography, case studies).",
        "research_question": "How do [population] experience or perceive [phenomenon]?",
        "inclusion": [
            "Qualitative or mixed-methods studies",
            "Primary research with original data collection",
            "Clearly describes participants and methodology",
            "Published in peer-reviewed venues",
        ],
        "exclusion": [
            "Purely quantitative studies with no qualitative component",
            "Secondary analyses or reviews of other qualitative work",
            "Grey literature without peer review",
        ],
        "note": "Pairs well with **Evidence Map** and **Narrative Synthesis** — themes matter "
                "more than pooled numbers here, so Statistical Meta-Analysis isn't recommended.",
    },
    {
        "key": "scoping_review",
        "label": "Scoping / mapping review",
        "description": "Broad map of what evidence exists on a topic, before committing to a focused review.",
        "research_question": "What is the nature and extent of research on [topic] in [context]?",
        "inclusion": [
            "Any study design that addresses the topic",
            "Published in any language with an available English abstract",
            "No date restriction (or specify a broad range)",
        ],
        "exclusion": [
            "Studies entirely off-topic despite keyword matches",
            "Duplicates across databases",
        ],
        "note": "Pairs well with **Evidence Map**, **Research Trend Forecaster**, and "
                "**Cross-Notebook Search** to connect findings to material you've already collected.",
    },
]

_SR_TEMPLATE_BY_KEY = {t["key"]: t for t in SR_TEMPLATES}


def get_template(key: str) -> Optional[Dict[str, Any]]:
    """Look up one template by its `key`; returns None if unknown."""
    return _SR_TEMPLATE_BY_KEY.get(key)
