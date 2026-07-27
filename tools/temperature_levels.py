"""
tools/temperature_levels.py
────────────────────────────
User-tunable "temperature level" for Research Notebook mode (Mode 2).

BeeSearch tunes each LLM call's temperature individually -- chunk/citation
grading and faithfulness checks are deterministic (0.0), while answer
generation, summaries, FAQs, and the Explain "storyteller" use small
non-zero temperatures suited to their task.

Rather than overriding those per-call values outright, a "temperature
level" shifts every non-zero call's baseline by the same delta, so the
relative tuning between tasks (e.g. "Explain" stays more exploratory than
a literal Q&A answer) is preserved at every level. Deterministic calls
(base temperature == 0.0) are always left untouched, regardless of level
-- grading and fact-checking never become "creative".

apply_temperature_level() is the single function every Notebook-mode
``_llm``/``_make_llm`` factory calls to turn its tuned base temperature
into the effective temperature for the user's chosen level.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

TEMPERATURE_LEVELS: Dict[str, Dict[str, object]] = {
    "precise": {
        "label": "Precise",
        "delta": None,  # special-cased: forces non-exempt calls to 0.0
        "description": (
            "Fully deterministic. The same question against the same sources "
            "always produces the same answer, word for word. Best when you "
            "need exact reproducibility."
        ),
    },
    "focused": {
        "label": "Focused",
        "delta": 0.0,
        "description": (
            "BeeSearch's default tuning. Factual answers, summaries, and "
            "extractions stay close to your source material, with minimal "
            "wording variation between runs."
        ),
    },
    "balanced": {
        "label": "Balanced",
        "delta": 0.2,
        "description": (
            "More natural, varied phrasing across answers, summaries, and "
            "explanations, while still grounded in your sources. Citation "
            "grading and fact-checking are unaffected."
        ),
    },
    "creative": {
        "label": "Creative",
        "delta": 0.4,
        "description": (
            "The most varied and exploratory phrasing -- useful for "
            "brainstorming, podcast-style explanations, and mind maps. "
            "Written answers may diverge further from exact source wording. "
            "Citation grading and fact-checking are unaffected."
        ),
    },
}

DEFAULT_TEMPERATURE_LEVEL = "focused"

# Display order for UI selectboxes and CLI help text.
LEVEL_ORDER: List[str] = ["precise", "focused", "balanced", "creative"]


def temperature_level_options() -> List[Tuple[str, str, str]]:
    """Return ``(key, label, description)`` for each level, in display order."""
    return [
        (key, str(TEMPERATURE_LEVELS[key]["label"]), str(TEMPERATURE_LEVELS[key]["description"]))
        for key in LEVEL_ORDER
    ]


def apply_temperature_level(base_temperature: float, level: str) -> float:
    """
    Adjust ``base_temperature`` according to the user's chosen level.

    Calls with ``base_temperature <= 0.0`` (deterministic grading and
    faithfulness checks) are always left at ``0.0``, regardless of level --
    grading never becomes "creative".

    For all other calls:
      - ``precise``  -> forced to ``0.0`` (fully deterministic)
      - ``focused``  -> unchanged (BeeSearch's tuned default)
      - ``balanced`` -> ``base_temperature + 0.2``, clamped to ``[0, 1]``
      - ``creative`` -> ``base_temperature + 0.4``, clamped to ``[0, 1]``

    An unrecognized ``level`` falls back to :data:`DEFAULT_TEMPERATURE_LEVEL`.
    """
    if base_temperature <= 0.0:
        return 0.0

    cfg = TEMPERATURE_LEVELS.get(level, TEMPERATURE_LEVELS[DEFAULT_TEMPERATURE_LEVEL])
    delta = cfg["delta"]
    if delta is None:  # "precise"
        return 0.0
    return max(0.0, min(1.0, base_temperature + float(delta)))
