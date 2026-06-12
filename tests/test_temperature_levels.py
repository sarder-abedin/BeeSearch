"""
tests/test_temperature_levels.py
──────────────────────────────────
Unit tests for tools/temperature_levels.py:

  - apply_temperature_level(): shifts a call's tuned base temperature by a
    level-dependent delta, clamped to [0, 1], while always leaving
    deterministic calls (base_temperature <= 0.0) at 0.0.

  - temperature_level_options() / TEMPERATURE_LEVELS / LEVEL_ORDER /
    DEFAULT_TEMPERATURE_LEVEL: the shared metadata consumed by the sidebar,
    CLI help, and /temperature slash command.

Pure stdlib — no network access or heavy deps required.
"""

from __future__ import annotations

import pytest

from tools.temperature_levels import (
    DEFAULT_TEMPERATURE_LEVEL,
    LEVEL_ORDER,
    TEMPERATURE_LEVELS,
    apply_temperature_level,
    temperature_level_options,
)


# ── Deterministic calls (base_temperature <= 0.0) are always exempt ───────────

@pytest.mark.parametrize("level", ["precise", "focused", "balanced", "creative", "unknown"])
def test_zero_base_temperature_always_stays_zero(level):
    assert apply_temperature_level(0.0, level) == 0.0


@pytest.mark.parametrize("level", ["precise", "focused", "balanced", "creative"])
def test_negative_base_temperature_clamped_to_zero(level):
    assert apply_temperature_level(-0.5, level) == 0.0


# ── Non-zero base temperature: per-level deltas ────────────────────────────────

def test_precise_forces_full_determinism():
    assert apply_temperature_level(0.3, "precise") == 0.0
    assert apply_temperature_level(0.7, "precise") == 0.0


def test_focused_is_unchanged_baseline():
    assert apply_temperature_level(0.3, "focused") == 0.3
    assert apply_temperature_level(0.7, "focused") == 0.7


def test_balanced_adds_0_2_delta():
    assert apply_temperature_level(0.3, "balanced") == pytest.approx(0.5)
    assert apply_temperature_level(0.7, "balanced") == pytest.approx(0.9)


def test_creative_adds_0_4_delta():
    assert apply_temperature_level(0.3, "creative") == pytest.approx(0.7)
    assert apply_temperature_level(0.1, "creative") == pytest.approx(0.5)


def test_results_are_clamped_to_one():
    assert apply_temperature_level(0.7, "creative") == 1.0  # 0.7 + 0.4 = 1.1 -> 1.0
    assert apply_temperature_level(0.9, "balanced") == 1.0  # 0.9 + 0.2 = 1.1 -> 1.0


def test_unknown_level_falls_back_to_default():
    assert apply_temperature_level(0.3, "not-a-real-level") == apply_temperature_level(0.3, DEFAULT_TEMPERATURE_LEVEL)


# ── Shared metadata used by UI / CLI ────────────────────────────────────────────

def test_default_level_is_focused_and_a_noop():
    assert DEFAULT_TEMPERATURE_LEVEL == "focused"
    assert TEMPERATURE_LEVELS[DEFAULT_TEMPERATURE_LEVEL]["delta"] == 0.0


def test_level_order_has_four_named_levels():
    assert LEVEL_ORDER == ["precise", "focused", "balanced", "creative"]
    assert set(LEVEL_ORDER) == set(TEMPERATURE_LEVELS)


def test_temperature_level_options_returns_label_and_description_per_level():
    options = temperature_level_options()
    assert [key for key, _, _ in options] == LEVEL_ORDER
    for key, label, description in options:
        assert label == TEMPERATURE_LEVELS[key]["label"]
        assert isinstance(description, str) and len(description) > 20
