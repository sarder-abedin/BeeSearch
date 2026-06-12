"""
tests/test_temperature_integration.py
────────────────────────────────────────
Integration-style checks that `temperature_level` is correctly threaded from
NotebookState / StoryState / settings dicts into the ChatOllama
``temperature=`` kwarg at every Research Notebook LLM factory:

  - agents.notebook_nodes._llm                Chat Q&A (answer_node)
  - agents.story_nodes._llm                   Explain / storyteller_node
  - agents.notebook_advanced._make_llm        Advanced tools (summary, FAQ, ...)
  - agents.notebook_pipeline_nodes._make_llm  7-agent pipeline

ChatOllama itself is mocked — no Ollama server or network access required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.notebook_state import create_notebook_state
from agents.story_state import create_story_state
from tools.temperature_levels import DEFAULT_TEMPERATURE_LEVEL, apply_temperature_level

LEVELS = ["precise", "focused", "balanced", "creative"]


@pytest.mark.parametrize("level", LEVELS)
def test_notebook_chat_llm_applies_temperature_level(level):
    state = create_notebook_state(
        user_message="What is X?", notebook_id="nb1", temperature_level=level,
    )
    with patch("agents.notebook_nodes.ChatOllama") as mock_chat:
        from agents.notebook_nodes import _llm
        _llm(state)
    _, kwargs = mock_chat.call_args
    assert kwargs["temperature"] == apply_temperature_level(0.3, level)


@pytest.mark.parametrize("level", LEVELS)
def test_storyteller_llm_applies_temperature_level(level):
    state = create_story_state(
        user_message="Explain X", session_id="s1", temperature_level=level,
    )
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        from agents.story_nodes import _llm
        _llm(state, temperature=0.7)
    _, kwargs = mock_chat.call_args
    assert kwargs["temperature"] == apply_temperature_level(0.7, level)


@pytest.mark.parametrize("level", LEVELS)
def test_notebook_advanced_make_llm_applies_temperature_level(level):
    settings = {"model": "llama3.1:8b", "num_ctx": 8192, "temperature_level": level}
    with patch("agents.notebook_advanced.ChatOllama") as mock_chat:
        from agents.notebook_advanced import _make_llm
        _make_llm(settings)
    _, kwargs = mock_chat.call_args
    assert kwargs["temperature"] == apply_temperature_level(0.3, level)


@pytest.mark.parametrize("level", LEVELS)
def test_pipeline_make_llm_applies_temperature_level(level):
    settings = {"model": "llama3.1:8b", "num_ctx": 8192, "temperature_level": level}
    with patch("langchain_ollama.ChatOllama") as mock_chat:
        from agents.notebook_pipeline_nodes import _make_llm
        _make_llm(settings)
    _, kwargs = mock_chat.call_args
    assert kwargs["temperature"] == apply_temperature_level(0.3, level)


def test_notebook_state_defaults_to_focused_when_unspecified():
    state = create_notebook_state(user_message="Q", notebook_id="nb1")
    assert state["temperature_level"] == DEFAULT_TEMPERATURE_LEVEL


def test_story_state_defaults_to_focused_when_unspecified():
    state = create_story_state(user_message="Q", session_id="s1")
    assert state["temperature_level"] == DEFAULT_TEMPERATURE_LEVEL


def test_missing_temperature_level_key_is_a_noop_like_pre_upgrade_callers():
    """A state dict from before this feature (no key at all) behaves as 'focused'."""
    state = create_notebook_state(user_message="Q", notebook_id="nb1")
    del state["temperature_level"]
    with patch("agents.notebook_nodes.ChatOllama") as mock_chat:
        from agents.notebook_nodes import _llm
        _llm(state)
    _, kwargs = mock_chat.call_args
    assert kwargs["temperature"] == 0.3  # unchanged from pre-feature baseline
