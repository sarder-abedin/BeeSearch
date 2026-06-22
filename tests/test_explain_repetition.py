"""
tests/test_explain_repetition.py
──────────────────────────────────
Unit tests for the Explain tab's repeated-clarification handling:
agents/story_nodes.py::repetition_tracker_node and ::concept_visualizer_node,
plus the StorytellerMemory.add_turn(explanation_style=...) persistence it relies on.

When a user re-asks the same question (or signals confusion: "I don't
understand", "still confused", ...), repetition_tracker_node overrides
explanation_style to something different from the immediately preceding
answer, and concept_visualizer_node renders an interactive Pyvis concept map
as a second, visual modality of explanation. Neither node makes a real
network/Ollama call in these tests — ChatOllama and pyvis are mocked.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from agents.story_memory import StorytellerMemory
from agents.story_nodes import (
    _concept_graph_to_pyvis_html,
    _extract_concept_graph_data,
    _is_confusion_phrase,
    _next_explanation_strategy,
    _question_similarity,
    _safe_label,
    _similar_to_recent_question,
    concept_visualizer_node,
    repetition_tracker_node,
)


# ── _question_similarity ─────────────────────────────────────────────────────

def test_question_similarity_high_for_paraphrase_sharing_topic_words():
    """Two phrasings of the same question score above the trigger threshold."""
    ratio = _question_similarity("What is backpropagation?", "How does backpropagation work?")
    assert ratio >= 0.4


def test_question_similarity_zero_for_unrelated_questions():
    ratio = _question_similarity("What is the capital of France?", "What is backpropagation?")
    assert ratio == 0.0


def test_question_similarity_one_for_identical_question():
    ratio = _question_similarity("What is backpropagation?", "What is backpropagation?")
    assert ratio == 1.0


def test_question_similarity_zero_when_either_side_has_no_meaningful_tokens():
    """A message that's only framing/stopwords (e.g. "What is this?") has no
    topic tokens to compare — similarity is 0, not a division-by-zero crash."""
    assert _question_similarity("What is this?", "What is backpropagation?") == 0.0
    assert _question_similarity("", "What is backpropagation?") == 0.0


# ── _is_confusion_phrase ──────────────────────────────────────────────────────

@pytest.mark.parametrize("message", [
    "I don't understand",
    "I dont understand this at all",
    "still confused about this",
    "What do you mean by that?",
    "Can you explain it differently?",
    "I'm still lost",
    "this still doesn't make sense",
])
def test_is_confusion_phrase_true_for_known_signals(message):
    assert _is_confusion_phrase(message) is True


@pytest.mark.parametrize("message", [
    "What is backpropagation?",
    "Can you explain backpropagation?",
    "Tell me more about transformers",
])
def test_is_confusion_phrase_false_for_ordinary_questions(message):
    assert _is_confusion_phrase(message) is False


# ── _similar_to_recent_question ───────────────────────────────────────────────

def test_similar_to_recent_question_finds_matching_prior_user_turn():
    history = [
        {"role": "user", "content": "What is backpropagation?"},
        {"role": "assistant", "content": "It's an algorithm..."},
    ]
    match = _similar_to_recent_question("How does backpropagation work?", history)
    assert match == "What is backpropagation?"


def test_similar_to_recent_question_ignores_assistant_turns():
    history = [{"role": "assistant", "content": "What is backpropagation? is a common question"}]
    assert _similar_to_recent_question("What is backpropagation?", history) == ""


def test_similar_to_recent_question_empty_when_no_match():
    history = [{"role": "user", "content": "What is the capital of France?"}]
    assert _similar_to_recent_question("What is backpropagation?", history) == ""


# ── _next_explanation_strategy ────────────────────────────────────────────────

def test_next_explanation_strategy_keeps_requested_when_different_from_last():
    assert _next_explanation_strategy("analogy", "simple") == "analogy"


def test_next_explanation_strategy_rotates_when_same_as_last():
    assert _next_explanation_strategy("simple", "simple") == "analogy"


def test_next_explanation_strategy_wraps_around_at_end_of_rotation():
    assert _next_explanation_strategy("debate", "debate") == "simple"


def test_next_explanation_strategy_handles_unknown_last_style():
    """A style not in the rotation (e.g. from a future/unrecognized value) is
    treated like "no information" — falls back to index -1 then wraps to 0."""
    assert _next_explanation_strategy("simple", "some_unknown_style") == "simple"


# ── _safe_label ────────────────────────────────────────────────────────────────

def test_safe_label_collapses_whitespace_and_trims_length():
    assert _safe_label("  Gradient   Descent\n\n") == "Gradient Descent"
    assert _safe_label("x" * 100, maxlen=10) == "x" * 10


def test_safe_label_handles_non_string_input():
    assert _safe_label(123) == "123"


# ── repetition_tracker_node ────────────────────────────────────────────────────

def test_repetition_tracker_false_on_first_message_even_with_confusion_phrase():
    """Confusion language on a user's very first message has nothing to be a
    repeat of — there's no prior assistant turn yet."""
    state = {"user_message": "I don't understand this topic", "conversation_history": [], "explanation_style": "simple"}
    result = repetition_tracker_node(state)
    assert result["is_repeat_clarification"] is False
    assert result["explanation_style"] == "simple"


def test_repetition_tracker_false_for_a_genuinely_new_question():
    history = [
        {"role": "user", "content": "What is backpropagation?"},
        {"role": "assistant", "content": "...", "explanation_style": "simple"},
    ]
    state = {"user_message": "What is reinforcement learning?", "conversation_history": history, "explanation_style": "simple"}
    result = repetition_tracker_node(state)
    assert result["is_repeat_clarification"] is False
    assert result["explanation_style"] == "simple"


def test_repetition_tracker_true_for_repeated_question_and_switches_style():
    history = [
        {"role": "user", "content": "What is backpropagation?"},
        {"role": "assistant", "content": "...", "explanation_style": "simple"},
    ]
    state = {
        "user_message": "How does backpropagation work?",
        "conversation_history": history,
        "explanation_style": "simple",
    }
    result = repetition_tracker_node(state)
    assert result["is_repeat_clarification"] is True
    assert result["repeated_question"] == "What is backpropagation?"
    assert result["explanation_style"] != "simple"


def test_repetition_tracker_true_for_explicit_confusion_signal():
    history = [
        {"role": "user", "content": "What is backpropagation?"},
        {"role": "assistant", "content": "...", "explanation_style": "analogy"},
    ]
    state = {"user_message": "I still don't get it", "conversation_history": history, "explanation_style": "analogy"}
    result = repetition_tracker_node(state)
    assert result["is_repeat_clarification"] is True
    # Requested style ("analogy") matches what was just tried — rotates to the next one.
    assert result["explanation_style"] == "walkthrough"


def test_repetition_tracker_keeps_requested_style_when_already_different_from_last():
    """User picked a different style than last time on their own — no need to
    force a rotation, their selection already satisfies "different"."""
    history = [
        {"role": "user", "content": "What is backpropagation?"},
        {"role": "assistant", "content": "...", "explanation_style": "simple"},
    ]
    state = {
        "user_message": "How does backpropagation work?",
        "conversation_history": history,
        "explanation_style": "debate",
    }
    result = repetition_tracker_node(state)
    assert result["is_repeat_clarification"] is True
    assert result["explanation_style"] == "debate"


def test_repetition_tracker_no_override_when_last_style_unknown():
    """Assistant turns saved before this feature existed have no
    explanation_style key — without knowing what was tried, the node keeps the
    user's current selection rather than guessing."""
    history = [
        {"role": "user", "content": "What is backpropagation?"},
        {"role": "assistant", "content": "..."},  # no explanation_style key
    ]
    state = {
        "user_message": "How does backpropagation work?",
        "conversation_history": history,
        "explanation_style": "simple",
    }
    result = repetition_tracker_node(state)
    assert result["is_repeat_clarification"] is True
    assert result["explanation_style"] == "simple"


def test_repetition_tracker_progress_bookkeeping():
    result = repetition_tracker_node({"user_message": "hi", "conversation_history": [], "completed_steps": ["context_loader"]})
    assert result["current_step"] == "repetition_tracker"
    assert result["completed_steps"] == ["context_loader", "repetition_tracker"]
    assert result["progress_pct"] == 25


# ── _concept_graph_to_pyvis_html (pyvis mocked — not installed in this sandbox) ─

def _install_fake_pyvis(monkeypatch):
    """Inject a fake pyvis.network module, mirroring how test_search_tools.py
    fakes duckduckgo_search for an optional dependency that may not be installed."""
    fake_net_instance = MagicMock()
    fake_net_instance.generate_html.return_value = "<html>concept map</html>"
    fake_network_module = types.ModuleType("pyvis.network")
    fake_network_module.Network = MagicMock(return_value=fake_net_instance)
    fake_pyvis_module = types.ModuleType("pyvis")
    fake_pyvis_module.network = fake_network_module
    monkeypatch.setitem(sys.modules, "pyvis", fake_pyvis_module)
    monkeypatch.setitem(sys.modules, "pyvis.network", fake_network_module)
    return fake_net_instance


def test_concept_graph_to_pyvis_html_adds_central_and_related_nodes(monkeypatch):
    fake_net = _install_fake_pyvis(monkeypatch)
    data = {
        "central": "Backpropagation",
        "related": [
            {"label": "Gradient Descent", "relation": "computes using"},
            {"label": "Chain Rule", "relation": "built on"},
        ],
    }
    html = _concept_graph_to_pyvis_html(data)
    assert html == "<html>concept map</html>"
    fake_net.add_node.assert_any_call(
        "Backpropagation", label="Backpropagation", color="#3B82F6", size=28, title="Central concept"
    )
    fake_net.add_node.assert_any_call(
        "Gradient Descent", label="Gradient Descent", color="#10B981", size=18, title="computes using"
    )
    assert fake_net.add_edge.call_count == 2


def test_concept_graph_to_pyvis_html_dedupes_related_label_matching_central(monkeypatch):
    """An LLM-extracted related item that duplicates the central concept's
    label would otherwise create a self-loop edge — skip it instead."""
    fake_net = _install_fake_pyvis(monkeypatch)
    data = {"central": "X", "related": [{"label": "X", "relation": "is"}, {"label": "Y", "relation": "supports"}]}
    _concept_graph_to_pyvis_html(data)
    assert fake_net.add_edge.call_count == 1


def test_concept_graph_to_pyvis_html_caps_related_items_at_six(monkeypatch):
    fake_net = _install_fake_pyvis(monkeypatch)
    data = {"central": "X", "related": [{"label": f"item{i}", "relation": "r"} for i in range(10)]}
    _concept_graph_to_pyvis_html(data)
    assert fake_net.add_edge.call_count == 6


def test_concept_graph_to_pyvis_html_raises_clear_error_when_pyvis_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyvis", None)
    monkeypatch.setitem(sys.modules, "pyvis.network", None)
    with pytest.raises(ImportError):
        _concept_graph_to_pyvis_html({"central": "X", "related": []})


# ── _extract_concept_graph_data ───────────────────────────────────────────────

def test_extract_concept_graph_data_parses_llm_json():
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = (
            '{"central": "Backpropagation", "related": [{"label": "Gradient Descent", "relation": "uses"}]}'
        )
        data = _extract_concept_graph_data("How does it work?", "Backpropagation is...", {"model_name": "m", "num_ctx": 4096})
    assert data["central"] == "Backpropagation"
    assert data["related"][0]["label"] == "Gradient Descent"


def test_extract_concept_graph_data_raises_when_no_json_found():
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = "I cannot do that."
        with pytest.raises(ValueError):
            _extract_concept_graph_data("q", "text", {"model_name": "m", "num_ctx": 4096})


# ── concept_visualizer_node ────────────────────────────────────────────────────

def test_concept_visualizer_node_is_noop_when_not_a_repeat():
    """Most turns aren't repeats — the node must not make any LLM call at all."""
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        result = concept_visualizer_node({"is_repeat_clarification": False, "completed_steps": []})
        mock_chat.assert_not_called()
    assert result["concept_visual_html"] == ""
    assert result["current_step"] == "concept_visualizer"


def test_concept_visualizer_node_renders_html_on_repeat(monkeypatch):
    _install_fake_pyvis(monkeypatch)
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = (
            '{"central": "Backpropagation", "related": [{"label": "Gradient Descent", "relation": "uses"}]}'
        )
        result = concept_visualizer_node({
            "is_repeat_clarification": True,
            "user_message": "How does it work?",
            "assistant_response": "Backpropagation is...",
            "model_name": "m",
            "num_ctx": 4096,
            "completed_steps": [],
        })
    assert result["concept_visual_html"] == "<html>concept map</html>"


def test_concept_visualizer_node_fails_safe_when_llm_call_errors():
    """Any extraction failure must never block the primary explanation that
    storyteller_node already produced — the turn should still complete."""
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.side_effect = RuntimeError("ollama unreachable")
        result = concept_visualizer_node({
            "is_repeat_clarification": True,
            "user_message": "q",
            "assistant_response": "text",
            "completed_steps": [],
        })
    assert result["concept_visual_html"] == ""


def test_concept_visualizer_node_fails_safe_when_pyvis_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyvis", None)
    monkeypatch.setitem(sys.modules, "pyvis.network", None)
    with patch("agents.story_nodes.ChatOllama") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = '{"central": "X", "related": []}'
        result = concept_visualizer_node({
            "is_repeat_clarification": True,
            "user_message": "q",
            "assistant_response": "text",
            "completed_steps": [],
        })
    assert result["concept_visual_html"] == ""


# ── StorytellerMemory.add_turn(explanation_style=...) persistence ────────────

def test_add_turn_persists_and_round_trips_explanation_style(tmp_path):
    mem = StorytellerMemory(db_path=tmp_path / "sessions.db")
    sid = mem.new_session(topic="Test")
    mem.add_turn(sid, role="user", content="What is X?")
    mem.add_turn(sid, role="assistant", content="X is...", explanation_style="analogy")

    history = mem.get_history(sid)
    assistant_turn = next(t for t in history if t["role"] == "assistant")
    assert assistant_turn["explanation_style"] == "analogy"


def test_add_turn_defaults_explanation_style_to_none_when_omitted(tmp_path):
    """Backward compatibility: turns saved without this new parameter must not
    crash readers that later .get() the key."""
    mem = StorytellerMemory(db_path=tmp_path / "sessions.db")
    sid = mem.new_session(topic="Test")
    mem.add_turn(sid, role="assistant", content="X is...")

    history = mem.get_history(sid)
    assert history[0].get("explanation_style") is None
