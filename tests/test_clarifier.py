"""tests/test_clarifier.py — Unit tests for tools/clarifier.py (fallback path)"""

import pytest

from tools.clarifier import (
    _fallback_questions,
    _validate_questions,
    generate_clarifying_questions,
)


# ── _fallback_questions ───────────────────────────────────────────────────────

class TestFallbackQuestions:
    @pytest.mark.parametrize("mode", ["document", "search", "hybrid", "proposal", "story", "wisdom"])
    def test_returns_list_for_all_modes(self, mode):
        qs = _fallback_questions(mode)
        assert isinstance(qs, list)
        assert len(qs) >= 1

    @pytest.mark.parametrize("mode", ["document", "search", "hybrid", "proposal", "story", "wisdom"])
    def test_each_question_has_required_keys(self, mode):
        for q in _fallback_questions(mode):
            for key in ("key", "question", "type", "options", "recommended"):
                assert key in q, f"Mode '{mode}', q '{q.get('key')}' missing '{key}'"

    @pytest.mark.parametrize("mode", ["document", "search", "hybrid", "proposal", "story", "wisdom"])
    def test_all_questions_are_select_type(self, mode):
        for q in _fallback_questions(mode):
            assert q["type"] == "select", (
                f"Mode '{mode}', key '{q['key']}': type is '{q['type']}', expected 'select'"
            )

    @pytest.mark.parametrize("mode", ["document", "search", "hybrid", "proposal", "story", "wisdom"])
    def test_select_type_has_options(self, mode):
        for q in _fallback_questions(mode):
            assert isinstance(q.get("options"), list)
            assert len(q["options"]) >= 2

    @pytest.mark.parametrize("mode", ["document", "search", "hybrid", "proposal", "story", "wisdom"])
    def test_recommended_matches_an_option(self, mode):
        for q in _fallback_questions(mode):
            assert q["recommended"] in q["options"], (
                f"Mode '{mode}', key '{q['key']}': recommended '{q['recommended']}' "
                f"not in options {q['options']}"
            )

    def test_unknown_mode_returns_generic_fallback(self):
        qs = _fallback_questions("nonexistent_mode")
        assert isinstance(qs, list)
        assert len(qs) >= 1

    def test_unknown_mode_fallback_has_recommended(self):
        for q in _fallback_questions("nonexistent_mode"):
            assert q.get("recommended") in q.get("options", [])

    @pytest.mark.parametrize("mode", ["document", "search", "hybrid", "proposal", "story", "wisdom"])
    def test_question_text_ends_with_question_mark(self, mode):
        for q in _fallback_questions(mode):
            assert q["question"].strip().endswith("?"), (
                f"Mode '{mode}', key '{q['key']}': question does not end with '?'"
            )

    @pytest.mark.parametrize("mode", ["document", "search", "hybrid", "proposal", "story", "wisdom"])
    def test_keys_are_snake_case(self, mode):
        import re
        pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        for q in _fallback_questions(mode):
            assert pattern.match(q["key"]), f"Key '{q['key']}' is not snake_case"

    def test_at_most_three_questions(self):
        for mode in ["document", "search", "hybrid", "proposal", "story", "wisdom"]:
            qs = _fallback_questions(mode)
            assert len(qs) <= 3, f"Mode '{mode}' returned {len(qs)} questions (max 3)"


# ── _validate_questions ───────────────────────────────────────────────────────

class TestValidateQuestions:
    def test_valid_select_question_passes_through(self):
        raw = [
            {
                "key": "audience",
                "question": "Who is the audience?",
                "type": "select",
                "options": ["A", "B", "C"],
                "recommended": "B",
            }
        ]
        result = _validate_questions(raw)
        assert len(result) == 1
        assert result[0]["recommended"] == "B"
        assert result[0]["type"] == "select"

    def test_text_type_converted_to_select(self):
        """A 'text' type question must be converted to 'select' with fallback options."""
        raw = [{"key": "goal", "question": "What do you want?", "type": "text"}]
        result = _validate_questions(raw)
        assert len(result) == 1
        assert result[0]["type"] == "select"
        assert len(result[0]["options"]) >= 2

    def test_missing_type_converted_to_select(self):
        raw = [{"key": "k", "question": "What?", "options": ["X", "Y"]}]
        result = _validate_questions(raw)
        assert result[0]["type"] == "select"

    def test_missing_recommended_defaults_to_first_option(self):
        raw = [{"key": "k", "question": "Q?", "type": "select", "options": ["A", "B", "C"]}]
        result = _validate_questions(raw)
        assert result[0]["recommended"] == "A"

    def test_recommended_not_in_options_reset_to_first(self):
        raw = [
            {
                "key": "k",
                "question": "Q?",
                "type": "select",
                "options": ["A", "B"],
                "recommended": "Z",
            }
        ]
        result = _validate_questions(raw)
        assert result[0]["recommended"] == "A"

    def test_non_list_input_returns_empty(self):
        assert _validate_questions("bad input") == []
        assert _validate_questions(None) == []

    def test_missing_key_drops_question(self):
        raw = [{"question": "No key question?", "type": "select", "options": ["A"]}]
        assert _validate_questions(raw) == []

    def test_missing_question_text_drops_entry(self):
        raw = [{"key": "k", "type": "select", "options": ["A"]}]
        assert _validate_questions(raw) == []

    def test_empty_options_gets_fallback_options(self):
        raw = [{"key": "k", "question": "Q?", "type": "select", "options": []}]
        result = _validate_questions(raw)
        assert len(result[0]["options"]) >= 2

    def test_original_dict_not_mutated(self):
        original = {"key": "k", "question": "Q?", "type": "text"}
        _validate_questions([original])
        assert original["type"] == "text"  # the original must not be modified


# ── generate_clarifying_questions (offline / fallback) ────────────────────────

class TestGenerateClarifyingQuestions:
    def test_falls_back_when_ollama_unreachable(self):
        """With a dead Ollama URL, should return hardcoded fallback questions."""
        qs = generate_clarifying_questions(
            goal="Understand attention mechanisms",
            mode="search",
            model_name="llama3.2:3b",
            ollama_base_url="http://localhost:1",  # guaranteed unreachable
            num_ctx=512,
        )
        assert isinstance(qs, list)
        assert len(qs) >= 1

    def test_fallback_result_has_required_keys(self):
        qs = generate_clarifying_questions(
            goal="Research protein folding",
            mode="document",
            model_name="llama3.2:3b",
            ollama_base_url="http://localhost:1",
            num_ctx=512,
        )
        for q in qs:
            for field in ("key", "question", "type", "options", "recommended"):
                assert field in q, f"Missing field '{field}' in fallback question"

    def test_fallback_all_questions_are_select_type(self):
        qs = generate_clarifying_questions(
            goal="Learn about neural networks",
            mode="story",
            model_name="llama3.2:3b",
            ollama_base_url="http://localhost:1",
            num_ctx=512,
        )
        for q in qs:
            assert q["type"] == "select"

    def test_fallback_recommended_in_options(self):
        qs = generate_clarifying_questions(
            goal="Write a proposal on climate change",
            mode="proposal",
            model_name="llama3.2:3b",
            ollama_base_url="http://localhost:1",
            num_ctx=512,
        )
        for q in qs:
            assert q["recommended"] in q["options"]

    def test_unknown_mode_does_not_raise(self):
        qs = generate_clarifying_questions(
            goal="Whatever",
            mode="totally_unknown_mode",
            model_name="llama3.2:3b",
            ollama_base_url="http://localhost:1",
            num_ctx=512,
        )
        assert isinstance(qs, list)
