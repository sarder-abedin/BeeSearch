"""tests/test_feedback_agent.py — Unit tests for agents/feedback_agent.py"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.feedback_agent import (
    MAX_FEEDBACK_ROUNDS,
    make_feedback_entry,
    refine_with_feedback,
)


class TestConstants:
    def test_max_rounds(self):
        assert MAX_FEEDBACK_ROUNDS == 3


class TestMakeFeedbackEntry:
    def test_structure(self):
        entry = make_feedback_entry(1, "make it shorter", "original text")
        assert entry["round"] == 1
        assert entry["feedback"] == "make it shorter"
        assert entry["previous_output"] == "original text"
        assert "timestamp" in entry

    def test_truncates_long_output(self):
        long_text = "x" * 5000
        entry = make_feedback_entry(2, "feedback", long_text)
        assert len(entry["previous_output"]) <= 3000

    def test_round_number(self):
        for r in [1, 2, 3]:
            entry = make_feedback_entry(r, "fb", "out")
            assert entry["round"] == r


class TestRefineWithFeedback:
    def _mock_llm(self, return_text: str):
        mock_response = MagicMock()
        mock_response.content = return_text
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    @patch("langchain_ollama.ChatOllama")
    def test_returns_refined_text(self, MockChatOllama):
        mock_llm = self._mock_llm("Refined output text")
        MockChatOllama.return_value = mock_llm

        result = refine_with_feedback(
            original_output="Original text",
            feedback="Make it shorter",
            mode="literature_search",
            model_name="llama3.1:8b",
            num_ctx=4096,
        )
        assert result == "Refined output text"

    @patch("langchain_ollama.ChatOllama")
    def test_passes_context_in_prompt(self, MockChatOllama):
        mock_llm = self._mock_llm("Refined")
        MockChatOllama.return_value = mock_llm

        refine_with_feedback(
            original_output="Output",
            feedback="Fix it",
            context="Some paper title",
            mode="wisdom",
            model_name="llama3.1:8b",
            num_ctx=4096,
        )
        call_args = mock_llm.invoke.call_args[0][0]
        human_content = call_args[1].content
        assert "Some paper title" in human_content

    @patch("langchain_ollama.ChatOllama")
    def test_returns_original_on_llm_failure(self, MockChatOllama):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionRefusedError("Ollama not running")
        MockChatOllama.return_value = mock_llm

        original = "The original output text"
        result = refine_with_feedback(
            original_output=original,
            feedback="Improve it",
            mode="systematic_review",
            model_name="llama3.1:8b",
            num_ctx=4096,
        )
        assert result == original

    @patch("langchain_ollama.ChatOllama")
    def test_all_mode_labels(self, MockChatOllama):
        mock_llm = self._mock_llm("ok")
        MockChatOllama.return_value = mock_llm

        for mode in ["literature_search", "wisdom", "systematic_review",
                     "notebook_pipeline", "proposal", "unknown_mode"]:
            result = refine_with_feedback(
                original_output="text",
                feedback="fb",
                mode=mode,
                model_name="llama3.1:8b",
                num_ctx=2048,
            )
            assert result == "ok"

    @patch("langchain_ollama.ChatOllama")
    def test_strips_whitespace_from_response(self, MockChatOllama):
        mock_llm = self._mock_llm("  \n Refined with whitespace \n  ")
        MockChatOllama.return_value = mock_llm

        result = refine_with_feedback("original", "feedback")
        assert result == "Refined with whitespace"

    @patch("langchain_ollama.ChatOllama")
    def test_truncates_long_original_in_prompt(self, MockChatOllama):
        mock_llm = self._mock_llm("ok")
        MockChatOllama.return_value = mock_llm

        long_output = "A" * 10000
        refine_with_feedback(
            original_output=long_output,
            feedback="shorten it",
            model_name="llama3.1:8b",
            num_ctx=4096,
        )
        human_content = mock_llm.invoke.call_args[0][0][1].content
        # Original is capped at 6000 chars in the prompt
        assert "A" * 6000 in human_content
        assert "A" * 6001 not in human_content


class TestFeedbackStateFields:
    """Verify that feedback fields are present in all relevant states."""

    def test_research_state_has_feedback_fields(self):
        from agents.state import ResearchState
        assert "feedback_history" in ResearchState.__annotations__
        assert "refinement_round" in ResearchState.__annotations__

    def test_wisdom_state_has_feedback_fields(self):
        from agents.wisdom_state import WisdomState
        assert "feedback_history" in WisdomState.__annotations__
        assert "refinement_round" in WisdomState.__annotations__

    def test_systematic_review_state_has_feedback_fields(self):
        from agents.systematic_review_state import SystematicReviewState
        assert "feedback_history" in SystematicReviewState.__annotations__
        assert "refinement_round" in SystematicReviewState.__annotations__

    def test_notebook_pipeline_state_has_feedback_fields(self):
        from agents.notebook_pipeline_state import NotebookPipelineState
        assert "feedback_history" in NotebookPipelineState.__annotations__
        assert "refinement_round" in NotebookPipelineState.__annotations__

    def test_proposal_gpt_state_has_feedback_fields(self):
        from agents.proposal_gpt_state import ProposalGPTState
        assert "feedback_history" in ProposalGPTState.__annotations__
        assert "refinement_round" in ProposalGPTState.__annotations__

    def test_factory_initialises_feedback_fields(self):
        from agents.state import create_initial_state
        s = create_initial_state("goal")
        assert s["feedback_history"] == []
        assert s["refinement_round"] == 0

    def test_wisdom_factory_initialises_feedback_fields(self):
        from agents.wisdom_state import create_wisdom_state
        s = create_wisdom_state("msg", "sid")
        assert s["feedback_history"] == []
        assert s["refinement_round"] == 0

    def test_sr_factory_initialises_feedback_fields(self):
        from agents.systematic_review_state import create_systematic_review_state
        s = create_systematic_review_state("question")
        assert s["feedback_history"] == []
        assert s["refinement_round"] == 0
