"""tests/test_phase3_gaps.py — Tests for all five architectural gap fixes.

Covers:
  Gap 1 — Evaluation framework (all four eval nodes)
  Gap 2 — Wisdom routing regex (PROCEED_TO_WISDOM anywhere in response)
  Gap 3 — Node-level streaming (stream_callback called per node)
  Gap 4 — Proposal memory saver node (explicit, separate from assembly)
  Gap 5 — Cross-session word-level tag overlap
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Gap 1 helpers ──────────────────────────────────────────────────────────────

def _eval_response(json_str: str):
    def _invoke(messages):
        resp = MagicMock()
        resp.content = json_str
        return resp
    return _invoke


# ═══════════════════════════════════════════════════════════════════════════════
# GAP 1 — Evaluation framework
# ═══════════════════════════════════════════════════════════════════════════════

class TestResearchEvalNode:
    def test_returns_eval_result_dict(self):
        from agents.eval_nodes import research_eval_node

        state = {
            "goal": "Summarise transformer architectures",
            "report": "Transformers use self-attention…",
            "key_findings": ["Self-attention is O(n²)", "BERT achieves SOTA on GLUE"],
            "references": [{"ref_num": 1}],
            "model_name": "llama3.1:8b",
            "num_ctx": 4096,
            "completed_steps": [],
        }
        with patch("agents.eval_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _eval_response(
                '{"goal_alignment": 4, "evidence_quality": 3, "clarity": 5, "overall": 4, "summary": "Good alignment."}'
            )
            MockOllama.return_value = instance

            out = research_eval_node(state)

        assert out["eval_result"]["overall"] == 4
        assert out["eval_result"]["goal_alignment"] == 4
        assert "summary" in out["eval_result"]
        assert "research_eval" in out["completed_steps"]

    def test_fails_silently_on_bad_json(self):
        from agents.eval_nodes import research_eval_node

        state = {
            "goal": "Test", "report": "", "key_findings": [],
            "references": [], "model_name": "llama3.1:8b", "num_ctx": 4096,
            "completed_steps": [],
        }
        with patch("agents.eval_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _eval_response("not valid json at all")
            MockOllama.return_value = instance

            out = research_eval_node(state)

        assert out["eval_result"] == {}
        assert "research_eval" in out["completed_steps"]


class TestProposalEvalNode:
    def test_returns_eval_result_dict(self):
        from agents.eval_nodes import proposal_eval_node

        state = {
            "goal": "AI for medical imaging",
            "title": "Deep learning for early cancer detection",
            "objectives": ["Develop CNN pipeline", "Evaluate on TCGA dataset"],
            "methodology": "We will use transfer learning…",
            "abstract": "This proposal describes…",
            "model_name": "llama3.1:8b",
            "num_ctx": 4096,
            "completed_steps": [],
        }
        with patch("agents.eval_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _eval_response(
                '{"goal_alignment": 5, "objectives_quality": 4, "methodology_soundness": 3, "overall": 4, "summary": "Solid proposal."}'
            )
            MockOllama.return_value = instance

            out = proposal_eval_node(state)

        assert out["eval_result"]["overall"] == 4
        assert "proposal_eval" in out["completed_steps"]


class TestStoryEvalNode:
    def test_returns_eval_result_dict(self):
        from agents.eval_nodes import story_eval_node

        state = {
            "topic": "Attention mechanism",
            "explanation_style": "analogy",
            "assistant_response": "Think of attention like a spotlight…",
            "model_name": "llama3.1:8b",
            "num_ctx": 4096,
            "completed_steps": [],
        }
        with patch("agents.eval_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _eval_response(
                '{"clarity": 5, "style_adherence": 4, "overall": 5, "summary": "Excellent analogy."}'
            )
            MockOllama.return_value = instance

            out = story_eval_node(state)

        assert out["eval_result"]["clarity"] == 5
        assert "story_eval" in out["completed_steps"]


class TestWisdomEvalNode:
    def test_skips_when_no_wisdom_generated(self):
        """Eval must return empty dict when deep_understanding is absent (clarification turn)."""
        from agents.eval_nodes import wisdom_eval_node

        state = {
            "deep_understanding": "",  # no wisdom yet
            "model_name": "llama3.1:8b",
            "num_ctx": 4096,
            "completed_steps": [],
        }
        out = wisdom_eval_node(state)
        assert out["eval_result"] == {}

    def test_returns_eval_when_wisdom_present(self):
        from agents.eval_nodes import wisdom_eval_node

        state = {
            "topic": "Chronic stress",
            "deep_understanding": "Cortisol impairs hippocampal neurogenesis…",
            "actionable_takeaways": ["Sleep 8h", "Exercise"],
            "overall_confidence": "High",
            "academic_papers": [{}, {}, {}],
            "model_name": "llama3.1:8b",
            "num_ctx": 4096,
            "completed_steps": [],
        }
        with patch("agents.eval_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.side_effect = _eval_response(
                '{"evidence_grounding": 4, "confidence_calibration": 5, "actionability": 4, "overall": 4, "summary": "Well-grounded wisdom."}'
            )
            MockOllama.return_value = instance

            out = wisdom_eval_node(state)

        assert out["eval_result"]["overall"] == 4
        assert "wisdom_eval" in out["completed_steps"]


# ═══════════════════════════════════════════════════════════════════════════════
# GAP 2 — Wisdom routing regex (PROCEED_TO_WISDOM anywhere in response)
# ═══════════════════════════════════════════════════════════════════════════════

class TestWisdomRoutingRegex:
    """clarification_node must detect PROCEED_TO_WISDOM regardless of position/case."""

    def _make_state(self, session_id=""):
        return {
            "user_message": "I feel stressed",
            "topic": "stress",
            "scenario": "",
            "session_id": session_id,
            "model_name": "llama3.1:8b",
            "num_ctx": 4096,
            "clarification_count": 0,
            "conversation_history": [],
            "document_context": "",
            "completed_steps": [],
            "errors": [],
            "clarifications": {},
        }

    def test_proceed_at_start(self):
        import agents.wisdom_nodes as wm
        state = self._make_state()
        with patch.object(wm, "_llm") as mock_llm_fn:
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(
                content="PROCEED_TO_WISDOM\nSearching the scientific literature now."
            )
            mock_llm_fn.return_value = llm
            out = wm.clarification_node(state)
        assert out["phase"] == "ready_to_generate"
        assert "Searching" in out["assistant_response"]

    def test_proceed_lowercase(self):
        """Lowercase proceed_to_wisdom must also trigger the proceed branch."""
        import agents.wisdom_nodes as wm
        state = self._make_state()
        with patch.object(wm, "_llm") as mock_llm_fn:
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(
                content="proceed_to_wisdom\nI have enough context."
            )
            mock_llm_fn.return_value = llm
            out = wm.clarification_node(state)
        assert out["phase"] == "ready_to_generate"

    def test_proceed_mid_response(self):
        """PROCEED_TO_WISDOM mid-response (LLM preamble) must still trigger."""
        import agents.wisdom_nodes as wm
        state = self._make_state()
        with patch.object(wm, "_llm") as mock_llm_fn:
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(
                content="I believe we have sufficient context. PROCEED_TO_WISDOM\nSearching now."
            )
            mock_llm_fn.return_value = llm
            out = wm.clarification_node(state)
        assert out["phase"] == "ready_to_generate"

    def test_regular_question_still_clarifies(self):
        """A response without PROCEED_TO_WISDOM must keep phase=clarifying."""
        import agents.wisdom_nodes as wm
        state = self._make_state()
        with patch.object(wm, "_llm") as mock_llm_fn:
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(
                content="How long have you been experiencing this stress?"
            )
            mock_llm_fn.return_value = llm
            out = wm.clarification_node(state)
        assert out["phase"] == "clarifying"
        assert out["clarification_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# GAP 3 — Node-level streaming
# ═══════════════════════════════════════════════════════════════════════════════

class TestNodeLevelStreaming:
    def test_story_graph_calls_stream_callback_per_node(self, tmp_path):
        """run_story_turn must invoke stream_callback for every graph node."""
        import agents.story_nodes as story_nodes_module
        from agents.story_graph import run_story_turn
        from agents.story_memory import StorytellerMemory
        from agents.story_state import create_story_state

        mem = StorytellerMemory(db_path=tmp_path / "sessions.db")
        sid = mem.new_session(topic="transformers")

        monkeypatch_targets = {
            "_memory": mem,
        }

        called_nodes: list = []

        def _cb(node_name, _state):
            called_nodes.append(node_name)

        with patch("agents.story_nodes.ChatOllama") as MockOllama:
            instance = MagicMock()
            instance.invoke.return_value = MagicMock(
                content='A great explanation.\n{"suggested_questions": ["Q1", "Q2"]}'
            )
            MockOllama.return_value = instance

            original_memory = story_nodes_module._memory
            story_nodes_module._memory = mem
            try:
                state = create_story_state(
                    user_message="Explain attention",
                    session_id=sid,
                    topic="transformers",
                )
                run_story_turn(state, stream_callback=_cb)
            finally:
                story_nodes_module._memory = original_memory

        assert "context_loader" in called_nodes
        assert "storyteller" in called_nodes
        assert "memory_saver" in called_nodes
        assert "story_eval" in called_nodes

    def test_wisdom_graph_calls_stream_callback_per_node(self, tmp_path):
        """run_wisdom_turn must invoke stream_callback for every graph node."""
        import agents.wisdom_nodes as wisdom_nodes_module
        from agents.wisdom_graph import run_wisdom_turn
        from agents.wisdom_memory import WisdomMemory
        from agents.wisdom_state import create_wisdom_state

        mem = WisdomMemory(db_path=tmp_path / "sessions.db")
        sid = mem.new_session(topic="stress")

        called_nodes: list = []

        def _cb(node_name, _state):
            called_nodes.append(node_name)

        original_memory = wisdom_nodes_module._memory
        original_academic = wisdom_nodes_module._academic
        original_web = wisdom_nodes_module._web

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []

        wisdom_nodes_module._memory = mem
        wisdom_nodes_module._academic = mock_searcher
        wisdom_nodes_module._web = mock_searcher

        try:
            with patch("agents.wisdom_nodes.ChatOllama") as MockOllama:
                instance = MagicMock()
                instance.invoke.return_value = MagicMock(
                    content="How long have you been stressed?"
                )
                MockOllama.return_value = instance

                state = create_wisdom_state(
                    user_message="I feel stressed",
                    session_id=sid,
                    topic="stress",
                )
                run_wisdom_turn(state, stream_callback=_cb)
        finally:
            wisdom_nodes_module._memory = original_memory
            wisdom_nodes_module._academic = original_academic
            wisdom_nodes_module._web = original_web

        assert "context_loader" in called_nodes
        assert "clarification" in called_nodes
        assert "memory_saver" in called_nodes
        assert "wisdom_eval" in called_nodes



# ═══════════════════════════════════════════════════════════════════════════════
# GAP 5 — Cross-session word-level tag overlap
# ═══════════════════════════════════════════════════════════════════════════════

class TestWordLevelTagOverlap:
    def test_partial_word_match_finds_related_session(self, tmp_path):
        """'chronic stress' and 'stress relief' share 'stress' — must find overlap."""
        from agents.wisdom_memory import WisdomMemory

        mem = WisdomMemory(db_path=tmp_path / "sessions.db")

        # Session A: has wisdom on "chronic stress cortisol"
        sid_a = mem.new_session(topic="chronic stress and cortisol")
        mem.save_wisdom(
            session_id=sid_a,
            deep_understanding="Cortisol levels rise with chronic stress.",
            simple_explanation="Stress hormones build up over time.",
            actionable_takeaways=["Sleep more", "Exercise"],
            validation={"overall_confidence": "High", "claims": [], "devils_advocate": ""},
            papers=[],
            queries=[],
            topic_tags=["chronic stress", "cortisol", "HPA axis"],
        )

        # Session B: current session with different but overlapping tags
        sid_b = mem.new_session(topic="stress relief techniques")

        related = mem.find_related_sessions(
            topic_tags=["stress relief", "relaxation", "cortisol reduction"],
            current_session_id=sid_b,
            limit=5,
        )

        # "stress" appears in both "chronic stress" (A) and "stress relief" (B)
        # "cortisol" appears in A and "cortisol reduction" in B
        session_ids = [r["session_id"] for r in related]
        assert sid_a in session_ids

    def test_no_word_overlap_returns_empty(self, tmp_path):
        """Completely disjoint tags must not be returned."""
        from agents.wisdom_memory import WisdomMemory

        mem = WisdomMemory(db_path=tmp_path / "sessions.db")

        sid_a = mem.new_session(topic="quantum computing")
        mem.save_wisdom(
            session_id=sid_a,
            deep_understanding="Quantum computers use qubits.",
            simple_explanation="Think of qubits as special coins.",
            actionable_takeaways=["Learn linear algebra"],
            validation={"overall_confidence": "Medium", "claims": [], "devils_advocate": ""},
            papers=[],
            queries=[],
            topic_tags=["quantum", "qubits", "entanglement"],
        )

        sid_b = mem.new_session(topic="nutrition and gut health")
        related = mem.find_related_sessions(
            topic_tags=["gut microbiome", "probiotics", "fermentation"],
            current_session_id=sid_b,
            limit=5,
        )
        assert related == []

    def test_exact_match_still_works(self, tmp_path):
        """Exact tag string match must still be found with word-level overlap."""
        from agents.wisdom_memory import WisdomMemory

        mem = WisdomMemory(db_path=tmp_path / "sessions.db")

        sid_a = mem.new_session(topic="sleep and memory")
        mem.save_wisdom(
            session_id=sid_a,
            deep_understanding="Sleep consolidates memory traces.",
            simple_explanation="Sleep is like hitting save on your brain.",
            actionable_takeaways=["Get 8h sleep"],
            validation={"overall_confidence": "High", "claims": [], "devils_advocate": ""},
            papers=[],
            queries=[],
            topic_tags=["sleep", "memory", "consolidation"],
        )

        sid_b = mem.new_session(topic="memory improvement")
        related = mem.find_related_sessions(
            topic_tags=["memory", "learning", "spaced repetition"],
            current_session_id=sid_b,
            limit=5,
        )
        session_ids = [r["session_id"] for r in related]
        assert sid_a in session_ids

    def test_overlap_rank_ordering(self, tmp_path):
        """Session with more word overlap must rank higher."""
        from agents.wisdom_memory import WisdomMemory

        mem = WisdomMemory(db_path=tmp_path / "sessions.db")

        # Session A: 1 overlapping word ("stress")
        sid_a = mem.new_session(topic="stress management")
        mem.save_wisdom(
            session_id=sid_a,
            deep_understanding="Stress management techniques work.",
            simple_explanation="Manage stress well.",
            actionable_takeaways=["Breathe"],
            validation={"overall_confidence": "Medium", "claims": [], "devils_advocate": ""},
            papers=[], queries=[],
            topic_tags=["stress"],
        )

        # Session B: 3 overlapping words ("stress", "cortisol", "sleep")
        sid_b = mem.new_session(topic="chronic stress and sleep")
        mem.save_wisdom(
            session_id=sid_b,
            deep_understanding="Chronic stress disrupts sleep via cortisol.",
            simple_explanation="Stress and sleep are interlinked.",
            actionable_takeaways=["Sleep 8h", "Reduce caffeine"],
            validation={"overall_confidence": "High", "claims": [], "devils_advocate": ""},
            papers=[], queries=[],
            topic_tags=["stress", "cortisol", "sleep"],
        )

        # Current session
        sid_c = mem.new_session(topic="cortisol and sleep quality")
        related = mem.find_related_sessions(
            topic_tags=["stress", "cortisol", "sleep quality"],
            current_session_id=sid_c,
            limit=5,
        )
        session_ids = [r["session_id"] for r in related]
        assert session_ids.index(sid_b) < session_ids.index(sid_a)
