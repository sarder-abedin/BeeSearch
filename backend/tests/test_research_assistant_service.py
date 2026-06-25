"""backend/tests/test_research_assistant_service.py
──────────────────────────────────────────────────────
Unit tests for backend/app/services/research_assistant_service.py — the
thin settings-dict-building wrapper around
agents.research_assistant.run_research_assistant.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.app.schemas.research_assistant import AskRequest
from backend.app.services.research_assistant_service import build_settings, run_ask


def test_build_settings_omits_optional_overrides_when_unset():
    """Omitted overrides defer to run_research_assistant's own fallback chain."""
    req = AskRequest(question="q")
    assert build_settings(req) == {"include_crossref": True}


def test_build_settings_includes_overrides_when_provided():
    req = AskRequest(
        question="q",
        model="llama3.1:8b",
        num_ctx=4096,
        temperature_level="creative",
        include_crossref=False,
    )
    assert build_settings(req) == {
        "include_crossref": False,
        "model": "llama3.1:8b",
        "num_ctx": 4096,
        "temperature_level": "creative",
    }


def test_run_ask_strips_question_and_forwards_settings_and_include_web():
    req = AskRequest(question="  what is x?  ", include_web=False)
    fake_result = {
        "question": "what is x?", "answer": "a", "citations": [], "sources": [],
        "academic_count": 0, "web_count": 0, "suggested_questions": [], "grounded": False,
    }
    with patch(
        "backend.app.services.research_assistant_service.run_research_assistant",
        return_value=fake_result,
    ) as mock_run:
        cb = MagicMock()
        result = run_ask(req, cb)

    assert result is fake_result
    mock_run.assert_called_once_with(
        "what is x?", {"include_crossref": True}, stream_callback=cb, include_web=False,
    )
