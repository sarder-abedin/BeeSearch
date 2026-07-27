"""backend/tests/test_notebook_explain_service.py
────────────────────────────────────────────────────
Unit tests for backend/app/services/notebook_explain_service.py.

Two independent memory stores are swapped in per test, mirroring the dual
access pattern the service itself uses (see module docstring):
  - ``mem``: a real StorytellerMemory backed by a temp SQLite file, swapped
    in for the service's own lazy ``_story_memory`` singleton.
  - ``notebook_mem``: a real NotebookMemory backed by a separate temp SQLite
    file, swapped in for the service's ``NotebookMemory`` class reference
    itself (``_ensure_session`` instantiates a fresh one per call, the same
    "tests patch the class" boundary test_notebook_advanced_service.py uses).

The only LLM-touching call (``run_story_turn``) is mocked at the service
module's own import site, the same boundary test_notebook_service.py uses
for ``run_notebook_turn``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.notebook_memory import NotebookMemory
from agents.story_memory import StorytellerMemory
from backend.app.schemas.notebook_explain import ExplainRequest
from backend.app.services import notebook_explain_service as service


@pytest.fixture()
def mem(tmp_path, monkeypatch) -> StorytellerMemory:
    instance = StorytellerMemory(tmp_path / "story.db")
    monkeypatch.setattr(service, "_story_memory", instance)
    return instance


@pytest.fixture()
def notebook_mem(tmp_path, monkeypatch) -> NotebookMemory:
    instance = NotebookMemory(tmp_path / "notebooks.db")
    monkeypatch.setattr(service, "NotebookMemory", lambda *a, **kw: instance)
    return instance


# ─────────────────────────────────────────────────────────────────────────────
# _ensure_session
# ─────────────────────────────────────────────────────────────────────────────

def test_ensure_session_creates_session_seeded_with_notebook_name(mem, notebook_mem):
    notebook_id = notebook_mem.new_notebook("My Notes")

    service._ensure_session(notebook_id)

    session = mem.load(notebook_id)
    assert session is not None
    assert session["topic"] == "My Notes"
    assert session["conversation"] == []


def test_ensure_session_defaults_topic_when_notebook_has_no_name(mem, notebook_mem):
    notebook_id = notebook_mem.new_notebook("")

    service._ensure_session(notebook_id)

    assert mem.load(notebook_id)["topic"] == "Untitled Notebook"


def test_ensure_session_is_noop_when_session_already_exists(mem, notebook_mem):
    notebook_id = notebook_mem.new_notebook("My Notes")
    service._ensure_session(notebook_id)
    mem.add_turn(notebook_id, "user", "hello")

    service._ensure_session(notebook_id)

    assert len(mem.load(notebook_id)["conversation"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# build_initial_state
# ─────────────────────────────────────────────────────────────────────────────

def test_build_initial_state_only_forwards_provided_overrides():
    req = ExplainRequest(notebook_id="nb1", message="  q  ")

    state = service.build_initial_state(req)

    assert state["user_message"] == "q"
    assert state["session_id"] == "nb1"
    assert state["explanation_style"] == "simple"
    assert state["explanation_level"] == "intermediate"
    assert state["model_name"] == "llama3.1:8b"  # create_story_state's own default
    assert state["temperature_level"] == "focused"  # DEFAULT_TEMPERATURE_LEVEL


def test_build_initial_state_forwards_overrides_when_given():
    req = ExplainRequest(
        notebook_id="nb1",
        message="q",
        explanation_style="analogy",
        explanation_level="expert",
        model="llama3.1:8b",
        num_ctx=4096,
        temperature_level="creative",
    )

    state = service.build_initial_state(req)

    assert state["explanation_style"] == "analogy"
    assert state["explanation_level"] == "expert"
    assert state["model_name"] == "llama3.1:8b"
    assert state["num_ctx"] == 4096
    assert state["temperature_level"] == "creative"


# ─────────────────────────────────────────────────────────────────────────────
# run_explain_turn
# ─────────────────────────────────────────────────────────────────────────────

def test_run_explain_turn_ensures_session_and_normalizes_citations(mem, notebook_mem):
    notebook_id = notebook_mem.new_notebook("My Notes")
    fake_final_state = {
        "assistant_response": "Answer [1] and [Source 1].",
        "citations": [
            {"n": 1, "doc_name": "a.txt", "page": 3, "snippet": "..."},
            {"n": "Source 1", "doc_name": "Some Paper", "snippet": "...", "url": "https://x"},
        ],
        "suggested_questions": ["Q1?"],
        "progress_pct": 100,
    }

    def fake_run(initial_state, stream_callback=None):
        if stream_callback:
            stream_callback("storyteller", {**fake_final_state, "progress_pct": 65})
        return fake_final_state

    cb = MagicMock()
    with patch(
        "backend.app.services.notebook_explain_service.run_story_turn", side_effect=fake_run
    ):
        req = ExplainRequest(notebook_id=notebook_id, message="What?")
        result = service.run_explain_turn(req, cb)

    # _ensure_session ran as a side effect
    assert mem.load(notebook_id) is not None

    assert result["assistant_response"] == "Answer [1] and [Source 1]."
    assert result["notebook_id"] == notebook_id
    assert result["user_message"] == "What?"
    assert result["citations"][0]["n"] == "1"
    assert result["citations"][0]["page_label"] == "p. 4"
    assert result["citations"][1]["n"] == "Source 1"
    assert result["citations"][1]["page_label"] == "n/a"

    cb.assert_any_call(
        "storyteller", {"label": "Composing explanation", "progress_pct": 65}
    )


def test_run_explain_turn_defaults_missing_citations_to_empty_list(mem, notebook_mem):
    notebook_id = notebook_mem.new_notebook("My Notes")

    def fake_run(initial_state, stream_callback=None):
        return {"assistant_response": "No sources yet.", "progress_pct": 65}

    with patch(
        "backend.app.services.notebook_explain_service.run_story_turn", side_effect=fake_run
    ):
        req = ExplainRequest(notebook_id=notebook_id, message="What?")
        result = service.run_explain_turn(req, MagicMock())

    assert result["citations"] == []


def test_run_explain_turn_unknown_node_name_falls_back_to_raw_label(mem, notebook_mem):
    notebook_id = notebook_mem.new_notebook("My Notes")

    def fake_run(initial_state, stream_callback=None):
        if stream_callback:
            stream_callback("a_brand_new_node", {"progress_pct": 42})
        return {"assistant_response": "Hi", "progress_pct": 100}

    cb = MagicMock()
    with patch(
        "backend.app.services.notebook_explain_service.run_story_turn", side_effect=fake_run
    ):
        req = ExplainRequest(notebook_id=notebook_id, message="What?")
        service.run_explain_turn(req, cb)

    cb.assert_any_call("a_brand_new_node", {"label": "a_brand_new_node", "progress_pct": 42})


# ─────────────────────────────────────────────────────────────────────────────
# get_history
# ─────────────────────────────────────────────────────────────────────────────

def test_get_history_empty_for_notebook_never_used_for_explain(mem):
    assert service.get_history("never-used") == []


def test_get_history_normalizes_citations(mem, notebook_mem):
    notebook_id = notebook_mem.new_notebook("My Notes")
    service._ensure_session(notebook_id)
    mem.add_turn(
        notebook_id,
        "assistant",
        "Answer [1].",
        suggested_questions=["Q1?"],
        explanation_style="simple",
        citations=[{"n": 1, "doc_name": "a.txt", "page": 0, "snippet": "..."}],
    )

    history = service.get_history(notebook_id)

    assert len(history) == 1
    assert history[0]["citations"][0]["n"] == "1"
    assert history[0]["citations"][0]["page_label"] == "p. 1"
