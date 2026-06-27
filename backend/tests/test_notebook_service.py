"""backend/tests/test_notebook_service.py
────────────────────────────────────────────
Unit tests for backend/app/services/notebook_service.py against a real
NotebookMemory backed by a temp SQLite file (swapped in for the module's
lazy singleton, the same "tests can inject a different instance" pattern
agents/notebook_nodes.py's own `_memory` singleton documents). The only
LLM-touching call (``run_notebook_turn``) is mocked at the
notebook_service module's own import site -- the same boundary
test_research_assistant_service.py uses for run_research_assistant.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.notebook_memory import NotebookMemory
from backend.app.schemas.notebook import ChatRequest, CreateNotebookRequest
from backend.app.services import notebook_service
from tools.hybrid_store import _stores


@pytest.fixture()
def mem(tmp_path, monkeypatch) -> NotebookMemory:
    instance = NotebookMemory(tmp_path / "notebooks.db")
    monkeypatch.setattr(notebook_service, "_memory", instance)
    return instance


def _txt(text: str) -> bytes:
    return text.encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Notebook CRUD
# ─────────────────────────────────────────────────────────────────────────────

def test_create_and_get_notebook_detail_round_trip(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="My Notes"))
    assert summary.name == "My Notes"
    assert summary.source_count == 0
    assert summary.turn_count == 0

    detail = notebook_service.get_notebook_detail(summary.notebook_id)
    assert detail is not None
    assert detail.notebook_id == summary.notebook_id
    assert detail.sources == []
    assert detail.conversation == []


def test_create_notebook_blank_name_defaults_to_untitled(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name=""))
    assert summary.name == "Untitled Notebook"


def test_get_notebook_detail_unknown_id_returns_none(mem):
    assert notebook_service.get_notebook_detail("nope") is None


def test_notebook_exists(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="X"))
    assert notebook_service.notebook_exists(summary.notebook_id) is True
    assert notebook_service.notebook_exists("nope") is False


def test_list_notebooks_returns_created_notebooks_newest_first(mem):
    notebook_service.create_notebook(CreateNotebookRequest(name="First"))
    second = notebook_service.create_notebook(CreateNotebookRequest(name="Second"))
    notebooks = notebook_service.list_notebooks()
    assert notebooks[0].notebook_id == second.notebook_id
    assert len(notebooks) == 2


def test_delete_notebook(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="X"))
    assert notebook_service.delete_notebook(summary.notebook_id) is True
    assert notebook_service.get_notebook_detail(summary.notebook_id) is None


def test_delete_notebook_unknown_id_returns_false(mem):
    assert notebook_service.delete_notebook("nope") is False


def test_rename_notebook(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="Old"))
    renamed = notebook_service.rename_notebook(summary.notebook_id, "New Name")
    assert renamed is not None
    assert renamed.name == "New Name"
    assert notebook_service.get_notebook_detail(summary.notebook_id).name == "New Name"


def test_rename_notebook_blank_name_keeps_current(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="Keep Me"))
    renamed = notebook_service.rename_notebook(summary.notebook_id, "   ")
    assert renamed.name == "Keep Me"


def test_rename_notebook_unknown_id_returns_none(mem):
    assert notebook_service.rename_notebook("nope", "New") is None


def test_get_history_empty_for_new_notebook(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="X"))
    assert notebook_service.get_history(summary.notebook_id) == []


# ─────────────────────────────────────────────────────────────────────────────
# Source upload / removal
# ─────────────────────────────────────────────────────────────────────────────

def test_upload_source_txt_adds_source_and_evicts_store(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="X"))
    _stores[f"notebook_{summary.notebook_id}"] = object()
    _stores[f"notebook_{summary.notebook_id}_bm25"] = object()

    result = notebook_service.upload_source(
        summary.notebook_id, "notes.txt", _txt("Hello world, this is page one.")
    )

    assert result.added is True
    assert result.duplicate is False
    assert result.source is not None
    assert result.source.filename == "notes.txt"
    assert result.source.total_chunks >= 1
    assert f"notebook_{summary.notebook_id}" not in _stores
    assert f"notebook_{summary.notebook_id}_bm25" not in _stores

    detail = notebook_service.get_notebook_detail(summary.notebook_id)
    assert detail.source_count == 1


def test_upload_source_duplicate_returns_added_false(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="X"))
    content = _txt("Same content twice.")
    notebook_service.upload_source(summary.notebook_id, "a.txt", content)

    result = notebook_service.upload_source(summary.notebook_id, "a.txt", content)

    assert result.added is False
    assert result.duplicate is True
    assert result.source is None
    assert notebook_service.get_notebook_detail(summary.notebook_id).source_count == 1


def test_upload_source_unknown_notebook_raises_keyerror(mem):
    with pytest.raises(KeyError):
        notebook_service.upload_source("nope", "a.txt", _txt("x"))


def test_upload_source_unsupported_file_type_raises_valueerror(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="X"))
    with pytest.raises(ValueError):
        notebook_service.upload_source(summary.notebook_id, "a.exe", b"binary junk")


def test_remove_source(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="X"))
    result = notebook_service.upload_source(summary.notebook_id, "a.txt", _txt("Hello"))
    _stores[f"notebook_{summary.notebook_id}"] = object()

    removed = notebook_service.remove_source(summary.notebook_id, result.source.doc_id)

    assert removed is True
    assert f"notebook_{summary.notebook_id}" not in _stores
    assert notebook_service.get_notebook_detail(summary.notebook_id).source_count == 0


def test_remove_source_unknown_doc_id_returns_false(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="X"))
    assert notebook_service.remove_source(summary.notebook_id, "nope") is False


# ─────────────────────────────────────────────────────────────────────────────
# page_label derivation (computed, never persisted -- see module docstring)
# ─────────────────────────────────────────────────────────────────────────────

def test_with_page_labels_computes_from_raw_page():
    out = notebook_service._with_page_labels([{"n": 1, "page": 3}])
    assert out[0]["page_label"] == "p. 4"


def test_with_page_labels_unknown_page_is_not_applicable():
    out = notebook_service._with_page_labels([{"n": 1, "page": -1}])
    assert out[0]["page_label"] == "n/a"


def test_with_page_labels_passes_through_none_and_empty():
    assert notebook_service._with_page_labels(None) is None
    assert notebook_service._with_page_labels([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Chat (build_initial_state + run_chat_turn)
# ─────────────────────────────────────────────────────────────────────────────

def test_build_initial_state_only_forwards_provided_overrides():
    req = ChatRequest(notebook_id="nb1", message="  q  ")
    state = notebook_service.build_initial_state(req)
    assert state["user_message"] == "q"
    assert state["notebook_id"] == "nb1"
    assert state["model_name"] == "llama3.1:8b"  # create_notebook_state's own default


def test_build_initial_state_forwards_overrides_when_given():
    req = ChatRequest(
        notebook_id="nb1",
        message="q",
        model="llama3.1:8b",
        num_ctx=4096,
        top_k=5,
        temperature_level="creative",
        include_web_search=True,
    )
    state = notebook_service.build_initial_state(req)
    assert state["model_name"] == "llama3.1:8b"
    assert state["num_ctx"] == 4096
    assert state["top_k"] == 5
    assert state["temperature_level"] == "creative"
    assert state["include_web_search"] is True


def test_run_chat_turn_adapts_callback_and_adds_page_labels(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="X"))
    fake_final_state = {
        "assistant_response": "Answer [1].",
        "citations": [{"n": 1, "doc_name": "a.txt", "page": 3, "snippet": "..."}],
        "suggested_questions": ["Q1?"],
        "progress_pct": 100,
    }

    def fake_run(initial_state, stream_callback=None):
        if stream_callback:
            stream_callback("answer", {**fake_final_state, "progress_pct": 80})
        return fake_final_state

    cb = MagicMock()
    with patch(
        "backend.app.services.notebook_service.run_notebook_turn", side_effect=fake_run
    ):
        req = ChatRequest(notebook_id=summary.notebook_id, message="What?")
        result = notebook_service.run_chat_turn(req, cb)

    assert result["assistant_response"] == "Answer [1]."
    assert result["citations"][0]["page_label"] == "p. 4"
    assert result["suggested_questions"] == ["Q1?"]
    cb.assert_any_call("answer", {"label": "Composing grounded answer", "progress_pct": 80})


def test_run_chat_turn_defaults_missing_citations_to_empty_list(mem):
    summary = notebook_service.create_notebook(CreateNotebookRequest(name="X"))

    def fake_run(initial_state, stream_callback=None):
        return {"assistant_response": "No sources yet.", "progress_pct": 80}

    with patch(
        "backend.app.services.notebook_service.run_notebook_turn", side_effect=fake_run
    ):
        req = ChatRequest(notebook_id=summary.notebook_id, message="What?")
        result = notebook_service.run_chat_turn(req, MagicMock())

    assert result["citations"] == []
