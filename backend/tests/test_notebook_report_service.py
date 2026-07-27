"""backend/tests/test_notebook_report_service.py
──────────────────────────────────────────────────
Unit tests for backend/app/services/notebook_report_service.py.

``agents.graph.run_research`` is mocked at the service module's own import
site, the same boundary every other phase's service test uses for its
pipeline entry point. ``NotebookMemory`` is swapped for a bare ``MagicMock``
whose ``.load()`` returns a hand-built notebook dict -- ``run_report`` only
ever calls ``.load()`` on it, so a full SQLite-backed instance (as other
phases' tests use, since their services call more than one NotebookMemory
method) isn't needed here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.schemas.notebook_report import ReportRequest
from backend.app.services import notebook_report_service as service


def _notebook(sources=None, chunks=None):
    return {"sources": sources or [], "chunks": chunks or []}


@pytest.fixture()
def notebook_mem(monkeypatch):
    instance = MagicMock()
    monkeypatch.setattr(service, "NotebookMemory", lambda *a, **kw: instance)
    return instance


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_mode
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_mode_no_sources_is_always_search():
    assert service._resolve_mode(_notebook(), include_academic=True) == "search"
    assert service._resolve_mode(_notebook(), include_academic=False) == "search"


def test_resolve_mode_with_sources_and_academic_is_hybrid():
    nb = _notebook(sources=[{"doc_id": "d1", "filename": "a.txt"}])
    assert service._resolve_mode(nb, include_academic=True) == "hybrid"


def test_resolve_mode_with_sources_and_no_academic_is_document():
    nb = _notebook(sources=[{"doc_id": "d1", "filename": "a.txt"}])
    assert service._resolve_mode(nb, include_academic=False) == "document"


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_references
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_references_stringifies_int_year():
    refs = [{"ref_num": 1, "year": 2023}]
    assert service._normalize_references(refs)[0]["year"] == "2023"


def test_normalize_references_defaults_falsy_year_to_empty_string():
    refs = [{"ref_num": 1, "year": ""}, {"ref_num": 2, "year": None}, {"ref_num": 3}]
    normalized = service._normalize_references(refs)
    assert [r["year"] for r in normalized] == ["", "", ""]


# ─────────────────────────────────────────────────────────────────────────────
# build_initial_state
# ─────────────────────────────────────────────────────────────────────────────

def test_build_initial_state_defaults_overrides_from_config():
    req = ReportRequest(notebook_id="nb1", goal="  What is X?  ")
    state = service.build_initial_state(req, _notebook())

    assert state["goal"] == "What is X?"
    assert state["mode"] == "search"
    assert state["uploaded_docs"] == []
    assert state["include_web_search"] is False
    assert state["model_name"] == service.cfg.ollama_model
    assert state["num_ctx"] == service.cfg.num_ctx
    assert state["embed_model"] == service.cfg.embedding_model


def test_build_initial_state_forwards_overrides_when_given():
    req = ReportRequest(
        notebook_id="nb1",
        goal="What is X?",
        include_web=True,
        model="llama3.1:8b",
        num_ctx=4096,
        embed_model="mxbai-embed-large",
    )
    state = service.build_initial_state(req, _notebook())

    assert state["include_web_search"] is True
    assert state["model_name"] == "llama3.1:8b"
    assert state["num_ctx"] == 4096
    assert state["embed_model"] == "mxbai-embed-large"


def test_build_initial_state_resolves_hybrid_mode_with_sources():
    req = ReportRequest(notebook_id="nb1", goal="What is X?", include_academic=True)
    nb = _notebook(sources=[{"doc_id": "d1", "filename": "a.txt"}])

    state = service.build_initial_state(req, nb)

    assert state["mode"] == "hybrid"


def test_build_initial_state_resolves_document_mode_without_academic():
    req = ReportRequest(notebook_id="nb1", goal="What is X?", include_academic=False)
    nb = _notebook(sources=[{"doc_id": "d1", "filename": "a.txt"}])

    state = service.build_initial_state(req, nb)

    assert state["mode"] == "document"


# ─────────────────────────────────────────────────────────────────────────────
# run_report
# ─────────────────────────────────────────────────────────────────────────────

def test_run_report_normalizes_references_and_echoes_notebook_id_and_mode(notebook_mem):
    notebook_mem.load.return_value = _notebook()
    fake_final_state = {
        "report": "# Report",
        "key_findings": ["Finding 1"],
        "references": [{"ref_num": 1, "year": 2023}],
        "eval_result": {"overall": 4},
        "errors": [],
        "progress_pct": 100,
    }

    def fake_run(initial_state, stream_callback=None):
        if stream_callback:
            stream_callback("report_generation", {**fake_final_state, "progress_pct": 95})
        return fake_final_state

    cb = MagicMock()
    with patch("backend.app.services.notebook_report_service.run_research", side_effect=fake_run):
        req = ReportRequest(notebook_id="nb1", goal="What is X?")
        result = service.run_report(req, cb)

    assert result["notebook_id"] == "nb1"
    assert result["mode"] == "search"
    assert result["references"][0]["year"] == "2023"
    cb.assert_any_call("report_generation", {"label": "Generating report", "progress_pct": 95})


def test_run_report_unknown_node_name_falls_back_to_raw_label(notebook_mem):
    notebook_mem.load.return_value = _notebook()

    def fake_run(initial_state, stream_callback=None):
        if stream_callback:
            stream_callback("a_brand_new_node", {"progress_pct": 42})
        return {"report": "", "references": [], "progress_pct": 100}

    cb = MagicMock()
    with patch("backend.app.services.notebook_report_service.run_research", side_effect=fake_run):
        req = ReportRequest(notebook_id="nb1", goal="What is X?")
        service.run_report(req, cb)

    cb.assert_any_call("a_brand_new_node", {"label": "a_brand_new_node", "progress_pct": 42})


def test_run_report_missing_notebook_treated_as_empty(notebook_mem):
    notebook_mem.load.return_value = None

    def fake_run(initial_state, stream_callback=None):
        return {"report": "", "references": [], "progress_pct": 100}

    with patch("backend.app.services.notebook_report_service.run_research", side_effect=fake_run):
        req = ReportRequest(notebook_id="missing", goal="What is X?")
        result = service.run_report(req, MagicMock())

    assert result["mode"] == "search"
