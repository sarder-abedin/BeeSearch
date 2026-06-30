"""backend/tests/test_notebook_pipeline_service.py
──────────────────────────────────────────────────
Unit tests for backend/app/services/notebook_pipeline_service.py.

``run_notebook_pipeline`` is mocked at the service module's own import site
(it does ``from agents.notebook_pipeline_graph import run_notebook_pipeline``
at module level, so that name is bound directly into
``notebook_pipeline_service``'s namespace) -- the same boundary
test_notebook_service.py uses for run_notebook_turn. The export helpers
(``render_dot_bytes`` / ``build_docx`` / ``build_pdf``) do their own LOCAL
imports inside the function body, so they're mocked at their *defining*
module instead -- the same pattern test_systematic_review_api.py uses for
``tools.prisma_report.generate_prisma_docx``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.app.schemas.notebook_pipeline import PipelineRequest
from backend.app.services import notebook_pipeline_service as service

# ─────────────────────────────────────────────────────────────────────────────
# build_initial_state
# ─────────────────────────────────────────────────────────────────────────────

def test_build_initial_state_only_forwards_provided_overrides():
    req = PipelineRequest(notebook_id="nb1")
    state = service.build_initial_state(req)

    assert state["notebook_id"] == "nb1"
    assert state["query"] == ""
    assert state["settings"] == {}
    # create_pipeline_state's own defaults, untouched.
    assert state["sources"] == []
    assert state["retrieval_mode"] == "empty"
    assert state["progress_pct"] == 0


def test_build_initial_state_forwards_overrides_when_given():
    req = PipelineRequest(
        notebook_id="nb1",
        query="explain the methodology",
        model="llama3.1:8b",
        num_ctx=4096,
        embed_model="nomic-embed-text",
        top_k=5,
        temperature_level="creative",
    )
    state = service.build_initial_state(req)

    assert state["query"] == "explain the methodology"
    assert state["settings"] == {
        "model": "llama3.1:8b",
        "num_ctx": 4096,
        "embed_model": "nomic-embed-text",
        "top_k": 5,
        "temperature_level": "creative",
    }


def test_build_initial_state_blank_query_defaults_to_empty_string():
    req = PipelineRequest(notebook_id="nb1", query="   ")
    state = service.build_initial_state(req)
    # Pydantic doesn't strip; build_initial_state forwards `req.query or ""` verbatim.
    assert state["query"] == "   "


# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline (callback adaptation)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_pipeline_adapts_callback_with_node_label_and_progress():
    fake_final_state = {
        "notebook_id": "nb1",
        "doc_count": 1,
        "cross_summary": "Summary.",
        "progress_pct": 100,
    }

    def fake_run(initial_state, stream_callback=None):
        if stream_callback:
            stream_callback("summarize", {**fake_final_state, "progress_pct": 29})
        return fake_final_state

    cb = MagicMock()
    with patch(
        "backend.app.services.notebook_pipeline_service.run_notebook_pipeline",
        side_effect=fake_run,
    ):
        result = service.run_pipeline(PipelineRequest(notebook_id="nb1"), cb)

    assert result["cross_summary"] == "Summary."
    assert result["notebook_id"] == "nb1"
    cb.assert_any_call("summarize", {"label": "Agent 2 — Summarization", "progress_pct": 29})


def test_run_pipeline_unknown_node_name_falls_back_to_raw_name():
    def fake_run(initial_state, stream_callback=None):
        if stream_callback:
            stream_callback("some_future_node", {"progress_pct": 50})
        return {}

    cb = MagicMock()
    with patch(
        "backend.app.services.notebook_pipeline_service.run_notebook_pipeline",
        side_effect=fake_run,
    ):
        service.run_pipeline(PipelineRequest(notebook_id="nb1"), cb)

    cb.assert_any_call("some_future_node", {"label": "some_future_node", "progress_pct": 50})


def test_run_pipeline_defaults_notebook_id_when_missing_from_result():
    def fake_run(initial_state, stream_callback=None):
        return {"doc_count": 0}  # no "notebook_id" key in the raw final state

    with patch(
        "backend.app.services.notebook_pipeline_service.run_notebook_pipeline",
        side_effect=fake_run,
    ):
        result = service.run_pipeline(PipelineRequest(notebook_id="nb-xyz"), MagicMock())

    assert result["notebook_id"] == "nb-xyz"


def test_run_pipeline_keeps_existing_notebook_id_from_result():
    def fake_run(initial_state, stream_callback=None):
        return {"notebook_id": "from-state"}

    with patch(
        "backend.app.services.notebook_pipeline_service.run_notebook_pipeline",
        side_effect=fake_run,
    ):
        result = service.run_pipeline(PipelineRequest(notebook_id="from-request"), MagicMock())

    assert result["notebook_id"] == "from-state"


# ─────────────────────────────────────────────────────────────────────────────
# Exports (sync -- no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def test_build_knowledge_graph_image_delegates_to_render_dot_bytes():
    with patch(
        "agents.notebook_advanced.render_dot_bytes", return_value=(b"PNGDATA", "")
    ) as mock_render:
        image, error = service.build_knowledge_graph_image("digraph{}", "png")

    assert image == b"PNGDATA"
    assert error == ""
    mock_render.assert_called_once_with("digraph{}", "png")


def test_build_study_guide_docx_delegates_to_build_docx():
    with patch("tools.export_tools.build_docx", return_value=b"DOCX") as mock_build:
        content = service.build_study_guide_docx("## Guide")

    assert content == b"DOCX"
    mock_build.assert_called_once_with("## Guide", [])


def test_build_study_guide_pdf_delegates_to_build_pdf():
    with patch("tools.export_tools.build_pdf", return_value=b"PDF") as mock_build:
        content = service.build_study_guide_pdf("## Guide")

    assert content == b"PDF"
    mock_build.assert_called_once_with("## Guide", [])
