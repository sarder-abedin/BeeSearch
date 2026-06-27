"""backend/tests/test_jobs.py
──────────────────────────────
Unit tests for the in-memory background job runner (backend/app/jobs.py)
that backs every long-running pipeline's job-id + polling endpoints.
"""

from __future__ import annotations

import time

from backend.app.jobs import create_job, get_job, run_in_background


def _wait_until_terminal(job, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while job.status not in ("done", "error") and time.monotonic() < deadline:
        time.sleep(0.02)


def test_create_job_returns_unique_queued_jobs():
    a = create_job()
    b = create_job()
    assert a.id != b.id
    assert a.status == "queued"
    assert get_job(a.id) is a


def test_get_job_returns_none_for_unknown_id():
    assert get_job("does-not-exist") is None


def test_run_in_background_reaches_done_with_result_and_stage_updates():
    job = create_job()

    def fn(stream_callback):
        stream_callback("searching", {"question": "q"})
        stream_callback("done", {"citations": 0})
        return {"answer": "ok"}

    run_in_background(job, fn)
    _wait_until_terminal(job)

    assert job.status == "done"
    assert job.result == {"answer": "ok"}
    assert job.stage == "done"
    assert job.stage_info == {"citations": 0}
    assert job.error is None


def test_run_in_background_captures_exception_as_error_status():
    job = create_job()

    def fn(stream_callback):
        raise RuntimeError("boom")

    run_in_background(job, fn)
    _wait_until_terminal(job)

    assert job.status == "error"
    assert job.error == "boom"
    assert job.result is None
