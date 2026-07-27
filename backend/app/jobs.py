"""backend/app/jobs.py
──────────────────────
In-memory background job runner shared by every long-running pipeline
endpoint (Mode 3 now; Mode 1/Mode 2 later). Wraps a blocking pipeline call
so an API route can return a job id immediately, while the frontend polls
``GET .../jobs/{id}`` for the same ``stream_callback(stage, info)`` progress
the CLI and Streamlit surfaces already render live (see
``agents/research_assistant.py``, ``main.py::_cmd_ask``).

In-memory and single-process by design, matching BeeSearch's local-first,
single-user architecture — jobs are lost on restart, which is acceptable
here (same trade-off the rest of the app makes outside of the SQLite-backed
Research Notebook memory).
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    stage: Optional[str] = None
    stage_info: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


_jobs: Dict[str, Job] = {}
_lock = threading.Lock()


def create_job() -> Job:
    job = Job(id=uuid.uuid4().hex)
    with _lock:
        _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def run_in_background(
    job: Job, fn: Callable[[Callable[[str, Dict[str, Any]], None]], Dict[str, Any]]
) -> None:
    """Run ``fn(stream_callback)`` on a background thread, updating *job* as it progresses.

    A plain thread (not asyncio) because the wrapped pipelines are themselves
    synchronous/blocking — sync httpx calls to Ollama and the search APIs —
    exactly as the CLI and Streamlit already invoke them.
    """

    def _on_stage(stage: str, info: Dict[str, Any]) -> None:
        with _lock:
            job.stage = stage
            job.stage_info = info

    def _worker() -> None:
        with _lock:
            job.status = "running"
        try:
            result = fn(_on_stage)
            with _lock:
                job.result = result
                job.status = "done"
        except Exception as e:
            logger.exception("Background job %s failed", job.id)
            with _lock:
                job.error = str(e)
                job.status = "error"

    threading.Thread(target=_worker, daemon=True, name=f"job-{job.id[:8]}").start()
