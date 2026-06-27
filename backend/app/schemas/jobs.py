"""backend/app/schemas/jobs.py
───────────────────────────────
Shared Pydantic shapes for the background-job + polling pattern used by
every long-running pipeline endpoint. Each pipeline's router defines its own
``result`` field on top of :class:`JobStatusBase` (the result shape differs
per pipeline) via a subclass — see ``schemas/research_assistant.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class JobCreated(BaseModel):
    job_id: str


class JobStatusBase(BaseModel):
    id: str
    status: str  # queued | running | done | error
    stage: Optional[str] = None
    stage_info: Dict[str, Any] = {}
    error: Optional[str] = None
