"""backend/app/routers/system.py
───────────────────────────────────
Hardware status, model recommendation, and settings metadata over HTTP --
the REST equivalent of ``ui/sidebar.py``'s top-level sections, consumed by
the React settings panel.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ..schemas.system import ShutdownResult, SystemStatusResponse
from ..services import system_service

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status", response_model=SystemStatusResponse)
def get_status(
    ram_override_gb: Optional[float] = Query(
        None, gt=0, description="Override detected RAM (GB) -- for Docker hosts under-reporting host RAM."
    ),
) -> SystemStatusResponse:
    return system_service.get_system_status(ram_override_gb=ram_override_gb)


@router.post("/shutdown", response_model=ShutdownResult)
def shutdown() -> ShutdownResult:
    return system_service.shutdown()
