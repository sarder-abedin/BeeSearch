"""backend/app/routers/health.py
──────────────────────────────────
Liveness probe + mock-LLM-mode visibility, so the frontend can show a dev
banner when running against the mock backend instead of a real Ollama.
"""

from __future__ import annotations

import os

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mock_llm": os.environ.get("BEESEARCH_MOCK_LLM") == "1"}
