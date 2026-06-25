"""backend/app/main.py
───────────────────────
FastAPI application entry point for BeeSearch's React frontend.

Added alongside the existing CLI (``main.py``) and Streamlit UI (``app.py``)
-- not a replacement for either. Reuses the same ``agents``/``config``/
``tools`` packages those two surfaces already call, so all three surfaces
share one set of pipelines.

Run (from the repo root):
    python -m uvicorn backend.app.main:app --reload --port 8000

Dev-only mock LLM + mock search, no Ollama or network required (see
``mock_llm.py`` / ``mock_search.py``):
    BEESEARCH_MOCK_LLM=1 python -m uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Must run before any `agents.*` module is imported, directly or transitively --
# those modules bind `from langchain_ollama import ChatOllama` into their own
# namespace at import time, so patching afterwards would be too late. (The
# mock search install has no such ordering constraint, but lives here too
# since it's gated by the same flag.)
if os.environ.get("BEESEARCH_MOCK_LLM") == "1":
    from .mock_llm import install_mock_llm
    from .mock_search import install_mock_search

    install_mock_llm()
    install_mock_search()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings

from .routers import health, research_assistant

cfg = get_settings()
logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(title="BeeSearch API", version="0.1.0")

_DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("BEESEARCH_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


app.include_router(health.router)
app.include_router(research_assistant.router)
