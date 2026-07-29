"""backend/app/services/system_service.py
────────────────────────────────────────────
Service layer behind ``GET /api/system/status`` and ``POST
/api/system/shutdown`` -- the REST equivalent of ``ui/sidebar.py``'s
hardware detection, model recommendation, and Safe Shutdown button.

Reuses ``config/hardware.py`` and ``tools/temperature_levels.py`` unmodified,
same as every other service module in this package. No caching here (unlike
the Streamlit sidebar's ``@st.cache_data(ttl=30)``) -- a REST client decides
its own polling/refresh cadence, and the hardware probe + Ollama ``/api/tags``
call are cheap enough to run per request.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from typing import Optional

from config.hardware import (
    KNOWN_EMBED_MODELS,
    KNOWN_MODELS,
    detect_hardware,
    get_all_pulled_embed_models,
    get_available_embed_models,
    get_available_models,
    get_model_suggestion,
    get_recommended_tier,
    recommend_config,
)
from config.settings import get_settings
from tools.shutdown import safe_shutdown
from tools.temperature_levels import DEFAULT_TEMPERATURE_LEVEL, temperature_level_options

from ..schemas.system import (
    EmbedModelInfo,
    HardwareInfo,
    ModelRecommendation,
    ModelSuggestion,
    SafeAlternative,
    ShutdownResult,
    SystemStatusResponse,
    TemperatureLevelOption,
    TierInfo,
)

logger = logging.getLogger(__name__)
cfg = get_settings()

CONTEXT_WINDOW_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]


def get_system_status(ram_override_gb: Optional[float] = None) -> SystemStatusResponse:
    hw = detect_hardware()
    if ram_override_gb:
        hw = {**hw, "ram_gb": float(ram_override_gb)}

    available_models = get_available_models(cfg.ollama_base_url)
    available_embed = get_available_embed_models(cfg.ollama_base_url)
    all_pulled_embed = get_all_pulled_embed_models(cfg.ollama_base_url)
    rec = recommend_config(hw, available_models)
    tier = get_recommended_tier(hw)

    safe_alt = rec.get("safe_alternative")

    # Known embed models with pulled flag
    known_embed_names = {m["name"] for m in KNOWN_EMBED_MODELS}
    embed_models = [EmbedModelInfo(**m, pulled=m["name"] in available_embed) for m in KNOWN_EMBED_MODELS]
    # Append unknown pulled embed models not already in the known list
    for name in all_pulled_embed:
        if name not in known_embed_names:
            embed_models.append(EmbedModelInfo(name=name, pulled=True))

    # Per-model config suggestions for all pulled chat models
    model_suggestions = {}
    for model_name in available_models:
        suggestion = get_model_suggestion(model_name)
        if suggestion:
            model_suggestions[model_name] = ModelSuggestion(**suggestion)

    return SystemStatusResponse(
        hardware=HardwareInfo(**hw),
        tier=TierInfo(**tier),
        recommendation=ModelRecommendation(
            model=rec.get("model"),
            num_ctx=rec["num_ctx"],
            reasoning=rec["reasoning"],
            hardware_note=rec["hardware_note"],
            pull_command=rec.get("pull_command"),
            can_run=rec["can_run"],
            tight_fit=rec.get("tight_fit", False),
            safe_alternative=SafeAlternative(name=safe_alt["name"], ram_gb=safe_alt["ram_gb"]) if safe_alt else None,
        ),
        available_models=available_models,
        embed_models=embed_models,
        model_suggestions=model_suggestions,
        temperature_levels=[
            TemperatureLevelOption(key=key, label=label, description=desc)
            for key, label, desc in temperature_level_options()
        ],
        default_temperature_level=DEFAULT_TEMPERATURE_LEVEL,
        context_window_options=CONTEXT_WINDOW_OPTIONS,
        vision_model=cfg.vision_model,
    )


def shutdown() -> ShutdownResult:
    """Flush DB handles and terminate this process shortly after returning.

    Unlike ``ui/sidebar.py``'s button (which runs in a separate Streamlit
    process and frees the *backend's* port from the outside), this endpoint
    runs *inside* the FastAPI process being shut down, so it never calls
    ``free_port()`` on its own port or on Ollama (11434) -- killing Ollama
    here would affect every other client using the same local Ollama
    instance, not just this server. It only flushes ChromaDB and then exits
    its own process.

    The process exit is deferred to a background thread so the HTTP response
    (confirming the shutdown was accepted) actually reaches the client before
    the server goes away -- mirrors the two-step confirm already required on
    the frontend before this is ever called.
    """
    safe_shutdown(ports=[], flush_db=True)

    def _terminate() -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Timer(0.5, _terminate).start()
    return ShutdownResult(message="Safe shutdown complete -- server is stopping.")
