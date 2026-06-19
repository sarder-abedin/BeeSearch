"""
config/__init__.py
────────────────────
Re-export surface for the `config` package.

Lets callers write `from config import get_settings, detect_hardware, ...`
instead of reaching into `config.settings` / `config.hardware` directly.
Both submodules are imported eagerly here (unlike `tools/__init__.py`'s
lazy `__getattr__` pattern) since neither pulls in heavy ML dependencies.
"""

from config.settings import Settings, get_settings
from config.hardware import (
    detect_hardware, get_available_models, recommend_config, KNOWN_MODELS,
    TIER_CONFIGS, get_recommended_tier,
)

__all__ = [
    "Settings", "get_settings",
    "detect_hardware", "get_available_models", "recommend_config", "KNOWN_MODELS",
    "TIER_CONFIGS", "get_recommended_tier",
]
