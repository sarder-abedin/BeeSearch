"""config/observability.py
─────────────────────────
Langfuse LLM observability integration (opt-in).

`get_langfuse_callbacks()` returns a list containing the shared CallbackHandler
when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in the environment,
or an empty list otherwise — so callers pass it directly to `callbacks=` without
any conditional logic at the call site.

Usage (in any ChatOllama factory):

    from config.observability import get_langfuse_callbacks

    return ChatOllama(..., callbacks=get_langfuse_callbacks())

Self-hosted setup:
    docker compose -f docker-compose.langfuse.yml up -d
    # Open http://localhost:3000, create account, copy API keys to .env
"""

from __future__ import annotations

from typing import Any

_handler = None  # module-level singleton; Langfuse's internal batch queue is thread-safe


def get_langfuse_callbacks() -> list[Any]:
    """Return [CallbackHandler] if Langfuse is configured, else [].

    The handler is created once and reused — creating a new one per LLM call
    would open a separate HTTP session each time.
    """
    from config.settings import get_settings

    cfg = get_settings()
    if not (cfg.langfuse_public_key and cfg.langfuse_secret_key):
        return []

    global _handler
    if _handler is None:
        # SDK v2 (langfuse.callback): accepts public_key/secret_key/host as kwargs.
        try:
            from langfuse.callback import CallbackHandler  # type: ignore[import]
            _handler = CallbackHandler(
                public_key=cfg.langfuse_public_key,
                secret_key=cfg.langfuse_secret_key,
                host=cfg.langfuse_host,
            )
            return [_handler]
        except ImportError:
            pass  # fall through to SDK v3 path
        except Exception:
            return []

        # SDK v3 (langfuse.langchain): reads credentials from env vars only.
        # Propagate the settings values into os.environ so the handler picks
        # them up even when pydantic-settings reads .env without touching os.environ.
        try:
            import os
            from langfuse.langchain import CallbackHandler  # type: ignore[import]
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", cfg.langfuse_public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", cfg.langfuse_secret_key)
            if cfg.langfuse_host:
                os.environ.setdefault("LANGFUSE_HOST", cfg.langfuse_host)
            _handler = CallbackHandler()
        except Exception:
            return []

    return [_handler] if _handler is not None else []


def flush_langfuse() -> None:
    """Flush any pending Langfuse events — call on process shutdown."""
    if _handler is not None:
        try:
            _handler.flush()
        except Exception:
            pass
