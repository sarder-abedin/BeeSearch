"""
tools/embeddings.py
───────────────────
Ollama-backed embedding client for hybrid RAG.

Uses Ollama's /api/embed endpoint to generate dense vector representations
of document chunks and search queries.  All embedding happens locally —
no cloud API keys required.

Default model: nomic-embed-text (768-dim, MIT licence, ~274 MB)
Other options: mxbai-embed-large (1024-dim), bge-m3 (1024-dim), all-minilm (384-dim)

Pull before first use:
    ollama pull nomic-embed-text
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import requests

logger = logging.getLogger(__name__)

_BATCH_SIZE = 32     # default chunks per /api/embed call
_MIN_BATCH  = 1      # floor when auto-reducing due to memory pressure


class OllamaEmbedder:
    """Thin wrapper around Ollama's /api/embed endpoint."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dim: int | None = None  # cached after first call

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_texts(
        self,
        texts: List[str],
        warning_callback: Optional[Callable[[str], None]] = None,
    ) -> List[List[float]]:
        """
        Embed a list of texts in batches.
        Returns a list of float vectors, one per input text.

        Automatically retries with half the batch size when the Ollama
        server returns a 5xx error (typically out-of-VRAM).  Emits a
        warning via *warning_callback* (if provided) each time it reduces
        the batch size so the UI can surface the degradation.

        Raises RuntimeError if the embedding model is not available.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = _BATCH_SIZE
        i = 0

        while i < len(texts):
            batch = texts[i : i + batch_size]
            try:
                batch_embs = self._call_api(batch)
                all_embeddings.extend(batch_embs)
                i += batch_size  # advance only on success
            except RuntimeError as exc:
                msg = str(exc)
                is_server_error = (
                    "500" in msg or "502" in msg or "503" in msg or "504" in msg
                    or "out of memory" in msg.lower()
                    or "oom" in msg.lower()
                )
                if is_server_error and batch_size > _MIN_BATCH:
                    batch_size = max(_MIN_BATCH, batch_size // 2)
                    warn = (
                        f"Embedding server returned an error (likely low VRAM). "
                        f"Retrying with batch size {batch_size}. "
                        f"Processing will be slower but should complete."
                    )
                    logger.warning(warn)
                    if warning_callback:
                        warning_callback(warn)
                else:
                    raise

        if all_embeddings:
            self._dim = len(all_embeddings[0])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int | None:
        """Vector dimensionality — available after the first embed call."""
        return self._dim

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and the model is pulled."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            base = self.model.split(":")[0]
            return any(m == self.model or m.startswith(base + ":") for m in models)
        except Exception:
            return False

    # ── Private ────────────────────────────────────────────────────────────────

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """POST to /api/embed and parse the response."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings:
                return embeddings
            # Fallback: older Ollama versions return a single "embedding" key
            single = data.get("embedding")
            if single:
                return [single]
            raise RuntimeError(f"Unexpected response from /api/embed: {data}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure `ollama serve` is running."
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise RuntimeError(
                    f"Embedding model '{self.model}' is not pulled. "
                    f"Run: ollama pull {self.model}"
                )
            raise RuntimeError(f"Ollama embedding error: {e}")
