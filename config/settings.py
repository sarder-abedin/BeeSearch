"""
config/settings.py
──────────────────
Centralised configuration loaded from environment variables (.env).
Using Pydantic BaseSettings so every value is typed and validated at
startup — no silent misconfigurations at runtime.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings  # pip install pydantic-settings

# Load .env from project root (two levels up from this file)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


class Settings(BaseSettings):
    # ── Local LLM ───────────────────────────────────────────
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.1:8b", alias="OLLAMA_MODEL")
    num_ctx: int = Field(32768, alias="NUM_CTX")

    # ── Hybrid RAG ───────────────────────────────────────────
    embedding_model: str = Field("nomic-embed-text", alias="EMBED_MODEL")
    chroma_persist_dir: str = Field("./outputs/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("research_embeddings", alias="CHROMA_COLLECTION_NAME")
    hybrid_top_k: int = Field(8, alias="HYBRID_TOP_K")

    # ── Docker / Deployment ──────────────────────────────────
    app_port: int = Field(8501, alias="APP_PORT")

    # ── Semantic Scholar ─────────────────────────────────────
    semantic_scholar_api_key: str = Field("", alias="SEMANTIC_SCHOLAR_API_KEY")
    semantic_scholar_base_url: str = Field(
        "https://api.semanticscholar.org/graph/v1",
        alias="SEMANTIC_SCHOLAR_BASE_URL",
    )

    # ── CrossRef ─────────────────────────────────────────────
    crossref_base_url: str = Field(
        "https://api.crossref.org/works", alias="CROSSREF_BASE_URL"
    )
    crossref_email: str = Field("researcher@example.com", alias="CROSSREF_EMAIL")

    # ── arXiv ────────────────────────────────────────────────
    arxiv_max_results: int = Field(10, alias="ARXIV_MAX_RESULTS")

    # ── Document Processing ──────────────────────────────────
    max_document_chunks: int = Field(500, alias="MAX_DOCUMENT_CHUNKS")
    chunk_size: int = Field(800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(150, alias="CHUNK_OVERLAP")
    docling_models_path: str = Field("models/docling", alias="DOCLING_MODELS_PATH")

    # ── Search ───────────────────────────────────────────────
    max_search_results: int = Field(8, alias="MAX_SEARCH_RESULTS")

    # ── Output ───────────────────────────────────────────────
    output_dir: str = Field("./outputs", alias="OUTPUT_DIR")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    class Config:
        env_file = str(_ROOT / ".env")
        env_file_encoding = "utf-8"
        populate_by_name = True

    # ── Helpers ──────────────────────────────────────────────
    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in [self.output_dir, self.chroma_persist_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    s = Settings()
    s.ensure_output_dirs()
    return s
