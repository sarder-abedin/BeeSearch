"""
tools/vector_store.py
─────────────────────
Manages a persistent ChromaDB vector store backed by local HuggingFace
sentence-transformer embeddings.

Why ChromaDB?
─────────────
  • Runs entirely locally — no API key, no cloud, no cost
  • Persists to disk between sessions
  • Supports metadata filtering (filter by doc_id, page, etc.)

Why sentence-transformers?
──────────────────────────
  • all-MiniLM-L6-v2 is 22 MB, fast on CPU, and very competitive
    on semantic similarity benchmarks
  • No API call — embeddings are computed locally every time

TUTORIAL NOTE
─────────────
The RAG (Retrieval-Augmented Generation) flow is:
  Store:    chunk text → embed → store vector + metadata in Chroma
  Retrieve: user query → embed → cosine-similarity search → top-k chunks

Those top-k chunks become the "context" in the LLM prompt, grounding
the model's answers in the actual document content.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from config.settings import get_settings
from tools.document_tools import DocumentChunk

logger = logging.getLogger(__name__)

# Lazy imports so the app starts even before heavy deps are installed
_chromadb = None
_SentenceTransformer = None


def _get_chroma():
    global _chromadb
    if _chromadb is None:
        import chromadb
        _chromadb = chromadb
    return _chromadb


def _get_st():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


# ── Embedding Function (ChromaDB-compatible) ──────────────────────────────────

class LocalEmbeddingFunction:
    """
    Wraps a sentence-transformers model into ChromaDB's EmbeddingFunction
    interface so we can plug it in without any cloud dependency.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # Lazy load on first call

    def _load(self):
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            ST = _get_st()
            self._model = ST(self.model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        self._load()
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


# ── Vector Store Manager ──────────────────────────────────────────────────────

class VectorStoreManager:
    """
    High-level interface for storing and retrieving document chunks.

    Typical usage
    ─────────────
    vsm = VectorStoreManager()
    vsm.add_chunks(processed_doc.chunks)
    results = vsm.similarity_search("climate change effects", top_k=5)
    """

    def __init__(self):
        cfg = get_settings()
        self._persist_dir = cfg.chroma_persist_dir
        self._collection_name = cfg.chroma_collection_name
        self._embed_fn = LocalEmbeddingFunction(cfg.embedding_model)
        self._client = None
        self._collection = None

    # ── Lazy initialisation ───────────────────────────────────

    def _ensure_collection(self):
        """Initialise ChromaDB client + collection on first use."""
        if self._collection is not None:
            return

        chromadb = _get_chroma()
        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # get_or_create is idempotent — safe to call every session
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},  # use cosine distance
        )
        logger.info(
            "ChromaDB collection '%s' ready (doc_count=%d)",
            self._collection_name,
            self._collection.count(),
        )

    # ── Write ─────────────────────────────────────────────────

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        """
        Embed and store document chunks in the vector store.

        Returns the number of chunks actually added (skips duplicates by chunk_id).
        """
        self._ensure_collection()

        # Filter out chunks already in the store
        existing_ids = set(self._collection.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("All chunks already in store — skipping.")
            return 0

        added = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )
            added += len(batch)

        logger.info("Added %d new chunks (skipped %d duplicates).", added, len(existing_ids))
        return added

    def clear_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to a specific document."""
        self._ensure_collection()
        results = self._collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info("Removed %d chunks for doc_id=%s", len(results["ids"]), doc_id)
            return len(results["ids"])
        return 0

    def clear_all(self) -> None:
        """Wipe the entire collection (useful for a fresh session)."""
        self._ensure_collection()
        self._client.delete_collection(self._collection_name)
        self._collection = None
        logger.warning("Vector store cleared.")

    # ── Read ──────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Parameters
        ----------
        query : natural-language question or topic
        top_k : number of results to return
        where : optional ChromaDB metadata filter dict

        Returns
        -------
        List of dicts with keys: text, metadata, distance, score
        """
        self._ensure_collection()

        kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": min(top_k, self.count())}
        if where:
            kwargs["where"] = where

        if kwargs["n_results"] == 0:
            return []

        raw = self._collection.query(**kwargs)
        results = []
        for doc, meta, dist in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            results.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "score": round(1 - dist, 4),  # cosine similarity (higher = better)
                }
            )
        return results

    def count(self) -> int:
        """Total number of chunks currently stored."""
        self._ensure_collection()
        return self._collection.count()

    def list_documents(self) -> List[Dict[str, str]]:
        """Return unique documents currently in the store."""
        self._ensure_collection()
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        seen: Dict[str, Dict] = {}
        for m in all_meta:
            src = m.get("source", "unknown")
            if src not in seen:
                seen[src] = m
        return list(seen.values())
