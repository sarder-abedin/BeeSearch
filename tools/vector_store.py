"""
tools/vector_store.py
─────────────────────
Manages a persistent ChromaDB vector store backed by local HuggingFace
sentence-transformer embeddings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from config.settings import get_settings
from tools.document_tools import DocumentChunk

logger = logging.getLogger(__name__)

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


class LocalEmbeddingFunction:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            ST = _get_st()
            self._model = ST(self.model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        self._load()
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


class VectorStoreManager:
    """
    High-level interface for storing and retrieving document chunks via ChromaDB.
    """

    def __init__(self):
        cfg = get_settings()
        self._persist_dir = cfg.chroma_persist_dir
        self._collection_name = cfg.chroma_collection_name
        self._embed_fn = LocalEmbeddingFunction(cfg.embedding_model)
        self._client = None
        self._collection = None

    def _ensure_collection(self):
        if self._collection is not None:
            return
        chromadb = _get_chroma()
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready (count=%d)", self._collection_name, self._collection.count())

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        self._ensure_collection()
        existing_ids = set(self._collection.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
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
        return added

    def clear_document(self, doc_id: str) -> int:
        self._ensure_collection()
        results = self._collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def clear_all(self) -> None:
        self._ensure_collection()
        self._client.delete_collection(self._collection_name)
        self._collection = None

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_collection()
        kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": min(top_k, self.count())}
        if where:
            kwargs["where"] = where
        if kwargs["n_results"] == 0:
            return []
        raw = self._collection.query(**kwargs)
        results = []
        for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
            results.append({"text": doc, "metadata": meta, "distance": dist, "score": round(1 - dist, 4)})
        return results

    def count(self) -> int:
        self._ensure_collection()
        return self._collection.count()

    def list_documents(self) -> List[Dict[str, str]]:
        self._ensure_collection()
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        seen: Dict[str, Dict] = {}
        for m in all_meta:
            src = m.get("source", "unknown")
            if src not in seen:
                seen[src] = m
        return list(seen.values())
