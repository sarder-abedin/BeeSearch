"""paper_graph/collection_store.py
──────────────────────────────────
In-memory store for Feature 2 (Discovery Network) collections.

Same trade-off as backend/app/jobs.py: in-memory and single-process by
design, matching BeeSearch's local-first, single-user architecture.
Collections are lost on server restart — acceptable here since the Research
Notebook's SQLite DB (outputs/memory/sessions.db) is the only durable store.

To wire collections into durable storage: add a `paper_graph_collections`
table to sessions.db using agents/notebook_memory.py's WAL-mode SQLite
helpers, serialise paper_nodes / edges as JSON columns, and replace the
dict operations below with SQL reads/writes.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .graph_builder import GraphEdge
from .s2_client import PaperNode


@dataclass
class Collection:
    id: str
    paper_ids: set = field(default_factory=set)
    paper_nodes: Dict[str, PaperNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)


class CollectionStore:
    """Thread-safe in-memory collection registry."""

    def __init__(self) -> None:
        self._collections: Dict[str, Collection] = {}
        self._lock = threading.Lock()

    def create(self, seed_nodes: List[PaperNode]) -> Collection:
        cid = uuid.uuid4().hex
        collection = Collection(id=cid)
        for node in seed_nodes:
            collection.paper_ids.add(node.id)
            collection.paper_nodes[node.id] = node
        with self._lock:
            self._collections[cid] = collection
        return collection

    def get(self, collection_id: str) -> Optional[Collection]:
        with self._lock:
            return self._collections.get(collection_id)

    def add_papers(
        self,
        collection_id: str,
        new_nodes: List[PaperNode],
        new_edges: List[GraphEdge],
    ) -> Optional[Collection]:
        with self._lock:
            col = self._collections.get(collection_id)
            if col is None:
                return None
            for node in new_nodes:
                if node.id not in col.paper_ids:
                    col.paper_ids.add(node.id)
                    col.paper_nodes[node.id] = node
            # Deduplicate edges by (source, target, edge_type)
            existing = {(e.source, e.target, e.edge_type) for e in col.edges}
            for edge in new_edges:
                key = (edge.source, edge.target, edge.edge_type)
                if key not in existing:
                    col.edges.append(edge)
                    existing.add(key)
            return col


# Module-level singleton — shared across all FastAPI requests in the process
_store: Optional[CollectionStore] = None
_store_lock = threading.Lock()


def get_store() -> CollectionStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = CollectionStore()
    return _store
