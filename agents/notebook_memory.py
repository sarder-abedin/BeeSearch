"""
agents/notebook_memory.py
──────────────────────────
Long-term persistence for the Research Notebook (Mode 8) — a NotebookLM-style
mode where users build a notebook from their own sources and chat with it
using grounded, cited retrieval.

State is split into two parts in SQLite:
  • meta_json column in notebooks table stores:
      {name, sources, conversation, created_at, last_modified}  (NO chunks)
  • Chunks stored separately in notebook_chunks table

This allows efficient querying without loading all chunk text when only
metadata is needed (e.g. list_notebooks).
"""

from __future__ import annotations

import logging
import random
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.session_db import _tx, init_db, pack, unpack

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(length: int = 8) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


class NotebookMemory:
    """
    Persistent store for Research Notebook sessions.

    Typical lifecycle
    -----------------
    mem = NotebookMemory()

    # Create a notebook
    nb_id = mem.new_notebook(name="Antibiotic Resistance")

    # Add a processed source (after DocumentProcessor + HybridStore indexing)
    mem.add_source(nb_id, processed_document, source_type="file")

    # Chat: append turns with citations
    mem.add_turn(nb_id, "user", "What datasets are used?")
    mem.add_turn(nb_id, "assistant", "The study uses ...[1]",
                 citations=[{"n": 1, "doc_name": "paper.pdf", "page": 4}])

    # Reload later
    notebook = mem.load(nb_id)
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path
        init_db(self._db_path)

    # ── Notebook management ───────────────────────────────────

    def new_notebook(self, name: str, notebook_id: str = "") -> str:
        """Create a new, empty notebook and return its id."""
        nb_id = notebook_id or _short_id()
        now = _now()
        meta: Dict[str, Any] = {
            "name": name.strip() or "Untitled Notebook",
            "sources": [],          # list of source metadata dicts
            "conversation": [],     # list of {role, content, timestamp, citations, ...}
            "created_at": now,
            "last_modified": now,
        }
        with _tx(self._db_path) as conn:
            conn.execute(
                "INSERT INTO notebooks VALUES (?,?,?,?,?,?,?)",
                (nb_id, now, now, meta["name"], 0, 0, pack(meta)),
            )
        logger.info("New notebook: %s — %s", nb_id, name[:60])
        return nb_id

    def load(self, notebook_id: str) -> Optional[Dict[str, Any]]:
        """Load a notebook by id. Returns None if not found."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
            if row is None:
                return None
            meta = unpack(row["meta_json"])
            chunks = conn.execute(
                """SELECT chunk_id, doc_id, doc_name, page_num, chunk_index, text
                   FROM notebook_chunks WHERE notebook_id=?""",
                (notebook_id,),
            ).fetchall()
        meta["chunks"] = [dict(c) for c in chunks]
        meta["notebook_id"] = notebook_id
        return meta

    def list_notebooks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return summary info for all notebooks, newest first."""
        with _tx(self._db_path) as conn:
            rows = conn.execute(
                """SELECT notebook_id, created_at, updated_at, name,
                          source_count, turn_count, meta_json
                   FROM notebooks
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        notebooks = []
        for row in rows:
            meta = unpack(row["meta_json"])
            notebooks.append({
                "notebook_id": row["notebook_id"],
                "name": row["name"] or "Untitled",
                "source_count": row["source_count"],
                "turn_count": row["turn_count"],
                "source_names": [s.get("filename", "") for s in meta.get("sources", [])],
                "created_at": row["created_at"],
                "last_modified": row["updated_at"],
            })
        return notebooks

    def delete(self, notebook_id: str) -> bool:
        with _tx(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            )
            return cursor.rowcount > 0

    def rename(self, notebook_id: str, new_name: str) -> bool:
        nb = self.load(notebook_id)
        if nb is None:
            return False
        cleaned = new_name.strip() or nb.get("name", "Untitled Notebook")
        now = _now()
        # Rebuild meta without notebook_id/chunks keys
        meta = {k: v for k, v in nb.items() if k not in ("notebook_id", "chunks")}
        meta["name"] = cleaned
        meta["last_modified"] = now
        with _tx(self._db_path) as conn:
            conn.execute(
                "UPDATE notebooks SET name=?, updated_at=?, meta_json=? WHERE notebook_id=?",
                (cleaned, now, pack(meta), notebook_id),
            )
        return True

    # ── Source management ─────────────────────────────────────

    def add_source(
        self,
        notebook_id: str,
        processed_doc: Any,
        source_type: str = "file",
        url: str = "",
    ) -> bool:
        """
        Append a processed source's metadata + chunks to the notebook.

        Parameters
        ----------
        processed_doc : a tools.document_tools.ProcessedDocument
        source_type   : "file" or "url"
        url           : original URL when source_type == "url"
        """
        with _tx(self._db_path) as conn:
            # Check notebook exists
            row = conn.execute(
                "SELECT meta_json, source_count FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
            if row is None:
                logger.warning("add_source: notebook %s not found", notebook_id)
                return False

            # Duplicate check
            dup = conn.execute(
                "SELECT 1 FROM notebook_chunks WHERE notebook_id=? AND doc_id=? LIMIT 1",
                (notebook_id, processed_doc.doc_id),
            ).fetchone()
            if dup is not None:
                logger.info(
                    "Source %s already in notebook %s — skipping",
                    processed_doc.filename, notebook_id,
                )
                return False

            meta = unpack(row["meta_json"])
            meta.setdefault("sources", []).append({
                "doc_id": processed_doc.doc_id,
                "filename": processed_doc.filename,
                "file_type": processed_doc.file_type,
                "source_type": source_type,
                "url": url,
                "total_pages": processed_doc.total_pages,
                "total_chunks": processed_doc.total_chunks,
                "content_md5": processed_doc.content_md5,
                "added_at": _now(),
            })
            now = _now()
            meta["last_modified"] = now
            source_count = len(meta["sources"])

            conn.execute(
                "UPDATE notebooks SET source_count=?, updated_at=?, meta_json=? WHERE notebook_id=?",
                (source_count, now, pack(meta), notebook_id),
            )

            # Insert chunks
            conn.executemany(
                """INSERT INTO notebook_chunks
                   (chunk_id, notebook_id, doc_id, doc_name, page_num, chunk_index, text)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    (ch.chunk_id, notebook_id, ch.doc_id, ch.doc_name,
                     ch.page_num, ch.chunk_index, ch.text)
                    for ch in processed_doc.chunks
                ],
            )

        logger.info(
            "Added source '%s' (%d chunks) to notebook %s",
            processed_doc.filename, processed_doc.total_chunks, notebook_id,
        )
        return True

    def remove_source(self, notebook_id: str, doc_id: str) -> bool:
        """Remove a source and all of its chunks from the notebook."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json, source_count FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
            if row is None:
                return False

            meta = unpack(row["meta_json"])
            before = len(meta.get("sources", []))
            meta["sources"] = [
                s for s in meta.get("sources", []) if s.get("doc_id") != doc_id
            ]
            if len(meta["sources"]) == before:
                return False

            now = _now()
            meta["last_modified"] = now
            source_count = len(meta["sources"])

            conn.execute(
                "DELETE FROM notebook_chunks WHERE notebook_id=? AND doc_id=?",
                (notebook_id, doc_id),
            )
            conn.execute(
                "UPDATE notebooks SET source_count=?, updated_at=?, meta_json=? WHERE notebook_id=?",
                (source_count, now, pack(meta), notebook_id),
            )

        logger.info("Removed source %s from notebook %s", doc_id, notebook_id)
        return True

    # ── Conversation ──────────────────────────────────────────

    def add_turn(
        self,
        notebook_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        suggested_questions: Optional[List[str]] = None,
    ) -> None:
        """Append a single conversation turn."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json, turn_count FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
            if row is None:
                logger.warning("add_turn: notebook %s not found", notebook_id)
                return
            meta = unpack(row["meta_json"])
            meta.setdefault("conversation", []).append({
                "role": role,
                "content": content,
                "timestamp": _now(),
                "citations": citations,
                "suggested_questions": suggested_questions,
            })
            now = _now()
            meta["last_modified"] = now
            turn_count = len(meta["conversation"])
            conn.execute(
                "UPDATE notebooks SET turn_count=?, updated_at=?, meta_json=? WHERE notebook_id=?",
                (turn_count, now, pack(meta), notebook_id),
            )

    def get_history(self, notebook_id: str, max_turns: int = 8) -> List[Dict]:
        """Return the last `max_turns` conversation turns."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
        if row is None:
            return []
        meta = unpack(row["meta_json"])
        return meta.get("conversation", [])[-max_turns:]
