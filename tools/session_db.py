"""tools/session_db.py
SQLite-backed session storage shared by all memory classes.
Single DB file: outputs/memory/sessions.db
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import orjson
    def _dumps(obj: Any) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_PASSTHROUGH_DATETIME)
    def _loads(data) -> Any:
        return orjson.loads(data)
except ImportError:
    import json as _j
    def _dumps(obj: Any) -> bytes:
        return _j.dumps(obj, default=str, ensure_ascii=False).encode()
    def _loads(data) -> Any:
        return _j.loads(data)

_DEFAULT_DB = Path("outputs/memory/sessions.db")

@contextmanager
def _tx(db_path: Path | None = None):
    path = db_path or _DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def pack(obj: Any) -> bytes:
    return _dumps(obj)

def unpack(data) -> Any:
    if data is None:
        return {}
    return _loads(data)

_DDL = """
CREATE TABLE IF NOT EXISTS research_sessions (
    session_id      TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    goal            TEXT DEFAULT '',
    mode            TEXT DEFAULT '',
    model_name      TEXT DEFAULT '',
    reference_count INTEGER DEFAULT 0,
    data_json       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_updated ON research_sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS notebooks (
    notebook_id  TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    name         TEXT DEFAULT 'Untitled Notebook',
    source_count INTEGER DEFAULT 0,
    turn_count   INTEGER DEFAULT 0,
    meta_json    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notebook_updated ON notebooks(updated_at DESC);

CREATE TABLE IF NOT EXISTS notebook_chunks (
    chunk_id    TEXT NOT NULL,
    notebook_id TEXT NOT NULL REFERENCES notebooks(notebook_id) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL,
    doc_name    TEXT NOT NULL,
    page_num    INTEGER DEFAULT 0,
    chunk_index INTEGER DEFAULT 0,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_nb ON notebook_chunks(notebook_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_pk ON notebook_chunks(notebook_id, chunk_id);
"""

def init_db(db_path: Path | None = None) -> None:
    """Create all tables and indexes if they do not already exist."""
    with _tx(db_path) as conn:
        conn.executescript(_DDL)
