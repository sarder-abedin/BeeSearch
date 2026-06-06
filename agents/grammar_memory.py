"""
agents/grammar_memory.py
─────────────────────────
SQLite persistence for Grammar Proofreading Mode (Mode 6).

All sessions are stored in a single SQLite database:
  outputs/memory/sessions.db  (grammar_sessions table)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.session_db import _tx, init_db, pack, unpack

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(length: int = 8) -> str:
    import random, string
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


class GrammarMemory:
    """
    Persistent store for Grammar Proofreading sessions.

    Typical lifecycle
    -----------------
    mem = GrammarMemory()
    session_id = mem.new_session(raw_text="The quick brown fox…")
    mem.save_result(session_id, final_state_dict)
    sessions = mem.list_sessions(limit=10)
    data = mem.load(session_id)
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path
        init_db(self._db_path)

    # ── Session management ────────────────────────────────────

    def new_session(
        self,
        raw_text: str = "",
        style_level: str = "professional_email",
        session_id: str = "",
    ) -> str:
        """Create a new grammar session and return its session_id."""
        sid = session_id or _short_id()
        now = _now()
        data: Dict[str, Any] = {
            "session_id": sid,
            "raw_text_excerpt": raw_text[:500],
            "style_level": style_level,
            "polished_text": "",
            "issues_found": [],
            "style_suggestions": [],
            "eval_result": {},
            "word_count": 0,
            "refinement_round": 0,
            "created_at": now,
            "last_modified": now,
        }
        with _tx(self._db_path) as conn:
            conn.execute(
                "INSERT INTO grammar_sessions VALUES (?,?,?,?,?,?,?,?)",
                (sid, now, now, style_level, 0, 0, 0, pack(data)),
            )
        logger.info("New grammar session: %s — style: %s", sid, style_level)
        return sid

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session by ID. Returns None if not found."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT data_json FROM grammar_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return unpack(row["data_json"])

    def save_result(self, session_id: str, state: Dict[str, Any]) -> None:
        """Persist the proofreading result from the final state dict."""
        existing = self.load(session_id) or {
            "session_id": session_id,
            "created_at": _now(),
        }
        now = _now()
        existing.update({
            "raw_text_excerpt": (state.get("raw_text") or "")[:500],
            "style_level": state.get("style_level", "professional_email"),
            "polished_text": state.get("polished_text", ""),
            "change_summary": state.get("change_summary", ""),
            "issues_found": state.get("issues_found", []),
            "style_suggestions": state.get("style_suggestions", []),
            "eval_result": state.get("eval_result", {}),
            "word_count": state.get("word_count", 0),
            "refinement_round": state.get("refinement_round", 0),
            "feedback_history": state.get("feedback_history", []),
            "last_modified": now,
        })
        style_level = existing.get("style_level", "professional_email")
        word_count = existing.get("word_count", 0)
        issues_count = len(existing.get("issues_found", []))
        has_result = 1 if existing.get("polished_text") else 0
        created_at = existing.get("created_at", now)
        with _tx(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO grammar_sessions
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, created_at, now, style_level,
                 word_count, issues_count, has_result, pack(existing)),
            )
        logger.info("Grammar result saved for session %s", session_id)

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return summary info for the most recent grammar sessions, newest first."""
        with _tx(self._db_path) as conn:
            rows = conn.execute(
                """SELECT session_id, created_at, updated_at, style_level,
                          word_count, issues_count, has_result, data_json
                   FROM grammar_sessions
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        sessions = []
        for row in rows:
            data = unpack(row["data_json"])
            sessions.append({
                "session_id": row["session_id"],
                "raw_text_excerpt": data.get("raw_text_excerpt", "")[:80],
                "style_level": row["style_level"],
                "word_count": row["word_count"],
                "issues_count": row["issues_count"],
                "refinement_round": data.get("refinement_round", 0),
                "has_result": bool(row["has_result"]),
                "created_at": row["created_at"],
                "last_modified": row["updated_at"],
            })
        return sessions

    def delete(self, session_id: str) -> bool:
        with _tx(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM grammar_sessions WHERE session_id=?",
                (session_id,),
            )
            return cursor.rowcount > 0
