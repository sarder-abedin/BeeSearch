"""
agents/story_memory.py
───────────────────────
Long-term conversation memory for the Research Partner (Mode 5).

All sessions are stored in a single SQLite database:
  outputs/memory/sessions.db  (story_sessions table)

The table tracks:
  • The topic being explored and any uploaded document context
  • Full conversation history (user + assistant turns)
  • Concepts already explained (so the LLM avoids redundant re-explanation)
  • Session metadata (created_at, last_modified)
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
    import random
    import string
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


class StorytellerMemory:
    """
    Persistent conversation store for Research Partner sessions.

    Typical lifecycle
    -----------------
    mem = StorytellerMemory()

    # Start a new session
    session_id = mem.new_session(topic="Transformers", document_context="...")

    # Add a turn after each user message + assistant response
    mem.add_turn(session_id, "user", "What is self-attention?")
    mem.add_turn(session_id, "assistant", "Think of it like...",
                 suggested_questions=["Follow-up 1", "Follow-up 2"])

    # Load on the next browser visit
    session = mem.load(session_id)
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path
        init_db(self._db_path)

    # ── Session management ────────────────────────────────────

    def new_session(
        self,
        topic: str,
        document_context: str = "",
        document_names: Optional[List[str]] = None,
        session_id: str = "",
    ) -> str:
        """
        Create a new storytelling session and return its session_id.
        """
        sid = session_id or _short_id()
        now = _now()
        data: Dict[str, Any] = {
            "session_id": sid,
            "topic": topic,
            "document_names": document_names or [],
            # Truncated raw text from uploaded docs, used as reference context
            "document_context": document_context[:2000] if document_context else "",
            "conversation": [],          # list of {role, content, timestamp, suggested_questions}
            "concepts_covered": [],      # concept names explained so far
            "created_at": now,
            "last_modified": now,
        }
        with _tx(self._db_path) as conn:
            conn.execute(
                "INSERT INTO story_sessions VALUES (?,?,?,?,?,?)",
                (sid, now, now, topic, 0, pack(data)),
            )
        logger.info("New story session: %s — topic: %s", sid, topic[:60])
        return sid

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session by ID. Returns None if not found."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT data_json FROM story_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return unpack(row["data_json"])

    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return summary info for the most recent story sessions, newest first."""
        with _tx(self._db_path) as conn:
            rows = conn.execute(
                """SELECT session_id, created_at, updated_at, topic, turn_count, data_json
                   FROM story_sessions
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        sessions = []
        for row in rows:
            data = unpack(row["data_json"])
            sessions.append({
                "session_id": row["session_id"],
                "topic": row["topic"] or "Untitled",
                "document_names": data.get("document_names", []),
                "turn_count": row["turn_count"],
                "concepts_covered": data.get("concepts_covered", []),
                "created_at": row["created_at"],
                "last_modified": row["updated_at"],
            })
        return sessions

    def delete(self, session_id: str) -> bool:
        with _tx(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM story_sessions WHERE session_id=?",
                (session_id,),
            )
            return cursor.rowcount > 0

    def rename(self, session_id: str, new_topic: str) -> bool:
        """Update the display topic for a session."""
        data = self.load(session_id)
        if data is None:
            return False
        now = _now()
        data["topic"] = new_topic.strip()
        data["last_modified"] = now
        with _tx(self._db_path) as conn:
            conn.execute(
                "UPDATE story_sessions SET topic=?, updated_at=?, data_json=? WHERE session_id=?",
                (new_topic.strip(), now, pack(data), session_id),
            )
        return True

    # ── Writing ───────────────────────────────────────────────

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        suggested_questions: Optional[List[str]] = None,
    ) -> None:
        """
        Append a single conversation turn to the session.

        Parameters
        ----------
        role                : "user" or "assistant"
        content             : The message text
        suggested_questions : For assistant turns — list of 2–3 follow-up questions
        """
        data = self.load(session_id)
        if data is None:
            logger.warning("add_turn: session %s not found", session_id)
            return

        turn: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": _now(),
            "suggested_questions": suggested_questions,
        }
        data.setdefault("conversation", []).append(turn)
        now = _now()
        data["last_modified"] = now
        turn_count = len(data["conversation"])
        with _tx(self._db_path) as conn:
            conn.execute(
                "UPDATE story_sessions SET updated_at=?, turn_count=?, data_json=? WHERE session_id=?",
                (now, turn_count, pack(data), session_id),
            )

    def add_concepts(self, session_id: str, concepts: List[str]) -> None:
        """
        Record newly explained concept names to avoid future redundancy.
        """
        data = self.load(session_id)
        if data is None:
            return

        existing = set(data.get("concepts_covered", []))
        existing.update(c.strip().lower() for c in concepts if c.strip())
        data["concepts_covered"] = sorted(existing)
        now = _now()
        data["last_modified"] = now
        with _tx(self._db_path) as conn:
            conn.execute(
                "UPDATE story_sessions SET updated_at=?, data_json=? WHERE session_id=?",
                (now, pack(data), session_id),
            )

    # ── Helpers ───────────────────────────────────────────────

    def get_history(self, session_id: str, max_turns: int = 8) -> List[Dict]:
        """Return the last `max_turns` conversation turns."""
        data = self.load(session_id)
        if not data:
            return []
        conversation = data.get("conversation", [])
        return conversation[-max_turns:]
