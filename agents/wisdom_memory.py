"""
agents/wisdom_memory.py
────────────────────────
SQLite persistence for Wisdom Mode sessions.

All sessions are stored in a single SQLite database:
  outputs/memory/sessions.db  (wisdom_sessions + wisdom_tags tables)
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


class WisdomMemory:
    """
    Persistent store for Wisdom Mode sessions.

    Typical lifecycle
    -----------------
    mem = WisdomMemory()

    # Start a new session
    session_id = mem.new_session(topic="Sleep and exam performance",
                                  scenario="I pull all-nighters before exams…")

    # Clarification phase — one turn per user message
    mem.add_turn(session_id, "user", "I have exams in 2 days…")
    mem.add_turn(session_id, "assistant", "How long have you been doing this?",
                 metadata={"is_question": True})

    # After wisdom is generated
    mem.save_wisdom(session_id, deep_understanding="…", simple_explanation="…", …)

    # Cross-session context (passive, used at synthesis time)
    related = mem.find_related_sessions(["sleep", "cognition"], session_id)
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path
        init_db(self._db_path)

    # ── Session management ────────────────────────────────────

    def new_session(
        self,
        topic: str = "",
        scenario: str = "",
        document_context: str = "",
        document_names: Optional[List[str]] = None,
        session_id: str = "",
    ) -> str:
        """Create a new wisdom session and return its session_id."""
        sid = session_id or _short_id()
        now = _now()
        data: Dict[str, Any] = {
            "session_id": sid,
            "topic": topic,
            "scenario": scenario,
            "document_names": document_names or [],
            "document_context": document_context[:3000] if document_context else "",
            "phase": "clarifying",
            "conversation": [],
            "wisdom_output": {},
            "knowledge_base": {"papers": [], "queries": []},
            "topic_tags": [],
            "created_at": now,
            "last_modified": now,
        }
        with _tx(self._db_path) as conn:
            conn.execute(
                "INSERT INTO wisdom_sessions VALUES (?,?,?,?,?,?,?)",
                (sid, now, now, topic, "clarifying", 0, pack(data)),
            )
        logger.info("New wisdom session: %s — topic: %s", sid, topic[:60])
        return sid

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session by ID. Returns None if not found."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT data_json FROM wisdom_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return unpack(row["data_json"])

    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return summary info for the most recent wisdom sessions, newest first."""
        with _tx(self._db_path) as conn:
            rows = conn.execute(
                """SELECT session_id, created_at, updated_at, topic, phase,
                          has_wisdom, data_json
                   FROM wisdom_sessions
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
                "phase": row["phase"],
                "topic_tags": data.get("topic_tags", []),
                "turn_count": len(data.get("conversation", [])),
                "has_wisdom": bool(row["has_wisdom"]),
                "created_at": row["created_at"],
                "last_modified": row["updated_at"],
            })
        return sessions

    def delete(self, session_id: str) -> bool:
        with _tx(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM wisdom_sessions WHERE session_id=?",
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
                "UPDATE wisdom_sessions SET topic=?, updated_at=?, data_json=? WHERE session_id=?",
                (new_topic.strip(), now, pack(data), session_id),
            )
        return True

    # ── Writing ───────────────────────────────────────────────

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Append a conversation turn. metadata fields are merged into the turn dict."""
        data = self.load(session_id)
        if data is None:
            logger.warning("add_turn: session %s not found", session_id)
            return
        turn: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": _now(),
        }
        if metadata:
            turn.update(metadata)
        data.setdefault("conversation", []).append(turn)
        now = _now()
        data["last_modified"] = now
        with _tx(self._db_path) as conn:
            conn.execute(
                "UPDATE wisdom_sessions SET updated_at=?, data_json=? WHERE session_id=?",
                (now, pack(data), session_id),
            )

    def save_wisdom(
        self,
        session_id: str,
        deep_understanding: str,
        simple_explanation: str,
        actionable_takeaways: List[str],
        validation: Dict,
        papers: List[Dict],
        queries: List[str],
        topic_tags: List[str],
    ) -> None:
        """Persist the generated wisdom output and update topic tags."""
        data = self.load(session_id) or {}
        data["wisdom_output"] = {
            "deep_understanding": deep_understanding,
            "simple_explanation": simple_explanation,
            "actionable_takeaways": actionable_takeaways,
            "validation": validation,
        }
        data["knowledge_base"] = {"papers": papers, "queries": queries}
        data["topic_tags"] = topic_tags
        data["phase"] = "done"
        now = _now()
        data["last_modified"] = now

        # Compute tag words for the wisdom_tags table
        words: set = set()
        for tag in topic_tags:
            words.update(tag.lower().split())

        with _tx(self._db_path) as conn:
            conn.execute(
                """UPDATE wisdom_sessions
                   SET updated_at=?, phase='done', has_wisdom=1, data_json=?
                   WHERE session_id=?""",
                (now, pack(data), session_id),
            )
            # Delete existing tags, then insert new ones
            conn.execute(
                "DELETE FROM wisdom_tags WHERE session_id=?",
                (session_id,),
            )
            conn.executemany(
                "INSERT INTO wisdom_tags VALUES (?,?)",
                [(session_id, w) for w in words],
            )
        logger.info("Wisdom saved for session %s — tags: %s", session_id, topic_tags[:5])

    def update_phase(self, session_id: str, phase: str) -> None:
        data = self.load(session_id) or {}
        data["phase"] = phase
        now = _now()
        data["last_modified"] = now
        with _tx(self._db_path) as conn:
            conn.execute(
                "UPDATE wisdom_sessions SET updated_at=?, phase=?, data_json=? WHERE session_id=?",
                (now, phase, pack(data), session_id),
            )

    # ── Cross-session context (passive) ───────────────────────

    def find_related_sessions(
        self,
        topic_tags: List[str],
        current_session_id: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find sessions whose topic tags overlap with the given tags.

        Uses word-level overlap: "chronic stress" and "stress relief" share
        the word "stress" and will match, even though the full tag strings differ.

        Returns sessions ranked by overlap count, for passive injection
        into the wisdom synthesis prompt.
        """
        if not topic_tags:
            return []

        target_words: set = set()
        for tag in topic_tags:
            target_words.update(tag.lower().split())

        if not target_words:
            return []

        placeholders = ",".join("?" * len(target_words))
        with _tx(self._db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT ws.session_id, ws.topic, ws.data_json,
                       COUNT(DISTINCT wt.word) AS overlap
                FROM wisdom_sessions ws
                JOIN wisdom_tags wt ON wt.session_id = ws.session_id
                WHERE wt.word IN ({placeholders})
                  AND ws.session_id != ?
                  AND ws.has_wisdom = 1
                GROUP BY ws.session_id
                ORDER BY overlap DESC
                LIMIT ?
                """,
                (*target_words, current_session_id, limit),
            ).fetchall()

        results = []
        for row in rows:
            data = unpack(row["data_json"])
            wo = data.get("wisdom_output", {})
            results.append({
                "session_id": row["session_id"],
                "topic": row["topic"],
                "overlap": row["overlap"],
                "topic_tags": data.get("topic_tags", []),
                "wisdom_snippet": wo.get("deep_understanding", "")[:400],
                "actionable_snippet": " | ".join(
                    wo.get("actionable_takeaways", [])[:2]
                ),
            })
        return results
