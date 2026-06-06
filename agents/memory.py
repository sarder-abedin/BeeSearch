"""
agents/memory.py
────────────────
Long-term memory for the Proposal Writer and Research sessions.

Design
──────
All sessions are stored in a single SQLite database:
  outputs/memory/sessions.db

ProposalMemory uses the proposal_sessions table.
ResearchMemory uses the research_sessions table.

This lets users close the browser, return later, pick up exactly where
they left off, and keep issuing revision instructions in the same thread.
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
    """Generate a compact alphanumeric session ID."""
    import random
    import string
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


class ProposalMemory:
    """
    Persistent store for proposal sessions.

    Each session maps to one row in the proposal_sessions table.

    Typical lifecycle
    -----------------
    mem = ProposalMemory()

    # First run — create session
    session_id = mem.new_session(goal="...", model="llama3.1:8b")
    mem.save_proposal(session_id, proposal_text, references)

    # Later run — load and revise
    session = mem.load(session_id)
    mem.add_revision(session_id, instruction="Make intro shorter", new_text="...")

    # List all sessions for UI selector
    sessions = mem.list_sessions()
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path
        init_db(self._db_path)

    # ── Session management ────────────────────────────────────

    def new_session(
        self,
        goal: str,
        model: str = "llama3.1:8b",
        instructions: str = "",
        session_id: str = "",
    ) -> str:
        """
        Create a new proposal session and return its session_id.
        Generates a short human-readable ID if none is provided.
        """
        sid = session_id or _short_id()
        now = _now()
        data: Dict[str, Any] = {
            "session_id": sid,
            "title": "",                   # filled after title node runs
            "goal": goal,
            "instructions": instructions,
            "model_name": model,
            "proposal_markdown": "",       # current draft
            "references": [],             # [{ref_num, apa, citation_key, ...}]
            "revision_history": [],       # [{timestamp, instruction, word_count}]
            "word_counts": {},            # {section: int}
            "created_at": now,
            "last_modified": now,
        }
        with _tx(self._db_path) as conn:
            conn.execute(
                "INSERT INTO proposal_sessions VALUES (?,?,?,?,?,?,?,?,?)",
                (sid, now, now, goal, "", model, 0, 0, pack(data)),
            )
        logger.info("New proposal session: %s", sid)
        return sid

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session by ID. Returns None if not found."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT data_json FROM proposal_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            logger.warning("Session not found: %s", session_id)
            return None
        return unpack(row["data_json"])

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return summary info for all stored sessions, newest first."""
        with _tx(self._db_path) as conn:
            rows = conn.execute(
                """SELECT session_id, created_at, updated_at, goal, title,
                          model_name, revision_count, has_proposal
                   FROM proposal_sessions
                   ORDER BY updated_at DESC""",
            ).fetchall()
        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row["session_id"],
                "title": row["title"] or row["goal"][:60] or "Untitled",
                "goal": (row["goal"] or "")[:80],
                "created_at": row["created_at"],
                "last_modified": row["updated_at"],
                "revision_count": row["revision_count"],
                "model_name": row["model_name"],
                "has_proposal": bool(row["has_proposal"]),
            })
        return sessions

    def delete(self, session_id: str) -> bool:
        with _tx(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM proposal_sessions WHERE session_id=?",
                (session_id,),
            )
            return cursor.rowcount > 0

    # ── Writing ───────────────────────────────────────────────

    def save_proposal(
        self,
        session_id: str,
        proposal_markdown: str,
        references: List[Dict],
        title: str = "",
        word_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Persist the current proposal draft and references."""
        data = self.load(session_id) or {}
        data["proposal_markdown"] = proposal_markdown
        data["references"] = references
        now = _now()
        data["last_modified"] = now
        if title:
            data["title"] = title
        if word_counts:
            data["word_counts"] = word_counts

        has_proposal = 1 if proposal_markdown else 0
        stored_title = data.get("title", "")
        goal = data.get("goal", "")
        model_name = data.get("model_name", "")
        revision_count = len(data.get("revision_history", []))
        created_at = data.get("created_at", now)

        with _tx(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO proposal_sessions
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (session_id, created_at, now, goal, stored_title,
                 model_name, revision_count, has_proposal, pack(data)),
            )

    def add_revision(
        self,
        session_id: str,
        instruction: str,
        new_proposal: str,
        new_references: Optional[List[Dict]] = None,
    ) -> None:
        """
        Record a revision in the session's history and update the draft.

        Stores the full previous proposal text so the user can always
        see what changed (and potentially roll back).
        """
        data = self.load(session_id) or {}
        old_text = data.get("proposal_markdown", "")

        entry: Dict[str, Any] = {
            "timestamp": _now(),
            "instruction": instruction,
            "previous_word_count": len(old_text.split()),
            "new_word_count": len(new_proposal.split()),
            # Store only first 500 chars of old text to keep JSON lean
            "previous_snippet": old_text[:500],
        }

        history: List = data.get("revision_history", [])
        history.append(entry)

        data["revision_history"] = history
        data["proposal_markdown"] = new_proposal
        now = _now()
        data["last_modified"] = now
        if new_references is not None:
            data["references"] = new_references

        revision_count = len(history)
        has_proposal = 1 if new_proposal else 0
        goal = data.get("goal", "")
        stored_title = data.get("title", "")
        model_name = data.get("model_name", "")
        created_at = data.get("created_at", now)

        with _tx(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO proposal_sessions
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (session_id, created_at, now, goal, stored_title,
                 model_name, revision_count, has_proposal, pack(data)),
            )
        logger.info(
            "Revision #%d saved for session %s (%d→%d words)",
            len(history), session_id,
            entry["previous_word_count"], entry["new_word_count"],
        )

    def update_field(self, session_id: str, **kwargs) -> None:
        """Update arbitrary top-level fields in a session."""
        data = self.load(session_id) or {}
        data.update(kwargs)
        now = _now()
        data["last_modified"] = now

        goal = data.get("goal", "")
        stored_title = data.get("title", "")
        model_name = data.get("model_name", "")
        revision_count = len(data.get("revision_history", []))
        has_proposal = 1 if data.get("proposal_markdown") else 0
        created_at = data.get("created_at", now)

        with _tx(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO proposal_sessions
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (session_id, created_at, now, goal, stored_title,
                 model_name, revision_count, has_proposal, pack(data)),
            )


class ResearchMemory:
    """
    Persistent store for research sessions (Modes 1–3).

    Each session maps to one row in the research_sessions table.

    Lets users return to a previous research run and review or re-export
    findings without re-running the full workflow.
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path
        init_db(self._db_path)

    def save_session(
        self,
        session_id: str,
        goal: str,
        report: str,
        references: List[Dict],
        key_findings: List[str],
        document_names: List[str],
        mode: str,
        model_name: str,
    ) -> None:
        """Persist a completed research session."""
        now = _now()
        data: Dict[str, Any] = {
            "session_id": session_id,
            "goal": goal,
            "report": report,
            "references": references,
            "key_findings": key_findings,
            "document_names": document_names,
            "mode": mode,
            "model_name": model_name,
            "created_at": now,
            "last_modified": now,
        }
        reference_count = len(references)
        with _tx(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO research_sessions
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, now, now, goal, mode, model_name,
                 reference_count, pack(data)),
            )
        logger.info("Research session saved: %s", session_id)

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a research session by ID. Returns None if not found."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT data_json FROM research_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return unpack(row["data_json"])

    def list_sessions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return summary info for the most recent research sessions."""
        with _tx(self._db_path) as conn:
            rows = conn.execute(
                """SELECT session_id, created_at, updated_at, goal, mode,
                          model_name, reference_count, data_json
                   FROM research_sessions
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        sessions = []
        for row in rows:
            data = unpack(row["data_json"])
            sessions.append({
                "session_id": row["session_id"],
                "goal": (row["goal"] or "")[:80],
                "mode": row["mode"],
                "model_name": row["model_name"],
                "reference_count": row["reference_count"],
                "document_names": data.get("document_names", []),
                "created_at": row["created_at"],
            })
        return sessions

    def delete(self, session_id: str) -> bool:
        with _tx(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM research_sessions WHERE session_id=?",
                (session_id,),
            )
            return cursor.rowcount > 0
