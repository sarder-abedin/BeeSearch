"""
agents/style_memory.py
──────────────────────
Persistent store for named Writing Style Profiles.

All profiles are stored in a single SQLite database:
  outputs/memory/sessions.db  (style_profiles table)

Lifecycle
─────────
  mem = StyleMemory()

  # Create a profile from uploaded documents
  profile_id = mem.create_profile(
      name="Academic Writing",
      documents=processed_docs,
      model_name="llama3.1:8b",
      ollama_base_url="http://localhost:11434",
      num_ctx=32768,
  )

  # Load by ID or name
  profile = mem.load(profile_id)
  profile = mem.load_by_name("Academic Writing")

  # List all profiles
  profiles = mem.list_profiles()

  # Delete
  mem.delete(profile_id)
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
    return "sp_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


class StyleMemory:
    """
    CRUD store for Writing Style Profiles.

    A style profile captures the writing characteristics of a user's documents
    and produces an injection_prompt that can be appended to LLM system prompts
    to make all AI-generated prose match the user's style.
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path
        init_db(self._db_path)

    # ── Profile creation ──────────────────────────────────────

    def create_profile(
        self,
        name: str,
        documents: list,
        model_name: str = "llama3.1:8b",
        ollama_base_url: str = "http://localhost:11434",
        num_ctx: int = 32768,
    ) -> str:
        """
        Analyse a list of ProcessedDocument objects, generate a Style Profile,
        save it, and return the profile_id.

        Parameters
        ----------
        name          : User-given name (e.g. "Grant Proposal Style")
        documents     : List of ProcessedDocument objects (from DocumentProcessor)
        model_name    : Ollama model to use for analysis
        ollama_base_url: Ollama server URL
        num_ctx       : LLM context window in tokens
        """
        from tools.style_profiler import analyse_writing_style

        profile_id = _short_id()
        now = _now()

        analysis_result = analyse_writing_style(
            documents=documents,
            model_name=model_name,
            ollama_base_url=ollama_base_url,
            num_ctx=num_ctx,
        )

        data: Dict[str, Any] = {
            "profile_id": profile_id,
            "name": name,
            "sample_documents": [getattr(d, "filename", str(d)) for d in documents],
            "model_name": model_name,
            "analysis": analysis_result.get("analysis", {}),
            "injection_prompt": analysis_result.get("injection_prompt", ""),
            "created_at": now,
            "last_modified": now,
        }

        name_lower = name.strip().lower()
        has_injection = 1 if data.get("injection_prompt") else 0

        with _tx(self._db_path) as conn:
            conn.execute(
                "INSERT INTO style_profiles VALUES (?,?,?,?,?,?,?)",
                (profile_id, now, now, name, name_lower, has_injection, pack(data)),
            )
        logger.info("Style profile '%s' created: %s", name, profile_id)
        return profile_id

    # ── Load ─────────────────────────────────────────────────

    def load(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Load a profile by its ID. Returns None if not found."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT data_json FROM style_profiles WHERE profile_id=?",
                (profile_id,),
            ).fetchone()
        if row is None:
            logger.warning("Style profile not found: %s", profile_id)
            return None
        return unpack(row["data_json"])

    def load_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Load the most recently created profile matching the given name."""
        name_lower = name.strip().lower()
        with _tx(self._db_path) as conn:
            row = conn.execute(
                """SELECT data_json FROM style_profiles
                   WHERE name_lower=?
                   ORDER BY updated_at DESC
                   LIMIT 1""",
                (name_lower,),
            ).fetchone()
        if row is None:
            return None
        return unpack(row["data_json"])

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Return summary info for all profiles, newest first."""
        with _tx(self._db_path) as conn:
            rows = conn.execute(
                """SELECT profile_id, created_at, updated_at, name,
                          has_injection, data_json
                   FROM style_profiles
                   ORDER BY updated_at DESC""",
            ).fetchall()
        profiles = []
        for row in rows:
            data = unpack(row["data_json"])
            profiles.append({
                "profile_id": row["profile_id"],
                "name": row["name"] or "Unnamed",
                "sample_documents": data.get("sample_documents", []),
                "created_at": row["created_at"],
                "last_modified": row["updated_at"],
                "has_injection": bool(row["has_injection"]),
            })
        return profiles

    def delete(self, profile_id: str) -> bool:
        with _tx(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM style_profiles WHERE profile_id=?",
                (profile_id,),
            )
            deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Style profile deleted: %s", profile_id)
        return deleted
