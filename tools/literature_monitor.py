"""
tools/literature_monitor.py
──────────────────────────────
Incremental literature monitoring for the Systematic Review pipeline.

Lets a user re-run an SR's search queries later and see only the papers
that are new since the last run. State (the research question, search
queries, and the set of citation keys already seen) is persisted as one
JSON file per monitor under `~/.beesearch/monitors/`, independent of the
SQLite session DB used elsewhere — monitors are meant to long-outlive any
single SR or Notebook session.
"""
from __future__ import annotations
import json, logging, os
from datetime import datetime
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
_MONITOR_DIR = os.path.join(os.path.expanduser("~"), ".beesearch", "monitors")

def _monitor_path(monitor_id: str) -> str:
    """Return the JSON file path for `monitor_id`, creating the monitors dir if needed."""
    os.makedirs(_MONITOR_DIR, exist_ok=True)
    return os.path.join(_MONITOR_DIR, f"{monitor_id}.json")

def save_monitor_state(monitor_id: str, research_question: str, search_queries: List[str], known_paper_keys: List[str]) -> None:
    """Merge `known_paper_keys` into any existing monitor state and write it to disk.

    Union-merges with previously known keys (rather than overwriting) so
    repeated runs accumulate the full set of papers ever seen, and
    preserves the original `created` timestamp across updates.
    """
    existing = load_monitor_state(monitor_id)
    all_keys = list(set((existing or {}).get("known_paper_keys", []) + known_paper_keys))
    state = {"monitor_id": monitor_id, "research_question": research_question, "search_queries": search_queries, "known_paper_keys": all_keys, "last_run": datetime.now().isoformat(), "created": (existing or {}).get("created", datetime.now().isoformat())}
    try:
        with open(_monitor_path(monitor_id), "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save monitor state: %s", e)

def load_monitor_state(monitor_id: str) -> Optional[Dict[str, Any]]:
    """Load a monitor's saved state, or `None` if it doesn't exist or fails to parse."""
    path = _monitor_path(monitor_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def list_monitors() -> List[Dict[str, Any]]:
    """List all saved monitors with summary fields, most recently run first."""
    os.makedirs(_MONITOR_DIR, exist_ok=True)
    monitors = []
    for fname in os.listdir(_MONITOR_DIR):
        if fname.endswith(".json"):
            s = load_monitor_state(fname[:-5])
            if s:
                monitors.append({"monitor_id": s["monitor_id"], "research_question": s.get("research_question", "")[:80], "last_run": s.get("last_run", ""), "paper_count": len(s.get("known_paper_keys", []))})
    return sorted(monitors, key=lambda x: x.get("last_run", ""), reverse=True)

def delete_monitor(monitor_id: str) -> None:
    """Delete a monitor's state file, if it exists."""
    path = _monitor_path(monitor_id)
    if os.path.exists(path):
        os.remove(path)

def find_new_papers(all_papers: List[Dict[str, Any]], known_keys: List[str]) -> List[Dict[str, Any]]:
    """Return the subset of `all_papers` whose citation_key is not in `known_keys`."""
    known_set = set(known_keys)
    return [p for p in all_papers if p.get("citation_key", "") not in known_set]

def monitor_id_from_question(research_question: str) -> str:
    """Derive a stable, filesystem-safe monitor ID from a research question.

    Hashing (rather than slugifying the text) avoids filename collisions
    and length/character issues across arbitrary user-entered questions.
    """
    import hashlib
    return hashlib.md5(research_question.strip().lower().encode()).hexdigest()[:12]
