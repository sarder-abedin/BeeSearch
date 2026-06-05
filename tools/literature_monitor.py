"""tools/literature_monitor.py — Incremental literature monitoring"""
from __future__ import annotations
import json, logging, os
from datetime import datetime
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
_MONITOR_DIR = os.path.join(os.path.expanduser("~"), ".researchbuddy", "monitors")

def _monitor_path(monitor_id: str) -> str:
    os.makedirs(_MONITOR_DIR, exist_ok=True)
    return os.path.join(_MONITOR_DIR, f"{monitor_id}.json")

def save_monitor_state(monitor_id: str, research_question: str, search_queries: List[str], known_paper_keys: List[str]) -> None:
    existing = load_monitor_state(monitor_id)
    all_keys = list(set((existing or {}).get("known_paper_keys", []) + known_paper_keys))
    state = {"monitor_id": monitor_id, "research_question": research_question, "search_queries": search_queries, "known_paper_keys": all_keys, "last_run": datetime.now().isoformat(), "created": (existing or {}).get("created", datetime.now().isoformat())}
    try:
        with open(_monitor_path(monitor_id), "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save monitor state: %s", e)

def load_monitor_state(monitor_id: str) -> Optional[Dict[str, Any]]:
    path = _monitor_path(monitor_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def list_monitors() -> List[Dict[str, Any]]:
    os.makedirs(_MONITOR_DIR, exist_ok=True)
    monitors = []
    for fname in os.listdir(_MONITOR_DIR):
        if fname.endswith(".json"):
            s = load_monitor_state(fname[:-5])
            if s:
                monitors.append({"monitor_id": s["monitor_id"], "research_question": s.get("research_question", "")[:80], "last_run": s.get("last_run", ""), "paper_count": len(s.get("known_paper_keys", []))})
    return sorted(monitors, key=lambda x: x.get("last_run", ""), reverse=True)

def delete_monitor(monitor_id: str) -> None:
    path = _monitor_path(monitor_id)
    if os.path.exists(path):
        os.remove(path)

def find_new_papers(all_papers: List[Dict[str, Any]], known_keys: List[str]) -> List[Dict[str, Any]]:
    known_set = set(known_keys)
    return [p for p in all_papers if p.get("citation_key", "") not in known_set]

def monitor_id_from_question(research_question: str) -> str:
    import hashlib
    return hashlib.md5(research_question.strip().lower().encode()).hexdigest()[:12]
