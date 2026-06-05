"""tools/sensitivity_analysis.py — Sensitivity analysis for systematic reviews"""
from __future__ import annotations
import copy
import logging
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)

def run_sensitivity_analysis(
    base_state: Dict[str, Any],
    scenario_name: str,
    modified_inclusion: Optional[List[str]] = None,
    modified_exclusion: Optional[List[str]] = None,
    quality_filter: Optional[str] = None,
    run_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    if quality_filter:
        original_table = base_state.get("evidence_table", [])
        if quality_filter == "High":
            filtered = [e for e in original_table if e.get("quality") == "High"]
        elif quality_filter in ("High+Medium", "Medium"):
            filtered = [e for e in original_table if e.get("quality") in ("High", "Medium")]
        else:
            filtered = original_table
        retained_pct = round(100 * len(filtered) / max(len(original_table), 1), 1)
        return {
            "scenario": scenario_name, "quality_filter": quality_filter,
            "original_n": len(original_table), "filtered_n": len(filtered),
            "pct_retained": retained_pct, "evidence_table": filtered,
            "note": f"Quality filter: {quality_filter} only — {retained_pct}% retained.",
        }
    if not run_fn:
        return {"scenario": scenario_name, "error": "No run function provided."}
    modified_state = copy.deepcopy(base_state)
    if modified_inclusion is not None:
        modified_state["inclusion_criteria"] = modified_inclusion
    if modified_exclusion is not None:
        modified_state["exclusion_criteria"] = modified_exclusion
    for k in ("search_queries","raw_papers","screened_papers","included_papers","excluded_papers","evidence_table","completed_steps","errors"):
        modified_state[k] = []
    modified_state["current_step"] = "start"
    modified_state["progress_pct"] = 0
    try:
        result = run_fn(modified_state)
        return {
            "scenario": scenario_name,
            "original_n": len(base_state.get("included_papers", [])),
            "new_n": len(result.get("included_papers", [])),
            "original_conclusion": base_state.get("conclusion", ""),
            "new_conclusion": result.get("conclusion", ""),
            "evidence_table": result.get("evidence_table", []),
        }
    except Exception as e:
        return {"scenario": scenario_name, "error": str(e)}

def build_sensitivity_scenarios(base_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_inc = base_state.get("inclusion_criteria", [])
    return [
        {"name": "High-quality studies only", "description": "Restrict to papers rated High quality.", "quality_filter": "High"},
        {"name": "High and Medium quality", "description": "Exclude only Low quality papers.", "quality_filter": "High+Medium"},
        {"name": "Stricter inclusion (recent papers)", "description": "Add recency constraint.", "modified_inclusion": base_inc + ["Published in the last 10 years"], "modified_exclusion": base_state.get("exclusion_criteria", [])},
    ]
