"""
tools/sensitivity_analysis.py
─────────────────────────────────
Sensitivity analysis for the Systematic Review pipeline.

Lets a user check how robust an SR's conclusion is to its inclusion
criteria by re-deriving the evidence table or conclusion under an
alternative scenario — e.g. restricting to high-quality studies only, or
tightening inclusion criteria — without mutating the original SR state.
"""
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
    """Re-derive the evidence table or conclusion under one alternative scenario.

    Two independent modes, chosen by which arguments are supplied:
      - `quality_filter` set: cheap, local re-filtering of
        `base_state["evidence_table"]` by quality rating — no LLM/pipeline
        re-run needed.
      - `quality_filter` unset: re-runs the full SR pipeline via `run_fn`
        against a deep copy of `base_state` with `modified_inclusion`/
        `modified_exclusion` substituted and all downstream fields reset,
        so the original `base_state` is left untouched.

    Returns a dict describing the scenario's outcome; on `run_fn` failure,
    returns `{"scenario": scenario_name, "error": str(e)}` instead of
    raising.
    """
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
    # Clear every downstream pipeline field so run_fn re-derives them from
    # scratch under the new criteria, rather than reusing stale results
    # computed under the original inclusion/exclusion rules.
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
    """Return a fixed set of standard sensitivity-analysis scenario definitions.

    Each dict is suitable for passing (minus "name"/"description") as
    keyword arguments to `run_sensitivity_analysis`. Scenarios cover the
    common SR robustness checks: quality-only filters and a stricter
    recency-based inclusion criterion.
    """
    base_inc = base_state.get("inclusion_criteria", [])
    return [
        {"name": "High-quality studies only", "description": "Restrict to papers rated High quality.", "quality_filter": "High"},
        {"name": "High and Medium quality", "description": "Exclude only Low quality papers.", "quality_filter": "High+Medium"},
        {"name": "Stricter inclusion (recent papers)", "description": "Add recency constraint.", "modified_inclusion": base_inc + ["Published in the last 10 years"], "modified_exclusion": base_state.get("exclusion_criteria", [])},
    ]
