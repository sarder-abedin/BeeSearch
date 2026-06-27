"""backend/app/services/systematic_review_service.py
─────────────────────────────────────────────────────────
Service layer for Mode 1 (Systematic Literature Review).

Mirrors ``research_assistant_service.py``'s pattern (build settings/state →
call the unmodified pipeline → return a raw dict the router wraps in a
Pydantic response) but for the 7-node PRISMA graph plus its 8 Explore-tool
deep-dives and 3 export formats.

Two things make this module less direct than Mode 3's:

1. ``run_systematic_review``'s ``stream_callback(node_name, final_state)``
   passes the *whole* evolving state on every node tick (unlike Mode 3's
   small per-call info dict), so :func:`run_sr` adapts it into the same
   ``(stage, info)`` shape ``backend.app.jobs.run_in_background`` expects --
   directly modeled on the ``stream_callback`` closure in
   ``ui/tabs/systematic_review.py::tab_systematic_review``.
2. The 8 Explore tools are deliberately *not* reached via one eager
   top-level import each -- every tool function is imported inside its own
   branch in :func:`run_explore_tool`, matching the lazy, function-local
   ``from tools.xxx import yyy`` pattern already used throughout
   ``ui/tabs/systematic_review.py`` (deferring optional/heavy deps -- plotly,
   pyvis, python-docx, reportlab -- until the specific tool that needs them
   actually runs).

Model/num_ctx resolution for Explore tools: the Streamlit tab reads the
*current sidebar settings* (not the original run's settings) when a deep-dive
button is clicked, since the user may have changed the model after running
the review. There is no persistent "current settings" in a stateless API, so
the natural analogue is: prefer an explicit override in the request's
``options`` (what the frontend's current model picker shows), falling back to
the values the original run actually used (``final_state["model_name"]`` /
``final_state["num_ctx"]``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from agents.systematic_review_graph import run_systematic_review
from agents.systematic_review_state import create_systematic_review_state

from ..schemas.systematic_review import (
    ExportRequest,
    GrammarCheckRequest,
    MetaAnalysisPoolRequest,
    SRRequest,
)

logger = logging.getLogger(__name__)

# Mirrors the `node_labels` dict in `ui/tabs/systematic_review.py::tab_systematic_review`.
NODE_LABELS: Dict[str, str] = {
    "query_generation":    "Generating search queries",
    "literature_search":   "Searching Google Scholar · arXiv · Semantic Scholar · CrossRef",
    "screening":           "Screening papers by title/abstract",
    "evidence_extraction": "Extracting evidence from papers",
    "quality_assessment":  "Assessing risk of bias, GRADE certainty, contradictions",
    "synthesis":           "Synthesising findings",
    "sr_eval":             "Evaluating review quality",
}


def _resolve_model(final_state: Dict[str, Any], options: Dict[str, Any]) -> tuple[str, int]:
    model = options.get("model") or final_state.get("model_name") or "llama3.1:8b"
    num_ctx = options.get("num_ctx") or final_state.get("num_ctx") or 32768
    return model, int(num_ctx)


def build_initial_state(req: SRRequest) -> Dict[str, Any]:
    """Build the initial ``SystematicReviewState`` for one request.

    Only forwards optional overrides that were actually provided, deferring
    to ``create_systematic_review_state``'s own defaults otherwise -- the
    same "only override when given" pattern as
    ``research_assistant_service.build_settings``.
    """
    kwargs: Dict[str, Any] = {}
    if req.model:
        kwargs["model_name"] = req.model
    if req.num_ctx is not None:
        kwargs["num_ctx"] = req.num_ctx
    if req.max_results is not None:
        kwargs["max_results"] = req.max_results
    if req.include_crossref is not None:
        kwargs["include_crossref"] = req.include_crossref
    return create_systematic_review_state(
        research_question=req.research_question.strip(),
        inclusion_criteria=list(req.inclusion_criteria),
        exclusion_criteria=list(req.exclusion_criteria),
        **kwargs,
    )


def run_sr(
    req: SRRequest, stream_callback: Callable[[str, Dict[str, Any]], None]
) -> Dict[str, Any]:
    """Run the Mode 1 PRISMA pipeline for one request; returns the raw final-state dict.

    Adapts ``run_systematic_review``'s ``(node_name, final_state)`` callback
    into the ``(stage, info)`` shape the job runner expects, extracting the
    same ``progress_pct``/``status_detail`` fields the Streamlit tab reads
    plus a friendly ``label`` from :data:`NODE_LABELS`.
    """
    initial_state = build_initial_state(req)

    def _adapter(node_name: str, final_state: Dict[str, Any]) -> None:
        stream_callback(node_name, {
            "label": NODE_LABELS.get(node_name, node_name),
            "progress_pct": final_state.get("progress_pct", 0),
            "status_detail": final_state.get("status_detail", ""),
        })

    return dict(run_systematic_review(initial_state, stream_callback=_adapter))


# ─────────────────────────────────────────────────────────────────────────────
# Explore tools (background-job dispatch)
# ─────────────────────────────────────────────────────────────────────────────

def _explore_citation_network(final_state: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    included = final_state.get("included_papers", [])
    if not included:
        return {"html": "", "stats": {}, "gap_candidates": [], "stance_counts": {}, "error": "No included papers to build a network from."}

    existing_html = final_state.get("citation_graph_html", "")
    if existing_html:
        return {"html": existing_html, "stats": {}, "gap_candidates": [], "stance_counts": {}}

    from tools.citation_network import (
        build_citation_network,
        classify_citation_stances,
        find_gap_candidates,
        network_stats,
        network_to_pyvis_html,
    )

    model, num_ctx = _resolve_model(final_state, options)
    G, meta, external_counts = build_citation_network(included, max_papers=25)
    stance_counts: Dict[str, Any] = {}
    if options.get("classify_stances") and G.number_of_edges():
        stance_counts = classify_citation_stances(G, meta, model, num_ctx)
    return {
        "html": network_to_pyvis_html(G, meta),
        "stats": network_stats(G),
        "gap_candidates": find_gap_candidates(external_counts),
        "stance_counts": stance_counts,
    }


def _explore_citation_context(final_state: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    included = final_state.get("included_papers", [])
    citing_idx = options.get("citing_idx")
    cited_idx = options.get("cited_idx")
    if citing_idx is None or cited_idx is None:
        raise ValueError("options must include 'citing_idx' and 'cited_idx'.")
    if not (0 <= citing_idx < len(included)) or not (0 <= cited_idx < len(included)):
        raise ValueError("citing_idx/cited_idx out of range for included_papers.")
    if citing_idx == cited_idx:
        raise ValueError("citing_idx and cited_idx must differ.")

    from tools.citation_context import extract_citation_context

    return extract_citation_context(included[citing_idx], included[cited_idx])


def _explore_reference_checking(final_state: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    evidence_table = final_state.get("evidence_table", [])
    if not evidence_table:
        return {"rob_table": [], "grade_results": {}, "contradictions": [], "source": "none"}

    rob_table = final_state.get("rob_table") or []
    grade_results = final_state.get("grade_results") or {}
    contradictions = final_state.get("contradictions")
    if contradictions is None:
        contradictions = []

    if rob_table or grade_results or contradictions:
        return {
            "rob_table": rob_table,
            "grade_results": grade_results,
            "contradictions": contradictions,
            "source": "existing",
        }

    from agents.contradiction_detector import detect_contradictions
    from agents.grade_assessment import grade_evidence_body
    from agents.risk_of_bias import assess_rob_batch

    model, num_ctx = _resolve_model(final_state, options)
    rq = final_state.get("research_question", "")
    rob_table = assess_rob_batch(evidence_table[:15], model, num_ctx)
    grade_results = grade_evidence_body(evidence_table, rq, rob_table, model, num_ctx)
    contradictions = detect_contradictions(evidence_table, rq, model, num_ctx)
    return {
        "rob_table": rob_table,
        "grade_results": grade_results,
        "contradictions": contradictions,
        "source": "recomputed",
    }


def _explore_preprint_status(final_state: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    included = final_state.get("included_papers", [])
    if not included:
        return {"results": [], "summary": {}}

    from tools.preprint_tracker import preprint_summary, track_preprints

    existing = final_state.get("preprint_tracking")
    results = existing if existing else track_preprints(included)
    return {"results": results, "summary": preprint_summary(results)}


def _explore_research_trends(final_state: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    from tools.trend_analyzer import analyze_trends, trend_to_chart_data

    rq = final_state.get("research_question", "")
    trend_data = analyze_trends(
        research_question=rq,
        search_queries=final_state.get("search_queries", []),
        corpus_papers=final_state.get("included_papers", []),
    )
    result: Dict[str, Any] = {
        "trend_data": trend_data,
        "chart_data": trend_to_chart_data(trend_data),
    }
    try:
        from tools.trend_analyzer import build_trend_figure
        fig = build_trend_figure(trend_data, rq)
        result["html"] = fig.to_html(full_html=False, include_plotlyjs="cdn")
    except ImportError:
        result["html"] = None
    return result


def _explore_concept_drift(final_state: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    all_papers = final_state.get("raw_papers", [])
    if not all_papers:
        return {"buckets": {}, "rising_terms": [], "declining_terms": [], "stable_terms": [], "llm_analysis": ""}

    from tools.concept_drift import detect_concept_drift

    model, num_ctx = _resolve_model(final_state, options)
    return detect_concept_drift(papers=all_papers, model_name=model, num_ctx=num_ctx)


def _explore_evidence_map(final_state: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    return build_evidence_map(final_state.get("evidence_table", []))


def _explore_meta_analysis(final_state: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Generic-dispatcher entry for Meta-Analysis -- runs the seed stage only.

    The LLM-draft and pool stages need extra request-specific input (edited
    rows, chosen measure) the generic ``options`` bag isn't shaped for; the
    dedicated ``/meta-analysis/seed``, ``/meta-analysis/draft`` and
    ``/meta-analysis/pool`` endpoints are the primary way to drive the full
    three-stage flow (see :func:`seed_meta_analysis`, :func:`draft_meta_analysis_rows`,
    :func:`pool_meta_analysis`).
    """
    return {"rows": seed_meta_analysis(final_state.get("evidence_table", []))}


_EXPLORE_DISPATCH: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    "citation_network": _explore_citation_network,
    "citation_context": _explore_citation_context,
    "reference_checking": _explore_reference_checking,
    "preprint_status": _explore_preprint_status,
    "research_trends": _explore_research_trends,
    "evidence_map": _explore_evidence_map,
    "meta_analysis": _explore_meta_analysis,
    "concept_drift": _explore_concept_drift,
}


def run_explore_tool(
    tool: str,
    final_state: Dict[str, Any],
    options: Dict[str, Any],
    stream_callback: Callable[[str, Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Dispatch one of the 8 Explore-tool deep-dives by name; returns its raw result dict."""
    handler = _EXPLORE_DISPATCH.get(tool)
    if handler is None:
        raise ValueError(f"Unknown explore tool: {tool!r}")
    stream_callback(tool, {"label": f"Running {tool}", "progress_pct": 0})
    result = handler(final_state, options)
    stream_callback(tool, {"label": f"Running {tool}", "progress_pct": 100})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Map (sync)
# ─────────────────────────────────────────────────────────────────────────────

def build_evidence_map(evidence_table: List[Dict[str, Any]]) -> Dict[str, Any]:
    from tools.evidence_map import build_evidence_map_data, evidence_map_to_plotly_html

    map_data = build_evidence_map_data(evidence_table)
    if map_data.get("total_studies", 0) == 0:
        return {"map_data": map_data, "html": None}
    try:
        html = evidence_map_to_plotly_html(map_data)
    except ImportError:
        html = None
    return {"map_data": map_data, "html": html}


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Analysis (sync seed + pool; LLM draft is a background job)
# ─────────────────────────────────────────────────────────────────────────────

def seed_meta_analysis(evidence_table: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from tools.meta_analysis import seed_meta_rows

    return seed_meta_rows(evidence_table)


def draft_meta_analysis_rows(
    final_state: Dict[str, Any],
    rows: List[Dict[str, Any]],
    measure: str,
    options: Dict[str, Any],
    stream_callback: Callable[[str, Dict[str, Any]], None],
) -> List[Dict[str, Any]]:
    """Best-effort LLM draft of effect/CI/N for each row from the matching paper's
    abstract -- the slow part of the meta-analysis flow, run as a background job.

    Mirrors the "Draft effect sizes from abstracts" button in
    `ui/tabs/systematic_review.py::_render_meta_analysis`.
    """
    from tools.meta_analysis import extract_effect_size_row

    included = final_state.get("included_papers", [])
    evidence_table = final_state.get("evidence_table", [])
    by_key = {p.get("citation_key"): p for p in included if p.get("citation_key")}
    evidence_by_key = {e.get("citation_key"): e for e in evidence_table if e.get("citation_key")}
    model, num_ctx = _resolve_model(final_state, options)

    rows = [dict(r) for r in rows]
    total = len(rows) or 1
    for i, row in enumerate(rows):
        ck = row.get("citation_key")
        paper = by_key.get(ck) or evidence_by_key.get(ck)
        if paper is not None:
            draft = extract_effect_size_row(paper, measure, model, num_ctx)
            if draft.get("found"):
                row["effect"] = draft["effect"]
                row["ci_low"] = draft["ci_low"]
                row["ci_high"] = draft["ci_high"]
                if draft.get("n") is not None:
                    row["n"] = draft["n"]
        stream_callback("meta_analysis_draft", {
            "label": "Drafting effect sizes",
            "progress_pct": int(100 * (i + 1) / total),
        })
    return rows


def get_measure_labels() -> Dict[str, str]:
    from tools.meta_analysis import MEASURE_LABELS

    return dict(MEASURE_LABELS)


def pool_meta_analysis(req: MetaAnalysisPoolRequest) -> Dict[str, Any]:
    from tools.meta_analysis import run_meta_analysis

    studies = [r.model_dump() for r in req.rows]
    result = run_meta_analysis(studies, measure=req.measure)
    forest_html: Optional[str] = None
    if result.get("ok"):
        try:
            from tools.meta_analysis import meta_analysis_to_forest_plotly
            forest_html = meta_analysis_to_forest_plotly(result, model=req.model)
        except ImportError:
            forest_html = None
    return {"result": result, "forest_html": forest_html}


# ─────────────────────────────────────────────────────────────────────────────
# Export: Markdown / DOCX / PDF / plain-language summaries
# ─────────────────────────────────────────────────────────────────────────────

def build_markdown(research_question: str, final_state: Dict[str, Any]) -> str:
    from tools.prisma_report import build_sr_markdown

    return build_sr_markdown(research_question, final_state)


def build_docx(final_state: Dict[str, Any], req: ExportRequest) -> bytes:
    from tools.prisma_report import generate_prisma_docx

    return generate_prisma_docx(final_state, author=req.author, institution=req.institution)


def build_pdf(final_state: Dict[str, Any], req: ExportRequest) -> bytes:
    from tools.prisma_report import generate_prisma_pdf

    return generate_prisma_pdf(final_state, author=req.author, institution=req.institution)


_SUMMARY_FORMATS = {"patient", "policy", "press", "all"}


def generate_plain_language_summary(
    final_state: Dict[str, Any], fmt: str, model: Optional[str], num_ctx: Optional[int]
) -> Dict[str, str]:
    """Generate one (or all three) plain-language summaries.

    Mirrors the format choice in `_tab_export`'s "Plain-Language Summaries"
    radio (`Patient / Public` / `Policy Brief` / `Press Release` / `All Three`).
    """
    if fmt not in _SUMMARY_FORMATS:
        raise ValueError(f"Unknown summary format: {fmt!r}. Expected one of {sorted(_SUMMARY_FORMATS)}.")

    from tools.plain_language import (
        generate_all_summaries,
        generate_patient_summary,
        generate_policy_brief,
        generate_press_release,
    )

    resolved_model, resolved_num_ctx = _resolve_model(final_state, {"model": model, "num_ctx": num_ctx})
    if fmt == "patient":
        return {"patient": generate_patient_summary(final_state, resolved_model, resolved_num_ctx)}
    if fmt == "policy":
        return {"policy": generate_policy_brief(final_state, resolved_model, resolved_num_ctx)}
    if fmt == "press":
        return {"press": generate_press_release(final_state, resolved_model, resolved_num_ctx)}
    return generate_all_summaries(final_state, resolved_model, resolved_num_ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Guided templates (read-only, no LLM/IO)
# ─────────────────────────────────────────────────────────────────────────────

def list_templates() -> List[Dict[str, Any]]:
    from tools.sr_templates import SR_TEMPLATES

    return SR_TEMPLATES


# ─────────────────────────────────────────────────────────────────────────────
# Grammar check gate (sync -- small, fast, capped-context LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def check_grammar(req: GrammarCheckRequest) -> Dict[str, Any]:
    """Mirrors `ui/helpers.py::_run_grammar_check`, the gate behind each of the
    SR tab's three guided inputs (`render_query_gate` on the research question,
    inclusion criteria, exclusion criteria)."""
    from tools.grammar_check import check_and_fix_grammar

    return check_and_fix_grammar(
        req.text,
        model_name=req.model or "",
        num_ctx=req.num_ctx or 8192,
        context_hint=req.context_hint,
    )
