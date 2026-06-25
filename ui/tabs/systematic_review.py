"""
ui/tabs/systematic_review.py
─────────────────────────────
Tab container for Mode 7 (user-facing Mode 1 — Systematic Literature Review).

Renders the entire PRISMA-style systematic review workflow: research question
+ inclusion/exclusion inputs, a run button that drives
`agents.systematic_review_graph.run_systematic_review`, and a tabbed results
view (Synthesis, Evidence, Explore, Write-up & Export) once a run completes.
The SR pipeline itself is stateless (see CLAUDE.md) — this module is the only
place SR results live, cached in `st.session_state["sr_last_result"]` so they
survive reruns triggered by widgets inside the results tabs.

Contents (in file order):
  1. Small render helpers       — `_render_prisma_flow`, `_render_evidence_table`
  2. Tab renderers              — Synthesis, Evidence (+ Abstract Screener),
                                   Explore (Citation Network, Preprint Status,
                                   Research Trends, Evidence Map, Meta-Analysis,
                                   Concept Drift), Write-up & Export
  3. Guided templates           — preset research-question/criteria starting
                                   points (`SR_TEMPLATES`) shown in an optional
                                   expander before the inputs
  4. Main entry point           — `tab_systematic_review(settings)`
  5. Markdown export helper     — `_build_sr_markdown`
"""

from __future__ import annotations

import logging
import time

import streamlit as st

from agents.systematic_review_graph import run_systematic_review
from agents.systematic_review_state import create_systematic_review_state
from ui.glossary import render_glossary_expander, term_help
from ui.helpers import render_eval_result, render_rag_reflection, render_query_gate

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Small render helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_prisma_flow(flow: dict) -> None:
    """Render the 5-stage PRISMA flow counts (Identified/Screened/Eligibility/Included/Excluded)
    as a row of `st.metric` columns. Pure display — reads only the `flow` dict passed in
    (`final_state["prisma_flow"]`), no session_state."""
    if not flow:
        return
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [
        ("Identified", "identified"), ("Screened", "screened"),
        ("Eligibility", "eligibility"), ("Included", "included"), ("Excluded", "excluded"),
    ]):
        col.metric(label, flow.get(key, 0))


def _render_evidence_table(evidence_table: list) -> None:
    """Render one expander per included paper (citation key, title, year, quality) with
    authors/journal/DOI, study design, sample size, relevance score, and key finding.
    Pure display — takes `final_state["evidence_table"]` as input, no session_state."""
    if not evidence_table:
        st.info("No papers were included in the review.")
        return
    for row in evidence_table:
        qual = row.get("quality", "Medium")
        ck = row.get("citation_key", "")
        title = row.get("title", "Untitled")
        year = row.get("year", "n.d.")
        with st.expander(f"[{ck}] {title[:70]} ({year}) — {qual}"):
            c1, c2 = st.columns([3, 1])
            with c1:
                auths = row.get("authors", [])
                st.markdown(f"**Authors:** {'; '.join(auths[:3])}{'…' if len(auths) > 3 else ''}")
                st.markdown(f"**Journal:** {row.get('journal') or 'N/A'}")
                if row.get("doi"):
                    st.markdown(f"**DOI:** [{row['doi']}](https://doi.org/{row['doi']})")
                elif row.get("url"):
                    st.markdown(f"[View paper]({row['url']})")
            with c2:
                st.metric("Study Design", row.get("study_design", "Unknown"))
                st.metric("Sample Size", row.get("sample_size", "Unknown"))
                st.metric("Relevance", f"{row.get('relevance_score', 3)}/5")
            if row.get("key_finding"):
                st.markdown(f"**Key finding:** {row['key_finding']}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab renderers (called after SR finishes)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_synthesis(final_state: dict, settings: dict) -> None:
    """Render the Synthesis results tab: key themes, narrative synthesis, research gaps,
    conclusion, limitations, and a feedback/refinement section for the synthesis text.

    Reads `final_state["session_id"]` to key the feedback widget and
    `session_state[f"_fb_output_{session_id}"]` (written by `render_feedback_section`)
    as the current draft to refine. Delegates rendering of the feedback UI itself to
    `ui.helpers.render_feedback_section`."""
    themes = final_state.get("key_themes", [])
    if themes:
        st.markdown("**Key Themes:**")
        for t in themes:
            st.markdown(f"- {t}")
        st.divider()

    grade_results = final_state.get("grade_results", {})
    if grade_results and grade_results.get("overall_grade"):
        grade = grade_results["overall_grade"]
        grade_color = {"High": "🟢", "Moderate": "🟡", "Low": "🟠", "Very low": "🔴"}.get(grade, "⚪")
        st.markdown(
            f"**Certainty of evidence (GRADE):** {grade_color} {grade} — "
            f"{grade_results.get('certainty_statement', '')}",
            help="Overall GRADE rating across all included studies. See the Explore → Risk & Certainty tab for per-domain detail.",
        )
        contradictions = final_state.get("contradictions", [])
        if contradictions:
            st.caption(f"⚠️ {len(contradictions)} conflicting finding(s) detected — see Explore → Risk & Certainty.")
        st.divider()

    st.subheader("Narrative Synthesis")
    st.markdown(final_state.get("narrative_synthesis", "*No synthesis generated.*"))
    st.divider()

    gaps = final_state.get("research_gaps", [])
    if gaps:
        st.markdown("**Research Gaps:**")
        for g in gaps:
            st.markdown(f"- {g}")
        st.divider()

    if final_state.get("conclusion"):
        st.subheader("Conclusion")
        st.markdown(final_state["conclusion"])
    if final_state.get("limitations"):
        st.subheader("Limitations of this Review")
        st.info(final_state["limitations"])

    from ui.helpers import render_feedback_section
    session_id = final_state.get("session_id", "sr")
    current = st.session_state.get(
        f"_fb_output_{session_id}", final_state.get("narrative_synthesis", "")
    )
    render_feedback_section(
        current_output=current,
        session_key=session_id,
        mode="systematic_review",
        model_name=final_state.get("model_name", "llama3.1:8b"),
        num_ctx=final_state.get("num_ctx", 32768),
        context="\n".join(
            p.get("title", "") + " " + p.get("abstract", "")[:200]
            for p in final_state.get("included_papers", [])[:5]
        ),
        key_suffix=f"_sr_{session_id}",
    )


def _tab_evidence(final_state: dict, settings: dict) -> None:
    """Render the Evidence results tab: the included-papers evidence table (via
    `_render_evidence_table`), an expander of excluded papers with exclusion reasons,
    and the Abstract Screener section (via `_render_abstract_screener`). No
    session_state of its own — purely composes other render helpers."""
    n_inc = len(final_state.get("included_papers", []))
    n_exc = len(final_state.get("excluded_papers", []))
    st.subheader(f"Evidence Table ({n_inc} included papers)", help=term_help("Quality score"))
    _render_evidence_table(final_state.get("evidence_table", []))

    if final_state.get("excluded_papers"):
        with st.expander(f"Excluded Papers ({n_exc})"):
            for p in final_state["excluded_papers"]:
                reason = p.get("exclusion_reason", "")
                st.markdown(f"- **{p.get('title','')[:70]}** ({p.get('year','n.d.')}) — _{reason}_")

    st.divider()
    _render_abstract_screener(final_state, settings)


def _render_abstract_screener(final_state: dict, settings: dict) -> None:
    """Render the Abstract Screener section: summary metrics (total/include/uncertain/
    exclude counts, mean score) plus a verdict-filterable list of every paper's LLM
    relevance score and rationale.

    Reads `final_state["screener_scores"]`. Writes/reads the verdict filter via the
    `screener_filter` selectbox widget key (no other session_state)."""
    st.subheader("Abstract Screener")
    st.markdown(
        "LLM relevance scores (0–100) for every paper retrieved before the inclusion/exclusion "
        "screening decision was made. Scores above 80 = clearly include; 50–79 = uncertain; "
        "below 50 = likely exclude."
    )
    screener_scores = final_state.get("screener_scores", [])
    if screener_scores:
        from tools.abstract_screener import screener_summary
        summary = screener_summary(screener_scores)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scored", summary["total"])
        c2.metric("Include", summary["include"])
        c3.metric("Uncertain", summary["uncertain"])
        c4.metric("Exclude", summary["exclude"])
        st.caption(f"Mean relevance score: {summary['mean_score']}/100")
        st.divider()
        verdict_filter = st.selectbox(
            "Show verdicts:", ["all", "include", "uncertain", "exclude"],
            key="screener_filter",
        )
        for r in screener_scores:
            if verdict_filter != "all" and r.get("verdict") != verdict_filter:
                continue
            paper = r.get("paper", {})
            score = r.get("score", 0)
            verdict = r.get("verdict", "")
            rationale = r.get("rationale", "")
            color = {"include": "[INCLUDE]", "uncertain": "[UNCERTAIN]", "exclude": "[EXCLUDE]"}.get(verdict, "")
            with st.expander(f"{color} [{score}/100] {paper.get('title','')[:70]}"):
                st.markdown(f"**Verdict:** {verdict.upper()}  |  **Score:** {score}/100")
                st.markdown(f"**Rationale:** {rationale}")
                if paper.get("abstract"):
                    st.markdown(f"**Abstract:** {paper['abstract'][:300]}…")
    else:
        st.info("Abstract screener scores will appear here after running the systematic review.")


def _render_citation_network_section(final_state: dict, settings: dict) -> None:
    """Render the Citation Network deep-dive tool: a button that calls
    `tools.citation_network.build_citation_network` (Semantic Scholar API) to build an
    ego network of citation links between included papers, then renders the cached
    pyvis HTML plus most-cited/isolated-paper/gap-candidate lists.

    Reuses any previously computed network from `final_state["citation_graph_html"]`
    (set by the SR pipeline itself) before offering to build a new one. Caches the
    freshly built network in `session_state["_cn_html"]`, `"_cn_stats"`, `"_cn_gaps"`
    so it survives reruns without re-querying Semantic Scholar. These keys are cleared
    whenever a new SR run starts (see `tab_systematic_review`)."""
    st.subheader("Citation Network", help=term_help("Citation network"))
    st.markdown(
        "Ego network showing citation links **between** the included papers. "
        "Node colour = quality (Green High · Amber Medium · Red Low). "
        "Requires Semantic Scholar API calls — click below to fetch them (~30s for 20 papers)."
    )
    included = final_state.get("included_papers", [])
    if not included:
        st.info("No included papers to build a network from.")
        return

    existing_html = final_state.get("citation_graph_html", "")
    if existing_html:
        st.components.v1.html(existing_html, height=520, scrolling=False)
        return

    classify = st.checkbox(
        "Classify citation stances (Smart Citations — LLM)",
        key="cn_classify_stances",
        help="Label each citation edge Supporting (green), Contrasting (red), or Mentioning (gray) "
             "by comparing the two papers' abstracts. Adds one LLM call per edge.",
    )
    if st.button("Build Citation Network", key="build_network"):
        model = settings.get("model", "llama3.1:8b")
        num_ctx = settings.get("num_ctx", 32768)
        with st.spinner("Querying Semantic Scholar for citation links…"):
            try:
                from tools.citation_network import (
                    build_citation_network,
                    classify_citation_stances,
                    network_to_pyvis_html,
                    network_stats,
                    find_gap_candidates,
                )
                G, meta, external_counts = build_citation_network(included, max_papers=25)
                stance_counts = {}
                if classify and G.number_of_edges():
                    with st.spinner("Classifying citation stances (Supporting / Contrasting / Mentioning)…"):
                        stance_counts = classify_citation_stances(G, meta, model, num_ctx)
                html = network_to_pyvis_html(G, meta)
                stats = network_stats(G)
                st.session_state["_cn_html"] = html
                st.session_state["_cn_stats"] = stats
                st.session_state["_cn_stances"] = stance_counts
                st.session_state["_cn_gaps"] = find_gap_candidates(external_counts)
                msg = (
                    f"Network built: {stats['nodes']} nodes, "
                    f"{stats['edges']} citation edges, "
                    f"{stats['isolated']} isolated papers."
                )
                if stance_counts.get("classified"):
                    msg += (
                        f" Stances — {stance_counts.get('Supporting', 0)} supporting, "
                        f"{stance_counts.get('Contrasting', 0)} contrasting, "
                        f"{stance_counts.get('Mentioning', 0)} mentioning."
                    )
                st.success(msg)
            except Exception as e:
                st.error(f"Citation network failed: {e}")

    if st.session_state.get("_cn_html"):
        st.components.v1.html(st.session_state["_cn_html"], height=520, scrolling=False)
        stance_counts = st.session_state.get("_cn_stances") or {}
        if stance_counts.get("classified"):
            st.caption(
                "Edge colours — 🟢 Supporting · 🔴 Contrasting · ⚪ Mentioning. "
                f"({stance_counts.get('Supporting', 0)} / {stance_counts.get('Contrasting', 0)} / "
                f"{stance_counts.get('Mentioning', 0)})"
            )
        stats = st.session_state.get("_cn_stats", {})
        if stats.get("most_cited"):
            st.markdown("**Most cited within corpus:**")
            for node, deg in stats["most_cited"]:
                st.markdown(f"- {node} — cited by {deg} included paper(s)")

        isolated_papers = stats.get("isolated_papers") or []
        if isolated_papers:
            with st.expander(
                f"Isolated papers ({len(isolated_papers)}) — no citation links to the rest of the corpus"
            ):
                for node in isolated_papers:
                    st.markdown(f"- {node}")

        gaps = st.session_state.get("_cn_gaps") or []
        if gaps:
            st.markdown(
                "**Frequently cited but not in your review — consider screening:**",
                help="Papers cited by 2+ of your included papers, but not themselves included.",
            )
            for g in gaps:
                label = g["title"]
                if g.get("year"):
                    label += f" ({g['year']})"
                if g.get("venue"):
                    label += f" — {g['venue']}"
                if g.get("url"):
                    label = f"[{label}]({g['url']})"
                st.markdown(f"- {label} — cited by {g['cited_by_count']} included paper(s)")


def _render_preprint_status_section(final_state: dict, settings: dict) -> None:
    """Render the Preprint Status deep-dive tool: a button that calls
    `tools.preprint_tracker.track_preprints` (CrossRef API) to check each included
    paper's publication status and flag retractions.

    Reuses `final_state["preprint_tracking"]` if already populated by the SR pipeline;
    otherwise caches the freshly fetched tracking list in `session_state["_pt_tracking"]`
    / `"_pt_summary"` so it survives reruns. These keys are cleared on a new SR run."""
    st.subheader("Preprint Status")
    st.markdown(
        "Checks each included paper against CrossRef to identify unverified preprints "
        "and flag any retractions. Requires CrossRef API calls — click below to check "
        "(~0.25s per paper)."
    )
    included = final_state.get("included_papers", [])
    if not included:
        st.info("No included papers.")
        return

    existing_tracking = final_state.get("preprint_tracking", [])
    if existing_tracking:
        _render_preprint_tracking(existing_tracking)
        return

    if st.button("Check Preprint Status", key="check_preprints"):
        with st.spinner("Querying CrossRef for publication status…"):
            try:
                from tools.preprint_tracker import track_preprints, preprint_summary
                tracking = track_preprints(included)
                summary = preprint_summary(tracking)
                st.session_state["_pt_tracking"] = tracking
                st.session_state["_pt_summary"] = summary
            except Exception as e:
                st.error(f"Preprint tracking failed: {e}")

    tracking = st.session_state.get("_pt_tracking")
    if tracking:
        _render_preprint_tracking(tracking)


def _render_preprint_tracking(tracking: list) -> None:
    """Render the preprint-status summary metrics and one expander per paper
    (status, note, published DOI/venue if applicable). Pure display, no session_state —
    shared by `_render_preprint_status_section` for both fresh and cached tracking data."""
    from tools.preprint_tracker import preprint_summary
    summary = preprint_summary(tracking)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Journal", summary.get("journal", 0))
    c2.metric("Published (was preprint)", summary.get("published", 0))
    c3.metric("Preprint only", summary.get("preprint", 0))
    c4.metric("Retracted", summary.get("retracted", 0))
    st.divider()
    for r in tracking:
        paper = r.get("paper", {})
        status = r.get("preprint_status", "unknown")
        label = {"journal": "[JOURNAL]", "published": "[PUBLISHED]", "preprint": "[PREPRINT]", "retracted": "[RETRACTED]"}.get(status, "[UNKNOWN]")
        with st.expander(f"{label} {paper.get('title','')[:70]} — {status.upper()}"):
            st.markdown(f"**Status:** {status}")
            st.markdown(f"**Note:** {r.get('note','')}")
            if r.get("published_doi"):
                st.markdown(f"**Published DOI:** [{r['published_doi']}](https://doi.org/{r['published_doi']})")
            if r.get("published_venue"):
                st.markdown(f"**Journal:** {r['published_venue']}")


def _seed_meta_rows(evidence_table: list) -> list:
    """Build initial editable rows (label + blank effect/CI/N) from the evidence table.
    Pure data transform — no Streamlit calls, no session_state."""
    import re
    rows = []
    for row in evidence_table:
        authors = row.get("authors", [])
        author_str = authors[0].split(",")[0] if authors else (row.get("citation_key") or "Unknown")
        year = row.get("year") or "n.d."
        label = f"{author_str} et al. ({year})" if len(authors) > 1 else f"{author_str} ({year})"
        n_match = re.search(r"\d+", str(row.get("sample_size") or ""))
        rows.append({
            "citation_key": row.get("citation_key", ""),
            "label": label,
            "effect": None,
            "ci_low": None,
            "ci_high": None,
            "n": int(n_match.group()) if n_match else None,
        })
    return rows


def _render_meta_analysis(final_state: dict, settings: dict) -> None:
    """Pool effect sizes from the evidence table into a forest plot (generic inverse-variance).

    Renders an editable `st.data_editor` of per-study effect/CI/N, an optional LLM
    "draft from abstracts" pass (`tools.meta_analysis.extract_effect_size_row`), and —
    once run — fixed-effect/random-effects estimates, heterogeneity stats, and a forest
    plot via `tools.meta_analysis.run_meta_analysis` / `meta_analysis_to_forest_plotly`.

    Session_state keys (all `_meta_*`, scoped to this single SR result — there is only
    ever one active review at a time, so no notebook/session id suffix is needed):
      - `_meta_table_hash` — MD5 of the evidence table's citation keys; when this
        changes (new SR run) the editable rows are reseeded from scratch.
      - `_meta_rows` — the editable row data backing the data_editor.
      - `_meta_seed_version` — bumped on every reseed/LLM-draft pass so the
        `_meta_editor_{version}` widget key changes, forcing Streamlit to remount the
        data_editor with the new rows instead of treating it as a continuation of the
        previous edit session.
      - `_meta_result` — the last `run_meta_analysis` output; cleared whenever the rows
        are reseeded or redrafted so a stale forest plot can't be shown against new data.
    """
    import hashlib

    import pandas as pd

    from tools.meta_analysis import MEASURE_LABELS, extract_effect_size_row, run_meta_analysis

    st.subheader("Statistical Meta-Analysis", help=term_help("Pooled effect / meta-analysis"))
    st.markdown(
        "Pool each study's reported effect size and 95% confidence interval into a single "
        "estimate with a forest plot — the standard step for combining evidence across studies. "
        "Enter the numbers from each paper's results (or draft them with the LLM below), then "
        "review and correct them before pooling — abstracts often round or omit statistics."
    )

    evidence_table = final_state.get("evidence_table", [])
    included = final_state.get("included_papers", [])
    if not evidence_table:
        st.info("Evidence table is empty — run the systematic review first.")
        return

    identity = [str(r.get("citation_key", "")) for r in evidence_table]
    table_hash = hashlib.md5("|".join(identity).encode("utf-8")).hexdigest()[:10]
    if st.session_state.get("_meta_table_hash") != table_hash:
        st.session_state["_meta_rows"] = _seed_meta_rows(evidence_table)
        st.session_state["_meta_table_hash"] = table_hash
        st.session_state["_meta_seed_version"] = st.session_state.get("_meta_seed_version", 0) + 1
        st.session_state.pop("_meta_result", None)

    measure_codes = list(MEASURE_LABELS.keys())
    measure = st.selectbox(
        "Effect measure", options=measure_codes,
        format_func=lambda code: MEASURE_LABELS[code],
        key="_meta_measure", help=term_help("Effect size"),
    )

    if st.button("Draft effect sizes from abstracts (LLM, best-effort)", key="meta_llm_draft"):
        by_key = {p.get("citation_key"): p for p in included if p.get("citation_key")}
        evidence_by_key = {e.get("citation_key"): e for e in evidence_table if e.get("citation_key")}
        rows = st.session_state["_meta_rows"]
        model = settings.get("model", "llama3.1:8b")
        num_ctx = settings.get("num_ctx", 32768)
        status_text = st.empty()
        progress_bar = st.progress(0)
        n_filled = 0
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
                    n_filled += 1
            pct = int(100 * (i + 1) / len(rows))
            progress_bar.progress(pct)
            status_text.markdown(f"**Drafting effect sizes…** `{pct}%` ({i + 1}/{len(rows)})")
        status_text.empty()
        progress_bar.empty()
        st.session_state["_meta_rows"] = rows
        st.session_state["_meta_seed_version"] = st.session_state.get("_meta_seed_version", 0) + 1
        st.session_state.pop("_meta_result", None)
        if n_filled:
            st.success(
                f"Drafted {n_filled} of {len(rows)} effect sizes from abstracts. Review carefully "
                "before pooling — abstracts often round or omit statistics."
            )
        else:
            st.warning("No usable effect sizes found in the abstracts. Enter them manually below.")

    st.caption("Edit any cell, or add/remove rows — leave a study blank to exclude it from pooling.")
    df = pd.DataFrame(st.session_state["_meta_rows"])
    for col in ("label", "effect", "ci_low", "ci_high", "n"):
        if col not in df.columns:
            df[col] = None
    editor_key = f"_meta_editor_{st.session_state.get('_meta_seed_version', 0)}"
    edited = st.data_editor(
        df[["label", "effect", "ci_low", "ci_high", "n"]],
        key=editor_key, num_rows="dynamic", use_container_width=True,
        column_config={
            "label": st.column_config.TextColumn("Study", width="medium"),
            "effect": st.column_config.NumberColumn(
                "Effect", format="%.3f",
                help=f"Point estimate as {MEASURE_LABELS[measure]}",
            ),
            "ci_low": st.column_config.NumberColumn("95% CI low", format="%.3f"),
            "ci_high": st.column_config.NumberColumn("95% CI high", format="%.3f"),
            "n": st.column_config.NumberColumn("N", format="%.0f", help="Sample size (optional)"),
        },
    )

    if st.button("Run Meta-Analysis", key="run_meta_analysis_btn", type="primary"):
        st.session_state["_meta_result"] = run_meta_analysis(edited.to_dict("records"), measure=measure)

    result = st.session_state.get("_meta_result")
    if not result:
        return
    if not result.get("ok"):
        st.warning(result.get("reason", "Could not pool these studies."))
        return

    st.divider()
    fe, rand_fx, het = result["fixed_effect"], result["random_effects"], result["heterogeneity"]
    st.markdown(f"**Pooled {result['measure_label']}** — k = {result['k']} studies")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fixed-effect", f"{fe['estimate']:.2f}",
              help=f"95% CI [{fe['ci_low']:.2f}, {fe['ci_high']:.2f}]")
    c2.metric("Random-effects", f"{rand_fx['estimate']:.2f}",
              help=f"95% CI [{rand_fx['ci_low']:.2f}, {rand_fx['ci_high']:.2f}]")
    c3.metric("I² (heterogeneity)", f"{het['i_squared']:.0f}%", help=term_help("Heterogeneity (I²)"))
    c4.metric("τ² (between-study variance)", f"{het['tau_squared']:.3f}")
    st.caption(f"Cochran's Q = {het['q']:.2f} (df = {het['df']}) · {het['interpretation']}")

    model_choice = st.radio(
        "Pooling model shown in the forest plot", options=["random", "fixed"],
        format_func=lambda m: ("Random-effects (recommended — allows for between-study variation)"
                               if m == "random" else "Fixed-effect (assumes one true underlying effect)"),
        key="_meta_model_choice", horizontal=True,
        help=term_help("Fixed-effect vs. random-effects model"),
    )

    st.subheader("Forest Plot", help=term_help("Forest plot"))
    try:
        from tools.meta_analysis import meta_analysis_to_forest_plotly
        import streamlit.components.v1 as components
        html = meta_analysis_to_forest_plotly(result, model=model_choice)
        components.html(html, height=130 + 42 * (result["k"] + 1) + 20, scrolling=False)
    except ImportError:
        from tools.meta_analysis import meta_analysis_to_forest_png
        png = meta_analysis_to_forest_png(result, model=model_choice)
        st.image(png, caption="Forest plot (matplotlib fallback)")


def _render_research_trends_section(final_state: dict, settings: dict) -> None:
    """Render the Research Trend Forecaster: a button that calls
    `tools.trend_analyzer.analyze_trends` (CrossRef field-wide year counts vs. this
    review's corpus) and plots the comparison with plotly (falls back to `st.json`
    if plotly isn't installed).

    Caches the fetched trend data in `session_state["_trend_data"]` / `"_trend_json"`
    so it survives reruns without re-querying CrossRef. Cleared on a new SR run."""
    rq = final_state.get("research_question", "")
    included = final_state.get("included_papers", [])

    st.subheader("Research Trend Forecaster")
    st.markdown(
        "Publication volume by year for this research area, sourced from CrossRef (field-wide) "
        "and compared to the papers retrieved in this SR run. Requires CrossRef API calls — "
        "click below to fetch them (a few seconds)."
    )

    if st.button("Analyze Trends", key="run_trends"):
        with st.spinner("Querying CrossRef for field-wide year counts…"):
            try:
                from tools.trend_analyzer import analyze_trends, trend_to_chart_data
                import json
                trend_data = analyze_trends(
                    research_question=rq,
                    search_queries=final_state.get("search_queries", []),
                    corpus_papers=included,
                )
                st.session_state["_trend_data"] = trend_data
                st.session_state["_trend_json"] = trend_to_chart_data(trend_data)
            except Exception as e:
                st.error(f"Trend analysis failed: {e}")

    if st.session_state.get("_trend_data"):
        td = st.session_state["_trend_data"]
        ca, cb, cc = st.columns(3)
        ca.metric("Field Trend", td.get("trend", "unknown").capitalize())
        cb.metric("Peak Year", td.get("peak_year", "N/A"))
        cc.metric("Total Publications (CrossRef)", f"{td.get('total_field', 0):,}")

        import json
        chart_json = st.session_state.get("_trend_json", "{}")
        chart = json.loads(chart_json)
        years = chart.get("years", [])
        if years:
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=years,
                    y=chart.get("field_counts", []),
                    name="Field-wide (CrossRef)",
                    marker_color="#2563EB",
                    opacity=0.7,
                ))
                fig.add_trace(go.Scatter(
                    x=years,
                    y=chart.get("corpus_counts", []),
                    name="This SR corpus",
                    mode="lines+markers",
                    marker=dict(color="#F59E0B", size=6),
                    line=dict(color="#F59E0B", width=2),
                ))
                fig.update_layout(
                    title=f"Publication Trend: {rq[:60]}…" if len(rq) > 60 else f"Publication Trend: {rq}",
                    xaxis=dict(title="Year", gridcolor="#E2E8F0"),
                    yaxis=dict(title="Publications", gridcolor="#E2E8F0"),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#F8FAFC",
                    font=dict(color="#334155"),
                    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#E2E8F0", borderwidth=1),
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("Install plotly (`pip install plotly`) to see the trend chart.")
                st.json(chart)


def _render_evidence_map_section(final_state: dict, settings: dict) -> None:
    """Render the Evidence Map: a Population x Intervention bubble chart built purely
    from `final_state["evidence_table"]` via `tools.evidence_map.build_evidence_map_data`
    / `evidence_map_to_plotly_html` (falls back to a matplotlib PNG if plotly isn't
    installed). No API calls, no caching — recomputed on every render since it's cheap."""
    st.subheader("Evidence Map", help=term_help("Evidence map"))
    st.markdown(
        "Bubble chart of evidence density across Population × Intervention dimensions. "
        "Bubble size = number of studies; colour = average quality. Renders instantly from "
        "this review's evidence table — no extra API calls."
    )
    evidence_table = final_state.get("evidence_table", [])

    if not evidence_table:
        st.info("Evidence table is empty — run the systematic review first.")
    else:
        try:
            from tools.evidence_map import build_evidence_map_data, evidence_map_to_plotly_html
            import streamlit.components.v1 as components
            map_data = build_evidence_map_data(evidence_table)
            if map_data["total_studies"] == 0:
                st.info("No evidence data to map.")
            else:
                ca, cb = st.columns(2)
                ca.metric("Populated Cells", map_data["total_cells"])
                cb.metric("Total Studies Mapped", map_data["total_studies"])
                try:
                    html = evidence_map_to_plotly_html(map_data)
                    components.html(html, height=480, scrolling=False)
                except ImportError:
                    from tools.evidence_map import evidence_map_to_png
                    png = evidence_map_to_png(map_data)
                    st.image(png, caption="Evidence Map (matplotlib fallback)")
        except Exception as e:
            st.error(f"Evidence map failed: {e}")


def _render_concept_drift_section(final_state: dict, settings: dict) -> None:
    """Render the Concept Drift Tracker: a button that calls
    `tools.concept_drift.detect_concept_drift` (LLM analysis pass over the full corpus,
    `final_state["raw_papers"]`) to surface rising/declining vocabulary across time
    buckets, plus an LLM narrative summary of the shifts.

    Caches the result in `session_state["_drift_data"]` so it survives reruns without
    re-running the LLM analysis. Cleared on a new SR run."""
    model = settings.get("model", "llama3.1:8b")
    num_ctx = settings.get("num_ctx", 32768)

    st.subheader("Concept Drift Tracker", help=term_help("Concept drift"))
    st.markdown(
        "Detects vocabulary shifts across time periods in the included papers — "
        "which terms are rising, which are declining. Runs an LLM analysis pass over "
        "the corpus (~tens of seconds depending on corpus size)."
    )

    all_papers = final_state.get("raw_papers", [])
    if not all_papers:
        st.info("No papers in corpus.")
    elif st.button("Detect Concept Drift", key="run_drift"):
        with st.spinner("Analysing vocabulary evolution across time buckets…"):
            try:
                from tools.concept_drift import detect_concept_drift
                drift = detect_concept_drift(
                    papers=all_papers,
                    model_name=model,
                    num_ctx=num_ctx,
                )
                st.session_state["_drift_data"] = drift
            except Exception as e:
                st.error(f"Concept drift analysis failed: {e}")

    drift = st.session_state.get("_drift_data")
    if drift:
        buckets = drift.get("buckets", {})
        if buckets:
            st.markdown("**Vocabulary by time period:**")
            for label, meta in list(buckets.items())[:6]:
                with st.expander(f"{label} — {meta['papers']} papers"):
                    st.markdown(", ".join(meta.get("top_terms", [])))

        col_r, col_d = st.columns(2)
        with col_r:
            st.markdown("**Rising terms** (becoming more prominent):")
            for r in drift.get("rising_terms", [])[:8]:
                st.markdown(f"- **{r['term']}** (+{r['growth']}) · {r['first_bucket']} → {r['last_bucket']}")
        with col_d:
            st.markdown("**Declining terms** (becoming less prominent):")
            for d in drift.get("declining_terms", [])[:8]:
                st.markdown(f"- **{d['term']}** ({d['growth']}) · {d['first_bucket']} → {d['last_bucket']}")

        if drift.get("llm_analysis"):
            st.divider()
            st.markdown("**LLM Analysis of Vocabulary Shifts:**")
            st.markdown(drift["llm_analysis"])


def _render_reference_checking_section(final_state: dict, settings: dict) -> None:
    """Render the Risk & Certainty deep-dive tool: per-paper risk of bias
    (RoB 2 / ROBINS-I), an overall GRADE certainty-of-evidence rating, and
    cross-paper contradiction detection.

    The SR pipeline's `quality_assessment_node` populates `rob_table`,
    `grade_results`, and `contradictions` on every run, so this section renders
    them directly from `final_state` when present. For older cached results that
    predate the pipeline node (no data in state), a button recomputes them
    on demand via the same `agents.risk_of_bias` / `agents.grade_assessment` /
    `agents.contradiction_detector` helpers, caching into
    `session_state["_rc_rob"]` / `"_rc_grade"` / `"_rc_contra"` (cleared on a new SR run)."""
    model = settings.get("model", "llama3.1:8b")
    num_ctx = settings.get("num_ctx", 32768)

    st.subheader("Risk of Bias · GRADE · Contradictions", help=term_help("Quality score"))
    st.markdown(
        "Per-paper **risk of bias** (RoB 2 for trials, ROBINS-I for observational studies), an "
        "overall **GRADE** certainty-of-evidence rating, and detected **contradictions** across "
        "the included papers. Computed automatically during the review — recompute below if needed."
    )

    evidence_table = final_state.get("evidence_table", [])
    if not evidence_table:
        st.info("No extracted evidence to assess.")
        return

    rob_table = final_state.get("rob_table") or st.session_state.get("_rc_rob") or []
    grade_results = final_state.get("grade_results") or st.session_state.get("_rc_grade") or {}
    contradictions = final_state.get("contradictions")
    if contradictions is None:
        contradictions = st.session_state.get("_rc_contra") or []

    if not (rob_table or grade_results or contradictions):
        if st.button("Assess Risk of Bias, GRADE & Contradictions", key="run_reference_checking"):
            rq = final_state.get("research_question", "")
            with st.spinner("Assessing risk of bias, GRADE certainty, and contradictions…"):
                try:
                    from agents.risk_of_bias import assess_rob_batch
                    from agents.grade_assessment import grade_evidence_body
                    from agents.contradiction_detector import detect_contradictions
                    rob_table = assess_rob_batch(evidence_table[:15], model, num_ctx)
                    grade_results = grade_evidence_body(evidence_table, rq, rob_table, model, num_ctx)
                    contradictions = detect_contradictions(evidence_table, rq, model, num_ctx)
                    st.session_state["_rc_rob"] = rob_table
                    st.session_state["_rc_grade"] = grade_results
                    st.session_state["_rc_contra"] = contradictions
                except Exception as e:
                    st.error(f"Reference checking failed: {e}")
        else:
            return

    _render_reference_checking_results(rob_table, grade_results, contradictions)


def _render_reference_checking_results(rob_table: list, grade_results: dict, contradictions: list) -> None:
    """Render the GRADE rating, per-paper risk-of-bias table, and contradiction
    cards. Pure display, no session_state — shared by the fresh-pipeline and
    recomputed code paths of `_render_reference_checking_section`."""
    if grade_results:
        grade = grade_results.get("overall_grade", "n/a")
        grade_color = {"High": "🟢", "Moderate": "🟡", "Low": "🟠", "Very low": "🔴"}.get(grade, "⚪")
        st.markdown(f"### Certainty of Evidence (GRADE): {grade_color} **{grade}**")
        if grade_results.get("certainty_statement"):
            st.info(grade_results["certainty_statement"])
        if grade_results.get("summary"):
            st.markdown(grade_results["summary"])
        domains = grade_results.get("domains", {})
        if domains:
            cols = st.columns(len(domains))
            for col, (dom, rating) in zip(cols, domains.items()):
                col.metric(dom.replace("_", " ").title(), rating)
        st.divider()

    if rob_table:
        st.markdown("**Risk of Bias (per paper):**")
        rob_icon = {"Low": "🟢", "Some concerns": "🟡", "High": "🔴"}
        for r in rob_table:
            overall = r.get("overall", "Some concerns")
            icon = rob_icon.get(overall, "⚪")
            with st.expander(f"{icon} {r.get('citation_key','')} — {r.get('title','')[:60]} ({r.get('tool','')}: {overall})"):
                for key, val in r.items():
                    if key in ("citation_key", "title", "tool", "overall", "justification"):
                        continue
                    st.markdown(f"- **{key.replace('_', ' ').title()}:** {val}")
                if r.get("justification"):
                    st.caption(r["justification"])
        st.divider()

    if contradictions:
        st.markdown(f"**Conflicting Findings ({len(contradictions)}):**")
        for c in contradictions:
            score = c.get("consensus_score", "?")
            with st.expander(f"⚠️ {c.get('claim','')[:80]} — consensus {score}/100"):
                pa = c.get("position_a", {})
                pb = c.get("position_b", {})
                st.markdown(f"- **Position A:** {pa.get('description','')} _(papers: {', '.join(pa.get('papers', []) or [])})_")
                st.markdown(f"- **Position B:** {pb.get('description','')} _(papers: {', '.join(pb.get('papers', []) or [])})_")
                if c.get("explanation"):
                    st.markdown(c["explanation"])
    elif rob_table or grade_results:
        st.success("No material contradictions detected across the included papers.")


def _render_citation_context_section(final_state: dict, settings: dict) -> None:
    """Render the Citation Context tool: pick a citing paper and a cited paper from
    the included set and surface the sentence(s) in the citing paper's open-access
    full text that reference the cited work (scite.ai-style snippets).

    Best-effort and open-access only — delegates to
    `tools.citation_context.extract_citation_context`, which fails safe to an
    "unavailable"/"not found" status when no open-access full text exists. Caches
    the last lookup in `session_state["_cc_result"]`."""
    st.subheader("Citation Context", help=term_help("Citation network"))
    st.markdown(
        "Find the exact sentence(s) where one included paper cites another, pulled from the "
        "**citing paper's open-access full text**. Best-effort: only works when an open-access "
        "PDF/HTML is available for the citing paper."
    )
    included = final_state.get("included_papers", [])
    if len(included) < 2:
        st.info("Need at least two included papers to look up a citation context.")
        return

    labels = [f"{p.get('citation_key') or p.get('title','')[:40]} ({p.get('year','n.d.')})" for p in included]
    col_a, col_b = st.columns(2)
    with col_a:
        citing_idx = st.selectbox("Citing paper (A)", range(len(included)),
                                  format_func=lambda i: labels[i], key="cc_citing")
    with col_b:
        cited_idx = st.selectbox("Cited paper (B)", range(len(included)),
                                 format_func=lambda i: labels[i], key="cc_cited")

    if citing_idx == cited_idx:
        st.caption("Pick two different papers.")
        return

    if st.button("Find Citation Context", key="run_citation_context"):
        with st.spinner("Fetching the citing paper's full text and searching for the citation…"):
            try:
                from tools.citation_context import extract_citation_context
                st.session_state["_cc_result"] = extract_citation_context(
                    included[citing_idx], included[cited_idx]
                )
            except Exception as e:
                st.error(f"Citation context lookup failed: {e}")

    result = st.session_state.get("_cc_result")
    if not result:
        return
    status = result.get("status")
    if status == "ok":
        st.success(f"Found {len(result['contexts'])} citing sentence(s).")
        for ctx in result["contexts"]:
            st.markdown(f"> {ctx['sentence']}")
            st.caption(f"matched on {ctx.get('matched_on','')}")
        if result.get("source_url"):
            st.caption(f"Source: [{result['source_url']}]({result['source_url']})")
    else:
        st.info(result.get("reason", "No citation context available."))


_EXPLORE_TOOLS = [
    ("citation_network", "Citation Network", _render_citation_network_section),
    ("citation_context", "Citation Context", _render_citation_context_section),
    ("reference_checking", "Risk & Certainty", _render_reference_checking_section),
    ("preprint_status", "Preprint Status", _render_preprint_status_section),
    ("research_trends", "Research Trends", _render_research_trends_section),
    ("evidence_map", "Evidence Map", _render_evidence_map_section),
    ("meta_analysis", "Meta-Analysis", _render_meta_analysis),
    ("concept_drift", "Concept Drift", _render_concept_drift_section),
]


def _tab_explore(final_state: dict, settings: dict) -> None:
    """Pick one deep-dive analysis tool to run on this review's corpus.

    Renders a horizontal radio of the six `_EXPLORE_TOOLS` entries and dispatches to
    the matching render function. Reads/writes the chosen tool via the `sr_explore_tool`
    widget key; each individual tool manages its own `_cn_*`/`_pt_*`/`_trend_*`/
    `_drift_*`/`_meta_*` session_state as documented on its own render function."""
    st.markdown(
        "Optional deep-dive tools that run on top of this review's corpus — pick one below. "
        "Each shows what it needs and roughly how long it takes before you run it."
    )
    labels = {key: label for key, label, _ in _EXPLORE_TOOLS}
    choice = st.radio(
        "Tool",
        options=list(labels.keys()),
        format_func=lambda k: labels[k],
        key="sr_explore_tool",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()
    for key, _, render_fn in _EXPLORE_TOOLS:
        if key == choice:
            render_fn(final_state, settings)
            break


def _tab_export(final_state: dict, rq: str, session_id: str, settings: dict) -> None:
    """Render the Write-up & Export tab: search queries used, Markdown export, the
    PRISMA 2020 DOCX/PDF manuscript report (`tools.prisma_report`), and plain-language
    summaries for patient/policy/press audiences (`tools.plain_language`).

    Session_state keys:
      - `_docx_bytes` / `_pdf_bytes` — generated report bytes, cached so the
        `st.download_button` below the Generate button has data to serve across the
        rerun that `st.button` triggers (the bytes aren't regenerated on every render).
      - `_pls_result` — dict of generated plain-language summaries (`patient`/`policy`/
        `press` keys), persisted the same way for its own download buttons.
    Widget keys `report_author` / `report_institution` / `pls_format` hold the DOCX/PDF
    author+institution fields and the plain-language summary format choice."""
    model = settings.get("model", "llama3.1:8b")
    num_ctx = settings.get("num_ctx", 32768)

    # Search queries
    st.subheader("Search Queries Used")
    for i, q in enumerate(final_state.get("search_queries", []), 1):
        st.markdown(f"{i}. {q}")

    st.divider()
    st.subheader("Export Systematic Review")

    # Markdown
    md_content = _build_sr_markdown(rq, final_state)
    st.download_button(
        label="Download as Markdown",
        data=md_content.encode("utf-8"),
        file_name=f"systematic_review_{session_id}.md",
        mime="text/markdown",
        use_container_width=True,
        key="dl_md",
    )

    st.divider()
    st.subheader("PRISMA 2020 Manuscript Report")
    author = st.text_input("Author name (optional)", key="report_author")
    institution = st.text_input("Institution (optional)", key="report_institution")

    col_docx, col_pdf = st.columns(2)

    with col_docx:
        if st.button("Generate DOCX Report", key="gen_docx", use_container_width=True):
            with st.spinner("Building PRISMA 2020 DOCX…"):
                try:
                    from tools.prisma_report import generate_prisma_docx
                    docx_bytes = generate_prisma_docx(final_state, author=author, institution=institution)
                    st.session_state["_docx_bytes"] = docx_bytes
                    st.success("DOCX ready.")
                except Exception as e:
                    st.error(f"DOCX generation failed: {e}")
        if st.session_state.get("_docx_bytes"):
            st.download_button(
                label="Download DOCX",
                data=st.session_state["_docx_bytes"],
                file_name=f"prisma_report_{session_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="dl_docx",
            )

    with col_pdf:
        if st.button("Generate PDF Report", key="gen_pdf", use_container_width=True):
            with st.spinner("Building PRISMA 2020 PDF…"):
                try:
                    from tools.prisma_report import generate_prisma_pdf
                    pdf_bytes = generate_prisma_pdf(final_state, author=author, institution=institution)
                    st.session_state["_pdf_bytes"] = pdf_bytes
                    st.success("PDF ready.")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
        if st.session_state.get("_pdf_bytes"):
            st.download_button(
                label="Download PDF",
                data=st.session_state["_pdf_bytes"],
                file_name=f"prisma_report_{session_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_pdf",
            )

    st.divider()
    st.subheader("Plain-Language Summaries")
    st.markdown("Generate lay-audience summaries for different audiences.")

    fmt = st.radio(
        "Format",
        ["Patient / Public", "Policy Brief", "Press Release", "All Three"],
        horizontal=True,
        key="pls_format",
    )

    if st.button("Generate Summary", key="gen_pls", use_container_width=True):
        with st.spinner("Generating plain-language summary…"):
            try:
                from tools.plain_language import (
                    generate_patient_summary,
                    generate_policy_brief,
                    generate_press_release,
                    generate_all_summaries,
                )
                if fmt == "Patient / Public":
                    result = {"patient": generate_patient_summary(final_state, model, num_ctx)}
                elif fmt == "Policy Brief":
                    result = {"policy": generate_policy_brief(final_state, model, num_ctx)}
                elif fmt == "Press Release":
                    result = {"press": generate_press_release(final_state, model, num_ctx)}
                else:
                    result = generate_all_summaries(final_state, model, num_ctx)
                st.session_state["_pls_result"] = result
            except Exception as e:
                st.error(f"Summary generation failed: {e}")

    pls = st.session_state.get("_pls_result", {})
    if pls.get("patient"):
        with st.expander("Patient / Public Summary", expanded=True):
            st.markdown(pls["patient"])
            st.download_button(
                "Download (txt)", pls["patient"].encode(),
                file_name=f"patient_summary_{session_id}.txt",
                key="dl_patient",
            )
    if pls.get("policy"):
        with st.expander("Policy Brief", expanded=True):
            st.markdown(pls["policy"])
            st.download_button(
                "Download (txt)", pls["policy"].encode(),
                file_name=f"policy_brief_{session_id}.txt",
                key="dl_policy",
            )
    if pls.get("press"):
        with st.expander("Press Release", expanded=True):
            st.markdown(pls["press"])
            st.download_button(
                "Download (txt)", pls["press"].encode(),
                file_name=f"press_release_{session_id}.txt",
                key="dl_press",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Guided templates — presets for common review types
# ─────────────────────────────────────────────────────────────────────────────
#
# Power-user note: choosing a template only pre-fills the research-question and
# inclusion/exclusion text areas below — every field stays freely editable, the
# raw CLI flags (--inclusion/--exclusion/etc.) are untouched, and "scratch" stays
# the default so nothing changes for users who never open this expander.

SR_TEMPLATES: list = [
    {
        "key": "clinical_rct",
        "label": "Clinical RCT review",
        "description": "Randomised controlled trials evaluating a clinical intervention in human participants.",
        "research_question": "What is the effect of [intervention] on [outcome] in [population]?",
        "inclusion": [
            "Randomised controlled trials (RCTs)",
            "Human participants",
            "Peer-reviewed, published 2010–present",
            "Reports the outcome of interest with quantitative results",
        ],
        "exclusion": [
            "Animal or in-vitro studies",
            "Case reports, editorials, conference abstracts only",
            "No control/comparison group",
            "Non-English publications",
        ],
        "note": "Pairs well with **Statistical Meta-Analysis** (pool effect sizes across trials) "
                "and **Preprint Status** (flags retracted or unpublished trials).",
    },
    {
        "key": "cs_survey",
        "label": "CS literature survey",
        "description": "Computer-science / engineering survey of methods, systems or benchmarks.",
        "research_question": "What approaches have been proposed for [task/problem], and how do they compare on [metric]?",
        "inclusion": [
            "Peer-reviewed papers or well-cited preprints (arXiv)",
            "Proposes, benchmarks, or surveys a method for the stated task",
            "Published within the last 10 years",
            "Reports quantitative results or a clear architectural contribution",
        ],
        "exclusion": [
            "Position papers / opinion pieces with no technical contribution",
            "Duplicate or superseded preprint versions",
            "Workshop posters with no accompanying results",
        ],
        "note": "Pairs well with **Citation Network**, **Concept Drift Tracker** and "
                "**Research Trend Forecaster** — CS moves fast, so track what's rising.",
    },
    {
        "key": "qual_synthesis",
        "label": "Qualitative evidence synthesis",
        "description": "Thematic synthesis of qualitative studies (interviews, ethnography, case studies).",
        "research_question": "How do [population] experience or perceive [phenomenon]?",
        "inclusion": [
            "Qualitative or mixed-methods studies",
            "Primary research with original data collection",
            "Clearly describes participants and methodology",
            "Published in peer-reviewed venues",
        ],
        "exclusion": [
            "Purely quantitative studies with no qualitative component",
            "Secondary analyses or reviews of other qualitative work",
            "Grey literature without peer review",
        ],
        "note": "Pairs well with **Evidence Map** and **Narrative Synthesis** — themes matter "
                "more than pooled numbers here, so Statistical Meta-Analysis isn't recommended.",
    },
    {
        "key": "scoping_review",
        "label": "Scoping / mapping review",
        "description": "Broad map of what evidence exists on a topic, before committing to a focused review.",
        "research_question": "What is the nature and extent of research on [topic] in [context]?",
        "inclusion": [
            "Any study design that addresses the topic",
            "Published in any language with an available English abstract",
            "No date restriction (or specify a broad range)",
        ],
        "exclusion": [
            "Studies entirely off-topic despite keyword matches",
            "Duplicates across databases",
        ],
        "note": "Pairs well with **Evidence Map**, **Research Trend Forecaster**, and "
                "**Cross-Notebook Search** to connect findings to material you've already collected.",
    },
]

_SR_TEMPLATE_BY_KEY = {t["key"]: t for t in SR_TEMPLATES}


def _consume_template_application() -> None:
    """Apply a queued template choice to the question/criteria widgets before they render.

    Must run before the `sr_question` / `sr_inclusion` / `sr_exclusion` text areas
    are instantiated (Streamlit forbids writing a widget's session-state key after
    the widget exists) — mirrors the `hw_apply_*` pattern used in ui/sidebar.py.
    """
    pending = st.session_state.pop("sr_apply_template", None)
    if not pending:
        return
    tmpl = _SR_TEMPLATE_BY_KEY.get(pending)
    if not tmpl:
        return
    st.session_state["sr_question"] = tmpl["research_question"]
    st.session_state["sr_inclusion"] = "\n".join(tmpl["inclusion"])
    st.session_state["sr_exclusion"] = "\n".join(tmpl["exclusion"])


def _render_template_picker() -> None:
    """Optional preset picker that pre-fills the question + criteria for common review types.

    Calls `_consume_template_application()` first to flush any pending choice from the
    previous run. Reads/writes the `sr_template_choice` selectbox key; on "Use this
    template" writes the chosen key to `session_state["sr_apply_template"]` and reruns
    so `_consume_template_application` can apply it before the input widgets render."""
    _consume_template_application()
    with st.expander("New to systematic reviews? Start from a guided template (optional)", expanded=False):
        st.caption(
            "Pick a starting point for a common review type — it pre-fills the research "
            "question and inclusion/exclusion criteria below, and you can edit anything "
            "afterwards. Prefer to write your own? Just skip this and start typing."
        )
        labels = {"_none": "— Start from scratch —"}
        labels.update({t["key"]: t["label"] for t in SR_TEMPLATES})
        choice = st.selectbox(
            "Template", options=list(labels.keys()), format_func=lambda k: labels[k],
            key="sr_template_choice",
        )
        if choice != "_none":
            tmpl = _SR_TEMPLATE_BY_KEY[choice]
            st.markdown(f"_{tmpl['description']}_")
            st.markdown(f"**Suggested research question:** {tmpl['research_question']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Inclusion criteria**")
                for item in tmpl["inclusion"]:
                    st.markdown(f"- {item}")
            with c2:
                st.markdown("**Exclusion criteria**")
                for item in tmpl["exclusion"]:
                    st.markdown(f"- {item}")
            st.caption(tmpl["note"])
            if st.button("Use this template", key="sr_apply_template_btn", type="primary"):
                st.session_state["sr_apply_template"] = choice
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def tab_systematic_review(settings: dict) -> None:
    """Mode 7 — PRISMA Systematic Review. Top-level entry point registered for this tab.

    Renders, in order: the glossary expander, the guided-template picker
    (`_render_template_picker`), the research-question + inclusion/exclusion inputs
    (each gated through `render_query_gate` for an optional grammar check), and the
    Run button. On click, builds a `SystematicReviewState` via
    `create_systematic_review_state` and drives it through
    `agents.systematic_review_graph.run_systematic_review`, streaming per-node progress
    into a progress bar / status line / step-log expander.

    Once a run completes (fresh or restored from cache), renders the shared results
    view: eval/RAG-reflection panels, the PRISMA flow metrics, and four result tabs
    (Synthesis, Evidence, Explore, Write-up & Export) via `_tab_synthesis`/
    `_tab_evidence`/`_tab_explore`/`_tab_export`.

    Session_state keys:
      - `sr_last_result` — the final SR state dict, persisted after every successful
        run so it survives reruns triggered by widgets inside the result tabs (e.g.
        the Explore tools' own buttons) without re-running the whole pipeline.
      - `sr_question` / `sr_inclusion` / `sr_exclusion` — the three input widgets'
        backing state; also the targets `_consume_template_application` writes into.
      - `_cn_html`, `_cn_stats`, `_cn_gaps`, `_pt_tracking`, `_pt_summary`,
        `_trend_data`, `_trend_json`, `_drift_data` — cached deep-dive results from the
        Explore tab's tools; explicitly cleared here at the start of a fresh run so a
        new corpus never shows a previous run's citation network / trends / drift.
    """
    st.header("Mode 7 — Systematic Review")
    st.markdown(
        "Conduct a **PRISMA-style systematic review** powered by local LLM inference (Ollama). "
        "Describe your research question and criteria below — BeeSearch searches Google "
        "Scholar, arXiv, Semantic Scholar and CrossRef, screens papers, extracts evidence, and "
        "synthesises the findings into a full review you can explore, analyse further, and export."
    )
    render_glossary_expander([
        "Systematic review", "PRISMA", "Inclusion / exclusion criteria", "Quality score",
        "Effect size", "Heterogeneity (I²)", "Forest plot", "Pooled effect / meta-analysis",
        "Fixed-effect vs. random-effects model", "Citation network", "Concept drift", "Evidence map",
    ])
    st.divider()

    # ── Guided templates ──────────────────────────────────────────────────────
    _render_template_picker()

    # ── Inputs ────────────────────────────────────────────────────────────────
    rq = st.text_area(
        "Research question",
        height=90,
        placeholder="e.g. What is the effect of sleep deprivation on working memory performance "
                    "in university students?",
        help=term_help("Systematic review"),
        key="sr_question",
    )
    rq_final, rq_ready = render_query_gate(rq, key="sr_question", settings=settings,
                                            context_hint="systematic literature review research question")
    st.caption(
        "These guide the screening step — be specific (study design, population, "
        "publication window, language, …) for sharper include/exclude decisions."
    )
    col_inc, col_exc = st.columns(2)
    with col_inc:
        st.markdown("**Inclusion criteria** *(one per line)*", help=term_help("Inclusion / exclusion criteria"))
        inc_raw = st.text_area(
            "Inclusion criteria",
            height=120,
            placeholder="Peer-reviewed empirical studies\nHuman participants\nPublished 2010–2024\n"
                        "English language",
            key="sr_inclusion",
            label_visibility="collapsed",
        )
        inc_final, inc_ready = render_query_gate(inc_raw, key="sr_inclusion", settings=settings,
                                                  context_hint="inclusion criteria for a systematic review")
    with col_exc:
        st.markdown("**Exclusion criteria** *(one per line)*", help=term_help("Inclusion / exclusion criteria"))
        exc_raw = st.text_area(
            "Exclusion criteria",
            height=120,
            placeholder="Animal studies\nCase reports\nConference abstracts only\n"
                        "Non-English publications",
            key="sr_exclusion",
            label_visibility="collapsed",
        )
        exc_final, exc_ready = render_query_gate(exc_raw, key="sr_exclusion", settings=settings,
                                                  context_hint="exclusion criteria for a systematic review")

    run_btn = st.button("Run Systematic Review", key="run_sr", type="primary", use_container_width=True)

    if run_btn and not rq.strip():
        st.warning("Please enter a research question.")
        return

    if run_btn and not (rq_ready and inc_ready and exc_ready):
        st.info("Please resolve the grammar suggestion(s) above, then click **Run Systematic Review** again.")
        return

    final_state = None

    if run_btn:
        # ── Run ───────────────────────────────────────────────────────────────
        inclusion = [l.strip() for l in inc_final.splitlines() if l.strip()]
        exclusion = [l.strip() for l in exc_final.splitlines() if l.strip()]

        initial_state = create_systematic_review_state(
            research_question=rq_final.strip(),
            inclusion_criteria=inclusion,
            exclusion_criteria=exclusion,
            model_name=settings["model"],
            num_ctx=settings["num_ctx"],
            max_results=settings.get("max_results", 8),
            include_crossref=settings.get("include_crossref", True),
        )

        # New corpus incoming — drop any cached deep-dive results from a
        # previous run so stale citation/trend/drift artefacts don't bleed through.
        for k in ("_cn_html", "_cn_stats", "_cn_gaps", "_cn_stances", "_cc_result",
                  "_pt_tracking", "_pt_summary", "_trend_data", "_trend_json", "_drift_data",
                  "_rc_rob", "_rc_grade", "_rc_contra"):
            st.session_state.pop(k, None)

        st.divider()
        st.subheader("Running Systematic Review…")
        status_text = st.empty()
        progress_bar = st.progress(0)
        step_log = st.expander("Step log", expanded=False)
        log_lines: list = []

        node_labels = {
            "query_generation":    "Generating search queries",
            "literature_search":   "Searching Google Scholar · arXiv · Semantic Scholar · CrossRef",
            "screening":           "Screening papers by title/abstract",
            "evidence_extraction": "Extracting evidence from papers",
            "quality_assessment":  "Assessing risk of bias, GRADE certainty, contradictions",
            "synthesis":           "Synthesising findings",
            "sr_eval":             "Evaluating review quality",
        }

        def stream_callback(node_name: str, state: dict) -> None:
            """Per-node progress callback passed to `run_systematic_review`; updates the
            progress bar, status line, and appends to the step-log expander as each
            graph node completes. Closes over `progress_bar`/`status_text`/`step_log`/
            `log_lines` from the enclosing `tab_systematic_review` call."""
            pct = state.get("progress_pct", 0)
            label = node_labels.get(node_name, node_name)
            detail = state.get("status_detail", "")
            progress_bar.progress(pct)
            status_text.markdown(f"**{label}…** `{pct}%`" + (f"  \n{detail}" if detail else ""))
            log_lines.append(f"{label} ({pct}%)" + (f" — {detail}" if detail else ""))
            with step_log:
                st.text("\n".join(log_lines))

        start = time.time()
        try:
            final_state = run_systematic_review(initial_state, stream_callback=stream_callback)
        except Exception as e:
            st.error(f"Workflow error: {e}")
            logger.exception("Systematic review failed")
            return

        elapsed = time.time() - start
        progress_bar.progress(100)
        status_text.markdown(f"**Done.** Finished in `{elapsed:.1f}s`")

        # Persist result so it survives reruns triggered by other widgets below
        st.session_state["sr_last_result"] = final_state
    else:
        final_state = st.session_state.get("sr_last_result")
        if not final_state:
            return

    # ── Results — single shared rendering path for fresh AND cached runs ──────
    for err in final_state.get("errors", []):
        st.warning(err)

    st.divider()
    render_eval_result(final_state.get("eval_result", {}), key_suffix="_sr")
    render_rag_reflection(final_state.get("rag_reflection_info"), key_suffix="_sr")
    st.divider()

    research_question = final_state.get("research_question", "")
    st.caption(f"**Reviewing:** {research_question}")
    st.subheader("PRISMA Flow", help=term_help("PRISMA"))
    _render_prisma_flow(final_state.get("prisma_flow", {}))
    n_included = len(final_state.get("included_papers", []))
    n_excluded = len(final_state.get("excluded_papers", []))
    st.caption(
        f"Queries: {len(final_state.get('search_queries', []))} · "
        f"Identified: {len(final_state.get('raw_papers', []))} · "
        f"Included: {n_included} · Excluded: {n_excluded}"
    )

    session_id = final_state.get("session_id", "sr")

    t_synthesis, t_evidence, t_explore, t_export = st.tabs([
        "Synthesis",
        "Evidence",
        "Explore",
        "Write-up & Export",
    ])

    with t_synthesis:
        _tab_synthesis(final_state, settings)

    with t_evidence:
        _tab_evidence(final_state, settings)

    with t_explore:
        _tab_explore(final_state, settings)

    with t_export:
        _tab_export(final_state, research_question, session_id, settings)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown export helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_sr_markdown(research_question: str, state: dict) -> str:
    """Build the full SR results (PRISMA flow table, key themes, narrative synthesis,
    research gaps, conclusion, limitations, evidence table) as a single Markdown
    string for the "Download as Markdown" button in `_tab_export`. Pure data → string
    transform — no Streamlit calls, no session_state."""
    from datetime import datetime
    lines = [
        "# Systematic Review Report",
        f"**Research Question:** {research_question}",
        f"*Generated: {datetime.today().strftime('%B %d, %Y')}*",
        "",
        "## PRISMA Flow",
        "",
    ]
    flow = state.get("prisma_flow", {})
    lines += [
        "| Stage | Count |",
        "| --- | --- |",
        f"| Identified | {flow.get('identified', 0)} |",
        f"| Screened | {flow.get('screened', 0)} |",
        f"| Eligibility | {flow.get('eligibility', 0)} |",
        f"| Included | {flow.get('included', 0)} |",
        f"| Excluded | {flow.get('excluded', 0)} |",
        "",
        "## Key Themes",
        "",
    ]
    for t in state.get("key_themes", []):
        lines.append(f"- {t}")

    lines += ["", "## Narrative Synthesis", "", state.get("narrative_synthesis", ""), ""]

    gaps = state.get("research_gaps", [])
    if gaps:
        lines += ["## Research Gaps", ""]
        for g in gaps:
            lines.append(f"- {g}")
        lines.append("")

    grade = state.get("grade_results", {})
    if grade and grade.get("overall_grade"):
        lines += [
            "## Certainty of Evidence (GRADE)", "",
            f"**Overall certainty:** {grade.get('overall_grade', 'n/a')}", "",
            grade.get("certainty_statement", ""), "",
            grade.get("summary", ""), "",
        ]

    contradictions = state.get("contradictions", [])
    if contradictions:
        lines += ["## Conflicting Findings", ""]
        for c in contradictions:
            lines.append(
                f"- **{c.get('claim', '')}** (consensus {c.get('consensus_score', '?')}/100): "
                f"{c.get('explanation', '')}"
            )
        lines.append("")

    rob_table = state.get("rob_table", [])
    if rob_table:
        lines += ["## Risk of Bias", "",
                  "| Citation | Tool | Overall |", "| --- | --- | --- |"]
        for r in rob_table:
            lines.append(f"| {r.get('citation_key', '')} | {r.get('tool', '')} | {r.get('overall', '')} |")
        lines.append("")

    if state.get("conclusion"):
        lines += ["## Conclusion", "", state["conclusion"], ""]

    if state.get("limitations"):
        lines += ["## Limitations", "", state["limitations"], ""]

    evidence = state.get("evidence_table", [])
    if evidence:
        lines += ["## Evidence Table", ""]
        lines += ["| Citation | Year | Design | Quality | Key Finding |",
                  "| --- | --- | --- | --- | --- |"]
        for row in evidence:
            ck = row.get("citation_key", "")
            yr = row.get("year", "n.d.")
            design = row.get("study_design", "")
            qual = row.get("quality", "")
            finding = row.get("key_finding", "")[:80]
            lines.append(f"| {ck} | {yr} | {design} | {qual} | {finding} |")
        lines.append("")

    return "\n".join(lines)
