"""ui/tabs/systematic_review.py — Mode 7: PRISMA Systematic Review"""

from __future__ import annotations

import logging
import time

import streamlit as st

from agents.systematic_review_graph import run_systematic_review
from agents.systematic_review_state import create_systematic_review_state
from ui.helpers import render_eval_result, render_rag_reflection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Small render helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_prisma_flow(flow: dict) -> None:
    if not flow:
        return
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [
        ("Identified", "identified"), ("Screened", "screened"),
        ("Eligibility", "eligibility"), ("Included", "included"), ("Excluded", "excluded"),
    ]):
        col.metric(label, flow.get(key, 0))


def _render_evidence_table(evidence_table: list) -> None:
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
    themes = final_state.get("key_themes", [])
    if themes:
        st.markdown("**Key Themes:**")
        for t in themes:
            st.markdown(f"- {t}")
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


def _tab_evidence(final_state: dict) -> None:
    n_inc = len(final_state.get("included_papers", []))
    n_exc = len(final_state.get("excluded_papers", []))
    st.subheader(f"Evidence Table ({n_inc} included papers)")
    _render_evidence_table(final_state.get("evidence_table", []))

    if final_state.get("excluded_papers"):
        with st.expander(f"Excluded Papers ({n_exc})"):
            for p in final_state["excluded_papers"]:
                reason = p.get("exclusion_reason", "")
                st.markdown(f"- **{p.get('title','')[:70]}** ({p.get('year','n.d.')}) — _{reason}_")


def _tab_discovery(final_state: dict, settings: dict) -> None:
    """Abstract Screener · Citation Network · Preprint Status."""
    rq = final_state.get("research_question", "")
    model = settings.get("model", "llama3.1:8b")
    num_ctx = settings.get("num_ctx", 32768)

    # ── Abstract Screener ─────────────────────────────────────────────────
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
            color = {"include": "🟢", "uncertain": "🟡", "exclude": "🔴"}.get(verdict, "⚪")
            with st.expander(f"{color} [{score}/100] {paper.get('title','')[:70]}"):
                st.markdown(f"**Verdict:** {verdict.upper()}  |  **Score:** {score}/100")
                st.markdown(f"**Rationale:** {rationale}")
                if paper.get("abstract"):
                    st.markdown(f"**Abstract:** {paper['abstract'][:300]}…")
    else:
        st.info("Abstract screener scores will appear here after running the systematic review.")

    st.divider()

    # ── Citation Network ──────────────────────────────────────────────────
    st.subheader("Citation Network")
    st.markdown(
        "Ego network showing citation links **between** the included papers. "
        "Green = High quality, Amber = Medium, Red = Low. "
        "Requires Semantic Scholar API calls (~30s for 20 papers)."
    )
    included = final_state.get("included_papers", [])
    if not included:
        st.info("No included papers to build a network from.")
    else:
        existing_html = final_state.get("citation_graph_html", "")
        if existing_html:
            st.components.v1.html(existing_html, height=520, scrolling=False)
        else:
            if st.button("Build Citation Network", key="build_network"):
                with st.spinner("Querying Semantic Scholar for citation links…"):
                    try:
                        from tools.citation_network import (
                            build_citation_network,
                            network_to_pyvis_html,
                            network_stats,
                        )
                        G, meta = build_citation_network(included, max_papers=25)
                        html = network_to_pyvis_html(G, meta)
                        stats = network_stats(G)
                        st.session_state["_cn_html"] = html
                        st.session_state["_cn_stats"] = stats
                        st.success(
                            f"Network built: {stats['nodes']} nodes, "
                            f"{stats['edges']} citation edges, "
                            f"{stats['isolated']} isolated papers."
                        )
                    except Exception as e:
                        st.error(f"Citation network failed: {e}")

            if st.session_state.get("_cn_html"):
                st.components.v1.html(st.session_state["_cn_html"], height=520, scrolling=False)
                stats = st.session_state.get("_cn_stats", {})
                if stats.get("most_cited"):
                    st.markdown("**Most cited within corpus:**")
                    for node, deg in stats["most_cited"]:
                        st.markdown(f"- {node} — cited by {deg} included paper(s)")

    st.divider()

    # ── Preprint Status ────────────────────────────────────────────────────
    st.subheader("Preprint Status")
    st.markdown(
        "Checks each included paper against CrossRef to identify unverified preprints "
        "and flag any retractions. Requires CrossRef API calls (~0.25s per paper)."
    )
    if not included:
        st.info("No included papers.")
    else:
        existing_tracking = final_state.get("preprint_tracking", [])
        if existing_tracking:
            _render_preprint_tracking(existing_tracking)
        else:
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
    from tools.preprint_tracker import preprint_summary
    summary = preprint_summary(tracking)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Journal", summary.get("journal", 0))
    c2.metric("Published (was preprint)", summary.get("published", 0))
    c3.metric("Preprint only", summary.get("preprint", 0))
    c4.metric("Retracted ⚠️", summary.get("retracted", 0))
    st.divider()
    for r in tracking:
        paper = r.get("paper", {})
        status = r.get("preprint_status", "unknown")
        icon = {"journal": "📰", "published": "✅", "preprint": "⚠️", "retracted": "🚫"}.get(status, "❓")
        with st.expander(f"{icon} {paper.get('title','')[:70]} — {status.upper()}"):
            st.markdown(f"**Status:** {status}")
            st.markdown(f"**Note:** {r.get('note','')}")
            if r.get("published_doi"):
                st.markdown(f"**Published DOI:** [{r['published_doi']}](https://doi.org/{r['published_doi']})")
            if r.get("published_venue"):
                st.markdown(f"**Journal:** {r['published_venue']}")


def _tab_trends(final_state: dict, settings: dict) -> None:
    """Research Trends · Evidence Map · Concept Drift."""
    rq = final_state.get("research_question", "")
    model = settings.get("model", "llama3.1:8b")
    num_ctx = settings.get("num_ctx", 32768)
    included = final_state.get("included_papers", [])

    # ── Research Trend Forecaster ─────────────────────────────────────────
    st.subheader("Research Trend Forecaster")
    st.markdown(
        "Publication volume by year for this research area, sourced from CrossRef (field-wide) "
        "and compared to the papers retrieved in this SR run."
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
                    marker_color="#1a6496",
                    opacity=0.7,
                ))
                fig.add_trace(go.Scatter(
                    x=years,
                    y=chart.get("corpus_counts", []),
                    name="This SR corpus",
                    mode="lines+markers",
                    marker=dict(color="#f39c12", size=6),
                    line=dict(color="#f39c12", width=2),
                ))
                fig.update_layout(
                    title=f"Publication Trend: {rq[:60]}…" if len(rq) > 60 else f"Publication Trend: {rq}",
                    xaxis_title="Year",
                    yaxis_title="Publications",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white"),
                    legend=dict(bgcolor="#1a1a2e"),
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("Install plotly (`pip install plotly`) to see the trend chart.")
                st.json(chart)

    st.divider()

    # ── Evidence Map ─────────────────────────────────────────────────────
    st.subheader("Evidence Map")
    st.markdown(
        "Bubble chart of evidence density across Population × Intervention dimensions. "
        "Bubble size = number of studies; colour = average quality."
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

    st.divider()

    # ── Concept Drift ─────────────────────────────────────────────────────
    st.subheader("Concept Drift Tracker")
    st.markdown(
        "Detects vocabulary shifts across time periods in the included papers — "
        "which terms are rising, which are declining."
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


def _tab_export(final_state: dict, rq: str, session_id: str, settings: dict) -> None:
    """Search Queries · Markdown · DOCX · PDF · Plain-Language Summaries."""
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
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def tab_systematic_review(settings: dict) -> None:
    """Mode 7 — PRISMA Systematic Review."""
    st.header("Mode 7 — Systematic Review")
    st.markdown(
        """
Conduct a **PRISMA-style systematic review** powered by local LLM inference (Ollama).
Searches Google Scholar · arXiv · Semantic Scholar · CrossRef, screens papers, extracts
evidence, synthesises findings, and provides advanced analysis tools.

**What you get:** PRISMA flow · Evidence table · Narrative synthesis · Abstract screener
scores · Citation network · Preprint status · Research trends · Evidence map ·
Concept drift · DOCX/PDF manuscript · Plain-language summaries
"""
    )
    st.divider()

    # ── Inputs ────────────────────────────────────────────────────────────────
    rq = st.text_area(
        "Research question",
        height=90,
        placeholder="e.g. What is the effect of sleep deprivation on working memory performance "
                    "in university students?",
        key="sr_question",
    )
    col_inc, col_exc = st.columns(2)
    with col_inc:
        st.markdown("**Inclusion criteria** *(one per line)*")
        inc_raw = st.text_area(
            "Inclusion criteria",
            height=120,
            placeholder="Peer-reviewed empirical studies\nHuman participants\nPublished 2010–2024\n"
                        "English language",
            key="sr_inclusion",
            label_visibility="collapsed",
        )
    with col_exc:
        st.markdown("**Exclusion criteria** *(one per line)*")
        exc_raw = st.text_area(
            "Exclusion criteria",
            height=120,
            placeholder="Animal studies\nCase reports\nConference abstracts only\n"
                        "Non-English publications",
            key="sr_exclusion",
            label_visibility="collapsed",
        )

    run_btn = st.button("Run Systematic Review", key="run_sr", type="primary", use_container_width=True)

    if run_btn and not rq.strip():
        st.warning("Please enter a research question.")
        return
    if not run_btn:
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    inclusion = [l.strip() for l in inc_raw.splitlines() if l.strip()]
    exclusion = [l.strip() for l in exc_raw.splitlines() if l.strip()]

    initial_state = create_systematic_review_state(
        research_question=rq.strip(),
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion,
        model_name=settings["model"],
        num_ctx=settings["num_ctx"],
    )

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
        "synthesis":           "Synthesising findings",
        "sr_eval":             "Evaluating review quality",
    }

    def stream_callback(node_name: str, state: dict) -> None:
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

    # Persist result for session reuse
    st.session_state["sr_last_result"] = final_state

    # ── Results ───────────────────────────────────────────────────────────────
    for err in final_state.get("errors", []):
        st.warning(err)

    st.divider()
    render_eval_result(final_state.get("eval_result", {}), key_suffix="_sr")
    render_rag_reflection(final_state.get("rag_reflection_info"), key_suffix="_sr")
    st.divider()

    st.subheader("PRISMA Flow")
    _render_prisma_flow(final_state.get("prisma_flow", {}))
    n_included = len(final_state.get("included_papers", []))
    n_excluded = len(final_state.get("excluded_papers", []))
    st.caption(
        f"Queries: {len(final_state.get('search_queries', []))} · "
        f"Identified: {len(final_state.get('raw_papers', []))} · "
        f"Included: {n_included} · Excluded: {n_excluded}"
    )

    session_id = initial_state.get("session_id", "sr")

    t_synthesis, t_evidence, t_discovery, t_trends, t_export = st.tabs([
        "Synthesis",
        "Evidence Table",
        "Discovery",
        "Trends & Analysis",
        "Export & Reports",
    ])

    with t_synthesis:
        _tab_synthesis(final_state, settings)

    with t_evidence:
        _tab_evidence(final_state)

    with t_discovery:
        _tab_discovery(final_state, settings)

    with t_trends:
        _tab_trends(final_state, settings)

    with t_export:
        _tab_export(final_state, rq, session_id, settings)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown export helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_sr_markdown(research_question: str, state: dict) -> str:
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
