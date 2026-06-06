"""ui/tabs/systematic_review.py — Systematic Review (PRISMA-style)"""

from __future__ import annotations

import logging
import time

import streamlit as st
import streamlit.components.v1 as components

from agents.systematic_review_graph import run_systematic_review
from agents.systematic_review_state import create_systematic_review_state
from ui.helpers import render_eval_result, render_rag_reflection

logger = logging.getLogger(__name__)


# ── Rendering helpers ─────────────────────────────────────────────────

def _render_prisma_metrics(flow: dict) -> None:
    if not flow:
        return
    cols = st.columns(5)
    labels = [
        ("Identified", "identified"),
        ("Screened", "screened"),
        ("Eligibility", "eligibility"),
        ("Included", "included"),
        ("Excluded", "excluded"),
    ]
    for col, (label, key) in zip(cols, labels):
        col.metric(label, flow.get(key, 0))


def _render_prisma_mermaid(flow: dict) -> None:
    """Render PRISMA flow as an interactive Mermaid diagram."""
    from tools.prisma_diagram import generate_prisma_mermaid, generate_prisma_dot

    mermaid_code = generate_prisma_mermaid(flow)
    html = f"""
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<div class="mermaid" style="max-width:640px;margin:auto;">
{mermaid_code}
</div>
<script>mermaid.initialize({{startOnLoad:true, theme:'default', securityLevel:'loose'}});</script>
"""
    components.html(html, height=460, scrolling=True)

    with st.expander("Graphviz version"):
        dot = generate_prisma_dot(flow)
        try:
            st.graphviz_chart(dot)
        except Exception:
            st.code(dot, language="dot")

    with st.expander("Mermaid source (paste into mermaid.live)"):
        st.code(mermaid_code, language="text")
        st.caption("→ [Open mermaid.live](https://mermaid.live) to render and export as PNG/SVG")


def _render_evidence_table(evidence_table: list) -> None:
    if not evidence_table:
        st.info("No papers were included in the review.")
        return

    for row in evidence_table:
        qual = row.get("quality", "Medium")
        title = row.get("title", "Untitled")
        year = row.get("year", "n.d.")
        ck = row.get("citation_key", "")
        design = row.get("study_design", "Unknown")
        n = row.get("sample_size", "Unknown")
        finding = row.get("key_finding", "")
        rel = row.get("relevance_score", 3)

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
                st.metric("Study Design", design)
                st.metric("Sample Size", n)
                st.metric("Relevance", f"{rel}/5")
            if finding:
                st.markdown(f"**Key finding:** {finding}")


def _render_rob_table(rob_table: list) -> None:
    if not rob_table:
        st.info("Risk of bias assessment was not performed (no papers included).")
        return

    icon_map = {"Low": "✅", "Some concerns": "⚠️", "High": "❌"}

    for rob in rob_table:
        overall = rob.get("overall", "Some concerns")
        icon = icon_map.get(overall, "⚠️")
        ck = rob.get("citation_key", "")
        title = rob.get("title", "")[:60]

        with st.expander(f"{icon} [{ck}] {title} — **{overall}** ({rob.get('tool','RoB 2')})"):
            st.markdown(f"**Tool:** {rob.get('tool', 'RoB 2')}")
            st.markdown(f"**Overall risk:** **{overall}**")
            if rob.get("justification"):
                st.markdown(f"**Justification:** {rob['justification']}")

            domain_keys = [
                k for k in rob
                if k not in ("tool", "overall", "justification", "citation_key", "title")
            ]
            if domain_keys:
                st.markdown("**Domain ratings:**")
                d_cols = st.columns(min(len(domain_keys), 3))
                for i, dk in enumerate(domain_keys):
                    rating = rob.get(dk, "")
                    col = d_cols[i % len(d_cols)]
                    domain_label = dk.replace("_", " ").title()
                    col.metric(domain_label[:24], rating)


def _render_grade(grade_results: dict) -> None:
    if not grade_results:
        st.info("GRADE assessment was not performed (no papers included).")
        return

    overall = grade_results.get("overall_grade", "Not assessed")
    grade_icons = {"High": "⭐⭐⭐⭐", "Moderate": "⭐⭐⭐", "Low": "⭐⭐", "Very low": "⭐"}

    c1, c2, c3 = st.columns(3)
    c1.metric("Starting Level", grade_results.get("starting_level", "?"))
    c2.metric("Overall GRADE", f"{grade_icons.get(overall, '')} {overall}")
    c3.metric("Studies", str(len(grade_results.get("domains", {}))))

    certainty = grade_results.get("certainty_statement", "")
    if certainty:
        st.info(f"**Certainty statement:** {certainty}")

    summary = grade_results.get("summary", "")
    if summary:
        st.markdown(f"**Summary:** {summary}")

    domains = grade_results.get("domains", {})
    if domains:
        st.markdown("**Domain downgrading decisions:**")
        domain_icons = {
            "no concern": "✅ No concern",
            "-1": "⬇️ Downgrade −1",
            "-2": "⬇️⬇️ Downgrade −2",
        }
        for domain, rating in domains.items():
            label = domain.replace("_", " ").title()
            icon_text = domain_icons.get(str(rating).strip(), str(rating))
            st.markdown(f"- **{label}:** {icon_text}")


def _render_contradictions(contradictions: list) -> None:
    if not contradictions:
        st.success("✅ No significant contradictions detected across included papers.")
        return

    st.markdown(f"**{len(contradictions)} area(s)** of conflicting evidence identified:")
    for i, c in enumerate(contradictions, 1):
        consensus = c.get("consensus_score", 50)
        with st.expander(f"{i}. {c.get('claim', '')}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Position A:**")
                pos_a = c.get("position_a", {})
                st.markdown(pos_a.get("description", ""))
                papers_a = pos_a.get("papers", [])
                if papers_a:
                    st.caption("Papers: " + ", ".join(papers_a))
            with col_b:
                st.markdown("**Position B:**")
                pos_b = c.get("position_b", {})
                st.markdown(pos_b.get("description", ""))
                papers_b = pos_b.get("papers", [])
                if papers_b:
                    st.caption("Papers: " + ", ".join(papers_b))
            st.progress(max(0, min(100, consensus)) / 100,
                        text=f"Consensus score: {consensus}/100")
            if c.get("explanation"):
                st.markdown(f"**Explanation:** {c['explanation']}")


# ── Main tab ──────────────────────────────────────────────────────────

def tab_systematic_review(settings: dict) -> None:
    st.header("Systematic Review")
    st.markdown(
        """
Conduct a **PRISMA-style systematic review** on any research question.
The agent searches academic databases, screens papers, extracts structured evidence,
and synthesises findings.

**What you get:**
- PRISMA flow diagram (interactive Mermaid + Graphviz)
- Evidence table with study design, quality rating, key finding
- Risk of Bias assessment (RoB 2 / ROBINS-I)
- GRADE evidence grading
- Contradiction detection across papers
- Narrative synthesis with inline citations
- Sensitivity analysis and incremental literature monitor
- OSF pre-registration template
- Quality self-evaluation scores
"""
    )

    st.divider()

    # ── Inputs ─────────────────────────────────────────────────────────────
    rq = st.text_area(
        "Research question",
        height=90,
        placeholder=(
            "e.g. What is the effect of sleep deprivation on working memory performance "
            "in university students?"
        ),
        key="sr_question",
    )

    col_inc, col_exc = st.columns(2)
    with col_inc:
        st.markdown("**Inclusion criteria** *(one per line)*")
        inc_raw = st.text_area(
            "Inclusion criteria",
            height=120,
            placeholder="Peer-reviewed empirical studies\nHuman participants\nPublished 2010–2024",
            key="sr_inclusion",
            label_visibility="collapsed",
        )
    with col_exc:
        st.markdown("**Exclusion criteria** *(one per line)*")
        exc_raw = st.text_area(
            "Exclusion criteria",
            height=120,
            placeholder="Animal studies\nCase reports\nConference abstracts only",
            key="sr_exclusion",
            label_visibility="collapsed",
        )

    run_btn = st.button(
        "Run Systematic Review", key="run_sr", type="primary", use_container_width=True,
    )

    if run_btn and not rq.strip():
        st.warning("Please enter a research question.")
        return

    if not run_btn:
        return

    # ── Parse criteria ──────────────────────────────────────────────────
    inclusion = [line.strip() for line in inc_raw.splitlines() if line.strip()]
    exclusion = [line.strip() for line in exc_raw.splitlines() if line.strip()]

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
        "literature_search":   "Searching arXiv, Semantic Scholar, CrossRef",
        "screening":           "Screening papers by title/abstract",
        "evidence_extraction": "Extracting evidence, RoB, GRADE & contradictions",
        "synthesis":           "Synthesising findings",
        "sr_eval":             "Evaluating review quality",
    }

    def stream_callback(node_name: str, state: dict) -> None:
        pct = state.get("progress_pct", 0)
        label = node_labels.get(node_name, node_name)
        detail = state.get("status_detail", "")
        progress_bar.progress(pct)
        if detail:
            status_text.markdown(f"**{label}…** `{pct}%`  \n{detail}")
        else:
            status_text.markdown(f"**{label}…** `{pct}%`")
        entry = f"{label} ({pct}%)" + (f" — {detail}" if detail else "")
        log_lines.append(entry)
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

    # Persist for SR→Notebook bridge
    st.session_state["sr_last_result"] = final_state

    # ── Results ──────────────────────────────────────────────────────────
    for err in final_state.get("errors", []):
        st.warning(err)

    st.divider()
    render_eval_result(final_state.get("eval_result", {}), key_suffix="_sr")
    render_rag_reflection(final_state.get("rag_reflection_info"), key_suffix="_sr")
    st.divider()

    # PRISMA metrics summary
    st.subheader("PRISMA Flow")
    _render_prisma_metrics(final_state.get("prisma_flow", {}))
    n_included = len(final_state.get("included_papers", []))
    n_excluded = len(final_state.get("excluded_papers", []))
    st.caption(
        f"Search queries: {len(final_state.get('search_queries', []))} · "
        f"Papers identified: {len(final_state.get('raw_papers', []))} · "
        f"Included: {n_included} · Excluded: {n_excluded}"
    )

    grade_overall = final_state.get("grade_results", {}).get("overall_grade", "n/a")
    rob_count = len(final_state.get("rob_table", []))
    contra_count = len(final_state.get("contradictions", []))
    st.caption(
        f"GRADE certainty: **{grade_overall}** · "
        f"RoB assessed: {rob_count} papers · "
        f"Contradictions: {contra_count}"
    )

    # ── Result tabs ─────────────────────────────────────────────────────
    (
        t_synthesis, t_evidence, t_prisma,
        t_rob, t_grade, t_contradictions,
        t_queries, t_advanced, t_export,
    ) = st.tabs([
        "Synthesis",
        "Evidence Table",
        "PRISMA Diagram",
        "Risk of Bias",
        "GRADE",
        "Contradictions",
        "Search Queries",
        "Advanced",
        "Export",
    ])

    # ── Tab 1: Synthesis ────────────────────────────────────────────────
    with t_synthesis:
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
        current_synthesis = st.session_state.get(
            f"_fb_output_{session_id}",
            final_state.get("narrative_synthesis", ""),
        )
        render_feedback_section(
            current_output=current_synthesis,
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

    # ── Tab 2: Evidence Table ──────────────────────────────────────────
    with t_evidence:
        st.subheader(f"Evidence Table ({n_included} included papers)")
        _render_evidence_table(final_state.get("evidence_table", []))

        if final_state.get("excluded_papers"):
            with st.expander(f"Excluded Papers ({n_excluded})"):
                for p in final_state["excluded_papers"]:
                    reason = p.get("exclusion_reason", "")
                    st.markdown(
                        f"- **{p.get('title','')[:70]}** ({p.get('year','n.d.')}) — _{reason}_"
                    )

    # ── Tab 3: PRISMA Diagram ────────────────────────────────────────
    with t_prisma:
        st.subheader("PRISMA 2020 Flow Diagram")
        flow = final_state.get("prisma_flow", {})
        if flow:
            _render_prisma_mermaid(flow)
        else:
            st.info("No PRISMA flow data available.")

    # ── Tab 4: Risk of Bias ─────────────────────────────────────────
    with t_rob:
        st.subheader("Risk of Bias Assessment")
        st.markdown(
            "Uses **Cochrane RoB 2** for randomised trials and **ROBINS-I** for "
            "observational studies. Domains rated: Low / Some concerns / High."
        )
        _render_rob_table(final_state.get("rob_table", []))

    # ── Tab 5: GRADE ─────────────────────────────────────────────────────
    with t_grade:
        st.subheader("GRADE Evidence Grading")
        st.markdown(
            "**G**rading of **R**ecommendations **A**ssessment, **D**evelopment and "
            "**E**valuation. RCT evidence starts High; observational studies start Low. "
            "Domains: risk of bias, inconsistency, indirectness, imprecision, publication bias."
        )
        _render_grade(final_state.get("grade_results", {}))

    # ── Tab 6: Contradictions ─────────────────────────────────────────
    with t_contradictions:
        st.subheader("Contradiction Analysis")
        st.markdown("Conflicting findings across included papers, with consensus scores.")
        _render_contradictions(final_state.get("contradictions", []))

    # ── Tab 7: Search Queries ────────────────────────────────────────
    with t_queries:
        st.subheader("Search Queries Used")
        for i, q in enumerate(final_state.get("search_queries", []), 1):
            st.markdown(f"{i}. {q}")

    # ── Tab 8: Advanced (Sensitivity + Monitor + Pre-registration) ─────────
    with t_advanced:
        adv_tabs = st.tabs(["Sensitivity Analysis", "Literature Monitor", "Pre-registration"])

        # ─ Sensitivity Analysis ─────────────────────────────────────
        with adv_tabs[0]:
            st.subheader("Sensitivity Analysis")
            st.markdown(
                "Test how robust your conclusions are by varying inclusion criteria "
                "or restricting to high-quality studies."
            )
            from tools.sensitivity_analysis import build_sensitivity_scenarios, run_sensitivity_analysis
            scenarios = build_sensitivity_scenarios(final_state)

            for scenario in scenarios:
                name = scenario.get("name", "")
                s_cache_key = f"sa_{name}_{initial_state.get('session_id', '')}"
                with st.expander(f"**{name}**"):
                    st.caption(scenario.get("description", ""))
                    if st.button(f"Run scenario", key=f"sa_btn_{hash(name)}_sr"):
                        with st.spinner("Running sensitivity analysis…"):
                            result = run_sensitivity_analysis(
                                final_state,
                                scenario_name=name,
                                modified_inclusion=scenario.get("modified_inclusion"),
                                modified_exclusion=scenario.get("modified_exclusion"),
                                quality_filter=scenario.get("quality_filter"),
                            )
                        st.session_state[s_cache_key] = result

                    sa_result = st.session_state.get(s_cache_key)
                    if sa_result:
                        if "error" in sa_result:
                            st.error(sa_result["error"])
                        else:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Original N", sa_result.get("original_n", 0))
                            c2.metric(
                                "After Filter",
                                sa_result.get("filtered_n", sa_result.get("new_n", 0)),
                            )
                            pct = sa_result.get("pct_retained", 100)
                            c3.metric("Retained", f"{pct}%")
                            if sa_result.get("note"):
                                st.info(sa_result["note"])
                            if sa_result.get("new_conclusion"):
                                with st.expander("Revised conclusion"):
                                    st.markdown(sa_result["new_conclusion"])

        # ─ Literature Monitor ────────────────────────────────────
        with adv_tabs[1]:
            st.subheader("Incremental Literature Monitor")
            st.markdown(
                "Save the current search state and return later to check for new papers "
                "published since the last run."
            )
            from tools.literature_monitor import (
                save_monitor_state, load_monitor_state,
                find_new_papers, monitor_id_from_question,
            )

            rq_val = final_state.get("research_question", "")
            monitor_id = monitor_id_from_question(rq_val)
            monitor_state = load_monitor_state(monitor_id)
            known_keys = [p.get("citation_key", "") for p in final_state.get("raw_papers", [])]

            if monitor_state:
                st.info(
                    f"**Last run:** {monitor_state.get('last_run', '')[:10]}  \n"
                    f"**Known papers:** {len(monitor_state.get('known_paper_keys', []))}"
                )

            m_col1, m_col2 = st.columns(2)
            if m_col1.button("Save search state", key="sr_save_monitor"):
                save_monitor_state(
                    monitor_id, rq_val,
                    final_state.get("search_queries", []),
                    known_keys,
                )
                st.success("Search state saved. Return later to check for new papers.")

            if monitor_state and m_col2.button("Check for new papers", key="sr_check_monitor"):
                from tools.search_tools import AcademicSearcher
                with st.spinner("Searching for new papers…"):
                    searcher = AcademicSearcher()
                    all_new = []
                    for q in monitor_state.get("search_queries", [rq_val])[:3]:
                        try:
                            papers = searcher.search(q, max_per_source=5)
                            for p in papers:
                                all_new.append({
                                    "citation_key": p.citation_key,
                                    "title": p.title,
                                    "year": p.year,
                                    "url": p.url,
                                })
                        except Exception:
                            pass
                    prev_keys = monitor_state.get("known_paper_keys", [])
                    new_papers = find_new_papers(all_new, prev_keys)

                if new_papers:
                    st.success(f"✨ Found **{len(new_papers)} new papers** since last run!")
                    for p in new_papers[:10]:
                        url = p.get("url", "")
                        title = p.get("title", "")
                        year = p.get("year", "")
                        if url:
                            st.markdown(f"- [{title}]({url}) ({year})")
                        else:
                            st.markdown(f"- **{title}** ({year})")
                else:
                    st.info("✅ No new papers found since last monitoring run.")

        # ─ Pre-registration ─────────────────────────────────────
        with adv_tabs[2]:
            st.subheader("SR Pre-Registration Template")
            st.markdown(
                "Generate an OSF-style pre-registration document for this systematic review. "
                "Register at [osf.io/registries](https://osf.io/registries) before beginning data collection."
            )
            from tools.preregistration import generate_preregistration, generate_prisma_checklist

            author_name = st.text_input(
                "Author name", key="sr_prereg_author",
                placeholder="Your Name / Institution",
            )
            pr_cache = f"prereg_{initial_state.get('session_id', '')}"
            pr_col1, pr_col2 = st.columns(2)

            if pr_col1.button("Generate Pre-registration", key="sr_gen_prereg", type="primary"):
                template = generate_preregistration(
                    final_state.get("research_question", ""),
                    final_state.get("inclusion_criteria", []),
                    final_state.get("exclusion_criteria", []),
                    final_state.get("search_queries", []),
                    author_name=author_name or "ResearchBuddy User",
                )
                st.session_state[pr_cache] = template

            if pr_col2.button("Generate PRISMA Checklist", key="sr_gen_checklist"):
                checklist = generate_prisma_checklist(final_state)
                st.session_state[f"{pr_cache}_checklist"] = checklist

            if pr_cache in st.session_state:
                template = st.session_state[pr_cache]
                st.markdown(template)
                st.download_button(
                    "Download pre-registration (.md)",
                    data=template,
                    file_name=f"preregistration_{initial_state.get('session_id','')}.md",
                    mime="text/markdown",
                    key="sr_dl_prereg",
                )

            if f"{pr_cache}_checklist" in st.session_state:
                st.divider()
                st.markdown(st.session_state[f"{pr_cache}_checklist"])

    # ── Tab 9: Export ────────────────────────────────────────────────────
    with t_export:
        st.subheader("Export Systematic Review")
        md_content = _build_sr_markdown(rq, final_state)
        st.download_button(
            label="Download as Markdown",
            data=md_content.encode("utf-8"),
            file_name=f"systematic_review_{initial_state.get('session_id','')}.md",
            mime="text/markdown",
            use_container_width=True,
        )


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
    ]

    grade = state.get("grade_results", {})
    if grade:
        lines += [
            "## GRADE Evidence Certainty",
            "",
            f"- **Starting level:** {grade.get('starting_level', '?')}",
            f"- **Overall grade:** {grade.get('overall_grade', '?')}",
            f"- {grade.get('certainty_statement', '')}",
            "",
        ]

    lines += ["## Key Themes", ""]
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
        lines += ["## Evidence Table", "",
                  "| Citation | Year | Design | Quality | Key Finding |",
                  "| --- | --- | --- | --- | --- |"]
        for row in evidence:
            lines.append(
                f"| {row.get('citation_key','')} | {row.get('year','n.d.')} | "
                f"{row.get('study_design','')} | {row.get('quality','')} | "
                f"{row.get('key_finding','')[:80]} |"
            )
        lines.append("")

    rob = state.get("rob_table", [])
    if rob:
        lines += ["## Risk of Bias Summary", "",
                  "| Citation | Tool | Overall |",
                  "| --- | --- | --- |"]
        for r in rob:
            lines.append(
                f"| {r.get('citation_key','')} | {r.get('tool','')} | {r.get('overall','')} |"
            )
        lines.append("")

    return "\n".join(lines)
