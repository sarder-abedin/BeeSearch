"""ui/tabs/systematic_review.py — Systematic Review (PRISMA-style)"""

from __future__ import annotations

import logging
import time

import streamlit as st

from agents.systematic_review_graph import run_systematic_review
from agents.systematic_review_state import create_systematic_review_state
from ui.helpers import render_eval_result, render_rag_reflection

logger = logging.getLogger(__name__)


def _render_prisma_flow(flow: dict) -> None:
    """Render the PRISMA flow diagram as a simple metric row."""
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


def _render_evidence_table(evidence_table: list) -> None:
    """Render the evidence extraction table."""
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


def tab_systematic_review(settings: dict) -> None:
    """
    Systematic Review — Simplified PRISMA Workflow.

    Automates a simplified systematic review workflow:
    1. Query generation from the research question
    2. Literature search (arXiv + Semantic Scholar)
    3. Title/abstract screening against inclusion/exclusion criteria
    4. Evidence extraction (study design, quality, key finding)
    5. Narrative synthesis, themes, gaps, and conclusion
    """
    st.header("Systematic Review")
    st.markdown(
        """
Conduct a simplified **PRISMA-style systematic review** on any research question.
The agent searches academic databases, screens papers against your criteria,
extracts structured evidence, and synthesises the findings.

**What you get:**
- PRISMA flow numbers (identified → screened → included)
- Evidence table: study design, quality rating, key finding per paper
- Narrative synthesis with inline citations
- Key themes, research gaps, and conclusion
- Quality self-evaluation scores
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

    run_btn = st.button(
        "Run Systematic Review", key="run_sr",
        type="primary", use_container_width=True,
    )

    if run_btn and not rq.strip():
        st.warning("Please enter a research question.")
        return

    if not run_btn:
        return

    # ── Parse criteria ────────────────────────────────────────────────────────
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
        "evidence_extraction": "Extracting evidence from papers",
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

    # ── Results ───────────────────────────────────────────────────────────────
    for err in final_state.get("errors", []):
        st.warning(err)

    st.divider()
    render_eval_result(final_state.get("eval_result", {}), key_suffix="_sr")
    render_rag_reflection(final_state.get("rag_reflection_info"), key_suffix="_sr")
    st.divider()

    # PRISMA flow
    st.subheader("PRISMA Flow")
    _render_prisma_flow(final_state.get("prisma_flow", {}))

    n_included = len(final_state.get("included_papers", []))
    n_excluded = len(final_state.get("excluded_papers", []))
    st.caption(
        f"Search queries used: {len(final_state.get('search_queries', []))} · "
        f"Papers identified: {len(final_state.get('raw_papers', []))} · "
        f"Included: {n_included} · Excluded: {n_excluded}"
    )

    # Main tabs
    t1, t2, t3, t4 = st.tabs([
        "Synthesis",
        "Evidence Table",
        "Search Queries",
        "Export",
    ])

    with t1:
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

        conclusion = final_state.get("conclusion", "")
        if conclusion:
            st.subheader("Conclusion")
            st.markdown(conclusion)

        limitations = final_state.get("limitations", "")
        if limitations:
            st.subheader("Limitations of this Review")
            st.info(limitations)

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

    with t2:
        st.subheader(f"Evidence Table ({n_included} included papers)")
        _render_evidence_table(final_state.get("evidence_table", []))

        if final_state.get("excluded_papers"):
            with st.expander(f"Excluded Papers ({n_excluded})"):
                for p in final_state["excluded_papers"]:
                    reason = p.get("exclusion_reason", "")
                    st.markdown(f"- **{p.get('title','')[:70]}** ({p.get('year','n.d.')}) — _{reason}_")

    with t3:
        st.subheader("Search Queries Used")
        for i, q in enumerate(final_state.get("search_queries", []), 1):
            st.markdown(f"{i}. {q}")

    with t4:
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
    """Build a full Markdown document for the systematic review."""
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
        f"| Stage | Count |",
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
        lines += ["| Citation | Year | Design | Quality | Key Finding |", "| --- | --- | --- | --- | --- |"]
        for row in evidence:
            ck = row.get("citation_key", "")
            yr = row.get("year", "n.d.")
            design = row.get("study_design", "")
            qual = row.get("quality", "")
            finding = row.get("key_finding", "")[:80]
            lines.append(f"| {ck} | {yr} | {design} | {qual} | {finding} |")
        lines.append("")

    return "\n".join(lines)
