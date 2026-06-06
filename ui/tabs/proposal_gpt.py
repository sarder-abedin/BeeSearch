"""
ui/tabs/proposal_gpt.py
────────────────────────
ProposalGPT — AI-assisted research proposal writing.

8-step workflow:
  Tab 1  Upload & Configure    — funding call + optional CVs/pubs/partners
  Tab 2  Call Analysis         — objectives, criteria matrix, compliance checklist
  Tab 3  Strategy              — win strategy, SWOT, reviewer perspective
  Tab 4  Literature            — literature review, state of art, gaps
  Tab 5  Draft Proposal        — all 20 sections with apply-improvement toggle
  Tab 6  Reviewer Simulation   — 5-reviewer scores + section-level feedback
  Tab 7  Compliance Check      — keyword coverage, missing sections, score
  Tab 8  Export                — DOCX / PDF / Markdown + budget CSV
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

import streamlit as st

from config.settings import get_settings
from ui.helpers import get_supported_file_types, process_uploads, render_eval_result, render_rag_reflection

logger = logging.getLogger(__name__)
cfg = get_settings()


# ── Session-state helpers ───────────────────────────────────────────────────────

_CTX_KEY = "_pgpt_run_ctx"
_RUN_KEY = "_pgpt_run_state"
_RESULT_KEY = "_pgpt_result"
_SESSION_KEY = "_pgpt_session_id"


def _get_result() -> dict | None:
    return st.session_state.get(_RESULT_KEY)


def _set_result(state: dict) -> None:
    st.session_state[_RESULT_KEY] = state


def _clear_result() -> None:
    for key in [_CTX_KEY, _RUN_KEY, _RESULT_KEY]:
        st.session_state.pop(key, None)


# ── Background runner ──────────────────────────────────────────────────────────

def _launch_pipeline(initial_state: dict) -> None:
    """Launch pipeline in a background thread."""
    from agents.proposal_gpt_graph import run_proposal_gpt
    from agents.proposal_gpt_state import ProposalGPTState

    run_data: dict = {"done": False, "result": None, "error": None,
                      "progress": 0, "status": "Initialising…", "log": []}
    st.session_state[_CTX_KEY] = {"start": time.time()}
    st.session_state[_RUN_KEY] = run_data

    _AGENT_LABELS = {
        "funding_call_analyzer":  "Agent 1 — Analysing funding call",
        "research_planner":       "Agent 2 — Building proposal strategy",
        "literature_review_agent":"Agent 3 — Reviewing literature",
        "proposal_writer":        "Agent 4 — Writing proposal sections",
        "impact_agent":           "Agent 5 — Generating impact sections",
        "budget_agent":           "Agent 6 — Building budget",
        "compliance_agent":       "Agent 7 — Checking compliance",
        "reviewer_agent":         "Agent 8 — Simulating reviewers",
        "improvement_agent":      "Agent 9 — Improving weak sections",
    }

    def _cb(node_name: str, state: dict) -> None:
        pct = state.get("progress_pct", 0)
        label = _AGENT_LABELS.get(node_name, node_name)
        run_data["progress"] = pct
        run_data["status"] = label
        run_data["log"].append(f"{label} ({pct}%)")

    def _run() -> None:
        try:
            result = run_proposal_gpt(ProposalGPTState(**initial_state), stream_callback=_cb)
            run_data["result"] = dict(result)
        except Exception as exc:
            run_data["error"] = str(exc)
            logger.exception("ProposalGPT pipeline failed")
        finally:
            run_data["done"] = True

    threading.Thread(target=_run, daemon=True).start()
    st.rerun()


# ── Upload helper ──────────────────────────────────────────────────────────────

def _extract_text_from_uploads(uploaded_files, settings: dict) -> list[str]:
    """Process uploaded files and return list of raw_text strings."""
    if not uploaded_files:
        return []
    processed = process_uploads(uploaded_files, settings)
    return [d.raw_text for d in processed if d.raw_text]


# ── Tab 1: Upload & Configure ─────────────────────────────────────────────────

def _tab_upload(settings: dict) -> None:
    st.markdown(
        "Upload the funding call document and optionally provide researcher context "
        "to personalise the proposal. The AI will analyse the call and generate a "
        "complete, publication-quality proposal."
    )

    if _get_result():
        st.success("A proposal has been generated. Switch to other tabs to review it.")
        if st.button("Start New Proposal", key="pgpt_new_proposal"):
            _clear_result()
            st.rerun()
        return

    # ── Funding Call ──────────────────────────────────────────────────────
    st.markdown("#### 1. Funding Call Document")
    col_upload, col_url = st.columns([2, 1])
    with col_upload:
        call_file = st.file_uploader(
            "Upload funding call (PDF, DOCX, TXT)",
            type=get_supported_file_types(settings.get("use_docling", True)),
            key="pgpt_call_file",
        )
    with col_url:
        call_url = st.text_input(
            "Or paste a URL",
            key="pgpt_call_url",
            placeholder="https://ec.europa.eu/…",
        )

    st.markdown("#### 2. Researcher & Project Context")
    user_ideas = st.text_area(
        "Research ideas and project description",
        key="pgpt_ideas",
        height=120,
        placeholder="Describe your research idea, preliminary results, unique angle, "
                    "why your team is best placed to do this…",
    )

    col_inst, col_agency = st.columns(2)
    with col_inst:
        institution_info = st.text_input(
            "Institution / Department",
            key="pgpt_institution",
            placeholder="e.g. Chalmers University of Technology, Dept. of Computer Science",
        )
    with col_agency:
        funding_agency = st.selectbox(
            "Funding Agency",
            ["Generic", "Horizon Europe", "ERC", "MSCA", "Vinnova",
             "VR (Swedish Research Council)", "Formas", "NSF", "UKRI", "DARPA"],
            key="pgpt_agency",
        )

    st.markdown("#### 3. Optional Supporting Documents")
    col_cv, col_pub, col_partner = st.columns(3)
    with col_cv:
        _pgpt_types = get_supported_file_types(settings.get("use_docling", True))
        cv_files = st.file_uploader(
            "CVs (optional)",
            type=_pgpt_types,
            accept_multiple_files=True,
            key="pgpt_cvs",
        )
    with col_pub:
        pub_files = st.file_uploader(
            "Publication lists (optional)",
            type=_pgpt_types,
            accept_multiple_files=True,
            key="pgpt_pubs",
        )
    with col_partner:
        partner_files = st.file_uploader(
            "Partner profiles (optional)",
            type=_pgpt_types,
            accept_multiple_files=True,
            key="pgpt_partners",
        )

    requirements = st.text_input(
        "Extra instructions (tone, word count, focus areas)",
        key="pgpt_requirements",
        placeholder="e.g. 'Focus on AI methods. Target 12 pages. Emphasise clinical applications.'",
    )

    st.divider()
    if st.button(
        "Generate Proposal",
        key="pgpt_generate",
        type="primary",
        use_container_width=True,
        help="Runs all 9 AI agents. Takes 5–15 minutes depending on model and context.",
    ):
        # ── Validate and collect inputs ───────────────────────────────────
        call_text = ""

        if call_file:
            with st.spinner("Processing funding call document…"):
                docs = process_uploads([call_file], settings)
            if docs:
                call_text = docs[0].raw_text
                call_filename = call_file.name
        elif call_url.strip():
            with st.spinner("Fetching funding call from URL…"):
                try:
                    from tools.document_tools import get_processor
                    from tools.web_loader import load_url_as_document
                    proc = get_processor(chunk_size=1000, overlap=100)
                    doc, err = load_url_as_document(call_url.strip(), proc)
                    if err:
                        st.error(f"Could not fetch URL: {err}")
                        return
                    call_text = doc.raw_text
                    call_filename = call_url.strip()[:60]
                except Exception as exc:
                    st.error(f"URL fetch failed: {exc}")
                    return

        if not call_text.strip():
            st.warning("Please upload a funding call document or paste a URL.")
            return

        with st.spinner("Processing supporting documents…"):
            cv_texts = _extract_text_from_uploads(cv_files, settings)
            pub_texts = _extract_text_from_uploads(pub_files, settings)
            partner_texts = _extract_text_from_uploads(partner_files, settings)

        from agents.proposal_gpt_state import create_proposal_gpt_state
        session_id = str(uuid.uuid4())[:8]
        st.session_state[_SESSION_KEY] = session_id

        initial = create_proposal_gpt_state(
            funding_call_text=call_text,
            user_ideas=user_ideas,
            requirements=requirements,
            funding_agency=funding_agency,
            model_name=settings["model"],
            num_ctx=settings["num_ctx"],
            session_id=session_id,
            cv_texts=cv_texts,
            publication_texts=pub_texts,
            institution_info=institution_info,
            partner_profiles=partner_texts,
            funding_call_filename=call_file.name if call_file else call_url[:40],
        )

        _launch_pipeline(dict(initial))


# ── Progress polling ───────────────────────────────────────────────────────────

def _render_progress() -> bool:
    """Render progress bar if pipeline is running. Returns True while running."""
    ctx = st.session_state.get(_CTX_KEY)
    if not ctx:
        return False

    run_state = st.session_state.get(_RUN_KEY)
    if not run_state:
        st.session_state.pop(_CTX_KEY, None)
        return False

    if not run_state["done"]:
        pct = run_state.get("progress", 0)
        status = run_state.get("status", "Running…")
        log = run_state.get("log", [])

        st.info("ProposalGPT is generating your proposal…")
        st.progress(pct / 100)
        st.markdown(f"**{status}** — `{pct}%`")
        if log:
            with st.expander("Agent Log", expanded=False):
                for entry in log:
                    st.caption(entry)

        time.sleep(3.0)
        st.rerun()
        return True

    # Done
    error = run_state.get("error")
    result = run_state.get("result")
    st.session_state.pop(_RUN_KEY, None)
    st.session_state.pop(_CTX_KEY, None)

    if error:
        st.error(f"Pipeline failed: {error}")
        return True

    _set_result(result)
    st.rerun()
    return True


# ── Tab 2: Call Analysis ───────────────────────────────────────────────────────

def _tab_call_analysis() -> None:
    result = _get_result()
    if not result:
        st.info("Generate a proposal first (Tab 1: Upload & Configure).")
        return

    st.markdown(f"**Funding Call:** {result.get('call_title', '—')}")
    st.markdown(f"**Agency:** {result.get('funding_agency', '—')} · **Deadline:** {result.get('deadline', 'Not specified')}")

    t_summary, t_objectives, t_criteria, t_checklist, t_success = st.tabs([
        "Summary", "Objectives", "Evaluation Matrix", "Checklist", "Success Factors"
    ])

    with t_summary:
        st.markdown(result.get("funding_summary", "*Not generated.*"))

    with t_objectives:
        objectives = result.get("call_objectives", [])
        if objectives:
            for i, obj in enumerate(objectives, 1):
                st.markdown(f"**{i}.** {obj}")
        else:
            st.info("No objectives extracted.")
        st.divider()
        outcomes = result.get("expected_outcomes", [])
        if outcomes:
            st.markdown("**Expected Outcomes:**")
            for o in outcomes:
                st.markdown(f"- {o}")

        keywords = result.get("keywords", [])
        if keywords:
            st.divider()
            st.markdown("**Strategic Keywords:**")
            st.markdown(" · ".join(f"`{k}`" for k in keywords))

    with t_criteria:
        st.markdown(result.get("evaluation_matrix", "*Not generated.*"))
        eligibility = result.get("eligibility_requirements", [])
        if eligibility:
            st.divider()
            st.markdown("**Eligibility Requirements:**")
            for e in eligibility:
                st.markdown(f"- {e}")

    with t_checklist:
        checklist = result.get("compliance_checklist", [])
        if checklist:
            for item in checklist:
                status = item.get("status", "⬜ Pending")
                st.markdown(f"{status} {item.get('item', '')}")
                if item.get("notes"):
                    st.caption(item["notes"])
        else:
            st.info("No checklist items.")

    with t_success:
        st.markdown(result.get("success_factors", "*Not generated.*"))


# ── Tab 3: Strategy ────────────────────────────────────────────────────────────

def _tab_strategy() -> None:
    result = _get_result()
    if not result:
        st.info("Generate a proposal first (Tab 1: Upload & Configure).")
        return

    t_win, t_swot, t_reviewer, t_risks = st.tabs([
        "Win Strategy", "SWOT", "Reviewer Perspective", "Risks"
    ])

    with t_win:
        st.markdown(result.get("win_strategy", "*Not generated.*"))
        hidden = result.get("hidden_priorities", [])
        if hidden:
            st.divider()
            st.markdown("**Hidden Priorities:**")
            for h in hidden:
                st.markdown(f"- {h}")

    with t_swot:
        st.markdown(result.get("swot_analysis", "*Not generated.*"))
        strengths = result.get("proposal_strengths", [])
        advantages = result.get("competitive_advantages", [])
        if strengths or advantages:
            col_s, col_a = st.columns(2)
            with col_s:
                if strengths:
                    st.markdown("**Proposal Strengths:**")
                    for s in strengths:
                        st.markdown(f"- {s}")
            with col_a:
                if advantages:
                    st.markdown("**Competitive Advantages:**")
                    for a in advantages:
                        st.markdown(f"- {a}")

    with t_reviewer:
        st.markdown(result.get("reviewer_perspective", "*Not generated.*"))
        expectations = result.get("reviewer_expectations", "")
        if expectations:
            st.divider()
            st.markdown("**Reviewer Expectations:**")
            st.markdown(expectations)

    with t_risks:
        risks = result.get("proposal_risks", [])
        if risks:
            st.markdown("| Risk | Likelihood | Mitigation |")
            st.markdown("|------|------------|------------|")
            for r in risks:
                st.markdown(
                    f"| {r.get('risk','')} | {r.get('likelihood','')} | "
                    f"{r.get('mitigation','')} |"
                )
        else:
            st.info("No risks identified.")


# ── Tab 4: Literature ──────────────────────────────────────────────────────────

def _tab_literature() -> None:
    result = _get_result()
    if not result:
        st.info("Generate a proposal first (Tab 1: Upload & Configure).")
        return

    t_lr, t_sota, t_gaps, t_refs = st.tabs([
        "Literature Review", "State of the Art", "Research Gaps", "References"
    ])

    with t_lr:
        st.markdown(result.get("literature_review", "*Not generated.*"))

    with t_sota:
        st.markdown(result.get("state_of_art", "*Not generated.*"))

    with t_gaps:
        gaps = result.get("research_gaps", [])
        if gaps:
            for i, g in enumerate(gaps, 1):
                st.markdown(f"**Gap {i}:** {g}")
        else:
            st.info("No research gaps identified.")

    with t_refs:
        refs = result.get("suggested_references", [])
        if refs:
            for i, r in enumerate(refs, 1):
                authors = ", ".join(r.get("authors", [])[:3])
                year = r.get("year", "")
                doi = r.get("doi", "")
                st.markdown(
                    f"**[{i}]** {r.get('title', '')} — *{authors}* ({year})"
                    + (f" · DOI: `{doi}`" if doi else "")
                )
        else:
            st.info("No references found.")


# ── Tab 5: Draft Proposal ─────────────────────────────────────────────────────

def _tab_draft_proposal() -> None:
    result = _get_result()
    if not result:
        st.info("Generate a proposal first (Tab 1: Upload & Configure).")
        return

    from tools.proposal_tools import SECTION_ORDER

    improved = result.get("improved_sections", {})
    use_improved = st.toggle(
        "Apply AI improvements to weak sections",
        value=True,
        key="pgpt_use_improved",
        help="Shows AI-rewritten versions of sections identified as weak by reviewers.",
    )

    # Work Packages / Deliverables / Milestones summary
    st.markdown("### Project Structure")
    col_wp, col_del, col_ms = st.columns(3)
    wps = result.get("work_packages", [])
    dels = result.get("deliverables", [])
    mss = result.get("milestones", [])
    col_wp.metric("Work Packages", len(wps))
    col_del.metric("Deliverables", len(dels))
    col_ms.metric("Milestones", len(mss))

    if wps:
        with st.expander("Work Packages"):
            for wp in wps:
                st.markdown(f"**{wp.get('id','')} — {wp.get('title','')}** ({wp.get('months','')})")
                st.caption(wp.get("description", ""))
                for task in wp.get("tasks", []):
                    st.markdown(f"  - {task}")

    if dels:
        with st.expander("Deliverables"):
            rows = ["| ID | Title | Type | Month | WP |", "|---|---|---|---|---|"]
            for d in dels:
                rows.append(f"| {d.get('id','')} | {d.get('title','')} | {d.get('type','')} | {d.get('month','')} | {d.get('wp','')} |")
            st.markdown("\n".join(rows))

    if mss:
        with st.expander("Milestones"):
            rows = ["| ID | Title | Month | Verification |", "|---|---|---|---|"]
            for m in mss:
                rows.append(f"| {m.get('id','')} | {m.get('title','')} | {m.get('month','')} | {m.get('verification','')} |")
            st.markdown("\n".join(rows))

    st.divider()
    st.markdown("### Proposal Sections")

    for field, heading in SECTION_ORDER:
        text = (improved.get(field) if use_improved else None) or result.get(field, "")
        if not text:
            continue
        is_improved = use_improved and field in improved
        label = f"**{heading}**" + (" *(AI-improved)*" if is_improved else "")
        with st.expander(label):
            st.markdown(text)


# ── Tab 6: Reviewer Simulation ────────────────────────────────────────────────

def _tab_reviewer_simulation() -> None:
    result = _get_result()
    if not result:
        st.info("Generate a proposal first (Tab 1: Upload & Configure).")
        return

    overall = result.get("overall_score", 0.0)
    score_color = "green" if overall >= 4.0 else "orange" if overall >= 3.0 else "red"
    st.markdown(
        f"<h2 style='color:{score_color}'>Overall Score: {overall:.1f} / 5.0</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(result.get("reviewer_report", "*No reviewer report generated.*"))

    improvement_plan = result.get("improvement_plan", "")
    if improvement_plan:
        st.divider()
        st.markdown("### Improvement Plan")
        st.markdown(improvement_plan)

    weak_sections = result.get("weak_sections", [])
    if weak_sections:
        st.divider()
        st.markdown("### Sections to Improve")
        for s in weak_sections:
            st.markdown(f"- {s.replace('_', ' ').title()}")


# ── Tab 7: Compliance Check ───────────────────────────────────────────────────

def _tab_compliance() -> None:
    result = _get_result()
    if not result:
        st.info("Generate a proposal first (Tab 1: Upload & Configure).")
        return

    score = result.get("compliance_score", 0)
    score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
    st.markdown(
        f"<h2 style='color:{score_color}'>Compliance Score: {score}/100</h2>",
        unsafe_allow_html=True,
    )

    col_pg, col_wc, col_kw = st.columns(3)
    col_pg.metric("Estimated Pages", result.get("page_estimate", 0))

    from tools.proposal_tools import SECTION_ORDER, section_word_count
    word_counts = section_word_count(result)
    total_words = sum(word_counts.values())
    col_wc.metric("Total Words", f"{total_words:,}")

    kw_cov = result.get("keyword_coverage", {})
    covered = sum(1 for v in kw_cov.values() if v)
    col_kw.metric("Keywords Covered", f"{covered}/{len(kw_cov)}")

    st.markdown(result.get("compliance_report", "*No compliance report generated.*"))

    issues = result.get("compliance_issues", [])
    if issues:
        st.divider()
        st.markdown("### Issues to Address")
        for issue in issues:
            st.warning(issue)

    # Word count by section
    if word_counts:
        with st.expander("Word Count by Section"):
            rows = ["| Section | Words |", "|---------|-------|"]
            for section, count in word_counts.items():
                if count > 0:
                    rows.append(f"| {section} | {count:,} |")
            st.markdown("\n".join(rows))


# ── Tab 8: Export ─────────────────────────────────────────────────────────────

def _tab_export() -> None:
    result = _get_result()
    if not result:
        st.info("Generate a proposal first (Tab 1: Upload & Configure).")
        return

    from tools.proposal_tools import assemble_full_proposal_md, build_proposal_docx, build_budget_csv

    session_id = result.get("session_id", "proposal")
    agency = result.get("funding_agency", "proposal")
    base_name = f"proposal_{agency.lower().replace(' ', '_')}_{session_id}"

    use_improved = st.toggle(
        "Apply AI improvements in exports",
        value=True,
        key="pgpt_export_improved",
    )

    st.markdown("### Download Proposal")
    col_md, col_docx, col_pdf = st.columns(3)

    # Markdown
    md_text = assemble_full_proposal_md(result, include_improved=use_improved)
    col_md.download_button(
        "Markdown (.md)",
        data=md_text.encode("utf-8"),
        file_name=f"{base_name}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # DOCX
    try:
        docx_bytes = build_proposal_docx(result, include_improved=use_improved)
        col_docx.download_button(
            "Word (.docx)",
            data=docx_bytes,
            file_name=f"{base_name}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    except Exception as e:
        col_docx.caption(f"DOCX unavailable: {e}")

    # PDF
    try:
        from tools.export_tools import build_pdf
        pdf_bytes = build_pdf(md_text, [])
        col_pdf.download_button(
            "PDF (.pdf)",
            data=pdf_bytes,
            file_name=f"{base_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        col_pdf.caption(f"PDF unavailable: {e}")

    st.divider()
    st.markdown("### Download Analysis Reports")
    col_win, col_review, col_comp = st.columns(3)

    win_md = (
        f"# Win Strategy Report\n\n{result.get('win_strategy','')}\n\n"
        f"# SWOT Analysis\n\n{result.get('swot_analysis','')}\n\n"
        f"# Reviewer Perspective\n\n{result.get('reviewer_perspective','')}"
    )
    col_win.download_button(
        "Strategy Report (.md)",
        data=win_md.encode("utf-8"),
        file_name=f"strategy_{session_id}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    review_md = result.get("reviewer_report", "") + "\n\n# Improvement Plan\n\n" + result.get("improvement_plan", "")
    col_review.download_button(
        "Reviewer Report (.md)",
        data=review_md.encode("utf-8"),
        file_name=f"reviewer_report_{session_id}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    col_comp.download_button(
        "Compliance Report (.md)",
        data=result.get("compliance_report", "").encode("utf-8"),
        file_name=f"compliance_{session_id}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### Download Budget")
    col_csv, col_bmd = st.columns(2)

    csv_bytes = build_budget_csv(result)
    col_csv.download_button(
        "Budget Spreadsheet (.csv)",
        data=csv_bytes,
        file_name=f"budget_{session_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    budget_md = (
        f"# Budget Summary\n\n{result.get('budget_summary_table','')}\n\n"
        f"# Budget Justification\n\n{result.get('budget_justification','')}"
    )
    col_bmd.download_button(
        "Budget Justification (.md)",
        data=budget_md.encode("utf-8"),
        file_name=f"budget_justification_{session_id}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()
    render_eval_result(result.get("eval_result", {}), key_suffix=f"_pgpt_{result.get('session_id','')}")
    render_rag_reflection(result.get("rag_reflection_info"), key_suffix=f"_pgpt_{result.get('session_id','')}")

    st.divider()
    st.markdown("### Proposal Package Summary")
    total_words = sum(
        len((result.get("improved_sections", {}).get(f) or result.get(f, "")).split())
        for f, _ in __import__("tools.proposal_tools", fromlist=["SECTION_ORDER"]).SECTION_ORDER
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Words", f"{total_words:,}")
    mc2.metric("Pages (est.)", result.get("page_estimate", 0))
    mc3.metric("Compliance", f"{result.get('compliance_score', 0)}/100")
    mc4.metric("Reviewer Score", f"{result.get('overall_score', 0):.1f}/5.0")

    st.divider()
    from ui.helpers import render_feedback_section
    session_id = result.get("session_id", "pgpt")
    # Assemble proposal text for refinement
    from tools.proposal_tools import assemble_full_proposal_md
    try:
        full_proposal = assemble_full_proposal_md(result)
    except Exception:
        full_proposal = result.get("executive_summary", "") + "\n\n" + result.get("methodology", "")
    render_feedback_section(
        current_output=full_proposal,
        session_key=f"pgpt_{session_id}",
        mode="proposal",
        model_name=result.get("model_name", "llama3.1:8b"),
        num_ctx=result.get("num_ctx", 32768),
        context=result.get("literature_review", "")[:1500],
        key_suffix=f"_pgpt_{session_id}",
    )


# ── Main entry point ───────────────────────────────────────────────────────────

def tab_proposal_gpt(settings: dict) -> None:
    """ProposalGPT — Mode 2: AI-assisted research proposal writing."""
    st.header("Proposal Writer")
    st.markdown(
        "Generate a complete, publication-quality research proposal from your funding call "
        "document. ProposalGPT runs **9 AI agents** in sequence: call analysis, strategy, "
        "literature review, proposal writing, impact, budget, compliance checking, "
        "reviewer simulation, and iterative improvement."
    )

    # ── Running state ──────────────────────────────────────────────────────
    if _render_progress():
        return

    (
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8
    ) = st.tabs([
        "Upload & Configure",
        "Call Analysis",
        "Strategy",
        "Literature",
        "Draft Proposal",
        "Reviewer Simulation",
        "Compliance",
        "Export",
    ])

    with tab1:
        _tab_upload(settings)
    with tab2:
        _tab_call_analysis()
    with tab3:
        _tab_strategy()
    with tab4:
        _tab_literature()
    with tab5:
        _tab_draft_proposal()
    with tab6:
        _tab_reviewer_simulation()
    with tab7:
        _tab_compliance()
    with tab8:
        _tab_export()
