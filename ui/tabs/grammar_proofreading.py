"""ui/tabs/grammar_proofreading.py — Mode 6: English Grammar Proofreading"""

from __future__ import annotations

import logging
import time
import uuid

import streamlit as st

from agents.grammar_graph import run_grammar_check
from agents.grammar_memory import GrammarMemory
from agents.grammar_state import create_grammar_state
from ui.helpers import (
    get_supported_file_types,
    process_uploads,
    render_eval_result,
    render_feedback_section,
    render_rag_reflection,
)

logger = logging.getLogger(__name__)

_LAST_RUN_KEY = "grammar_last_run"

_STYLE_OPTIONS = {
    "Professional Email": "professional_email",
    "Academic": "academic",
    "Formal": "formal",
    "Informal": "informal",
}

_FOCUS_OPTIONS = ["Grammar", "Punctuation", "Spelling", "Style", "Clarity"]

_NODE_LABELS = {
    "text_loader":      "Analysing input text",
    "grammar_analysis": "Detecting grammar, spelling & punctuation issues",
    "polish":           "Rewriting for clarity and fluency",
    "style_advisor":    "Generating style improvement tips",
    "grammar_eval":     "Evaluating output quality",
}


def _severity_icon(severity: str) -> str:
    return {"error": "🔴", "warning": "🟡", "suggestion": "🔵"}.get(severity.lower(), "⚪")


def _render_issues_table(issues: list) -> None:
    if not issues:
        st.success("No issues detected.")
        return
    for iss in issues:
        sev = iss.get("severity", "suggestion")
        icon = _severity_icon(sev)
        itype = iss.get("type", "?").title()
        original = iss.get("original", "")
        suggestion = iss.get("suggestion", "")
        explanation = iss.get("explanation", "")
        with st.expander(f"{icon} [{itype}] \"{original[:60]}{'…' if len(original) > 60 else ''}\""):
            col_a, col_b = st.columns(2)
            col_a.markdown(f"**Original:**  \n`{original}`")
            col_b.markdown(f"**Suggestion:**  \n`{suggestion}`")
            if explanation:
                st.caption(explanation)


def _render_style_tips(suggestions: list) -> None:
    if not suggestions:
        st.info("No additional style suggestions.")
        return
    for tip in suggestions:
        cat = tip.get("category", "General").title()
        suggestion = tip.get("suggestion", "")
        rationale = tip.get("rationale", "")
        with st.expander(f"✨ [{cat}] {suggestion[:80]}"):
            if rationale:
                st.caption(rationale)


def tab_grammar_proofreading(settings: dict) -> None:
    """
    Mode 6 — English Grammar Proofreading.

    The agent rewrites the user's text for clarity, fluency, and correctness
    suited to the chosen writing context. The polished text is the primary output
    with downloadable Markdown. A feedback loop allows iterative refinement.
    """
    st.header("Mode 6 — English Grammar Proofreading")
    st.markdown(
        """
Paste or upload any English text and get a **professionally polished version** — rewritten for
clarity, fluency, and correctness in your chosen writing context.

**Four writing contexts:** Academic · Professional Email · Formal · Informal
**What you get:** Polished rewrite · Detected issues · Style tips · Quality score · Feedback refinement
"""
    )

    st.divider()

    # ── Input section ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 1])

    with col_left:
        free_text = st.text_area(
            "Paste your text here",
            height=250,
            placeholder=(
                "Paste any English text — an email, essay, report, paragraph, or document excerpt. "
                "The agent will proofread and rewrite it to match your chosen writing context."
            ),
            key="grammar_free_text",
        )

    with col_right:
        style_label = st.selectbox(
            "Writing context",
            options=list(_STYLE_OPTIONS.keys()),
            index=0,
            help=(
                "Academic — journal papers, theses\n"
                "Professional Email — emails, memos, cover letters\n"
                "Formal — legal docs, official letters\n"
                "Informal — blogs, personal writing"
            ),
            key="grammar_style",
        )
        style_level = _STYLE_OPTIONS[style_label]

        selected_focus = st.multiselect(
            "Focus areas",
            options=_FOCUS_OPTIONS,
            default=[],
            help="Leave empty to cover all areas",
            key="grammar_focus",
        )
        focus_areas = [f.lower() for f in selected_focus]

    # ── File upload ───────────────────────────────────────────────────────────
    _grammar_types = get_supported_file_types(settings.get("use_docling", True))
    _fmt_label = ", ".join(t.upper() for t in _grammar_types)
    with st.expander(f"Or upload a document ({_fmt_label})", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload document",
            type=_grammar_types,
            accept_multiple_files=False,
            key="grammar_upload",
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.info(f"File ready: **{uploaded_files.name}**")

    # ── Resolve input text ────────────────────────────────────────────────────
    raw_text = free_text.strip()
    from_file = False
    source_filename = ""

    if uploaded_files and not raw_text:
        with st.spinner(f"Extracting text from {uploaded_files.name}…"):
            try:
                docs = process_uploads([uploaded_files], settings)
                if docs:
                    raw_text = docs[0].raw_text
                    from_file = True
                    source_filename = uploaded_files.name
            except Exception as e:
                st.error(f"Could not process file: {e}")
                return

    run_btn = st.button("Proofread", type="primary", use_container_width=True, key="grammar_run")

    if run_btn and not raw_text:
        st.warning("Please paste some text or upload a document first.")
        return

    if run_btn:
        # ── Session & pipeline ────────────────────────────────────────────────
        session_id = str(uuid.uuid4())

        initial_state = create_grammar_state(
            raw_text=raw_text,
            session_id=session_id,
            model_name=settings.get("model", ""),
            num_ctx=settings.get("num_ctx", 0),
            style_level=style_level,
            focus_areas=focus_areas,
        )

        st.divider()
        status_text = st.empty()
        progress_bar = st.progress(0)
        step_log_expander = st.expander("Step log", expanded=False)
        log_lines: list = []

        def stream_callback(node_name: str, state: dict) -> None:
            pct = state.get("progress_pct", 0)
            label = _NODE_LABELS.get(node_name, node_name)
            detail = state.get("status_detail", "")
            progress_bar.progress(pct)
            if detail:
                status_text.markdown(f"**{label}…** `{pct}%`  \n{detail}")
            else:
                status_text.markdown(f"**{label}…** `{pct}%`")
            entry = f"{label} ({pct}%)" + (f" — {detail}" if detail else "")
            log_lines.append(entry)
            with step_log_expander:
                st.text("\n".join(log_lines))

        start = time.time()
        try:
            final_state = run_grammar_check(initial_state, stream_callback=stream_callback)
        except Exception as e:
            st.error(f"Proofreading failed: {e}")
            logger.exception("Grammar check failed")
            return

        elapsed = time.time() - start
        progress_bar.progress(100)
        status_text.markdown(f"**Done.** Finished in `{elapsed:.1f}s`")

        # Save to memory
        try:
            mem = GrammarMemory()
            mem.save_result(session_id, dict(final_state))
        except Exception as e:
            logger.warning("Could not save grammar session: %s", e)

        # Persist everything needed for display across reruns (e.g. after feedback)
        st.session_state[_LAST_RUN_KEY] = {
            "final_state": dict(final_state),
            "raw_text": raw_text,
            "from_file": from_file,
            "source_filename": source_filename,
            "style_label": style_label,
            "session_id": session_id,
        }

        for err in final_state.get("errors", []):
            st.warning(err)

    elif _LAST_RUN_KEY not in st.session_state:
        return

    # ── Unpack persisted state (works on first run and on reruns after feedback) ─
    _last = st.session_state[_LAST_RUN_KEY]
    final_state = _last["final_state"]
    raw_text = _last["raw_text"]
    from_file = _last["from_file"]
    source_filename = _last["source_filename"]
    style_label = _last["style_label"]
    session_id = _last["session_id"]

    st.divider()

    # ── Output tabs ───────────────────────────────────────────────────────────
    # Always show the latest version: refined text takes priority over the original polished text
    fb_key = f"_fb_output_grammar_{session_id}"
    polished = st.session_state.get(fb_key) or final_state.get("polished_text", "")
    change_summary = final_state.get("change_summary", "")
    issues = final_state.get("issues_found", [])
    style_tips = final_state.get("style_suggestions", [])
    word_count = final_state.get("word_count", 0)

    t1, t2, t3, t4 = st.tabs([
        "📝 Polished Text",
        "🔍 Issues Found",
        "✨ Style Tips",
        "📊 Summary",
    ])

    with t1:
        if from_file and source_filename:
            st.caption(f"Source: {source_filename} — polished for **{style_label}** context")

        if polished:
            st.markdown(polished)
        else:
            st.info("No polished output was generated.")

        if polished:
            st.divider()
            dl_col1, dl_col2 = st.columns(2)
            out_name = source_filename.rsplit(".", 1)[0] if from_file else "proofread"
            dl_col1.download_button(
                "Download as Markdown",
                data=polished,
                file_name=f"{out_name}_polished.md",
                mime="text/markdown",
                key="grammar_dl_md",
            )
            dl_col2.download_button(
                "Download as TXT",
                data=polished,
                file_name=f"{out_name}_polished.txt",
                mime="text/plain",
                key="grammar_dl_txt",
            )

    with t2:
        issue_count = len(issues)
        if issue_count:
            severity_counts = {}
            for iss in issues:
                sev = iss.get("severity", "suggestion")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            cols = st.columns(len(severity_counts) + 1)
            cols[0].metric("Total Issues", issue_count)
            for col, (sev, cnt) in zip(cols[1:], severity_counts.items()):
                col.metric(sev.title() + "s", cnt)
            st.divider()

        if change_summary:
            st.subheader("What was changed")
            st.markdown(change_summary)
            st.divider()

        st.subheader("Detected issues")
        _render_issues_table(issues)

    with t3:
        _render_style_tips(style_tips)

    with t4:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Words", f"{word_count:,}")
        m2.metric("Issues Found", len(issues))
        m3.metric("Style Tips", len(style_tips))
        m4.metric("Writing Context", style_label)
        if final_state.get("sentence_count"):
            st.caption(f"Approx. {final_state['sentence_count']} sentences detected in original text.")

    # ── Quality metrics ───────────────────────────────────────────────────────
    render_eval_result(final_state.get("eval_result", {}), key_suffix=f"_grammar_{session_id}")
    render_rag_reflection(final_state.get("rag_reflection_info"), key_suffix=f"_grammar_{session_id}")

    # ── Feedback revision loop (unlimited rounds) ─────────────────────────────
    if polished:
        render_feedback_section(
            current_output=polished,
            session_key=f"grammar_{session_id}",
            mode="grammar_proofreading",
            model_name=settings.get("model", ""),
            num_ctx=settings.get("num_ctx", 0),
            context=raw_text,
            key_suffix=f"_grammar_{session_id}",
            max_rounds=None,
        )
