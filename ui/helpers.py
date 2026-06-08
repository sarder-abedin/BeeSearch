"""
ui/helpers.py
─────────────
Shared rendering helpers used across multiple UI tabs.

Covers: file upload processing, Socratic clarification form,
citation download buttons, reference lists, key findings, and report display.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from config.settings import get_settings
from tools.citation_tools import refs_to_bibtex, refs_to_ris
from tools.clarifier import generate_clarifying_questions
from tools.document_tools import DocumentProcessor, get_processor, _peek_pdf_pages

logger = logging.getLogger(__name__)
cfg = get_settings()

# ── Supported file-type lists ──────────────────────────────────────────────────

_BASE_FILE_TYPES = ["pdf", "docx", "doc", "txt", "md"]
_DOCLING_EXTRA_TYPES = ["pptx", "xlsx", "csv", "html", "png", "jpg", "jpeg"]


def get_supported_file_types(use_docling: bool = False) -> list:
    """Return the file extension list for st.file_uploader's type= parameter."""
    types = list(_BASE_FILE_TYPES)
    if use_docling:
        types.extend(_DOCLING_EXTRA_TYPES)
    return types


# ── File helpers ───────────────────────────────────────────────────────────────

def save_upload(uploaded_file) -> Path:
    """Write a Streamlit UploadedFile to a temp file; return its path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def process_uploads(uploaded_files, settings: dict) -> list:
    """Process a list of Streamlit uploaded files into ProcessedDocuments.

    Routes to DoclingProcessor when settings["use_docling"] is True, otherwise
    uses the fast pdfplumber-based DocumentProcessor.

    Streamlit passes DeletedFile sentinel objects when files are removed from
    the uploader widget.  These have no .name / .getbuffer(), so we skip them.
    """
    use_docling = settings.get("use_docling", True)
    use_ocr = settings.get("use_ocr", False)
    large_doc_threshold = settings.get("large_doc_page_threshold", cfg.large_doc_page_threshold)

    if use_docling:
        ocr_label = " + OCR" if use_ocr else ""
        st.caption(f"Using Docling{ocr_label} for advanced parsing (pdfplumber fallback for PDFs > {large_doc_threshold} pages)…")

    processed = []
    for f in uploaded_files:
        if not hasattr(f, "name") or not hasattr(f, "getbuffer"):
            continue
        try:
            file_obj = io.BytesIO(f.getbuffer())

            # For large PDFs, auto-fall back to pdfplumber to avoid loading
            # Docling's ML models (~500 MB) into RAM for long documents.
            effective_docling = use_docling
            if use_docling and Path(f.name).suffix.lower() == ".pdf":
                page_count = _peek_pdf_pages(file_obj)
                file_obj.seek(0)
                if page_count > large_doc_threshold:
                    effective_docling = False
                    st.info(
                        f"{f.name}: {page_count} pages — switching to lightweight parser "
                        f"(threshold: {large_doc_threshold} pages)"
                    )

            processor = get_processor(
                use_docling=effective_docling,
                use_ocr=use_ocr,
                chunk_size=settings.get("chunk_size", 800),
                overlap=settings.get("chunk_overlap", 150),
                max_raw_chars=200_000,
                max_pages=300,
            )
            doc = processor.process_file(Path(f.name), file_obj=file_obj)
            processed.append(doc)
            st.success(f"{f.name} — {doc.total_chunks} chunks extracted")
        except Exception as e:
            st.error(f"Failed to process {getattr(f, 'name', repr(f))}: {e}")
    return processed


# ── Socratic clarification form ────────────────────────────────────────────────

_OTHER_OPTION = "Other (please specify)"


def render_clarification_form(mode_key: str, goal: str, settings: dict) -> dict:
    """Render Socratic clarification questions as multiple-choice radio widgets.

    Each question shows 3–4 preset options plus an 'Other' fallback for custom
    input. The agent's recommended answer is pre-selected and shown as a hint.

    Returns the answers dict (may be empty if questions not yet generated).
    Call AFTER the goal text_area. Manages session_state keys internally.
    """
    q_key = f"clarify_q_{mode_key}"
    g_key = f"clarify_goal_{mode_key}"

    existing_questions = st.session_state.get(q_key, [])
    stored_goal = st.session_state.get(g_key, "")

    if existing_questions and stored_goal != goal.strip():
        del st.session_state[q_key]
        existing_questions = []

    answers: dict = {}

    if existing_questions:
        st.markdown("##### Clarify Your Requirements")
        st.caption(
            "Select the option that best fits — or choose **Other** to enter your own answer."
        )
        for q in existing_questions:
            key = q.get("key", "q")
            field_key = f"clarify_ans_{mode_key}_{key}"
            other_key = f"clarify_other_{mode_key}_{key}"

            options: list = list(q.get("options") or [])
            recommended: str = q.get("recommended", "")

            # Default to recommended index; fall back to first option
            default_idx = options.index(recommended) if recommended in options else 0

            st.markdown(f"**{q['question']}**")
            if recommended:
                st.caption(f"Suggested: **{recommended}**")

            selected = st.radio(
                "",
                options + [_OTHER_OPTION],
                index=default_idx,
                key=field_key,
                label_visibility="collapsed",
            )

            if selected == _OTHER_OPTION:
                custom = st.text_input(
                    "Please specify:",
                    key=other_key,
                    placeholder="Enter your answer…",
                )
                answers[key] = custom.strip()
            else:
                answers[key] = selected

        if st.button("Reset questions", key=f"clarify_reset_{mode_key}"):
            del st.session_state[q_key]
            if g_key in st.session_state:
                del st.session_state[g_key]
            st.rerun()
    else:
        if st.button(
            "Clarify Requirements",
            key=f"clarify_btn_{mode_key}",
            help="Generate 2–3 focused questions to sharpen the output.",
        ):
            if not goal.strip():
                st.warning("Please enter your goal first, then clarify requirements.")
            else:
                with st.spinner("Generating clarifying questions…"):
                    qs = generate_clarifying_questions(
                        goal=goal,
                        mode=mode_key,
                        model_name=settings.get("model", cfg.ollama_model),
                        ollama_base_url=cfg.ollama_base_url,
                        num_ctx=min(settings.get("num_ctx", cfg.num_ctx), 4096),
                    )
                st.session_state[q_key] = qs
                st.session_state[g_key] = goal.strip()
                st.rerun()

    return answers


# ── Citation export ────────────────────────────────────────────────────────────

def render_citation_downloads(references: list, key_suffix: str = "") -> None:
    """Render BibTeX and RIS download buttons side by side."""
    if not references:
        return
    col_bib, col_ris = st.columns(2)
    with col_bib:
        st.download_button(
            label="Export BibTeX (.bib)",
            data=refs_to_bibtex(references),
            file_name=f"references{key_suffix}.bib",
            mime="text/plain",
            use_container_width=True,
            key=f"bib_{key_suffix}",
        )
    with col_ris:
        st.download_button(
            label="Export RIS (.ris)",
            data=refs_to_ris(references),
            file_name=f"references{key_suffix}.ris",
            mime="text/plain",
            use_container_width=True,
            key=f"ris_{key_suffix}",
        )


# ── Reference list ─────────────────────────────────────────────────────────────

def render_references(references: list, key_suffix: str = "") -> None:
    """Render the reference list with citation export + expandable entries."""
    if not references:
        st.info("No academic references found for this run.")
        return

    st.subheader(f"References ({len(references)})")
    render_citation_downloads(references, key_suffix=key_suffix)
    st.divider()

    _source_label = {
        "arxiv": "arXiv preprint",
        "semantic_scholar": "Peer-reviewed",
        "crossref": "CrossRef",
    }

    for ref in references:
        with st.expander(f"[{ref['ref_num']}] {ref['title'][:80]}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Authors:** {'; '.join(ref['authors'][:5])}")
                st.markdown(f"**Journal/Venue:** {ref.get('journal') or 'N/A'}")
                st.markdown(f"**Year:** {ref.get('year') or 'N/A'}")
                if ref.get("doi"):
                    st.markdown(f"**DOI:** [{ref['doi']}](https://doi.org/{ref['doi']})")
                if ref.get("url"):
                    st.markdown(f"**URL:** [{ref['url'][:50]}]({ref['url']})")
                if ref.get("abstract_snippet"):
                    st.markdown(f"**Abstract:** _{ref['abstract_snippet']}_")
            with col2:
                source = _source_label.get(ref.get("source", ""), "Unknown")
                st.caption(f"**Source:** {source}")
                if ref.get("citation_count") is not None:
                    st.metric("Citations", ref["citation_count"])
            st.code(ref["apa"], language=None)


# ── Quality evaluation ────────────────────────────────────────────────────────

def render_eval_result(eval_result: dict, key_suffix: str = "") -> None:
    """Render quality evaluation scores as compact metrics in an expander."""
    if not eval_result or not eval_result.get("overall"):
        return

    overall = eval_result.get("overall", 0)
    summary = eval_result.get("summary", "")

    if overall >= 4:
        score_label = f"Quality Score: {overall}/5 — Good"
    elif overall >= 3:
        score_label = f"Quality Score: {overall}/5 — Adequate"
    else:
        score_label = f"Quality Score: {overall}/5 — Needs improvement"

    with st.expander(
        f"{score_label} — {summary[:60]}{'…' if len(summary) > 60 else ''}",
        expanded=False,
    ):
        dimension_keys = [
            k for k in eval_result
            if k not in ("overall", "summary", "ragchecker_faithfulness")
        ]
        if dimension_keys:
            cols = st.columns(len(dimension_keys) + 1)
            for col, key in zip(cols, dimension_keys):
                col.metric(key.replace("_", " ").title(), f"{eval_result[key]}/5")
            cols[-1].metric("Overall", f"{overall}/5")
        else:
            st.metric("Overall", f"{overall}/5")
        if summary:
            st.caption(summary)

        # RAGchecker faithfulness section
        rag_faith = eval_result.get("ragchecker_faithfulness")
        if rag_faith and not rag_faith.get("skipped") and rag_faith.get("faithfulness_score") is not None:
            score = rag_faith["faithfulness_score"]
            pct = int(score * 100)
            checked = rag_faith.get("checked_claims", 0)
            supported = rag_faith.get("supported_claims", 0)
            st.markdown(
                f"**RAGchecker Faithfulness: {pct}%** — "
                f"{supported}/{checked} claims supported by sources"
            )


def render_rag_reflection(rag_reflection_info, key_suffix: str = "") -> None:
    """Render self-reflective RAG metadata in a collapsed expander."""
    if not rag_reflection_info:
        return

    # Normalise: chunk-based modes return List[Dict], paper-based return Dict
    if isinstance(rag_reflection_info, list):
        entries = rag_reflection_info
    else:
        entries = [rag_reflection_info]

    total_retrieved = sum(e.get("total_retrieved", e.get("papers_retrieved", 0)) for e in entries)
    total_relevant = sum(e.get("total_relevant", e.get("papers_after_grading", 0)) for e in entries)
    if total_retrieved == 0:
        return

    pct = int(100 * total_relevant / total_retrieved) if total_retrieved else 0
    label = f"Self-Reflective RAG — {total_relevant}/{total_retrieved} items passed grading ({pct}%)"

    with st.expander(label, expanded=False):
        cols = st.columns(3)
        cols[0].metric("Retrieved", total_retrieved)
        cols[1].metric("Relevant", total_relevant)
        cols[2].metric("Pass Rate", f"{pct}%")

        for i, entry in enumerate(entries):
            cycles = entry.get("cycles")
            skipped = entry.get("grading_skipped", False)
            rewritten = entry.get("rewritten_queries", [])
            query = entry.get("query", "")

            details = []
            if query:
                details.append(f"**Query:** {query[:120]}{'…' if len(query) > 120 else ''}")
            if cycles is not None:
                details.append(f"**Cycles:** {cycles}")
            if rewritten:
                details.append(f"**Query rewritten:** {rewritten[0][:80]}{'…' if len(rewritten[0]) > 80 else ''}")
            if skipped:
                details.append("_Grading skipped (all items returned true — silent LLM failure)_")

            if details and len(entries) > 1:
                st.markdown(f"**Query {i + 1}:** " + " · ".join(details))
            elif details:
                st.markdown("  \n".join(details))


# ── Key findings + report ──────────────────────────────────────────────────────

def render_key_findings(findings: list) -> None:
    if not findings:
        return
    st.subheader("Key Findings")
    for i, f in enumerate(findings, 1):
        st.markdown(f"**{i}.** {f}")


def render_report(report: str, session_id: str) -> None:
    """Display the report with a Markdown download button."""
    st.subheader("Full Research Report")
    st.markdown(report, unsafe_allow_html=False)

    out_path = Path(cfg.output_dir) / f"report_{session_id}.md"
    out_path.write_text(report, encoding="utf-8")

    st.download_button(
        label="Download Report (Markdown)",
        data=report,
        file_name=f"research_report_{session_id}.md",
        mime="text/markdown",
    )


# ── Feedback & Refinement ─────────────────────────────────────────────────────

def render_feedback_section(
    current_output: str,
    session_key: str,
    mode: str,
    model_name: str,
    num_ctx: int,
    context: str = "",
    key_suffix: str = "",
    max_rounds: int | None = 3,
) -> str:
    """
    Render a feedback input section below any mode's output.

    Stores refinement history in st.session_state (ephemeral — not persisted to disk).
    Returns the current (possibly refined) output string.

    Parameters
    ----------
    current_output : The output text currently shown to the user
    session_key    : Unique key for this result (typically session_id)
    mode           : Mode name for refine_with_feedback() prompt customisation
    model_name     : Ollama model name
    num_ctx        : LLM context window size
    context        : Extra context injected into the refinement prompt (not shown to user)
    key_suffix     : Added to all widget keys to avoid Streamlit key collisions
    max_rounds     : Maximum refinement rounds. None = unlimited (default = MAX_FEEDBACK_ROUNDS).
    """
    from agents.feedback_agent import refine_with_feedback, make_feedback_entry, MAX_FEEDBACK_ROUNDS

    history_key = f"_fb_history_{session_key}"
    output_key = f"_fb_output_{session_key}"

    if history_key not in st.session_state:
        st.session_state[history_key] = []
    if output_key not in st.session_state:
        st.session_state[output_key] = current_output

    history: list = st.session_state[history_key]
    refined_output: str = st.session_state[output_key]
    round_num: int = len(history)

    st.divider()

    limit_reached = max_rounds is not None and round_num >= max_rounds
    round_label = f"round {round_num + 1}" + (f" / {max_rounds}" if max_rounds is not None else "")

    if limit_reached:
        st.info(
            f"Maximum {max_rounds} refinements reached for this session. "
            "Start a new session to continue refining."
        )
    else:
        with st.expander(
            f"Refine this output  ({round_label})",
            expanded=(round_num == 0),
        ):
            feedback_text = st.text_area(
                "Your feedback:",
                placeholder=(
                    "e.g. 'Make the methodology section more detailed', "
                    "'Add more focus on climate adaptation', "
                    "'Simplify the language for a non-specialist audience'"
                ),
                height=100,
                key=f"_fb_input_{key_suffix}",
            )
            col_btn, col_hint = st.columns([1, 3])
            with col_btn:
                clicked = st.button(
                    "Refine",
                    key=f"_fb_btn_{key_suffix}",
                    type="primary",
                    disabled=not feedback_text.strip(),
                    use_container_width=True,
                )
            with col_hint:
                if max_rounds is not None:
                    remaining = max_rounds - round_num
                    st.caption(f"{remaining} refinement(s) remaining in this session.")

            if clicked and feedback_text.strip():
                with st.spinner(f"Refining ({round_label})…"):
                    new_output = refine_with_feedback(
                        original_output=refined_output,
                        feedback=feedback_text.strip(),
                        context=context,
                        mode=mode,
                        model_name=model_name,
                        num_ctx=num_ctx,
                    )
                entry = make_feedback_entry(round_num + 1, feedback_text.strip(), refined_output)
                st.session_state[history_key].append(entry)
                st.session_state[output_key] = new_output
                st.rerun()

    if history:
        with st.expander(f"Revision history  ({len(history)} round(s))", expanded=False):
            for rev in reversed(history):
                st.markdown(f"**Round {rev['round']} feedback:** _{rev['feedback']}_")
                with st.expander(
                    f"Previous output before round {rev['round']}", expanded=False
                ):
                    preview = rev["previous_output"]
                    st.markdown(preview[:2000] + ("…" if len(preview) > 2000 else ""))
                st.divider()

    return st.session_state[output_key]


# ── Grammar check gate (optional pre-submission query correction) ──────────────

def _run_grammar_check(text: str, settings: dict) -> dict:
    from tools.grammar_check import check_and_fix_grammar
    return check_and_fix_grammar(
        text, model_name=settings.get("model", ""), num_ctx=settings.get("num_ctx", 8192),
    )


def _render_grammar_suggestion(cached: dict, *, key: str) -> None:
    """Renders the original-vs-suggested comparison and Accept / Keep / Edit controls.
    Mutates `cached["final"]` in st.session_state and reruns once the user decides."""
    st.info("✏️ **Suggested grammar fixes** — review below, then choose how to proceed.")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Your version")
        st.markdown(f"> {cached['source']}")
    with c2:
        st.caption("Suggested correction")
        st.markdown(f"> {cached['corrected']}")

    edit_flag = f"_gc_editing_{key}"
    b1, b2, b3 = st.columns(3)
    if b1.button("✓ Use suggested", key=f"_gc_accept_{key}", use_container_width=True):
        cached["final"] = cached["corrected"]
        st.session_state.pop(edit_flag, None)
        st.rerun()
    if b2.button("Keep my original", key=f"_gc_keep_{key}", use_container_width=True):
        cached["final"] = cached["source"]
        st.session_state.pop(edit_flag, None)
        st.rerun()
    if b3.button("✏️ Edit it myself", key=f"_gc_editbtn_{key}", use_container_width=True):
        st.session_state[edit_flag] = True

    if st.session_state.get(edit_flag):
        edited = st.text_area(
            "Your edited version", value=cached["corrected"], key=f"_gc_editta_{key}", height=100,
        )
        if st.button("Use this edited version", key=f"_gc_editconfirm_{key}", type="primary"):
            cached["final"] = edited.strip()
            st.session_state.pop(edit_flag, None)
            st.rerun()


def render_query_gate(raw_text: str, *, key: str, settings: dict) -> tuple[str, bool]:
    """
    Optional grammar-check gate for a text_input/text_area query, paired with a
    "Run"-style action button elsewhere on the page.

    Renders a small radio toggle ("Use as typed" / "Check grammar before
    running"). When checking is enabled, runs a quick LLM grammar/spelling pass
    the first time it sees this exact text, and shows an inline review with
    three resolutions: accept the suggestion, keep the original, or hand-edit it.

    Returns ``(text_to_submit, ready)``:
      - ``ready=True``  → safe to proceed; ``text_to_submit`` is what should be
        processed (the original if checking is off / text unchanged / user kept
        it, or the user-approved/edited correction otherwise).
      - ``ready=False`` → a suggestion is awaiting the user's decision; the
        caller's action should be held off (e.g. show "resolve the suggestion
        above, then click Run again").

    `key` must be unique per field (used to namespace widgets and cached state).
    """
    mode = st.radio(
        "Grammar check",
        options=["as_typed", "check_fix"],
        format_func=lambda m: "Use as typed" if m == "as_typed" else "✓ Check grammar before running",
        horizontal=True,
        label_visibility="collapsed",
        key=f"_gc_mode_{key}",
    )
    if mode == "as_typed" or not raw_text.strip():
        return raw_text, True

    state_key = f"_gc_state_{key}"
    cached = st.session_state.get(state_key)

    if cached is None or cached["source"] != raw_text:
        with st.spinner("Checking grammar…"):
            result = _run_grammar_check(raw_text, settings)
        if not result["changed"]:
            st.session_state[state_key] = {"source": raw_text, "corrected": raw_text, "final": raw_text}
            return raw_text, True
        st.session_state[state_key] = {"source": raw_text, "corrected": result["corrected"], "final": None}
        cached = st.session_state[state_key]

    if cached["final"] is not None:
        if cached["final"] != cached["source"]:
            st.caption(f"✓ Using corrected version: _{cached['final'][:140]}{'…' if len(cached['final']) > 140 else ''}_")
        return cached["final"], True

    _render_grammar_suggestion(cached, key=key)
    return raw_text, False


def render_chat_gate(raw_message: str | None, *, key: str, settings: dict) -> str | None:
    """
    Optional grammar-check gate for a chat-style input (st.chat_input / pending
    follow-up question). Renders a small radio toggle once per call; pass it
    whatever the chat input returned (often None).

    When checking is enabled and a fresh message arrives, the message is held
    back, quickly grammar-checked, and shown for the user's approval (accept /
    keep original / hand-edit) before it is sent into the conversation.

    Returns the message that should be processed THIS run, or None if nothing
    should be sent yet (checking disabled + no message, or a suggestion is still
    awaiting the user's decision).
    """
    mode = st.radio(
        "Grammar check",
        options=["as_typed", "check_fix"],
        format_func=lambda m: "Send as typed" if m == "as_typed" else "✓ Check grammar before sending",
        horizontal=True,
        label_visibility="collapsed",
        key=f"_gc_mode_{key}",
    )
    state_key = f"_gc_chatstate_{key}"
    cached = st.session_state.get(state_key)

    if mode == "as_typed":
        if cached:
            st.session_state.pop(state_key, None)
            return cached["source"]
        return raw_message if (raw_message and raw_message.strip()) else None

    # mode == "check_fix"
    if raw_message and raw_message.strip():
        st.session_state[state_key] = {"source": raw_message.strip(), "corrected": None, "final": None}
        cached = st.session_state[state_key]

    if not cached:
        return None

    if cached["final"] is not None:
        final = cached["final"]
        st.session_state.pop(state_key, None)
        return final

    if cached["corrected"] is None:
        with st.spinner("Checking grammar…"):
            result = _run_grammar_check(cached["source"], settings)
        if not result["changed"]:
            st.session_state.pop(state_key, None)
            return cached["source"]
        cached["corrected"] = result["corrected"]

    _render_grammar_suggestion(cached, key=key)
    return None
