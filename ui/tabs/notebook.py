"""ui/tabs/notebook.py — Mode 8: Research Notebook (NotebookLM-style grounded Q&A)"""

from __future__ import annotations

import logging

import streamlit as st

from agents.notebook_graph import run_notebook_turn
from agents.notebook_memory import NotebookMemory
from agents.notebook_state import create_notebook_state
from config.settings import get_settings
from tools.hybrid_store import _stores as _hybrid_stores
from ui.glossary import render_glossary_expander, term_help
from ui.helpers import (
    get_supported_file_types, process_uploads, render_eval_result, render_rag_reflection,
    render_query_gate, render_chat_gate,
)

logger = logging.getLogger(__name__)
cfg = get_settings()


# ── Ingestion helpers ───────────────────────────────────────────────────────────

def _index_and_store(notebook_id: str, processed_docs: list, settings: dict,
                     source_type: str = "file", url: str = "") -> int:
    """
    Persist processed docs to NotebookMemory (JSON) and evict the cached
    in-memory store so it is rebuilt fresh at the next query.

    Embedding and FAISS indexing are deferred to query time (retrieve_node)
    to avoid OOM during file addition in memory-constrained environments.
    Returns the number of new sources actually added.
    """
    if not processed_docs:
        return 0

    mem = NotebookMemory()
    added = 0
    for doc in processed_docs:
        if mem.add_source(notebook_id, doc, source_type=source_type, url=url):
            added += 1

    if added:
        # Evict the stale in-memory store so retrieve_node rebuilds with new chunks.
        _hybrid_stores.pop(f"notebook_{notebook_id}", None)
        _hybrid_stores.pop(f"notebook_{notebook_id}_bm25", None)

    return added


def _render_citations(citations: list) -> None:
    """Render a compact citation list under an assistant answer."""
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for c in citations:
            page = c.get("page", 0)
            url = c.get("url", "")
            if url:
                page_label = f"[{c.get('doc_name', url)[:60]}]({url})"
                st.markdown(f"**[{c.get('n')}]** {page_label}")
            else:
                page_label = f"p.{page}" if isinstance(page, int) and page >= 0 else "n/a"
                st.markdown(f"**[{c.get('n')}] {c.get('doc_name', 'unknown')}** · {page_label}")
            snippet = c.get("snippet", "")
            if snippet:
                st.caption(snippet)


# ── Advanced-feature tab helpers ────────────────────────────────────────────────

def _gen_button(label: str, key: str, cache_key: str, settings: dict,
                notebook_id: str, fn, *fn_args):
    """
    Render a Generate / Regenerate button pair.  Calls `fn(notebook_id,
    *fn_args, settings)` and stores the result under `cache_key` in
    session_state.  Returns (result_or_None, error_str).
    """
    col_gen, col_clr = st.columns([4, 1])
    btn_label = "Regenerate" if cache_key in st.session_state else label
    if col_gen.button(btn_label, key=key, type="primary", use_container_width=True):
        with st.spinner("Working…"):
            result, err = fn(notebook_id, *fn_args, settings)
        if err:
            st.error(err)
            return None, err
        st.session_state[cache_key] = result
    if col_clr.button("Clear", key=f"{key}_clr"):
        st.session_state.pop(cache_key, None)
        st.rerun()
    return st.session_state.get(cache_key), ""


def _docx_pdf_buttons(markdown_text: str, base_name: str, key_prefix: str) -> None:
    """Render DOCX and PDF download buttons for a markdown text block."""
    try:
        from tools.export_tools import build_docx, build_pdf
        cols = st.columns(2)
        try:
            docx_bytes = build_docx(markdown_text, [])
            cols[0].download_button(
                "Download .docx",
                data=docx_bytes,
                file_name=f"{base_name}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"{key_prefix}_docx",
            )
        except Exception as e:
            cols[0].caption(f"DOCX unavailable: {e}")
        try:
            pdf_bytes = build_pdf(markdown_text, [])
            cols[1].download_button(
                "Download .pdf",
                data=pdf_bytes,
                file_name=f"{base_name}.pdf",
                mime="application/pdf",
                key=f"{key_prefix}_pdf",
            )
        except Exception as e:
            cols[1].caption(f"PDF unavailable: {e}")
    except ImportError:
        st.caption("DOCX/PDF export not available — install reportlab and python-docx.")


def _dot_export_buttons(dot_string: str, base_name: str, key_prefix: str) -> None:
    """Render DOT, PNG, SVG download buttons for a graphviz DOT string."""
    from agents.notebook_advanced import render_dot_bytes
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download .dot",
        data=dot_string,
        file_name=f"{base_name}.dot",
        mime="text/plain",
        key=f"{key_prefix}_dot",
    )
    png, png_err = render_dot_bytes(dot_string, "png")
    if not png_err:
        c2.download_button(
            "Download .png",
            data=png,
            file_name=f"{base_name}.png",
            mime="image/png",
            key=f"{key_prefix}_png",
        )
    else:
        c2.caption("PNG unavailable")
    svg, svg_err = render_dot_bytes(dot_string, "svg")
    if not svg_err:
        c3.download_button(
            "Download .svg",
            data=svg,
            file_name=f"{base_name}.svg",
            mime="image/svg+xml",
            key=f"{key_prefix}_svg",
        )
    else:
        c3.caption("SVG unavailable")


def _render_section_qa(
    active_id: str, doc_id: str, section_idx: int,
    section_title: str, section_chunks: list, level: str, settings: dict,
) -> None:
    """Q&A interface scoped to a single section, inside an expander."""
    from agents.section_summary import answer_section_question
    qa_hist_key = f"nb_sec_qa_{active_id}_{doc_id}_{section_idx}"
    qa_history = st.session_state.get(qa_hist_key, [])

    st.caption("Ask a follow-up question about this section:")
    for turn in qa_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    q_key = f"nb_sec_q_{active_id}_{doc_id}_{section_idx}_{len(qa_history)}"
    q_input = st.text_input(
        "Question",
        key=q_key,
        placeholder=f"Ask about '{section_title[:40]}'…",
        label_visibility="collapsed",
    )
    if st.button("Ask", key=f"nb_sec_ask_{active_id}_{doc_id}_{section_idx}",
                 type="secondary"):
        if q_input.strip():
            with st.spinner("Thinking…"):
                answer = answer_section_question(
                    section_title, section_chunks, q_input.strip(), level,
                    qa_history,
                    settings.get("model", cfg.ollama_model),
                    settings.get("num_ctx", cfg.num_ctx),
                )
            st.session_state[qa_hist_key] = qa_history + [
                {"role": "user", "content": q_input.strip()},
                {"role": "assistant", "content": answer},
            ]
            st.session_state[f"nb_sec_last_qa_{active_id}_{doc_id}"] = section_idx
            st.rerun()


def _render_section_breakdown(
    active_id: str, doc_id: str, level: str,
    breakdown: list, review: list | None, settings: dict,
) -> None:
    """Render per-section expanders: summary, claim questions, Q&A, expert review."""
    last_qa_idx = st.session_state.get(f"nb_sec_last_qa_{active_id}_{doc_id}")
    n = len(breakdown)
    st.caption(
        f"{n} section{'s' if n != 1 else ''} detected · "
        f"{level.title()} level"
        + (" · Expert review available" if review else "")
    )

    for idx, sec in enumerate(breakdown):
        title = sec["title"]
        summary = sec["summary"]
        claim_questions = sec.get("claim_questions", [])
        sec_chunks = sec["chunks"]
        expanded = (idx == 0) or (idx == last_qa_idx)

        with st.expander(f"**{title}**", expanded=expanded):
            # ── Plain-language summary ────────────────────────────────────
            st.markdown(summary)

            # ── Claim questions (auto-generated) ─────────────────────────
            if claim_questions:
                st.divider()
                st.markdown("**Critical Questions — unclear or unsupported claims:**")
                qa_hist_key = f"nb_sec_qa_{active_id}_{doc_id}_{idx}"
                from agents.section_summary import answer_section_question
                for qi, q in enumerate(claim_questions):
                    if st.button(
                        q,
                        key=f"nb_sec_cq_{active_id}_{doc_id}_{idx}_{qi}",
                        use_container_width=True,
                    ):
                        qa_history = st.session_state.get(qa_hist_key, [])
                        with st.spinner("Answering…"):
                            answer = answer_section_question(
                                title, sec_chunks, q, level, qa_history,
                                settings.get("model", cfg.ollama_model),
                                settings.get("num_ctx", cfg.num_ctx),
                            )
                        st.session_state[qa_hist_key] = qa_history + [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": answer},
                        ]
                        st.session_state[f"nb_sec_last_qa_{active_id}_{doc_id}"] = idx
                        st.rerun()

            # ── Expert review ─────────────────────────────────────────────
            if review and idx < len(review):
                rev = review[idx]
                st.divider()
                st.markdown("**Expert Review**")
                col_s, col_w = st.columns(2)
                with col_s:
                    st.markdown("**Strengths**")
                    st.markdown(rev.get("strengths") or "—")
                with col_w:
                    st.markdown("**Weaknesses**")
                    st.markdown(rev.get("weaknesses") or "—")
                col_l, col_i = st.columns(2)
                with col_l:
                    st.markdown("**Limitations**")
                    st.markdown(rev.get("limitations") or "—")
                with col_i:
                    st.markdown("**How to Improve**")
                    st.markdown(rev.get("improvements") or "—")

            # ── User Q&A ──────────────────────────────────────────────────
            st.divider()
            _render_section_qa(
                active_id, doc_id, idx, title, sec_chunks, level, settings,
            )


def _tab_cross_summary(active_id: str, notebook: dict, settings: dict) -> None:
    from agents.notebook_advanced import generate_cross_document_summary
    st.markdown(
        "Synthesizes **all** notebook sources into a unified markdown summary "
        "covering common themes, complementary contributions, contradictions, "
        "and key takeaways."
    )
    cache_key = f"nb_summary_{active_id}"
    result, _ = _gen_button(
        "Generate Summary", f"nb_gen_summary_{active_id}", cache_key,
        settings, active_id, generate_cross_document_summary,
    )
    if result:
        st.markdown(result)
        nb_name = notebook.get("name", "notebook")
        st.download_button(
            "Download (.md)",
            data=result,
            file_name=f"summary_{nb_name}.md",
            mime="text/markdown",
            key=f"nb_dl_summary_{active_id}",
        )
        _docx_pdf_buttons(result, f"summary_{nb_name}", f"nb_summary_{active_id}")

    # ── Section-by-Section Breakdown ─────────────────────────────────────────
    st.divider()
    st.markdown("#### Section-by-Section Breakdown")
    st.caption(
        "Select one source to drill into its structure section by section. "
        "Each section gets a plain-language summary, auto-generated critical "
        "questions about unclear or unsupported claims, an interactive Q&A, "
        "and optional expert reviewer feedback."
    )

    sources = notebook.get("sources", [])
    if not sources:
        st.info("Add at least one source to enable section breakdown.")
        return

    src_options = {s["filename"]: s["doc_id"] for s in sources}
    chosen_filename = st.selectbox(
        "Select source to break down",
        list(src_options.keys()),
        key=f"nb_sec_src_{active_id}",
    )
    chosen_doc_id = src_options[chosen_filename]

    level = st.radio(
        "Explanation level",
        options=["novice", "intermediate", "expert"],
        format_func=lambda x: {
            "novice": "Novice",
            "intermediate": "Intermediate",
            "expert": "Expert",
        }[x],
        index=1,
        horizontal=True,
        key=f"nb_sec_level_{active_id}",
    )

    breakdown_key = f"nb_sec_{active_id}_{chosen_doc_id}_{level}"
    review_key = f"nb_sec_rev_{active_id}_{chosen_doc_id}"
    breakdown = st.session_state.get(breakdown_key)
    review = st.session_state.get(review_key)

    col_gen, col_rev, col_clr = st.columns([3, 3, 1])

    # ── Generate / regenerate section breakdown ───────────────────────────────
    if col_gen.button(
        "Regenerate Breakdown" if breakdown else "Generate Section Breakdown",
        key=f"nb_sec_gen_{active_id}",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.pop(breakdown_key, None)
        st.session_state.pop(review_key, None)

        from agents.section_summary import (
            detect_sections_hybrid,
            generate_section_claim_questions,
            get_doc_chunks,
            summarize_section,
        )
        doc_chunks = get_doc_chunks(notebook, chosen_doc_id)
        if not doc_chunks:
            st.warning("No content found for this source.")
        else:
            with st.spinner("Detecting document sections…"):
                sections = detect_sections_hybrid(
                    doc_chunks,
                    model_name=settings.get("model", cfg.ollama_model),
                    num_ctx=settings.get("num_ctx", cfg.num_ctx),
                )

            result = []
            progress = st.progress(0, text="Analysing sections…")
            n_sec = len(sections)
            for i, (title, sec_chunks) in enumerate(sections):
                frac = (i + 1) / n_sec
                progress.progress(frac, text=f"[{i+1}/{n_sec}] {title[:50]}…")
                summary = summarize_section(
                    title, sec_chunks, level,
                    settings.get("model", cfg.ollama_model),
                    settings.get("num_ctx", cfg.num_ctx),
                )
                claim_qs = generate_section_claim_questions(
                    title, sec_chunks,
                    settings.get("model", cfg.ollama_model),
                    settings.get("num_ctx", cfg.num_ctx),
                )
                result.append({
                    "title": title,
                    "summary": summary,
                    "claim_questions": claim_qs,
                    "chunks": sec_chunks,
                })
            progress.empty()
            st.session_state[breakdown_key] = result
            st.rerun()

    # ── Generate / regenerate expert review ───────────────────────────────────
    rev_disabled = breakdown is None
    if col_rev.button(
        "Regenerate Expert Review" if review else "Generate Expert Review",
        key=f"nb_sec_rev_gen_{active_id}",
        use_container_width=True,
        disabled=rev_disabled,
        help="Generate section breakdown first, then expert review becomes available.",
    ):
        st.session_state.pop(review_key, None)
        current_breakdown = st.session_state.get(breakdown_key, [])
        from agents.section_summary import review_section
        review_result = []
        progress = st.progress(0, text="Generating expert reviews…")
        n_sec = len(current_breakdown)
        for i, sec in enumerate(current_breakdown):
            progress.progress(
                (i + 1) / n_sec,
                text=f"[{i+1}/{n_sec}] Reviewing: {sec['title'][:50]}…",
            )
            rev = review_section(
                sec["title"], sec["chunks"],
                settings.get("model", cfg.ollama_model),
                settings.get("num_ctx", cfg.num_ctx),
            )
            review_result.append(rev)
        progress.empty()
        st.session_state[review_key] = review_result
        st.rerun()

    if col_clr.button("Clear", key=f"nb_sec_clr_{active_id}"):
        st.session_state.pop(breakdown_key, None)
        st.session_state.pop(review_key, None)
        st.rerun()

    # ── Render results ────────────────────────────────────────────────────────
    breakdown = st.session_state.get(breakdown_key)
    review = st.session_state.get(review_key)
    if breakdown:
        _render_section_breakdown(
            active_id, chosen_doc_id, level, breakdown, review, settings,
        )


def _tab_faq(active_id: str, notebook: dict, settings: dict) -> None:
    st.markdown(
        "Auto-generates frequently asked questions with grounded answers "
        "drawn from your notebook sources."
    )
    cache_key = f"nb_faq_{active_id}"
    from agents.notebook_advanced import generate_faq

    n_q = st.slider("Number of questions", 4, 16, 8, key=f"nb_faq_n_{active_id}")

    col_gen, col_clr = st.columns([4, 1])
    btn_label = "Regenerate FAQ" if cache_key in st.session_state else "Generate FAQ"
    if col_gen.button(btn_label, key=f"nb_gen_faq_{active_id}", type="primary",
                      use_container_width=True):
        with st.spinner("Generating FAQ…"):
            items, err = generate_faq(active_id, settings, n_questions=n_q)
        if err:
            st.error(err)
        else:
            st.session_state[cache_key] = items
    if col_clr.button("Clear", key=f"nb_clr_faq_{active_id}"):
        st.session_state.pop(cache_key, None)
        st.rerun()

    items = st.session_state.get(cache_key)
    if items:
        src_names = [s["filename"] for s in notebook.get("sources", [])]
        for faq_item in items:
            with st.expander(f"**{faq_item.get('question', '')}**"):
                st.markdown(faq_item.get("answer", ""))
                cited = faq_item.get("sources", [])
                if cited:
                    labels = [
                        src_names[n - 1] for n in cited
                        if isinstance(n, int) and 1 <= n <= len(src_names)
                    ]
                    if labels:
                        st.caption("Sources: " + ", ".join(labels))

        # Export as markdown
        faq_md = "\n\n".join(
            f"### {it.get('question','')}\n{it.get('answer','')}" for it in items
        )
        st.download_button(
            "Download FAQ (.md)",
            data=faq_md,
            file_name=f"faq_{notebook.get('name','notebook')}.md",
            mime="text/markdown",
            key=f"nb_dl_faq_{active_id}",
        )


def _tab_literature_review(active_id: str, notebook: dict, settings: dict) -> None:
    st.markdown(
        "Generates a formal academic-style literature review with structured "
        "sections: introduction, background, methodology, key findings, "
        "critical analysis, and conclusion."
    )
    cache_key = f"nb_litreview_{active_id}"
    from agents.notebook_advanced import generate_literature_review

    result, _ = _gen_button(
        "Generate Literature Review", f"nb_gen_lr_{active_id}", cache_key,
        settings, active_id, generate_literature_review,
    )
    if result:
        st.markdown(result)
        nb_name = notebook.get("name", "notebook")
        st.download_button(
            "Download (.md)",
            data=result,
            file_name=f"literature_review_{nb_name}.md",
            mime="text/markdown",
            key=f"nb_dl_lr_{active_id}",
        )
        _docx_pdf_buttons(result, f"literature_review_{nb_name}", f"nb_lr_{active_id}")


def _tab_mindmap(active_id: str, notebook: dict, settings: dict) -> None:
    st.markdown(
        "Extracts key concepts and their relationships from your sources and "
        "renders them as an interactive mind map."
    )
    cache_key = f"nb_mindmap_{active_id}"
    from agents.notebook_advanced import generate_mindmap

    result, _ = _gen_button(
        "Generate Mind Map", f"nb_gen_mm_{active_id}", cache_key,
        settings, active_id, generate_mindmap,
    )
    if result:
        try:
            st.graphviz_chart(result)
        except Exception as e:
            st.error(f"Could not render mind map: {e}")
            st.code(result, language="dot")
        _dot_export_buttons(result, f"mindmap_{notebook.get('name','notebook')}",
                            f"nb_mm_{active_id}")
        with st.expander("DOT source"):
            st.code(result, language="dot")


def _tab_audio(active_id: str, notebook: dict, settings: dict) -> None:
    st.markdown(
        "Generates a spoken-word summary script (~300 words, ~2 min) "
        "and synthesizes it to a downloadable **.wav** audio file."
    )
    cache_key = f"nb_audio_{active_id}"
    wav_key = f"nb_audio_wav_{active_id}"
    nb_name = notebook.get("name", "notebook")
    from agents.notebook_advanced import generate_audio_summary, synthesize_speech

    result, _ = _gen_button(
        "Generate Audio Script", f"nb_gen_audio_{active_id}", cache_key,
        settings, active_id, generate_audio_summary,
    )
    if result:
        st.markdown(f"**Word count:** {len(result.split())}")
        st.text_area(
            "Audio script",
            value=result,
            height=240,
            key=f"nb_audio_ta_{active_id}",
        )
        dl_col, play_col, wav_col = st.columns(3)
        dl_col.download_button(
            "Download script (.txt)",
            data=result,
            file_name=f"audio_summary_{nb_name}.txt",
            mime="text/plain",
            key=f"nb_dl_audio_{active_id}",
        )

        # Browser TTS via Web Speech API
        if play_col.button("▶ Play in browser", key=f"nb_play_{active_id}"):
            import json as _json
            st.components.v1.html(
                f"""<script>
var text = {_json.dumps(result)};
if (window.speechSynthesis) {{
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.rate = 0.9;
    window.speechSynthesis.speak(u);
}} else {{
    alert("Your browser does not support speech synthesis.");
}}
</script>""",
                height=0,
            )

        # pyttsx3 WAV synthesis
        if wav_col.button("Synthesize .wav", key=f"nb_synth_{active_id}"):
            with st.spinner("Synthesizing audio (~30 s)…"):
                wav_bytes, wav_err = synthesize_speech(result)
            if wav_err:
                st.error(wav_err)
            else:
                st.session_state[wav_key] = wav_bytes

        if wav_key in st.session_state:
            st.audio(st.session_state[wav_key], format="audio/wav")
            st.download_button(
                "Download .wav",
                data=st.session_state[wav_key],
                file_name=f"audio_summary_{nb_name}.wav",
                mime="audio/wav",
                key=f"nb_dl_wav_{active_id}",
            )


def _tab_compare(active_id: str, notebook: dict, settings: dict) -> None:
    sources = notebook.get("sources", [])
    if len(sources) < 2:
        st.info("Add at least two sources to use source comparison.")
        return

    st.markdown("Select two sources to compare side-by-side.")
    options = {s["filename"]: s["doc_id"] for s in sources}
    names = list(options.keys())

    col_a, col_b = st.columns(2)
    sel_a = col_a.selectbox("Source A", names, index=0, key=f"nb_cmp_a_{active_id}")
    sel_b = col_b.selectbox(
        "Source B", names,
        index=min(1, len(names) - 1),
        key=f"nb_cmp_b_{active_id}",
    )

    doc_a = options[sel_a]
    doc_b = options[sel_b]
    cache_key = f"nb_compare_{active_id}_{doc_a}_{doc_b}"

    from agents.notebook_advanced import compare_sources

    result, _ = _gen_button(
        "Compare Sources", f"nb_gen_cmp_{active_id}", cache_key,
        settings, active_id, compare_sources, doc_a, doc_b,
    )
    if result:
        st.markdown(result)
        nb_name = notebook.get("name", "notebook")
        st.download_button(
            "Download comparison (.md)",
            data=result,
            file_name=f"comparison_{nb_name}.md",
            mime="text/markdown",
            key=f"nb_dl_cmp_{active_id}",
        )


def _tab_knowledge_graph(active_id: str, notebook: dict, settings: dict) -> None:
    st.markdown(
        "Extracts entities (concepts, methods, datasets, authors) and their "
        "relationships from your sources and visualises them as a knowledge graph."
    )
    st.markdown(
        "**Legend:** "
        "<span style='color:#3b82f6'>■ Concept</span> "
        "<span style='color:#10b981'>■ Method</span> "
        "<span style='color:#f59e0b'>■ Dataset</span> "
        "<span style='color:#8b5cf6'>■ Author</span> "
        "<span style='color:#ef4444'>■ Institution</span>",
        unsafe_allow_html=True,
    )
    cache_key = f"nb_kgraph_{active_id}"
    from agents.notebook_advanced import extract_knowledge_graph

    result, _ = _gen_button(
        "Extract Knowledge Graph", f"nb_gen_kg_{active_id}", cache_key,
        settings, active_id, extract_knowledge_graph,
    )
    if result:
        try:
            st.graphviz_chart(result)
        except Exception as e:
            st.error(f"Could not render knowledge graph: {e}")
            st.code(result, language="dot")
        _dot_export_buttons(result, f"knowledge_graph_{notebook.get('name','notebook')}",
                            f"nb_kg_{active_id}")
        with st.expander("DOT source"):
            st.code(result, language="dot")


def _tab_timeline(active_id: str, notebook: dict, settings: dict) -> None:
    st.markdown(
        "Extracts a **chronological timeline** of events, discoveries, and milestones "
        "from your notebook sources."
    )
    cache_key = f"nb_timeline_{active_id}"
    from agents.notebook_advanced import extract_timeline

    items, _ = _gen_button(
        "Extract Timeline", f"nb_gen_tl_{active_id}", cache_key,
        settings, active_id, extract_timeline,
    )
    if items:
        src_names = [s["filename"] for s in notebook.get("sources", [])]
        md_lines = ["| Year | Event | Significance | Source |",
                    "|------|-------|-------------|--------|"]
        for item in items:
            year = item.get("year", "n.d.")
            event = item.get("event", "").replace("|", "\\|")
            sig = item.get("significance", "").replace("|", "\\|")
            src_n = item.get("source", 0)
            src_label = (
                src_names[src_n - 1][:20] if isinstance(src_n, int) and 1 <= src_n <= len(src_names)
                else "—"
            )
            md_lines.append(f"| {year} | {event} | {sig} | {src_label} |")
        table_md = "\n".join(md_lines)
        st.markdown(table_md)
        st.download_button(
            "Download (.md)",
            data=table_md,
            file_name=f"timeline_{notebook.get('name','notebook')}.md",
            mime="text/markdown",
            key=f"nb_dl_tl_{active_id}",
        )


def _tab_study_comparison(active_id: str, notebook: dict, settings: dict) -> None:
    st.markdown(
        "Generates a **structured comparison table** across all sources: "
        "research type, sample/data scope, methodology, key findings, and limitations."
    )
    cache_key = f"nb_studytable_{active_id}"
    from agents.notebook_advanced import generate_study_comparison

    result, _ = _gen_button(
        "Generate Study Comparison", f"nb_gen_st_{active_id}", cache_key,
        settings, active_id, generate_study_comparison,
    )
    if result:
        st.markdown(result)
        nb_name = notebook.get("name", "notebook")
        st.download_button(
            "Download (.md)",
            data=result,
            file_name=f"study_comparison_{nb_name}.md",
            mime="text/markdown",
            key=f"nb_dl_st_{active_id}",
        )
        _docx_pdf_buttons(result, f"study_comparison_{nb_name}", f"nb_st_{active_id}")


def _tab_pipeline(active_id: str, notebook: dict, settings: dict) -> None:
    """
    Run the full 7-agent LangGraph pipeline and display all outputs.

    Agents and their outputs:
      1 Document Ingestion   → source/chunk inventory
      2 Summarization        → per-doc + cross-document summary
      3 Retrieval            → top-k relevant chunks
      4 Citation Verification → claim verification report
      5 Knowledge Graph      → rendered DOT graph
      6 Study Guide          → key concepts / glossary / Q&A
      7 Podcast Script       → two-speaker dialogue
    """
    from agents.notebook_pipeline_graph import run_notebook_pipeline
    from agents.notebook_pipeline_state import create_pipeline_state

    st.markdown(
        "Runs all **7 analysis agents** in sequence, each passing its results to the "
        "next via LangGraph state. Produces a summary, citation verification report, "
        "knowledge graph, study guide, and podcast script in one click."
    )

    if not notebook.get("sources"):
        st.info("Add at least one source to this notebook before running the pipeline.")
        return

    query = st.text_input(
        "Focus query (optional)",
        key=f"nb_pipeline_query_{active_id}",
        placeholder="e.g. 'key findings on attention mechanisms'",
        help="Agent 3 (Retrieval) uses this query to surface the most relevant chunks. "
             "Leave blank for a broad overview.",
    )
    query_final, query_ready = render_query_gate(query, key=f"nb_pipeline_query_{active_id}", settings=settings)

    cache_key = f"nb_pipeline_{active_id}"
    col_run, col_clr = st.columns([4, 1])
    btn_label = "Run Pipeline Again" if cache_key in st.session_state else "Run Full Pipeline"

    if col_run.button(btn_label, key=f"nb_run_pipeline_{active_id}",
                      type="primary", use_container_width=True):
        if not query_ready:
            st.info("Please resolve the grammar suggestion above, then click **Run Full Pipeline** again.")
        else:
            st.session_state.pop(cache_key, None)
            initial = create_pipeline_state(
                notebook_id=active_id,
                settings=settings,
                query=query_final.strip(),
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            _AGENT_LABELS = {
                "ingest": "Agent 1 — Document Ingestion",
                "summarize": "Agent 2 — Summarization",
                "retrieve": "Agent 3 — Retrieval",
                "verify_citations": "Agent 4 — Citation Verification",
                "build_kg": "Agent 5 — Knowledge Graph",
                "generate_study_guide": "Agent 6 — Study Guide",
                "generate_podcast": "Agent 7 — Podcast Script",
            }

            def _cb(node_name: str, partial_state: dict) -> None:
                pct = partial_state.get("progress_pct", 0)
                label = _AGENT_LABELS.get(node_name, node_name)
                progress_bar.progress(pct / 100)
                status_text.caption(f"{label} ({pct}%)")

            try:
                with st.spinner("Running pipeline…"):
                    result = run_notebook_pipeline(initial, stream_callback=_cb)
                progress_bar.progress(1.0)
                status_text.caption("Pipeline complete.")
                st.session_state[cache_key] = dict(result)
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                return

    if col_clr.button("Clear", key=f"nb_clr_pipeline_{active_id}"):
        st.session_state.pop(cache_key, None)
        st.rerun()

    state = st.session_state.get(cache_key)
    if not state:
        return

    render_eval_result(state.get("eval_result", {}), key_suffix=f"_nb_pipeline_{active_id}")
    render_rag_reflection(state.get("rag_reflection_info"), key_suffix=f"_nb_pipeline_{active_id}")

    # Show any errors at the top
    errors = state.get("errors", [])
    if errors:
        with st.expander(f"{len(errors)} warning(s)", expanded=False):
            for e in errors:
                st.caption(e)

    nb_name = notebook.get("name", "notebook")

    (
        ptab_ingest, ptab_summary, ptab_retrieve,
        ptab_citations, ptab_kg, ptab_study, ptab_podcast,
    ) = st.tabs([
        "Ingestion",
        "Summary",
        "Retrieval",
        "Citations",
        "Knowledge Graph",
        "Study Guide",
        "Podcast",
    ])

    with ptab_ingest:
        st.markdown(f"**{state.get('ingestion_summary', 'No summary.')}**")
        sources = state.get("sources", [])
        if sources:
            for src in sources:
                st.caption(
                    f"**{src['filename']}** · "
                    f"{src.get('total_chunks', 0)} chunks · "
                    f"{src.get('file_type', '')} · added {src.get('added_at', '')[:10]}"
                )

    with ptab_summary:
        cross = state.get("cross_summary", "")
        per_doc = state.get("per_doc_summaries", {})
        if cross:
            st.markdown(cross)
            st.download_button(
                "Download (.md)", data=cross,
                file_name=f"pipeline_summary_{nb_name}.md",
                mime="text/markdown", key=f"nb_pl_sum_dl_{active_id}",
            )
            _docx_pdf_buttons(cross, f"pipeline_summary_{nb_name}", f"nb_pl_sum_{active_id}")
        if per_doc:
            st.divider()
            st.markdown("**Per-document summaries**")
            for fname, summary in per_doc.items():
                with st.expander(fname):
                    st.markdown(summary or "_No summary generated._")

    with ptab_retrieve:
        chunks = state.get("retrieved_chunks", [])
        mode = state.get("retrieval_mode", "empty")
        st.caption(f"Retrieval mode: **{mode}** · {len(chunks)} chunk(s) retrieved")
        for i, chunk in enumerate(chunks, 1):
            with st.expander(
                f"[{i}] {chunk.get('doc_name', '')} — p.{chunk.get('page_num', '?')}"
            ):
                st.markdown(chunk.get("text", ""))

    with ptab_citations:
        report = state.get("citation_report", "")
        if report:
            st.markdown(report)
            st.download_button(
                "Download (.md)", data=report,
                file_name=f"citation_report_{nb_name}.md",
                mime="text/markdown", key=f"nb_pl_cit_dl_{active_id}",
            )

    with ptab_kg:
        dot = state.get("knowledge_graph_dot", "")
        if dot:
            st.graphviz_chart(dot)
            _dot_export_buttons(dot, f"pipeline_kg_{nb_name}", f"nb_pl_kg_{active_id}")
        else:
            st.info("Knowledge graph not available.")

    with ptab_study:
        guide = state.get("study_guide", "")
        if guide:
            st.markdown(guide)
            st.download_button(
                "Download (.md)", data=guide,
                file_name=f"study_guide_{nb_name}.md",
                mime="text/markdown", key=f"nb_pl_sg_dl_{active_id}",
            )
            _docx_pdf_buttons(guide, f"study_guide_{nb_name}", f"nb_pl_sg_{active_id}")
            # Feedback refinement for study guide
            from ui.helpers import render_feedback_section
            render_feedback_section(
                current_output=state.get("study_guide", ""),
                session_key=f"nb_pipeline_{active_id}",
                mode="notebook_pipeline",
                model_name=settings.get("model", "llama3.1:8b"),
                num_ctx=settings.get("num_ctx", 32768),
                context=state.get("cross_summary", ""),
                key_suffix=f"_nb_pipeline_{active_id}",
            )
        else:
            st.info("Study guide not available.")

    with ptab_podcast:
        script = state.get("podcast_script", "")
        if script:
            st.markdown(f"```\n{script}\n```")
            st.download_button(
                "Download (.txt)", data=script,
                file_name=f"podcast_script_{nb_name}.txt",
                mime="text/plain", key=f"nb_pl_pod_dl_{active_id}",
            )
            # TTS playback via browser Web Speech API
            if st.button("Read aloud (browser TTS)",
                         key=f"nb_pl_pod_tts_{active_id}"):
                import streamlit.components.v1 as components
                safe = script.replace("`", "").replace("\\", "\\\\").replace('"', '\\"')
                components.html(
                    f"""<script>
                    var u=new SpeechSynthesisUtterance("{safe[:3000]}");
                    u.rate=0.9; window.speechSynthesis.speak(u);
                    </script>""",
                    height=0,
                )
        else:
            st.info("Podcast script not available.")


def _rebuild_processed_docs(notebook: dict) -> list:
    """Reconstruct ProcessedDocument objects from stored notebook chunks."""
    from tools.document_tools import DocumentChunk, ProcessedDocument
    chunks_by_doc: dict = {}
    for c in notebook.get("chunks", []):
        chunks_by_doc.setdefault(c["doc_id"], []).append(c)
    src_by_id = {s["doc_id"]: s for s in notebook.get("sources", [])}
    docs = []
    for doc_id, raw_chunks in chunks_by_doc.items():
        src = src_by_id.get(doc_id, {})
        filename = src.get("filename", doc_id)
        sorted_chunks = sorted(raw_chunks, key=lambda c: (c.get("page_num", 0), c.get("chunk_index", 0)))
        doc_chunks = [
            DocumentChunk(
                chunk_id=c["chunk_id"],
                doc_id=doc_id,
                doc_name=filename,
                page_num=c.get("page_num", 0),
                chunk_index=c.get("chunk_index", 0),
                text=c.get("text", ""),
                metadata=c.get("metadata", {}),
            )
            for c in sorted_chunks
        ]
        raw_text = "\n\n".join(c.get("text", "") for c in sorted_chunks)
        docs.append(ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            total_chunks=len(doc_chunks),
            chunks=doc_chunks,
            raw_text=raw_text,
            content_md5=src.get("content_md5", ""),
        ))
    return docs


def _tab_research_report(active_id: str, notebook: dict, settings: dict) -> None:
    """Generate a structured research report grounded in notebook sources."""
    st.markdown(
        "Generate a full research report grounded in your notebook sources "
        "and optionally augmented with peer-reviewed papers from arXiv and Semantic Scholar."
    )

    goal = st.text_input(
        "Research goal or question",
        key=f"nb_rpt_goal_{active_id}",
        placeholder="e.g. 'Summarise key findings on transformer attention mechanisms'",
    )
    goal_final, goal_ready = render_query_gate(goal, key=f"nb_rpt_goal_{active_id}", settings=settings)
    col1, col2 = st.columns(2)
    include_academic = col1.toggle(
        "Search academic sources",
        value=True,
        key=f"nb_rpt_academic_{active_id}",
        help="Search arXiv + Semantic Scholar for peer-reviewed papers",
    )
    include_web = col2.toggle(
        "Include web search",
        value=False,
        key=f"nb_rpt_web_{active_id}",
        help="Also search the web via Google",
    )

    sources = notebook.get("sources", [])
    if not sources:
        mode = "search"
        st.info("No sources in this notebook — will search academic literature only.")
    elif include_academic:
        mode = "hybrid"
    else:
        mode = "document"

    cache_key = f"nb_rpt_{active_id}"

    col_run, col_clr = st.columns([4, 1])
    btn_label = "Regenerate Report" if cache_key in st.session_state else "Generate Research Report"
    if col_run.button(btn_label, key=f"nb_rpt_run_{active_id}", type="primary", use_container_width=True):
        if not goal.strip():
            st.warning("Please enter a research goal.")
        elif not goal_ready:
            st.info("Please resolve the grammar suggestion above, then click **Generate Research Report** again.")
        else:
            try:
                from agents.graph import run_research
                from agents.state import create_initial_state
            except ModuleNotFoundError:
                st.error(
                    "The Research Report agent isn't available in this build "
                    "(its `agents.graph` / `agents.state` modules are missing). "
                    "Try **Pipeline** or **Chat** for grounded answers from your sources instead."
                )
                logger.warning("Research Report unavailable: agents.graph/agents.state not found")
                return

            processed_docs = _rebuild_processed_docs(notebook)
            initial_state = create_initial_state(
                goal=goal_final.strip(),
                uploaded_docs=processed_docs,
                mode=mode,
                include_web_search=include_web,
                model_name=settings["model"],
                num_ctx=settings["num_ctx"],
                embed_model=settings.get("embed_model", cfg.embedding_model),
            )

            prog = st.progress(0)
            status = st.empty()

            _step_labels = {
                "document_ingestion":    "Indexing notebook sources",
                "query_generation":      "Generating search queries",
                "academic_search":       "Searching arXiv + Semantic Scholar",
                "web_search":            "Searching the web",
                "document_analysis":     "Analysing sources",
                "reference_compilation": "Compiling references",
                "report_generation":     "Generating report",
                "research_eval":         "Evaluating quality",
            }

            def _cb(node_name: str, state: dict) -> None:
                pct = state.get("progress_pct", 0)
                prog.progress(pct / 100)
                status.caption(_step_labels.get(node_name, node_name.replace("_", " ").title()) + f" ({pct}%)")

            with st.spinner("Running research workflow…"):
                try:
                    final_state = run_research(initial_state, stream_callback=_cb)
                except Exception as exc:
                    st.error(f"Research workflow failed: {exc}")
                    logger.exception("Notebook research report failed")
                    return

            prog.progress(1.0)
            status.empty()
            st.session_state[cache_key] = dict(final_state)

    if col_clr.button("Clear", key=f"nb_rpt_clr_{active_id}"):
        st.session_state.pop(cache_key, None)
        st.rerun()

    result = st.session_state.get(cache_key)
    if not result:
        return

    from ui.helpers import render_key_findings, render_references, render_report
    render_key_findings(result.get("key_findings", []))
    st.divider()
    rt_report, rt_refs = st.tabs(["Report", "References"])
    with rt_report:
        render_report(result.get("report", ""), f"nb_rpt_{active_id}")
    with rt_refs:
        render_references(result.get("references", []), key_suffix=f"_nb_rpt_{active_id}")


def _tab_explain(active_id: str, notebook: dict, settings: dict) -> None:
    """Conversational science communicator grounded in the notebook's sources."""
    try:
        from agents.story_graph import run_story_turn
        from agents.story_memory import StorytellerMemory
        from agents.story_state import create_story_state
    except ModuleNotFoundError:
        st.info(
            "The Explain agent isn't available in this build "
            "(its `agents.story_*` modules are missing). "
            "Try the **Chat** tab to ask grounded questions about your sources instead."
        )
        logger.warning("Explain tab unavailable: agents.story_* modules not found")
        return

    st.markdown(
        "Ask questions about your notebook sources in plain language. "
        "Choose an explanation **style** — simple language, an extended analogy, "
        "a step-by-step walkthrough, or a structured debate — and an audience "
        "**level** — novice, intermediate, or expert — and the agent tailors its "
        "response to both."
    )

    memory = StorytellerMemory()
    nb_name = notebook.get("name", "Notebook")

    explanation_style = st.radio(
        "Explanation style",
        options=["simple", "analogy", "walkthrough", "debate"],
        format_func=lambda x: {
            "simple":      "Simple Language",
            "analogy":     "Extended Analogy",
            "walkthrough": "Step-by-Step",
            "debate":      "For vs. Against",
        }[x],
        horizontal=True,
        key=f"nb_explain_style_{active_id}",
    )
    explanation_level = st.radio(
        "Explanation level",
        options=["novice", "intermediate", "expert"],
        format_func=lambda x: {
            "novice":       "Novice",
            "intermediate": "Intermediate",
            "expert":       "Expert",
        }[x],
        index=1,
        horizontal=True,
        key=f"nb_explain_level_{active_id}",
    )

    # Resolve or auto-create a story session linked to this notebook
    session_key = f"nb_explain_sid_{active_id}"
    effective_sid = st.session_state.get(session_key)

    if effective_sid:
        session_data = memory.load(effective_sid)
        if not session_data:
            # Session was deleted — clear reference
            st.session_state.pop(session_key, None)
            effective_sid = None
            session_data = None
    else:
        session_data = None

    # Show existing conversation
    if session_data:
        st.caption(f"Session: {session_data.get('topic', nb_name)[:50]}")
        concepts = session_data.get("concepts_covered", [])
        if concepts:
            st.caption("Concepts covered: " + ", ".join(concepts[:8]))
        st.divider()
        for turn in session_data.get("conversation", []):
            role = turn.get("role", "user")
            with st.chat_message(role):
                st.markdown(turn.get("content", ""))
                if role == "assistant":
                    qs = turn.get("suggested_questions") or []
                    if qs:
                        st.markdown("**Follow-up questions:**")
                        for q in qs:
                            if st.button(q, key=f"nb_exp_sq_{hash(q + turn.get('content', '')[:20])}_{active_id}"):
                                st.session_state[f"nb_explain_pending_{active_id}"] = q
                                st.rerun()
    else:
        st.info("Type your first question below to start an explanation session grounded in this notebook.")

    # Chat input
    pending = st.session_state.pop(f"nb_explain_pending_{active_id}", None)
    user_input = st.chat_input(
        placeholder=f"Ask anything about {nb_name}…",
        key=f"nb_explain_chat_{active_id}",
    )
    message = render_chat_gate(pending or user_input, key=f"nb_explain_{active_id}", settings=settings)
    if not message:
        return

    # Auto-create session if none exists
    if not effective_sid:
        # Build doc context from notebook chunks (first 500 chars per chunk, max 2000 total)
        doc_context_parts = []
        total_chars = 0
        for c in notebook.get("chunks", []):
            snippet = c.get("text", "")[:500]
            if total_chars + len(snippet) > 2000:
                break
            doc_context_parts.append(snippet)
            total_chars += len(snippet)
        doc_context = "\n\n---\n".join(doc_context_parts)[:2000]
        doc_names = [s["filename"] for s in notebook.get("sources", [])]

        effective_sid = memory.new_session(
            topic=nb_name,
            document_context=doc_context,
            document_names=doc_names,
        )
        st.session_state[session_key] = effective_sid
        session_data = memory.load(effective_sid)

    with st.chat_message("user"):
        st.markdown(message)

    state = create_story_state(
        user_message=message,
        session_id=effective_sid,
        topic=nb_name,
        model_name=settings["model"],
        num_ctx=settings["num_ctx"],
        explanation_style=explanation_style,
        explanation_level=explanation_level,
    )

    step_log = st.empty()
    done_steps: list = []

    def _cb(node_name: str, _state: dict) -> None:
        done_steps.append(node_name.replace("_", " ").title())
        step_log.caption(" → ".join(done_steps))

    with st.spinner("Explaining…"):
        try:
            final = run_story_turn(state, stream_callback=_cb)
        except Exception as exc:
            st.error(f"Error: {exc}")
            logger.exception("Notebook explain failed")
            return

    step_log.empty()

    with st.chat_message("assistant"):
        st.markdown(final.get("assistant_response", ""))
        for q in final.get("suggested_questions", []):
            if st.button(q, key=f"nb_exp_newsq_{hash(q + message[:20])}_{active_id}"):
                st.session_state[f"nb_explain_pending_{active_id}"] = q
                st.rerun()

    st.rerun()


# ── Cross-notebook search ────────────────────────────────────────────────────────

def _render_cross_notebook_search(memory: NotebookMemory, settings: dict) -> None:
    """
    "Search across everything I've ever uploaded" — a single search box that
    looks through every source in every notebook at once, not just the one
    that's currently open.

    Runs NotebookMemory.search_all_notebooks(), a lightweight keyword search
    over the shared notebook_chunks table — instant, with no per-notebook
    index to build first. Each hit can jump straight to its source notebook.
    """
    with st.expander("Search across all notebooks", expanded=False):
        st.caption(
            "Search the full text of every source in every notebook you've ever "
            "created — handy when you can't remember which notebook a paper or "
            "passage lives in."
        )
        col_q, col_n = st.columns([4, 1])
        with col_q:
            query = st.text_input(
                "Search all notebooks", key="xns_query",
                placeholder="e.g. transformer attention mechanism",
                label_visibility="collapsed",
            )
            query_final, query_ready = render_query_gate(query, key="xns_query", settings=settings)
        with col_n:
            limit = st.number_input(
                "Max results", min_value=5, max_value=100, value=20, step=5,
                key="xns_limit", label_visibility="collapsed",
                help="Maximum number of matching passages to show.",
            )
        run = st.button("Search all notebooks", key="xns_run", type="primary")

        if run:
            if not query_ready:
                st.info("Please resolve the grammar suggestion above, then click **Search all notebooks** again.")
            else:
                q = query_final.strip()
                if not q:
                    st.warning("Enter at least one search term.")
                    st.session_state.pop("xns_results", None)
                else:
                    with st.spinner("Searching every notebook you have…"):
                        st.session_state["xns_results"] = memory.search_all_notebooks(q, limit=int(limit))
                        st.session_state["xns_query_used"] = q

        results = st.session_state.get("xns_results")
        if results is not None:
            q_used = st.session_state.get("xns_query_used", "")
            if not results:
                st.info(f"No matches for **{q_used}** in any notebook yet — try a shorter or different phrase.")
            else:
                nb_count = len({h["notebook_id"] for h in results})
                st.success(
                    f"{len(results)} matching passage(s) for **{q_used}** "
                    f"across {nb_count} notebook(s)."
                )
                for i, hit in enumerate(results):
                    header = (
                        f"[{hit['notebook_name'][:28]}] {hit['doc_name'][:42]} "
                        f"— p.{hit['page_num']} · {hit['matched_terms']} term(s) matched"
                    )
                    with st.expander(header):
                        st.markdown(hit["snippet"])
                        if st.button(
                            f"Open “{hit['notebook_name'][:28]}” notebook",
                            key=f"xns_open_{i}_{hit['notebook_id']}_{hit['doc_id']}_{hit['chunk_index']}",
                        ):
                            st.session_state["nb_jump_to"] = hit["notebook_id"]
                            st.rerun()


# ── Main tab ─────────────────────────────────────────────────────────────────────

def tab_notebook(settings: dict) -> None:
    """
    Research Notebook.

    Layout: 2-column (1/3 sources | 2/3 chat + advanced tabs).
    Advanced tabs: Summary, FAQ, Literature Review, Mind Map,
                   Audio Script, Source Comparison, Knowledge Graph,
                   Research Report, Explain.
    """
    st.header("Research Notebook")
    st.markdown(
        """
Build a notebook from your own sources — PDFs, Word docs, text, Markdown, or
web pages — then ask questions. Every answer is grounded **only** in your
sources and cites the exact document and page. Conversations are saved per
notebook, so you can return any time.

Use the **advanced tabs** on the right for summaries, FAQs, literature reviews,
mind maps, audio scripts, source comparisons, knowledge graphs, full **Research Reports**
(grounded in your sources and augmented with academic literature), and an **Explain** tab
for conversational science communication with multiple explanation styles.
"""
    )
    render_glossary_expander([
        "RAG (Retrieval-Augmented Generation)", "Hybrid retrieval", "Embedding model",
        "Context window", "Chunking", "Docling / OCR", "Quality score", "Faithfulness",
    ])

    memory = NotebookMemory()

    # A cross-notebook search hit may queue a "jump to notebook" request.
    # Apply it — and clear the cached selector label so the staleness-recovery
    # logic below re-derives the correct label — before the selectbox renders.
    _jump_to = st.session_state.pop("nb_jump_to", None)
    if _jump_to:
        st.session_state["nb_active_id"] = _jump_to
        st.session_state.pop("nb_selector", None)

    _render_cross_notebook_search(memory, settings)

    src_col, chat_col = st.columns([1, 2])

    # ── Left: notebook selector + source management ──────────────────────────
    with src_col:
        st.markdown("#### Notebook")

        notebooks = memory.list_notebooks()
        options = {"+ New notebook": None}
        for nb in notebooks:
            label = f"{nb['name'][:30]} ({nb['source_count']} src, {nb['turn_count']} turns)"
            options[label] = nb["notebook_id"]

        # The selectbox key caches the display label. Labels change on every rerun
        # after a file upload or question (source_count / turn_count increments).
        # When the cached label is no longer in options Streamlit silently resets
        # to index 0 ("+ New notebook"), losing the active notebook context.
        #
        # Fix: only update the cached label when it is STALE (not present in the
        # current options dict). When the user has just changed their selection,
        # Streamlit has already written the new label into session state — we must
        # not overwrite it, or the selectbox will snap back to the old notebook.
        _active_id_hint = st.session_state.get("nb_active_id")
        if _active_id_hint and st.session_state.get("nb_selector") not in options:
            for _label, _nb_id in options.items():
                if _nb_id == _active_id_hint:
                    st.session_state["nb_selector"] = _label
                    break

        chosen_label = st.selectbox(
            "Select notebook", list(options.keys()), key="nb_selector",
        )
        selected_id = options[chosen_label]

        # Keep nb_active_id in sync with whatever the user selects
        if selected_id:
            st.session_state["nb_active_id"] = selected_id

        if selected_id is None:
            new_name = st.text_input(
                "Notebook name", key="nb_new_name", placeholder="e.g. Antibiotic Resistance",
            )
            if st.button("Create notebook", key="nb_create", type="primary", use_container_width=True):
                nb_id = memory.new_notebook(new_name or "Untitled Notebook")
                st.session_state["nb_active_id"] = nb_id
                # Drop the cached selector label (same trick "jump to notebook" uses
                # below) so the staleness recovery picks the new notebook's label on
                # the next run — otherwise the selector is left reading "+ New
                # notebook" (still a *valid* option) while the panel beneath it
                # shows the notebook that was just created.
                st.session_state.pop("nb_selector", None)
                st.rerun()

        active_id = selected_id or st.session_state.get("nb_active_id")

        if active_id:
            notebook = memory.load(active_id)
            if not notebook:
                st.error("Notebook not found.")
                st.session_state.pop("nb_active_id", None)
                return

            st.divider()
            st.markdown("#### Sources", help=term_help("Chunking"))

            # ── Upload files ──────────────────────────────────
            files = st.file_uploader(
                "Add files",
                type=get_supported_file_types(settings.get("use_docling", True)),
                accept_multiple_files=True,
                key=f"nb_files_{active_id}",
                help="Files are indexed automatically when you ask your first question, or click the button below to add them now.",
            )
            # Filter out Streamlit DeletedFile sentinels before counting/processing
            valid_files = [f for f in (files or []) if hasattr(f, "name") and hasattr(f, "getbuffer")]
            if valid_files:
                st.caption(
                    f"📎 {len(valid_files)} file(s) ready — will be indexed automatically when "
                    f"you ask a question, or click below to add now."
                )
                if st.button("Add files now", key=f"nb_add_files_{active_id}",
                            use_container_width=True):
                    with st.spinner("Processing and indexing documents…"):
                        processed = process_uploads(valid_files, settings)
                        added = _index_and_store(active_id, processed, settings, source_type="file")
                    if added > 0:
                        st.success(f"Added {added} source(s).")
                        st.rerun()
                    else:
                        st.warning("No new sources were added (may be duplicates).")

            # ── Add a web page (optional) ─────────────────────
            with st.expander("Add a specific web page (optional)"):
                url = st.text_input("Page URL", key=f"nb_url_{active_id}",
                                    placeholder="https://…")
                if st.button("Fetch and add", key=f"nb_add_url_{active_id}",
                             use_container_width=True):
                    if not url.strip():
                        st.warning("Enter a URL first.")
                    else:
                        from tools.document_tools import DocumentProcessor
                        from tools.web_loader import load_url_as_document
                        with st.spinner("Fetching page…"):
                            processor = DocumentProcessor(
                                chunk_size=settings.get("chunk_size", cfg.chunk_size),
                                overlap=settings.get("chunk_overlap", cfg.chunk_overlap),
                            )
                            doc, err = load_url_as_document(url.strip(), processor)
                        if err:
                            st.error(err)
                        else:
                            added = _index_and_store(active_id, [doc], settings,
                                                     source_type="url", url=url.strip())
                            st.success(f"Added: {doc.filename}")
                            st.rerun()

            # ── Auto web search toggle ────────────────────────
            auto_web_key = f"nb_auto_web_{active_id}"
            auto_web_search = st.toggle(
                "Auto web search",
                value=st.session_state.get(auto_web_key, False),
                key=auto_web_key,
                help=(
                    "When enabled, the agent automatically searches the web (DuckDuckGo) "
                    "for relevant pages to supplement your notebook sources on each question."
                ),
            )

            # ── Source list ───────────────────────────────────
            sources = notebook.get("sources", [])
            if sources:
                st.caption(f"{len(sources)} source(s) in this notebook:")
                for s in sources:
                    cols = st.columns([5, 1])
                    cols[0].markdown(
                        f"**{s['filename'][:32]}**  \n"
                        f"<span style='color:#64748B;font-size:0.75rem'>"
                        f"{s.get('file_type','')} · {s.get('total_chunks',0)} chunks</span>",
                        unsafe_allow_html=True,
                    )
                    if cols[1].button("Remove", key=f"nb_rm_{s['doc_id']}", help="Remove source"):
                        memory.remove_source(active_id, s["doc_id"])
                        _hybrid_stores.pop(f"notebook_{active_id}", None)
                        _hybrid_stores.pop(f"notebook_{active_id}_bm25", None)
                        st.rerun()
            else:
                if auto_web_search:
                    st.info(
                        "No uploaded sources — that's fine. "
                        "**Auto web search** is on: the agent will search Google for each question."
                    )
                else:
                    st.info("No sources yet. Add files or a web page above, "
                            "or enable **Auto web search** to let the agent search Google.")

            # ── Notebook actions ──────────────────────────────
            st.divider()
            with st.expander("Rename / Delete"):
                rename = st.text_input("Rename notebook", value=notebook.get("name", ""),
                                       key=f"nb_rename_{active_id}")
                if st.button("Save name", key=f"nb_save_name_{active_id}"):
                    memory.rename(active_id, rename)
                    st.rerun()
                if st.button("Delete notebook", key=f"nb_delete_{active_id}"):
                    memory.delete(active_id)
                    st.session_state.pop("nb_active_id", None)
                    from tools.hybrid_store import _stores
                    _stores.pop(f"notebook_{active_id}", None)
                    st.rerun()

    # ── Right: chat + advanced tabs ───────────────────────────────────────────
    with chat_col:
        if not active_id:
            st.info("Create or select a notebook on the left to begin.")
            return

        notebook = memory.load(active_id)
        if not notebook:
            st.info("Create or select a notebook on the left to begin.")
            return

        st.markdown(f"**{notebook.get('name', 'Notebook')}**")
        src_names = [s["filename"] for s in notebook.get("sources", [])]
        if src_names:
            st.caption("Grounded in: " + ", ".join(n[:24] for n in src_names[:6])
                       + (" …" if len(src_names) > 6 else ""))
        st.divider()

        (
            tab_chat, tab_summary, tab_faq, tab_litreview,
            tab_mindmap, tab_audio, tab_compare, tab_kgraph,
            tab_timeline, tab_study_table, tab_pipeline,
            tab_research_report, tab_explain,
        ) = st.tabs([
            "Chat",
            "Summary",
            "FAQ",
            "Lit Review",
            "Mind Map",
            "Audio",
            "Compare",
            "Graph",
            "Timeline",
            "Study Table",
            "Pipeline",
            "Research Report",
            "Explain",
        ])

        # ── Tab 1: Chat ───────────────────────────────────────
        with tab_chat:
            for turn in notebook.get("conversation", []):
                role = turn.get("role", "user")
                with st.chat_message(role):
                    st.markdown(turn.get("content", ""))
                    if role == "assistant":
                        _render_citations(turn.get("citations") or [])
                        qs = turn.get("suggested_questions") or []
                        for q in qs:
                            if st.button(q, key=f"nb_sq_{hash(q + turn.get('content','')[:20])}"):
                                st.session_state["nb_pending_q"] = q
                                st.rerun()

            pending = st.session_state.pop("nb_pending_q", None)
            typed = st.chat_input(
                placeholder="Ask a question about your sources…", key="nb_chat_input",
            )
            message = render_chat_gate(pending or typed, key="nb_chat", settings=settings)
            if not message:
                pass  # fall through to other tabs rendering
            else:
                auto_web = st.session_state.get(f"nb_auto_web_{active_id}", False)

                # Auto-ingest any files waiting in the uploader before answering.
                # Filter out Streamlit DeletedFile sentinels (no .name / .getbuffer).
                _raw_pending = st.session_state.get(f"nb_files_{active_id}") or []
                pending_files = [f for f in _raw_pending if hasattr(f, "name") and hasattr(f, "getbuffer")]
                if pending_files:
                    with st.spinner(
                        f"Indexing {len(pending_files)} uploaded file(s) before answering…"
                    ):
                        processed = process_uploads(pending_files, settings)
                        added = _index_and_store(active_id, processed, settings, source_type="file")
                    if added > 0:
                        st.toast(f"✅ Indexed {added} new source(s) from upload.")
                        notebook = memory.load(active_id)  # reload with updated sources
                    elif processed:
                        st.toast("ℹ️ Uploaded files already in notebook (skipping duplicates).")

                if not notebook.get("sources") and not auto_web:
                    st.warning(
                        "Add at least one source before asking questions, "
                        "or enable **Auto web search** above to let the agent search Google."
                    )
                else:
                    with st.chat_message("user"):
                        st.markdown(message)

                    state = create_notebook_state(
                        user_message=message,
                        notebook_id=active_id,
                        model_name=settings["model"],
                        num_ctx=settings["num_ctx"],
                        embed_model=settings.get("embed_model", cfg.embedding_model),
                        top_k=settings.get("hybrid_top_k", cfg.hybrid_top_k),
                        include_web_search=auto_web,
                    )

                    labels = {
                        "retrieve": "Searching sources",
                        "answer": "Composing grounded answer",
                        "save": "Saving",
                    }
                    step_log = st.empty()
                    done: list = []

                    def _cb(node_name: str, _state: dict) -> None:
                        done.append(labels.get(node_name, node_name))
                        step_log.caption(" → ".join(done))

                    with st.spinner("Searching your notebook…"):
                        try:
                            final = run_notebook_turn(state, stream_callback=_cb)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            logger.exception("Notebook graph failed")
                            return
                    step_log.empty()

                    with st.chat_message("assistant"):
                        st.markdown(final.get("assistant_response", ""))
                        _render_citations(final.get("citations", []))
                        for q in final.get("suggested_questions", []):
                            if st.button(q, key=f"nb_newsq_{hash(q + message[:20])}"):
                                st.session_state["nb_pending_q"] = q
                                st.rerun()

                    render_eval_result(
                        dict(final).get("eval_result", {}),
                        key_suffix=f"_nb_chat_{active_id}",
                    )
                    render_rag_reflection(
                        dict(final).get("rag_reflection_info"),
                        key_suffix=f"_nb_chat_{active_id}",
                    )
                    st.rerun()

        # ── Tab 2: Cross-document summary ─────────────────────
        with tab_summary:
            _tab_cross_summary(active_id, notebook, settings)

        # ── Tab 3: FAQ ────────────────────────────────────────
        with tab_faq:
            _tab_faq(active_id, notebook, settings)

        # ── Tab 4: Literature review ──────────────────────────
        with tab_litreview:
            _tab_literature_review(active_id, notebook, settings)

        # ── Tab 5: Mind map ───────────────────────────────────
        with tab_mindmap:
            _tab_mindmap(active_id, notebook, settings)

        # ── Tab 6: Audio summary ──────────────────────────────
        with tab_audio:
            _tab_audio(active_id, notebook, settings)

        # ── Tab 7: Source comparison ──────────────────────────
        with tab_compare:
            _tab_compare(active_id, notebook, settings)

        # ── Tab 8: Knowledge graph ────────────────────────────
        with tab_kgraph:
            _tab_knowledge_graph(active_id, notebook, settings)

        # ── Tab 9: Timeline ───────────────────────────────────
        with tab_timeline:
            _tab_timeline(active_id, notebook, settings)

        # ── Tab 10: Study comparison table ────────────────────
        with tab_study_table:
            _tab_study_comparison(active_id, notebook, settings)

        # ── Tab 11: Full 7-agent pipeline ─────────────────────
        with tab_pipeline:
            _tab_pipeline(active_id, notebook, settings)

        # ── Tab 12: Research report (absorbs Modes 1 & 3) ────────────────────
        with tab_research_report:
            _tab_research_report(active_id, notebook, settings)

        # ── Tab 13: Explain (absorbs Mode 5 storytelling) ─────────────────────
        with tab_explain:
            _tab_explain(active_id, notebook, settings)
