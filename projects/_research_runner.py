"""projects/_research_runner.py — Shared research workflow runner for Modes 1-3.

Root cause of the previous bug
────────────────────────────────
st.rerun() inside a `while not done` loop restarts the entire Streamlit
script. On the next render the Run button is no longer pressed, so
run_research_workflow() is never re-entered and the poll loop never
continues. The fix: store all run context in session_state at launch,
then call resume_active_run() unconditionally at the top of each mode's
run() function so every rerender picks up the active run.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid

import streamlit as st

logger = logging.getLogger(__name__)

_STEP_LABELS = {
    "document_ingestion":    "Indexing documents (FAISS + BM25)",
    "query_generation":      "Generating search queries",
    "academic_search":       "Searching arXiv, Semantic Scholar, CrossRef",
    "web_search":            "Searching web (DuckDuckGo)",
    "document_analysis":     "Analysing documents (Hybrid RAG)",
    "reference_compilation": "Compiling references",
    "report_generation":     "Writing report",
    "research_eval":         "Evaluating output quality",
}


# ── Internal session-state key helpers ────────────────────────────────────────

def _ctx_key(mode_key: str) -> str:
    return f"_run_ctx_{mode_key}"


def _run_key(mode_key: str) -> str:
    return f"_run_state_{mode_key}"


# ── Public API ─────────────────────────────────────────────────────────────────

def has_active_run(mode_key: str) -> bool:
    """Return True if a workflow is currently in progress for *mode_key*.

    Check this BEFORE rendering any form elements so the form is never
    added to the Streamlit element tree during an active run — avoids
    the visual artifact where form + progress bar appear together.
    """
    return _ctx_key(mode_key) in st.session_state


def run_research_workflow(
    goal, uploaded_files, mode, clarifications, with_sr,
    settings, memory_dir, mode_key,
) -> None:
    """
    Start the research graph in a background thread and immediately rerun
    so that resume_active_run() takes over on the next render.

    All context needed for display is stored in session_state so it
    survives across st.rerun() calls.
    """
    from ui.helpers import process_uploads
    from agents.state import create_initial_state
    from agents.graph import run_research

    ck = _ctx_key(mode_key)
    rk = _run_key(mode_key)

    # Guard: don't start a second thread if one is already running
    if ck in st.session_state:
        return

    processed_docs: list = []
    if uploaded_files:
        with st.spinner("Processing uploaded documents…"):
            processed_docs = process_uploads(uploaded_files, settings)

    session_id = str(uuid.uuid4())[:8]
    initial_state = create_initial_state(
        goal=goal,
        uploaded_docs=processed_docs,
        mode=mode,
        include_web_search=settings["include_web"],
        session_id=session_id,
        model_name=settings["model"],
        num_ctx=settings["num_ctx"],
        embed_model=settings["embed_model"],
        style_profile=settings.get("style_profile"),
        clarifications=clarifications,
    )

    # run_data is a plain Python dict captured by both closures below.
    # The background thread mutates it directly — never via st.session_state —
    # because background threads lack a Streamlit ScriptRunContext, so
    # st.session_state writes from threads are silently dropped (causing the
    # "missing ScriptRunContext!" warning and progress stuck at 0%).
    run_data: dict = {
        "done":     False,
        "result":   None,
        "error":    None,
        "progress": 0,
        "status":   "Starting…",
        "log":      [],
    }

    # Store everything the display step will need.
    # run_data is stored by reference — the main thread reads the same object
    # that the background thread mutates.
    st.session_state[ck] = {
        "run_key":        rk,
        "session_id":     session_id,
        "goal":           goal,
        "with_sr":        with_sr,
        "mode_key":       mode_key,
        "memory_dir":     memory_dir,
        "mode":           mode,
        "processed_docs": processed_docs,
        "start":          time.time(),
    }
    st.session_state[rk] = run_data  # store reference, not a copy

    def _stream_callback(node_name: str, state: dict) -> None:
        """Update run_data directly — no st.session_state access from thread."""
        pct = state.get("progress_pct", 0)
        label = _STEP_LABELS.get(node_name, node_name.replace("_", " ").title())
        detail = state.get("status_detail", "")
        run_data["progress"] = pct
        run_data["status"]   = label
        run_data["detail"]   = detail
        entry = f"{label} ({pct}%)" + (f" — {detail}" if detail else "")
        run_data["log"].append(entry)

    def _run() -> None:
        try:
            result = run_research(initial_state, stream_callback=_stream_callback)
            run_data["result"] = result
        except Exception as exc:
            run_data["error"] = str(exc)
            logger.exception("Research workflow failed")
        finally:
            run_data["done"] = True

    threading.Thread(target=_run, daemon=True).start()
    st.rerun()


def resume_active_run(mode_key: str, settings: dict) -> bool:
    """
    Call this at the top of every mode's run() function.

    If a workflow is in progress for this mode:
      - While running: render a live progress bar and call st.rerun() to poll.
      - When done: render the full results.
      - Returns True so the caller can `return` immediately (skipping the form).

    If no run is active, returns False and the caller renders the form normally.
    """
    ck = _ctx_key(mode_key)
    ctx = st.session_state.get(ck)
    if not ctx:
        return False

    rk = ctx["run_key"]
    run_state = st.session_state.get(rk)
    if not run_state:
        # Stale context — clean up
        st.session_state.pop(ck, None)
        return False

    # ── Still running — show live progress ────────────────────────────────────
    if not run_state["done"]:
        pct    = run_state.get("progress", 0)
        status = run_state.get("status", "Starting…")
        log    = run_state.get("log", [])

        st.subheader("Research Workflow Running")
        st.progress(pct)
        detail = run_state.get("detail", "")
        if detail:
            st.markdown(f"**Running:** {status}  `{pct}%`  \n{detail}")
        else:
            st.markdown(f"**Running:** {status}  `{pct}%`")
        if log:
            with st.expander("Step-by-step Log", expanded=False):
                st.text("\n".join(log))

        time.sleep(2.0)
        st.rerun()
        return True  # unreachable after rerun, but satisfies type checker

    # ── Done — collect results and clean up ───────────────────────────────────
    final_state    = run_state.get("result")
    error          = run_state.get("error")
    processed_docs = ctx.get("processed_docs", [])

    # ── Persist result so feedback section survives rerenders ────────────────
    elapsed = time.time() - ctx["start"]
    st.session_state[f"_stored_{mode_key}"] = {
        "final_state":    dict(final_state) if final_state else {},
        "processed_docs": ctx.get("processed_docs", []),
        "session_id":     ctx["session_id"],
        "goal":           ctx["goal"],
        "with_sr":        ctx.get("with_sr", False),
        "mode":           ctx.get("mode", mode_key),
        "elapsed":        elapsed,
    }

    st.session_state.pop(rk, None)
    st.session_state.pop(ck, None)

    if error:
        st.error(f"Workflow error: {error}")
        return True

    st.progress(100)
    st.markdown(f"**Complete.** Finished in `{elapsed:.1f}s`")

    # Persist session to disk
    try:
        from agents.memory import ResearchMemory
        ResearchMemory(memory_dir=ctx["memory_dir"]).save_session(
            session_id=ctx["session_id"],
            goal=ctx["goal"],
            report=final_state.get("report", ""),
            references=final_state.get("references", []),
            key_findings=final_state.get("key_findings", []),
            document_names=[d.filename for d in processed_docs],
            mode=ctx["mode"],
            model_name=settings["model"],
        )
    except Exception as exc:
        logger.warning("Could not save research session: %s", exc)

    display_research_results(
        final_state, processed_docs,
        ctx["session_id"], settings, elapsed,
        ctx["goal"], ctx["with_sr"], mode_key,
    )
    return True


def show_stored_result(mode_key: str, settings: dict) -> bool:
    """
    Re-render a stored research result with a feedback section below it.

    Call after `has_active_run()` returns False — shows persistent results
    from the last completed run so the user can refine the output without
    re-running the full pipeline.

    Returns True if a stored result was found and rendered (caller should return).
    Returns False if nothing is stored (caller should show the input form).
    """
    stored = st.session_state.get(f"_stored_{mode_key}")
    if not stored:
        return False

    final_state    = stored["final_state"]
    processed_docs = stored["processed_docs"]
    session_id     = stored["session_id"]
    goal           = stored["goal"]

    if st.button("New Search", key=f"_new_search_{mode_key}"):
        # Clear stored result and all feedback state for this session
        st.session_state.pop(f"_stored_{mode_key}", None)
        for k in [k for k in st.session_state if k.endswith(f"_{session_id}")]:
            st.session_state.pop(k, None)
        st.rerun()
        return True

    display_research_results(
        final_state,
        processed_docs,
        session_id,
        settings,
        stored["elapsed"],
        goal,
        stored["with_sr"],
        mode_key,
    )

    # ── Feedback section ──────────────────────────────────────────────────────
    from ui.helpers import render_feedback_section

    output_key = f"_fb_output_{session_id}"
    current_report = st.session_state.get(output_key, final_state.get("report", ""))

    context_parts = [
        f"{r.get('title', '')} {r.get('abstract_snippet', '')}"
        for r in final_state.get("references", [])[:6]
    ]
    context = " ".join(context_parts)

    refined = render_feedback_section(
        current_report,
        session_key=session_id,
        mode="literature_search",
        model_name=settings["model"],
        num_ctx=settings["num_ctx"],
        context=context,
        key_suffix=f"_{mode_key}_{session_id}",
    )

    if refined != current_report:
        st.subheader("Refined Report")
        st.markdown(refined)
        st.download_button(
            label="Download Refined Report (Markdown)",
            data=refined,
            file_name=f"refined_report_{session_id}.md",
            mime="text/markdown",
            key=f"_dl_refined_{session_id}",
        )

    return True


# ── Display helpers (unchanged logic, extracted for reuse) ─────────────────────

def display_research_results(
    final_state, processed_docs, session_id, settings, elapsed, goal, with_sr, mode_key
):
    from ui.helpers import (
        render_eval_result, render_key_findings, render_rag_reflection,
        render_references, render_report,
    )

    st.divider()
    errors = final_state.get("errors", [])
    if errors:
        with st.expander("Warnings"):
            for e in errors:
                st.warning(e)

    render_key_findings(final_state.get("key_findings", []))
    st.divider()

    per_doc_analyses = final_state.get("per_doc_analyses", {})
    rag_label = "RAG Fallback" if final_state.get("rag_fallback") else "RAG Chunks"
    tab_labels = ["Full Report", "References", rag_label]
    if per_doc_analyses:
        tab_labels.insert(2, "Per-Document Analysis")

    result_tabs = st.tabs(tab_labels)
    tab_idx = 0
    with result_tabs[tab_idx]:
        render_report(final_state.get("report", ""), session_id)
    tab_idx += 1
    with result_tabs[tab_idx]:
        render_references(final_state.get("references", []), key_suffix=f"_{session_id}")
    tab_idx += 1
    if per_doc_analyses:
        with result_tabs[tab_idx]:
            st.subheader("Per-Document Analysis")
            for filename, doc_analysis in per_doc_analyses.items():
                with st.expander(filename):
                    st.markdown(doc_analysis)
            if final_state.get("multi_doc_synthesis"):
                st.divider()
                st.subheader("Cross-Document Synthesis")
                st.markdown(final_state["multi_doc_synthesis"])
        tab_idx += 1
    with result_tabs[tab_idx]:
        _render_rag_tab(final_state, processed_docs, settings)

    render_eval_result(final_state.get("eval_result", {}), key_suffix=f"_{mode_key}_{session_id}")
    render_rag_reflection(final_state.get("rag_reflection_info"), key_suffix=f"_{mode_key}_{session_id}")

    if with_sr:
        _run_inline_sr(goal, settings, session_id)

    st.divider()
    st.subheader("Workflow Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Documents", len(processed_docs))
    m2.metric("Queries", len(final_state.get("search_queries", [])))
    m3.metric("Papers Found", len(final_state.get("academic_papers", [])))
    m4.metric("References", len(final_state.get("references", [])))
    m5.metric("Time (s)", f"{elapsed:.1f}")


def _render_rag_tab(final_state, processed_docs, settings):
    retrieved_chunks = final_state.get("retrieved_chunks", [])
    rag_fallback = final_state.get("rag_fallback", False)
    embed_model = settings.get("embed_model", "nomic-embed-text")

    # ── Status banner ─────────────────────────────────────────────────────
    if not processed_docs:
        st.info(
            "No documents were uploaded — this run used search-only mode (no RAG).  \n"
            "Upload documents in Modes 1, 3, or 4 to activate Hybrid RAG."
        )
        return

    if rag_fallback:
        st.warning(
            f"**Vectorless fallback** was used — the embedding model `{embed_model}` "
            f"is not pulled in Ollama, so document text was injected directly without indexing.  \n"
            f"To enable full Hybrid RAG, run:  \n"
            f"```\nollama pull {embed_model}\n```"
        )
    else:
        total_chunks = len(retrieved_chunks)
        total_chars = sum(len(c["text"]) for c in retrieved_chunks)
        num_docs = len({c["doc_name"] for c in retrieved_chunks}) if retrieved_chunks else len(processed_docs)
        st.success(
            f"**Hybrid RAG active** — FAISS (dense) + BM25 (keyword) + ChromaDB (cache)  \n"
            f"Embedding model: `{embed_model}` · "
            f"{total_chunks} chunks retrieved across {num_docs} document(s) · "
            f"{total_chars:,} chars"
        )

    if retrieved_chunks:
        st.divider()
        docs_seen: dict = {}
        for chunk in retrieved_chunks:
            docs_seen.setdefault(chunk["doc_name"], []).append(chunk)
        for doc_name, chunks in docs_seen.items():
            with st.expander(f"{doc_name} — {len(chunks)} chunks"):
                for chunk in chunks:
                    score = chunk.get("rrf_score", 0.0)
                    st.markdown(
                        f"**Page {chunk['page_num']} · Chunk {chunk['chunk_index']}**"
                        f"  —  RRF score `{score:.4f}`"
                    )
                    st.text(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
                    st.divider()
    elif not rag_fallback:
        st.info("No chunks were retrieved for this run.")


def _run_inline_sr(goal, settings, session_id):
    from agents.systematic_review_graph import run_systematic_review
    from agents.systematic_review_state import create_systematic_review_state
    from ui.tabs.systematic_review import _render_prisma_flow, _render_evidence_table

    st.divider()
    st.subheader("Systematic Review (PRISMA)")
    sr_state = create_systematic_review_state(
        research_question=goal,
        model_name=settings["model"],
        num_ctx=settings["num_ctx"],
    )
    sr_status   = st.empty()
    sr_progress = st.progress(0)

    def _sr_cb(node_name, state):
        pct = state.get("progress_pct", 0)
        label = node_name.replace("_", " ").title()
        detail = state.get("status_detail", "")
        sr_progress.progress(pct)
        if detail:
            sr_status.markdown(f"**{label}…** `{pct}%`  \n{detail}")
        else:
            sr_status.markdown(f"**{label}…** `{pct}%`")

    try:
        sr_final = run_systematic_review(sr_state, stream_callback=_sr_cb)
    except Exception as exc:
        st.error(f"Systematic review error: {exc}")
        return

    sr_progress.progress(100)
    sr_status.markdown("**Systematic Review complete.**")
    _render_prisma_flow(sr_final.get("prisma_flow", {}))
    sr_t1, sr_t2 = st.tabs(["Synthesis", "Evidence Table"])
    with sr_t1:
        st.markdown(sr_final.get("narrative_synthesis", "*No synthesis.*"))
    with sr_t2:
        _render_evidence_table(sr_final.get("evidence_table", []))
