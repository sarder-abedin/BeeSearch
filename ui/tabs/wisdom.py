"""ui/tabs/wisdom.py — Mode 6: Wisdom Mode"""

from __future__ import annotations

import logging

import streamlit as st

from agents.wisdom_graph import run_wisdom_turn
from agents.wisdom_memory import WisdomMemory
from agents.wisdom_state import create_wisdom_state
from agents.systematic_review_graph import run_systematic_review
from agents.systematic_review_state import create_systematic_review_state
from ui.helpers import get_supported_file_types, process_uploads, render_clarification_form, render_eval_result, render_rag_reflection

logger = logging.getLogger(__name__)


def render_wisdom_output(state_or_output: dict, key_suffix: str = "") -> None:
    """
    Render the four Wisdom tabs: Scientific View, Simple View, Action Steps, Validation.

    Accepts either a WisdomState dict (fresh generation) or the `wisdom_output`
    sub-dict loaded from memory (session reload).
    """
    if "wisdom_output" in state_or_output:
        wo = state_or_output["wisdom_output"]
        val = wo.get("validation", {})
        deep_understanding   = wo.get("deep_understanding", "")
        simple_explanation   = wo.get("simple_explanation", "")
        actionable_takeaways = wo.get("actionable_takeaways", [])
        wisdom_claims        = val.get("claims", [])
        devils_advocate      = val.get("devils_advocate", "")
        overall_confidence   = val.get("overall_confidence", "Medium")
    else:
        deep_understanding   = state_or_output.get("deep_understanding", "")
        simple_explanation   = state_or_output.get("simple_explanation", "")
        actionable_takeaways = state_or_output.get("actionable_takeaways", [])
        wisdom_claims        = state_or_output.get("wisdom_claims", [])
        devils_advocate      = state_or_output.get("devils_advocate", "")
        overall_confidence   = state_or_output.get("overall_confidence", "Medium")

    if not deep_understanding:
        return

    conf_label = {"High": "High", "Medium": "Medium", "Low": "Low"}.get(overall_confidence, "Medium")

    w1, w2, w3, w4 = st.tabs([
        "Scientific View",
        "Simple View",
        "Action Steps",
        f"Validation ({conf_label})",
    ])

    with w1:
        st.markdown(deep_understanding)
    with w2:
        st.markdown(simple_explanation)
    with w3:
        if actionable_takeaways:
            for i, step in enumerate(actionable_takeaways, 1):
                st.markdown(f"**{i}.** {step}")
        else:
            st.info("No action steps were generated for this session.")
    with w4:
        st.markdown(f"**Overall Confidence:** {conf_label}")
        if wisdom_claims:
            st.divider()
            for claim in wisdom_claims:
                conf  = claim.get("confidence", "Medium")
                st.markdown(
                    f"**{conf}** &nbsp;|&nbsp; *{claim.get('consensus', '')}*"
                )
                st.markdown(f"> {claim.get('claim', '')}")
                if claim.get("source"):
                    st.caption(claim['source'])
                st.divider()
        if devils_advocate:
            st.markdown("**Devil's Advocate**")
            st.info(devils_advocate)


def tab_wisdom(settings: dict) -> None:
    """
    Mode 6 — Wisdom Mode.

    Phase 1 (Clarification): agent asks up to 3 Socratic questions.
    Phase 2 (Generation): searches literature, synthesises wisdom, validates claims.
    Phase 3 (Follow-up): full chat Q&A with wisdom output as context.

    Long-term memory: each session persists as outputs/memory/wisdom_<id>.json.
    Sessions are created automatically on the first message — no pre-entry required.
    """
    st.header("Mode 6 — Wisdom Mode")
    st.markdown(
        """
Transform knowledge into wisdom. Describe a scenario or question — the agent will
ask a few clarifying questions, then search the **scientific literature** to synthesise:

- **Scientific explanation** — mechanisms, evidence, citations
- **Simple version** — everyday language, concrete analogies
- **Action steps** — evidence-based things you can do right now
- **Validation** — confidence scores, consensus labels, devil's advocate

Conversations are **saved automatically** and past sessions quietly inform new ones.
"""
    )

    memory = WisdomMemory()
    ctrl_col, chat_col = st.columns([1, 2])

    # ── Left panel ────────────────────────────────────────────────────────────
    with ctrl_col:
        st.markdown("#### Session")

        sessions = memory.list_sessions()
        session_options: dict = {"New session": None}
        for s in sessions:
            phase_label = "[Done]" if s["phase"] == "done" else "[Active]"
            label = f"{phase_label} {s['topic'][:32]} ({s['created_at'][:10]})"
            session_options[label] = s["session_id"]

        chosen_label = st.selectbox(
            "Load session",
            list(session_options.keys()),
            key="wisdom_session_selector",
        )
        active_session_id = session_options[chosen_label]

        # ── New session: optional document upload only ────────────────────────
        if active_session_id is None:
            st.caption("Type your question in the chat to start a new session automatically.")
            with st.expander("Upload documents (optional)"):
                wisdom_files = st.file_uploader(
                    "Upload context documents",
                    type=get_supported_file_types(settings.get("use_docling", True)),
                    accept_multiple_files=True,
                    key="wisdom_uploads",
                )
            # Store files in session_state so the chat handler can access them
            if wisdom_files:
                st.session_state["wisdom_pending_files"] = wisdom_files
            elif "wisdom_pending_files" not in st.session_state:
                st.session_state["wisdom_pending_files"] = []

        # ── Existing session info ─────────────────────────────────────────────
        if active_session_id:
            session_data = memory.load(active_session_id)
            if session_data:
                # ── Inline rename ─────────────────────────────────────────────
                with st.expander("Rename session"):
                    new_name = st.text_input(
                        "Session name",
                        value=session_data.get("topic", ""),
                        key=f"wisdom_rename_{active_session_id}",
                        label_visibility="collapsed",
                    )
                    if st.button("Save name", key=f"wisdom_save_name_{active_session_id}"):
                        if new_name.strip():
                            memory.rename(active_session_id, new_name.strip())
                            st.success("Renamed.")
                            st.rerun()

                tags = session_data.get("topic_tags", [])
                if tags:
                    st.markdown("**Topic tags:**")
                    st.caption(", ".join(tags[:8]))

                phase = session_data.get("phase", "clarifying")
                phase_display = {
                    "clarifying": "Awaiting clarification",
                    "done": "Wisdom generated",
                }.get(phase, phase)
                st.caption(f"**Phase:** {phase_display}")

                papers = session_data.get("knowledge_base", {}).get("papers", [])
                if papers:
                    st.caption(f"**Sources used:** {len(papers)}")

                # ── Related sessions ──────────────────────────────────────────
                if tags:
                    related = memory.find_related_sessions(tags, active_session_id, limit=3)
                    if related:
                        st.markdown("**Related sessions:**")
                        for rel in related:
                            rel_label = f"{rel['topic'][:35]}… ({rel.get('created_at','')[:10]})"
                            rel_phase = "[Done]" if rel.get("phase") == "done" else "[Active]"
                            if st.button(
                                f"{rel_phase} {rel_label}",
                                key=f"load_related_{rel['session_id']}",
                                help="Load this related session",
                                use_container_width=True,
                            ):
                                st.session_state["wisdom_session_id"] = rel["session_id"]
                                st.rerun()

                # ── Systematic Review on topic ────────────────────────────────
                st.markdown("#### Systematic Review")
                if st.button(
                    "Run PRISMA Review on this topic",
                    key="wisdom_run_sr",
                    help="Run a full PRISMA systematic review on the wisdom topic.",
                    use_container_width=True,
                ):
                    st.session_state["wisdom_trigger_sr"] = session_data.get("topic", "")

                st.divider()
                if st.button("Delete Session", key="wisdom_delete"):
                    memory.delete(active_session_id)
                    st.session_state.pop("wisdom_session_id", None)
                    st.rerun()

    # ── Resolve active session ─────────────────────────────────────────────────
    effective_session_id = active_session_id or st.session_state.get("wisdom_session_id")

    with chat_col:
        if effective_session_id:
            session_data = memory.load(effective_session_id)
            if not session_data:
                st.error("Session not found. Please create a new session.")
                return

            topic = session_data.get("topic", "")
            st.markdown(f"**Topic:** {topic}")
            if session_data.get("document_names"):
                st.caption(f"Documents: {', '.join(session_data['document_names'])}")

            st.divider()

            wisdom_output = session_data.get("wisdom_output", {})
            conversation = session_data.get("conversation", [])

            for turn in conversation:
                role      = turn.get("role", "user")
                content   = turn.get("content", "")
                has_wisdom = turn.get("has_wisdom", False)
                with st.chat_message(role):
                    st.markdown(content)
                    if has_wisdom and wisdom_output:
                        render_wisdom_output(
                            {"wisdom_output": wisdom_output},
                            key_suffix=f"_hist_{effective_session_id}",
                        )
        else:
            st.info("Load an existing session or type your question below to start a new one.")

        # ── Chat input (works for both existing and new sessions) ─────────────
        placeholder = (
            f"Ask or answer about {session_data.get('topic', '')}…"
            if effective_session_id
            else "What would you like to explore? Type to start a new session…"
        )

        user_input = st.chat_input(
            placeholder=placeholder,
            key="wisdom_chat_input",
        )

        if not user_input:
            # ── Handle PRISMA trigger (no chat input needed) ──────────────────
            sr_topic = st.session_state.pop("wisdom_trigger_sr", None)
            if sr_topic:
                _run_wisdom_systematic_review(sr_topic, settings)
            return

        # ── Auto-create session if none active ────────────────────────────────
        if not effective_session_id:
            auto_topic = user_input.strip()[:60].rstrip("?").strip() or "Wisdom Session"
            doc_context = ""
            doc_names: list = []
            pending_files = st.session_state.pop("wisdom_pending_files", [])
            if pending_files:
                with st.spinner("Processing documents…"):
                    processed = process_uploads(pending_files, settings)
                    doc_names = [d.filename for d in processed]
                    doc_context = "\n\n---\n".join(
                        d.raw_text[:2000] for d in processed
                    )[:3000]

            effective_session_id = memory.new_session(
                topic=auto_topic,
                scenario=user_input.strip(),
                document_context=doc_context,
                document_names=doc_names,
            )
            st.session_state["wisdom_session_id"] = effective_session_id
            session_data = memory.load(effective_session_id)

        topic = session_data.get("topic", "") if session_data else ""

        with st.chat_message("user"):
            st.markdown(user_input)

        initial_state = create_wisdom_state(
            user_message=user_input,
            session_id=effective_session_id,
            topic=topic,
            model_name=settings["model"],
            num_ctx=settings["num_ctx"],
            clarifications=st.session_state.get(
                f"wisdom_clarifications_{effective_session_id}", {}
            ),
        )

        node_labels = {
            "context_loader":   "Loading session context",
            "clarification":    "Wisdom Oracle is thinking",
            "knowledge_search": "Searching scientific literature",
            "wisdom_synthesis": "Synthesising wisdom",
            "wisdom_validator": "Validating claims",
            "wisdom_followup":  "Preparing follow-up",
            "memory_saver":     "Saving session",
            "wisdom_eval":      "Evaluating output quality",
        }

        step_log = st.empty()
        completed: list = []

        def _wisdom_stream_callback(node_name: str, _state: dict) -> None:
            completed.append(node_labels.get(node_name, node_name))
            step_log.caption(" → ".join(completed))

        with st.spinner("Wisdom Oracle is working…"):
            try:
                final_state = run_wisdom_turn(
                    initial_state,
                    stream_callback=_wisdom_stream_callback,
                )
            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Wisdom graph failed")
                return

        step_log.empty()

        assistant_response = final_state.get("assistant_response", "")
        phase = final_state.get("phase", "clarifying")

        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            if phase == "done" and final_state.get("deep_understanding"):
                render_wisdom_output(final_state, key_suffix=f"_new_{effective_session_id}")
            render_eval_result(
                final_state.get("eval_result", {}),
                key_suffix=f"_wisdom_{effective_session_id}",
            )
            render_rag_reflection(
                final_state.get("rag_reflection_info"),
                key_suffix=f"_wisdom_{effective_session_id}",
            )

        # Feedback refinement — only available after full wisdom generation
        if phase == "done" and final_state.get("deep_understanding"):
            from ui.helpers import render_feedback_section
            context = "\n".join(
                p.get("title", "") + ": " + p.get("abstract", "")[:200]
                for p in final_state.get("academic_papers", [])[:4]
            )
            render_feedback_section(
                current_output=assistant_response,
                session_key=f"wisdom_{effective_session_id}",
                mode="wisdom",
                model_name=initial_state.get("model_name", "llama3.1:8b"),
                num_ctx=initial_state.get("num_ctx", 32768),
                context=context,
                key_suffix=f"_wisdom_{effective_session_id}",
            )

        st.rerun()  # refresh chat after turn


def _run_wisdom_systematic_review(topic: str, settings: dict) -> None:
    """Run and render a PRISMA systematic review on a wisdom topic."""
    from ui.tabs.systematic_review import _render_prisma_flow, _render_evidence_table

    st.divider()
    st.subheader(f"Systematic Review — {topic}")

    sr_state = create_systematic_review_state(
        research_question=topic,
        model_name=settings["model"],
        num_ctx=settings["num_ctx"],
    )
    sr_status = st.empty()
    sr_progress = st.progress(0)

    def _cb(node_name: str, state: dict) -> None:
        pct = state.get("progress_pct", 0)
        sr_progress.progress(pct)
        sr_status.markdown(f"**{node_name.replace('_', ' ').title()}…** `{pct}%`")

    try:
        sr_final = run_systematic_review(sr_state, stream_callback=_cb)
    except Exception as exc:
        st.error(f"Systematic Review error: {exc}")
        return

    sr_progress.progress(100)
    sr_status.markdown("**Systematic Review complete.**")
    _render_prisma_flow(sr_final.get("prisma_flow", {}))

    sr_t1, sr_t2 = st.tabs(["Synthesis", "Evidence Table"])
    with sr_t1:
        for theme in sr_final.get("key_themes", []):
            st.markdown(f"- {theme}")
        if sr_final.get("key_themes"):
            st.divider()
        st.markdown(sr_final.get("narrative_synthesis", "*No synthesis generated.*"))
        if sr_final.get("conclusion"):
            st.divider()
            st.markdown(f"**Conclusion:** {sr_final['conclusion']}")
    with sr_t2:
        _render_evidence_table(sr_final.get("evidence_table", []))
