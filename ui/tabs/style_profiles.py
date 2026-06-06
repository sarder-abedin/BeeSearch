"""ui/tabs/style_profiles.py — Writing Style Profiles management tab"""

from __future__ import annotations

import logging

import streamlit as st

from agents.style_memory import StyleMemory
from config.settings import get_settings
from ui.helpers import get_supported_file_types, process_uploads

logger = logging.getLogger(__name__)
cfg = get_settings()


def tab_style_profiles(settings: dict) -> None:
    """
    Style Profiles management tab.

    Lets users create named writing style profiles from their own document samples.
    The active profile is selected in the sidebar and injected into every LLM
    writing call across all modes.
    """
    st.header("Writing Style Profiles")
    st.markdown(
        """
Upload 2–5 of your own documents (past papers, reports, proposals) to teach the assistant
your writing style. The AI will analyse tone, structure, vocabulary, and citation patterns,
then apply that style to every report and proposal it writes — in all research modes.

**No model retraining required.** Style profiles work via prompt injection and are
stored as JSON files in `outputs/memory/style_*.json`.
"""
    )

    style_mem = StyleMemory()
    sub_create, sub_manage = st.tabs(["Create New Profile", "Manage Profiles"])

    # ── Create ─────────────────────────────────────────────────────────────────
    with sub_create:
        st.markdown("### New Style Profile")

        profile_name = st.text_input(
            "Profile name",
            placeholder="e.g. Academic Writing, Grant Proposals, Technical Reports",
            key="style_profile_name",
        )

        st.markdown("**Upload writing samples** *(2–5 documents recommended)*")
        st.caption(
            "Use documents that represent how *you* write — past papers, reports, proposals. "
            "The system reads the first ~3,000 characters per document."
        )
        style_docs = st.file_uploader(
            "Upload writing samples",
            type=get_supported_file_types(settings.get("use_docling", True)),
            accept_multiple_files=True,
            key="style_profile_uploads",
            label_visibility="collapsed",
        )

        if style_docs:
            st.success(f"{len(style_docs)} document(s) ready for analysis.")
            for f in style_docs:
                st.caption(f"  • {f.name}")

        create_btn = st.button(
            "Analyse Style & Create Profile",
            key="create_style_profile",
            type="primary",
            use_container_width=True,
            disabled=not (profile_name.strip() and style_docs),
        )

        if create_btn:
            if not profile_name.strip():
                st.warning("Please enter a profile name.")
            elif not style_docs:
                st.warning("Please upload at least one writing sample document.")
            else:
                with st.spinner(
                    f"Analysing writing style across {len(style_docs)} document(s)…  "
                    "This takes ~30–60 seconds."
                ):
                    try:
                        processed = process_uploads(style_docs, settings)
                        if not processed:
                            st.error("Could not extract text from any of the uploaded files.")
                        else:
                            profile_id = style_mem.create_profile(
                                name=profile_name.strip(),
                                documents=processed,
                                model_name=settings["model"],
                                ollama_base_url=cfg.ollama_base_url,
                                num_ctx=settings["num_ctx"],
                            )
                            profile = style_mem.load(profile_id)
                            st.success(
                                f"Profile **'{profile_name.strip()}'** created! "
                                f"Select it in the sidebar to activate it."
                            )
                            if profile:
                                with st.expander("Style Analysis Summary"):
                                    analysis = profile.get("analysis", {})
                                    tone  = analysis.get("tone_formality", {})
                                    struct = analysis.get("structure_format", {})
                                    vocab = analysis.get("vocabulary_complexity", {})
                                    cit   = analysis.get("citation_evidence", {})

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Tone & Formality**")
                                        if tone.get("register"):
                                            st.caption(f"Register: {tone['register']}")
                                        if tone.get("person"):
                                            st.caption(f"Person: {tone['person']}")
                                        if tone.get("hedging"):
                                            st.caption(f"Hedging: {tone['hedging']}")
                                        st.markdown("**Vocabulary**")
                                        if vocab.get("technical_density"):
                                            st.caption(f"Technical density: {vocab['technical_density']}")
                                        if vocab.get("avoids"):
                                            st.caption(f"Avoids: {vocab['avoids']}")
                                    with col2:
                                        st.markdown("**Structure & Format**")
                                        if struct.get("paragraph_length"):
                                            st.caption(f"Paragraphs: {struct['paragraph_length']}")
                                        if struct.get("transitions"):
                                            st.caption(f"Transitions: {struct['transitions']}")
                                        st.markdown("**Citation Style**")
                                        if cit.get("citation_density"):
                                            st.caption(f"Density: {cit['citation_density']}")
                                        if cit.get("citation_placement"):
                                            st.caption(f"Placement: {cit['citation_placement']}")

                                with st.expander("Injection Prompt"):
                                    st.text(profile.get("injection_prompt", ""))
                            st.rerun()
                    except Exception as e:
                        st.error(f"Style analysis failed: {e}")
                        logger.exception("Style profile creation failed")

    # ── Manage ─────────────────────────────────────────────────────────────────
    with sub_manage:
        profiles = style_mem.list_profiles()

        if not profiles:
            st.info(
                "No style profiles yet. Create one in the 'Create New Profile' tab "
                "by uploading 2–5 of your own writing samples."
            )
            return

        st.markdown(f"**{len(profiles)} saved profile(s)**")
        st.caption(
            "Select a profile in the sidebar (Settings → Writing Style Profile) to activate it. "
            "The active profile is injected into every LLM writing call."
        )

        for p in profiles:
            with st.expander(
                f"**{p['name']}** | `{p['profile_id']}` | "
                f"{len(p.get('sample_documents', []))} doc(s) | {p['created_at'][:10]}"
            ):
                col1, col2 = st.columns([4, 1])
                with col1:
                    docs = p.get("sample_documents", [])
                    if docs:
                        st.markdown("**Based on:**")
                        for d in docs:
                            st.caption(f"  • {d}")
                    full = style_mem.load(p["profile_id"])
                    if full:
                        analysis = full.get("analysis", {})
                        if analysis:
                            tone = analysis.get("tone_formality", {})
                            if tone.get("register"):
                                st.caption(f"Tone: {tone['register']} | Person: {tone.get('person','')}")
                            struct = analysis.get("structure_format", {})
                            if struct.get("paragraph_length"):
                                st.caption(f"Structure: {struct['paragraph_length']}")
                        with st.expander("Show injection prompt"):
                            st.text(full.get("injection_prompt", ""))
                with col2:
                    if st.button(
                        "Delete",
                        key=f"del_style_{p['profile_id']}",
                        use_container_width=True,
                    ):
                        style_mem.delete(p["profile_id"])
                        st.success(f"Deleted '{p['name']}'")
                        st.rerun()
