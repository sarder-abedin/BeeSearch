"""
ui/sidebar.py
─────────────
Sidebar: hardware detection, model/context/RAG settings,
writing style profile picker, recent research sessions.
"""

from __future__ import annotations

import logging

import streamlit as st

from agents.style_memory import StyleMemory
from config.hardware import (
    KNOWN_EMBED_MODELS, TIER_CONFIGS, detect_hardware, get_available_embed_models,
    get_available_models, get_recommended_tier, recommend_config,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()


@st.cache_data(ttl=30, show_spinner=False)
def _load_system_info():
    """Detect hardware + query Ollama for available models (cached 30 s)."""
    hw = detect_hardware()
    available = get_available_models(cfg.ollama_base_url)
    rec = recommend_config(hw, available)
    available_embed = get_available_embed_models(cfg.ollama_base_url)
    return hw, available, rec, available_embed


def render_sidebar() -> dict:
    """Render sidebar controls; return configuration dict."""
    with st.sidebar:
        st.title("Settings")

        hw, available_models, _cached_rec, available_embed = _load_system_info()

        if "hw_apply_model" in st.session_state:
            st.session_state["sidebar_model"] = st.session_state.pop("hw_apply_model")
        if "hw_apply_ctx" in st.session_state:
            st.session_state["sidebar_num_ctx"] = st.session_state.pop("hw_apply_ctx")
        if "hw_apply_all" in st.session_state:
            _all = st.session_state.pop("hw_apply_all")
            st.session_state["sidebar_model"] = _all["model"]
            st.session_state["sidebar_num_ctx"] = _all["num_ctx"]
            st.session_state["sidebar_top_k_applied"] = _all["hybrid_top_k"]
            st.session_state["sidebar_max_results_applied"] = _all["max_results"]
            st.session_state["sidebar_chunk_size_applied"] = _all["chunk_size"]
            st.session_state["sidebar_chunk_overlap_applied"] = _all["chunk_overlap"]

        # ── Apply RAM override to recommendation ──────────────
        _ram_override = st.session_state.get("hw_ram_override_gb", 0.0)
        if _ram_override and _ram_override != hw["ram_gb"]:
            hw_effective = dict(hw)
            hw_effective["ram_gb"] = float(_ram_override)
            rec = recommend_config(hw_effective, available_models)
            tier = get_recommended_tier(hw_effective)
        else:
            rec = _cached_rec
            tier = get_recommended_tier(hw)

        # ── Hardware panel ────────────────────────────────────
        gpu_labels = {
            "apple_silicon": "Apple Silicon (Metal)",
            "nvidia": "NVIDIA (CUDA)",
            "cpu": "CPU only",
        }
        with st.expander("Hardware", expanded=True):
            _display_ram = _ram_override if _ram_override else hw["ram_gb"]
            c1, c2 = st.columns(2)
            c1.metric(
                "RAM",
                f"{_display_ram:.0f} GB",
                delta="override" if _ram_override and _ram_override != hw["ram_gb"] else None,
            )
            c2.metric("Accelerator", gpu_labels.get(hw["gpu_type"], "Unknown"))
            st.caption(f"CPU: {hw['cpu']}")
            st.caption(f"OS: {hw['os']} ({hw['arch']})")
            _tier_colours = {"maximum": "green", "high": "blue", "standard": "orange", "low": "red"}
            _tc = _tier_colours.get(tier["tier"], "gray")
            st.markdown(
                f"**Performance tier:** :{_tc}[{tier['label']}]  \n"
                f"<small>{tier['description']}</small>",
                unsafe_allow_html=True,
            )

            # Docker RAM override — shown when running inside a container
            if hw.get("in_docker"):
                st.info(
                    f"Running in **Docker** — detected {hw['ram_gb']:.0f} GB "
                    "(container allocation, not host RAM).  \n"
                    "Enter your machine's actual RAM below for accurate model recommendations."
                )
                _override_val = float(_ram_override or hw["ram_gb"])
                _new_override = st.number_input(
                    "Actual RAM (GB)",
                    min_value=1.0,
                    max_value=512.0,
                    value=_override_val,
                    step=8.0,
                    key="hw_ram_input",
                    help=(
                        "Mac unified memory: 8 / 16 / 24 / 32 / 36 / 48 / 64 / 96 / 128 GB.  \n"
                        "Check Apple menu → About This Mac."
                    ),
                )
                _col_apply, _col_clear = st.columns(2)
                with _col_apply:
                    if st.button("Apply", key="apply_ram_override", use_container_width=True):
                        st.session_state["hw_ram_override_gb"] = float(_new_override)
                        st.cache_data.clear()
                        st.rerun()
                with _col_clear:
                    if _ram_override and st.button(
                        "Clear", key="clear_ram_override", use_container_width=True
                    ):
                        st.session_state.pop("hw_ram_override_gb", None)
                        st.cache_data.clear()
                        st.rerun()

        # ── Recommendation banner ─────────────────────────────
        st.markdown("#### Model Recommendation")
        if rec["can_run"]:
            tight = rec.get("tight_fit", False)
            safe_alt = rec.get("safe_alternative")

            if tight and safe_alt:
                st.warning(rec["reasoning"])
                options = [rec["model"], safe_alt["name"]]
                labels = {
                    rec["model"]: f"{rec['model']} — higher capability, tight fit",
                    safe_alt["name"]: (
                        f"{safe_alt['name']} — {safe_alt['ram_gb']} GB, "
                        f"more headroom ({safe_alt['note']})"
                    ),
                }
                chosen = st.radio(
                    "Select model:",
                    options=options,
                    format_func=lambda x: labels[x],
                    key="hw_model_choice",
                    index=0,
                )
                chosen_ctx = rec["num_ctx"] if chosen == rec["model"] else safe_alt["num_ctx"]
                col_apply, col_refresh = st.columns([2, 1])
                with col_apply:
                    if st.button("Apply Selection", key="apply_hw_rec", use_container_width=True):
                        st.session_state["hw_apply_model"] = chosen
                        st.session_state["hw_apply_ctx"] = chosen_ctx
                        st.rerun()
                with col_refresh:
                    if st.button("Refresh", key="refresh_hw",
                                 help="Re-check Ollama for new models",
                                 use_container_width=True):
                        st.cache_data.clear()
                        st.rerun()
            else:
                st.success(f"**{rec['model']}** — {rec['reasoning']}")
                col_apply, col_refresh = st.columns([2, 1])
                with col_apply:
                    if st.button("Apply Recommendation", key="apply_hw_rec", use_container_width=True):
                        st.session_state["hw_apply_model"] = rec["model"]
                        st.session_state["hw_apply_ctx"] = rec["num_ctx"]
                        st.rerun()
                with col_refresh:
                    if st.button("Refresh", key="refresh_hw",
                                 help="Re-check Ollama for new models",
                                 use_container_width=True):
                        st.cache_data.clear()
                        st.rerun()
        else:
            st.warning(f"No compatible models found. {rec['hardware_note']}")
            if rec["pull_command"]:
                st.markdown("**Pull the recommended model:**")
                st.code(rec["pull_command"], language="bash")
            if st.button("Refresh after pulling", key="refresh_hw_nopull", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        # ── Recommended configuration (all settings) ─────────
        st.divider()
        st.markdown("#### Recommended Configuration")
        st.caption(
            f"Tier **{tier['label']}** — {tier['description']}.  \n"
            "These settings are tuned for your hardware across all modes."
        )
        _rec_model = rec.get("model") or "—"
        _rec_ctx = tier["num_ctx"]
        _rec_top_k = tier["hybrid_top_k"]
        _rec_chunk = tier["chunk_size"]
        _rec_max = tier["max_results"]
        _r1, _r2 = st.columns(2)
        _r1.metric("Context (tokens)", f"{_rec_ctx:,}")
        _r2.metric("Chunks per query", _rec_top_k)
        _r3, _r4 = st.columns(2)
        _r3.metric("Chunk size (chars)", _rec_chunk)
        _r4.metric("Max papers", _rec_max)
        if _rec_model != "—":
            st.caption(f"Recommended model: **{_rec_model}**")
        if rec.get("can_run") and _rec_model != "—":
            if st.button(
                "Apply All Recommended Settings",
                key="apply_all_hw",
                use_container_width=True,
                type="primary",
                help=(
                    f"Sets model={_rec_model}, context={_rec_ctx:,} tokens, "
                    f"chunks/query={_rec_top_k}, chunk size={_rec_chunk}, "
                    f"max papers={_rec_max}"
                ),
            ):
                st.session_state["hw_apply_all"] = {
                    "model": _rec_model,
                    "num_ctx": _rec_ctx,
                    "hybrid_top_k": _rec_top_k,
                    "max_results": _rec_max,
                    "chunk_size": _rec_chunk,
                    "chunk_overlap": tier["chunk_overlap"],
                }
                st.rerun()

        # ── Model selector ────────────────────────────────────
        st.divider()
        st.markdown("#### LLM Model")

        if available_models:
            current = st.session_state.get("sidebar_model")
            if current and current not in available_models:
                st.session_state["sidebar_model"] = available_models[0]

            def _model_label(name: str) -> str:
                suffix = " (recommended)" if name == rec.get("model") else ""
                return f"{name}{suffix}"

            model = st.selectbox(
                "Active model",
                options=available_models,
                format_func=_model_label,
                key="sidebar_model",
                help="Only models already pulled via `ollama pull` appear here.",
            )
        else:
            st.warning(
                "Ollama is not reachable or no models are pulled. "
                "Start Ollama and pull a model, then click Refresh."
            )
            model = st.text_input(
                "Model name (manual entry)",
                value=rec.get("model") or "llama3.2:3b",
                key="sidebar_model_manual",
                help="Ollama is unreachable — enter the model name manually.",
            )

        # ── Context window ────────────────────────────────────
        st.divider()
        st.markdown("#### Context Window")
        ctx_options = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

        if "sidebar_num_ctx" in st.session_state:
            applied = st.session_state["sidebar_num_ctx"]
            if applied not in ctx_options:
                st.session_state["sidebar_num_ctx"] = min(ctx_options, key=lambda x: abs(x - applied))

        num_ctx = st.select_slider(
            "Tokens",
            options=ctx_options,
            key="sidebar_num_ctx",
            help=(
                "Controls how many retrieved chunks fit into the LLM context. "
                "Match to your model's maximum. Reduce if you hit out-of-memory errors."
            ),
        )

        # ── Hybrid RAG ────────────────────────────────────────
        st.divider()
        st.markdown("#### Hybrid RAG")
        st.caption(
            "Combines **dense search** (FAISS vector index) with **keyword search** (BM25), "
            "then merges results via Reciprocal Rank Fusion. "
            "Used automatically when you upload documents."
        )

        # Build selector: available models first, then unavailable
        _embed_options = []
        _embed_help_lines = []
        for m in KNOWN_EMBED_MODELS:
            pulled = m["name"] in available_embed
            label = m["name"] if pulled else f"{m['name']} (not pulled)"
            _embed_options.append(label)
            status = "Pulled" if pulled else "Not pulled"
            _embed_help_lines.append(
                f"{m['name']} — {m['dim']}d, {m['size_gb']} GB  [{status}]  {m['note']}"
            )

        # Default to first available model; fall back to first option
        _default_idx = 0
        for i, m in enumerate(KNOWN_EMBED_MODELS):
            if m["name"] in available_embed:
                _default_idx = i
                break

        _embed_label = st.selectbox(
            "Embedding model",
            _embed_options,
            index=_default_idx,
            help="\n".join(_embed_help_lines),
            key="sidebar_embed_model",
        )
        # Strip the "(not pulled)" suffix to get the actual model name
        embed_model = _embed_label.replace(" (not pulled)", "")

        # Status badge: is the chosen model ready?
        _chosen_available = embed_model in available_embed
        if _chosen_available:
            st.success(
                f"Hybrid RAG ready — FAISS + BM25 + ChromaDB  "
                f"(`{embed_model}`)"
            )
        else:
            st.warning(
                f"Embedding model not pulled — documents will use vectorless fallback.  \n"
                f"Run: `ollama pull {embed_model}`"
            )
            if st.button(
                "Copy pull command",
                key="copy_pull_cmd",
                help=f"ollama pull {embed_model}",
                use_container_width=True,
            ):
                st.code(f"ollama pull {embed_model}", language="bash")

        _top_k_default = st.session_state.pop("sidebar_top_k_applied", 8)
        hybrid_top_k = st.slider(
            "Chunks per query",
            min_value=3, max_value=20, value=_top_k_default,
            key="sidebar_hybrid_top_k",
            help=(
                "How many document chunks are retrieved per search query "
                "and passed to the LLM. Higher = more context, slower."
            ),
        )

        # ── Search settings ───────────────────────────────────
        st.divider()
        st.markdown("#### Search Settings")
        _max_results_default = st.session_state.pop("sidebar_max_results_applied", 6)
        max_results = st.slider("Max papers per query", 3, 15, _max_results_default,
                                key="sidebar_max_results")
        include_web = st.toggle(
            "Include Google web search",
            value=False,
            help=(
                "Supplements academic results with Google Search via the FastAPI service. "
                "Start with: uvicorn tools.google_search_service:app --port 8000"
            ),
        )
        include_crossref = st.toggle(
            "Include CrossRef search",
            value=False,
            help="Adds CrossRef (DOI database) to literature search (slower).",
        )

        # ── Document settings ─────────────────────────────────
        st.divider()
        st.markdown("#### Document Settings")

        use_docling = st.toggle(
            "Advanced Parsing (Docling)",
            value=True,
            key="sidebar_use_docling",
            help=(
                "Docling provides layout-aware PDF parsing, table extraction as "
                "Markdown, and support for PPTX, XLSX, HTML, and image files. "
                "Models are cached in `models/docling/` after first download."
            ),
        )

        if use_docling:
            use_ocr = st.toggle(
                "Enable OCR (slower)",
                value=False,
                key="sidebar_use_ocr",
                help=(
                    "Extracts text from scanned PDFs and image files using EasyOCR. "
                    "Requires an additional ~200 MB model download on first use."
                ),
            )
            st.caption(
                "Docling models cached in `models/docling/`.  \n"
                "First run may take a few minutes while models download."
            )
            # Chunk size/overlap are not used by Docling's HybridChunker, but we
            # store the tier-recommended value so it takes effect if Docling is
            # disabled later.
            chunk_size = st.session_state.pop("sidebar_chunk_size_applied", 800)
            chunk_overlap = st.session_state.pop("sidebar_chunk_overlap_applied", 150)
        else:
            use_ocr = False
            _chunk_size_default = st.session_state.pop("sidebar_chunk_size_applied", 800)
            _chunk_overlap_default = st.session_state.pop("sidebar_chunk_overlap_applied", 150)
            chunk_size = st.slider("Chunk size (chars)", 400, 1200, _chunk_size_default,
                                   step=100, key="sidebar_chunk_size")
            chunk_overlap = st.slider("Chunk overlap (chars)", 50, 300, _chunk_overlap_default,
                                      step=25, key="sidebar_chunk_overlap")

        # ── Writing Style Profile ─────────────────────────────
        st.divider()
        st.markdown("#### Writing Style Profile")
        _style_mem = StyleMemory()
        _all_profiles = _style_mem.list_profiles()
        _profile_options = ["None (default style)"] + [p["name"] for p in _all_profiles]
        _selected_profile_name = st.selectbox(
            "Active profile",
            options=_profile_options,
            key="sidebar_style_profile",
            help=(
                "Select a writing style profile to make all AI-generated prose match "
                "your writing style. Create profiles via 'Manage Writing Style Profiles' on the home page."
            ),
        )
        _active_profile: dict | None = None
        if _selected_profile_name != "None (default style)":
            _active_profile = _style_mem.load_by_name(_selected_profile_name)
            if _active_profile:
                st.caption(
                    f"Active: {_selected_profile_name} "
                    f"({len(_active_profile.get('sample_documents', []))} document(s) analysed)"
                )
            else:
                st.caption("Profile not found — please re-create it.")
                _active_profile = None

        # ── Safe Shutdown ─────────────────────────────────────
        st.divider()
        st.markdown("#### Safe Shutdown")
        st.caption(
            "Flushes ChromaDB, releases ports 8501, 8000, and 11434, "
            "then stops the server cleanly."
        )
        if not st.session_state.get("_shutdown_confirm"):
            if st.button(
                "Shut Down Safely",
                key="sidebar_shutdown_btn",
                use_container_width=True,
                help="Click once to arm, then confirm.",
            ):
                st.session_state["_shutdown_confirm"] = True
                st.rerun()
        else:
            st.warning("This will stop the server. Sessions are saved to disk.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Confirm Shutdown", key="shutdown_yes", type="primary", use_container_width=True):
                    import os
                    import signal as _signal
                    from tools.shutdown import safe_shutdown
                    st.info("Shutting down — you can close this tab.")
                    safe_shutdown(ports=[8000, 11434], flush_db=True)
                    os.kill(os.getpid(), _signal.SIGTERM)
            with col_no:
                if st.button("Cancel", key="shutdown_no", use_container_width=True):
                    st.session_state.pop("_shutdown_confirm", None)
                    st.rerun()

        # ── Recent Research Sessions ──────────────────────────
        st.divider()
        st.markdown("#### Recent Sessions")
        try:
            from agents.memory import ResearchMemory
            recent_sessions = ResearchMemory().list_sessions(limit=5)
            if recent_sessions:
                for sess in recent_sessions:
                    mode_label = {"document": "Doc", "search": "Search", "hybrid": "Hybrid"}.get(
                        sess.get("mode", ""), "Research"
                    )
                    goal_text = sess["goal"][:42] + ("…" if len(sess["goal"]) > 42 else "")
                    date_str = sess.get("created_at", "")[:10]
                    ref_count = sess.get("reference_count", 0)
                    if st.button(
                        f"[{mode_label}] {goal_text}\n{date_str} · {ref_count} refs",
                        key=f"load_rsess_{sess['session_id']}",
                        use_container_width=True,
                        help=f"Reload session from {date_str}",
                    ):
                        st.session_state["load_research_session_id"] = sess["session_id"]
                        st.rerun()
            else:
                st.caption("No saved sessions yet. Run a research workflow to save one.")
        except Exception as e:
            logger.debug("Could not load recent sessions: %s", e)

        st.divider()
        st.markdown("#### About")
        st.markdown(
            "**Agentic Research Assistant**  \n"
            "Local AI · Ollama · LangGraph · Hybrid RAG  \n"
            "arXiv · Semantic Scholar · CrossRef"
        )

    return {
        "model": model,
        "num_ctx": num_ctx,
        "embed_model": embed_model,
        "hybrid_top_k": hybrid_top_k,
        "max_results": max_results,
        "include_web": include_web,
        "include_crossref": include_crossref,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_docling": use_docling,
        "use_ocr": use_ocr,
        "style_profile": _active_profile,
    }
