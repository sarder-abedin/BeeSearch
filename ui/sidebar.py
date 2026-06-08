"""
ui/sidebar.py — BeeSearch sidebar: hardware detection, model settings.
"""

from __future__ import annotations

import logging

import streamlit as st

from config.hardware import (
    KNOWN_EMBED_MODELS, TIER_CONFIGS, detect_hardware, get_available_embed_models,
    get_available_models, get_recommended_tier, recommend_config,
)
from config.settings import get_settings
from ui.glossary import term_help

logger = logging.getLogger(__name__)
cfg = get_settings()


@st.cache_data(ttl=30, show_spinner=False)
def _load_system_info():
    hw = detect_hardware()
    available = get_available_models(cfg.ollama_base_url)
    rec = recommend_config(hw, available)
    available_embed = get_available_embed_models(cfg.ollama_base_url)
    return hw, available, rec, available_embed


def render_sidebar() -> dict:
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
            st.session_state["sidebar_large_doc_threshold_applied"] = _all["large_doc_page_threshold"]

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
            c1.metric("RAM", f"{_display_ram:.0f} GB",
                      delta="override" if _ram_override and _ram_override != hw["ram_gb"] else None)
            c2.metric("Accelerator", gpu_labels.get(hw["gpu_type"], "Unknown"))
            st.caption(f"CPU: {hw['cpu']}")
            st.caption(f"OS: {hw['os']} ({hw['arch']})")
            _tc = {"maximum": "green", "high": "blue", "standard": "orange", "low": "red"}.get(tier["tier"], "gray")
            st.markdown(
                f"**Performance tier:** :{_tc}[{tier['label']}]  \n"
                f"<small>{tier['description']}</small>",
                unsafe_allow_html=True,
                help=term_help("Hardware tier"),
            )

            if hw.get("in_docker"):
                st.info(
                    f"Running in **Docker** — detected {hw['ram_gb']:.0f} GB "
                    "(container allocation, not host RAM).  \n"
                    "Enter your machine's actual RAM below for accurate model recommendations."
                )
                _override_val = float(_ram_override or hw["ram_gb"])
                _new_override = st.number_input(
                    "Actual RAM (GB)", min_value=1.0, max_value=512.0,
                    value=_override_val, step=8.0, key="hw_ram_input",
                )
                _col_apply, _col_clear = st.columns(2)
                with _col_apply:
                    if st.button("Apply", key="apply_ram_override", use_container_width=True):
                        st.session_state["hw_ram_override_gb"] = float(_new_override)
                        st.cache_data.clear()
                        st.rerun()
                with _col_clear:
                    if _ram_override and st.button("Clear", key="clear_ram_override", use_container_width=True):
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
                    safe_alt["name"]: f"{safe_alt['name']} — {safe_alt['ram_gb']} GB, more headroom",
                }
                chosen = st.radio("Select model:", options=options,
                                  format_func=lambda x: labels[x],
                                  key="hw_model_choice", index=0)
                chosen_ctx = rec["num_ctx"] if chosen == rec["model"] else safe_alt["num_ctx"]
                col_apply, col_refresh = st.columns([2, 1])
                with col_apply:
                    if st.button("Apply Selection", key="apply_hw_rec", use_container_width=True):
                        st.session_state["hw_apply_model"] = chosen
                        st.session_state["hw_apply_ctx"] = chosen_ctx
                        st.rerun()
                with col_refresh:
                    if st.button("Refresh", key="refresh_hw", use_container_width=True):
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
                    if st.button("Refresh", key="refresh_hw", use_container_width=True):
                        st.cache_data.clear()
                        st.rerun()
        else:
            st.warning(f"No compatible models found. {rec['hardware_note']}")
            if rec["pull_command"]:
                st.code(rec["pull_command"], language="bash")
            if st.button("Refresh after pulling", key="refresh_hw_nopull", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        # ── Recommended configuration ─────────────────────────
        st.divider()
        st.markdown("#### Recommended Configuration")
        _rec_model = rec.get("model") or "—"
        _rec_ctx = tier["num_ctx"]
        _rec_top_k = tier["hybrid_top_k"]
        _rec_chunk = tier["chunk_size"]
        _rec_max = tier["max_results"]
        _rec_threshold = tier["large_doc_page_threshold"]
        _r1, _r2 = st.columns(2)
        _r1.metric("Context (tokens)", f"{_rec_ctx:,}")
        _r2.metric("Chunks per query", _rec_top_k)
        _r3, _r4 = st.columns(2)
        _r3.metric("Chunk size (chars)", _rec_chunk)
        _r4.metric("Max papers", _rec_max)
        st.caption(f"Large-doc switch to fast parsing: pages > **{_rec_threshold}**")
        if _rec_model != "—":
            st.caption(f"Recommended model: **{_rec_model}**")
        if rec.get("can_run") and _rec_model != "—":
            if st.button("Apply All Recommended Settings", key="apply_all_hw",
                         use_container_width=True, type="primary"):
                st.session_state["hw_apply_all"] = {
                    "model": _rec_model, "num_ctx": _rec_ctx,
                    "hybrid_top_k": _rec_top_k, "max_results": _rec_max,
                    "chunk_size": _rec_chunk, "chunk_overlap": tier["chunk_overlap"],
                    "large_doc_page_threshold": _rec_threshold,
                }
                st.rerun()

        # ── LLM Model ─────────────────────────────────────────
        st.divider()
        st.markdown("#### LLM Model")
        if available_models:
            current = st.session_state.get("sidebar_model")
            if current and current not in available_models:
                st.session_state["sidebar_model"] = available_models[0]

            def _model_label(name: str) -> str:
                return f"{name} (recommended)" if name == rec.get("model") else name

            model = st.selectbox("Active model", options=available_models,
                                 format_func=_model_label, key="sidebar_model")
        else:
            st.warning("Ollama is not reachable or no models are pulled.")
            model = st.text_input("Model name (manual entry)",
                                  value=rec.get("model") or "llama3.2:3b",
                                  key="sidebar_model_manual")

        # ── Context window ────────────────────────────────────
        st.divider()
        st.markdown("#### Context Window", help=term_help("Context window"))
        ctx_options = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
        if "sidebar_num_ctx" in st.session_state:
            applied = st.session_state["sidebar_num_ctx"]
            if applied not in ctx_options:
                st.session_state["sidebar_num_ctx"] = min(ctx_options, key=lambda x: abs(x - applied))
        num_ctx = st.select_slider("Tokens", options=ctx_options, key="sidebar_num_ctx")

        # ── Hybrid RAG ────────────────────────────────────────
        st.divider()
        st.markdown("#### Hybrid RAG", help=term_help("Hybrid retrieval"))
        st.caption("FAISS dense + BM25 sparse search, fused with Reciprocal Rank Fusion.")

        _embed_options, _embed_help_lines = [], []
        for m in KNOWN_EMBED_MODELS:
            pulled = m["name"] in available_embed
            _embed_options.append(m["name"] if pulled else f"{m['name']} (not pulled)")
            _embed_help_lines.append(
                f"{m['name']} — {m['dim']}d, {m['size_gb']} GB  "
                f"[{'Pulled' if pulled else 'Not pulled'}]  {m['note']}"
            )
        _default_idx = next(
            (i for i, m in enumerate(KNOWN_EMBED_MODELS) if m["name"] in available_embed), 0
        )
        _embed_label = st.selectbox("Embedding model", _embed_options, index=_default_idx,
                                    help="\n".join(_embed_help_lines), key="sidebar_embed_model")
        embed_model = _embed_label.replace(" (not pulled)", "")
        if embed_model in available_embed:
            st.success(f"Hybrid RAG ready — FAISS + BM25 + ChromaDB (`{embed_model}`)")
        else:
            st.warning(f"Run: `ollama pull {embed_model}`")

        _top_k_default = st.session_state.pop("sidebar_top_k_applied", 8)
        hybrid_top_k = st.slider("Chunks per query", min_value=3, max_value=20,
                                 value=_top_k_default, key="sidebar_hybrid_top_k")

        # ── Search settings ───────────────────────────────────
        st.divider()
        st.markdown("#### Search Settings")
        _max_results_default = st.session_state.pop("sidebar_max_results_applied", 6)
        max_results = st.slider("Max papers per query", 3, 20, _max_results_default,
                                key="sidebar_max_results")
        include_crossref = st.toggle("Include CrossRef search", value=False)

        # ── Document settings ─────────────────────────────────
        st.divider()
        st.markdown("#### Document Settings", help=term_help("Docling / OCR"))
        use_docling = st.toggle("Advanced Parsing (Docling)", value=True, key="sidebar_use_docling")
        if use_docling:
            use_ocr = st.toggle("Enable OCR (slower)", value=False, key="sidebar_use_ocr")
            chunk_size = st.session_state.pop("sidebar_chunk_size_applied", 800)
            chunk_overlap = st.session_state.pop("sidebar_chunk_overlap_applied", 150)
        else:
            use_ocr = False
            _cs = st.session_state.pop("sidebar_chunk_size_applied", 800)
            _co = st.session_state.pop("sidebar_chunk_overlap_applied", 150)
            chunk_size = st.slider("Chunk size (chars)", 400, 1200, _cs, step=100, key="sidebar_chunk_size")
            chunk_overlap = st.slider("Chunk overlap (chars)", 50, 300, _co, step=25, key="sidebar_chunk_overlap")
        _threshold_default = st.session_state.pop("sidebar_large_doc_threshold_applied", cfg.large_doc_page_threshold)
        large_doc_page_threshold = st.slider(
            "Large-PDF page threshold", min_value=10, max_value=300,
            value=_threshold_default, step=10, key="sidebar_large_doc_threshold",
            help="PDFs with more pages than this automatically switch from Docling to the "
                 "lighter pdfplumber parser, avoiding a large memory spike on big documents.",
        )

        # ── Safe Shutdown ─────────────────────────────────────
        st.divider()
        if not st.session_state.get("_shutdown_confirm"):
            if st.button("Shut Down Safely", key="sidebar_shutdown_btn", use_container_width=True):
                st.session_state["_shutdown_confirm"] = True
                st.rerun()
        else:
            st.warning("This will stop the server. Sessions are saved to disk.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Confirm Shutdown", key="shutdown_yes", type="primary", use_container_width=True):
                    import os, signal as _signal
                    from tools.shutdown import safe_shutdown
                    safe_shutdown(ports=[8000, 11434], flush_db=True)
                    os.kill(os.getpid(), _signal.SIGTERM)
            with col_no:
                if st.button("Cancel", key="shutdown_no", use_container_width=True):
                    st.session_state.pop("_shutdown_confirm", None)
                    st.rerun()

        st.divider()
        st.markdown(
            "**BeeSearch**  \n"
            "Local AI · Ollama · LangGraph · Hybrid RAG  \n"
            "Google Scholar · arXiv · Semantic Scholar · CrossRef"
        )

    return {
        "model": model,
        "num_ctx": num_ctx,
        "embed_model": embed_model,
        "hybrid_top_k": hybrid_top_k,
        "max_results": max_results,
        "include_web": False,
        "include_crossref": include_crossref,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_docling": use_docling,
        "use_ocr": use_ocr,
        "large_doc_page_threshold": large_doc_page_threshold,
        "style_profile": None,
    }
