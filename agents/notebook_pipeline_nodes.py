"""
agents/notebook_pipeline_nodes.py
────────────────────────────────────
Seven LangGraph nodes for the Mode 8 Research Notebook pipeline.

  START
    │
    ▼  Agent 1
  [ingest]              Load sources & chunks from NotebookMemory
    │
    ▼  Agent 2
  [summarize]           Per-doc summaries + cross-document synthesis
    │
    ▼  Agent 3
  [retrieve]            Hybrid RAG (FAISS + BM25 + RRF) for focus query
    │
    ▼  Agent 4
  [verify_citations]    Verify summary claims against source material
    │
    ▼  Agent 5
  [build_kg]            Entity–relationship knowledge graph → DOT string
    │
    ▼  Agent 6
  [study_guide]         Key Concepts · Glossary · Q&A · Quick Summary
    │
    ▼  Agent 7
  [podcast]             Two-speaker podcast episode dialogue
    │
   END

Reuse policy
────────────
• _make_llm / _invoke / _sources_context : imported from notebook_advanced.py
• _rebuild_docs_from_chunks              : imported from notebook_nodes.py
• _knowledge_graph_to_dot / _parse_json_object_from_llm : imported from notebook_advanced.py
• HybridStore                            : via tools.hybrid_store.get_or_create_store
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List

from agents.notebook_memory import NotebookMemory
from agents.notebook_pipeline_state import NotebookPipelineState
from config.settings import get_settings
from tools.temperature_levels import DEFAULT_TEMPERATURE_LEVEL, apply_temperature_level

logger = logging.getLogger(__name__)
cfg = get_settings()

_TOTAL_STEPS = 7

# ── Lazy memory singleton — tests can monkeypatch ──────────────────────────────

_memory: NotebookMemory | None = None


def _get_memory() -> NotebookMemory:
    global _memory
    if _memory is None:
        _memory = NotebookMemory()
    return _memory


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm(settings: dict, temperature: float = 0.3, num_predict: int = 2048):
    """ChatOllama factory — mirrors notebook_advanced._make_llm."""
    import httpx
    from langchain_ollama import ChatOllama
    level = settings.get("temperature_level", DEFAULT_TEMPERATURE_LEVEL)
    return ChatOllama(
        model=settings.get("model", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=apply_temperature_level(temperature, level),
        num_predict=num_predict,
        num_ctx=settings.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(300.0)},
    )


def _invoke(llm, system: str, human: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    return llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=human)]
    ).content.strip()


def _sources_context_from_state(
    state: NotebookPipelineState,
    max_chars_per_doc: int = 3_500,
) -> str:
    """Build numbered source context block from state's chunks — delegates to
    notebook_advanced._sources_context via a minimal notebook dict."""
    from agents.notebook_advanced import _sources_context
    return _sources_context(
        {"sources": state.get("sources", []), "chunks": state.get("chunks", [])},
        max_chars_per_doc=max_chars_per_doc,
    )


def _parse_json_array(raw: str) -> Any:
    """Extract the first JSON array from LLM output; falls back to full parse."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return json.loads(raw)


def _progress(step: int) -> int:
    """Return progress % for step (1-indexed, out of _TOTAL_STEPS)."""
    return min(100, round(step / _TOTAL_STEPS * 100))


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Document Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def ingestion_node(state: NotebookPipelineState) -> Dict[str, Any]:
    """
    Load all sources and chunks from the notebook JSON into pipeline state.
    No LLM call — pure I/O.
    """
    notebook_id = state.get("notebook_id", "")
    errors: List[str] = list(state.get("errors", []))
    completed: List[str] = list(state.get("completed_steps", []))

    notebook = _get_memory().load(notebook_id)
    if not notebook:
        errors.append(f"Notebook '{notebook_id}' not found.")
        return {
            "errors": errors,
            "current_step": "ingest",
            "progress_pct": _progress(1),
            "completed_steps": completed,
        }

    sources: List[Dict] = notebook.get("sources", [])
    chunks: List[Dict] = notebook.get("chunks", [])
    doc_count = len(sources)
    source_names = [s["filename"] for s in sources]
    ingestion_summary = (
        f"Loaded {doc_count} source(s) with {len(chunks)} chunk(s) "
        f"from notebook '{notebook.get('name', notebook_id)}'. "
        + (f"Sources: {', '.join(source_names)}." if source_names else "No sources found.")
    )
    logger.info("ingestion_node: %s", ingestion_summary)

    return {
        "sources": sources,
        "chunks": chunks,
        "doc_count": doc_count,
        "ingestion_summary": ingestion_summary,
        "current_step": "ingest",
        "completed_steps": completed + ["ingest"],
        "progress_pct": _progress(1),
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Summarization
# ─────────────────────────────────────────────────────────────────────────────

def summarization_node(state: NotebookPipelineState) -> Dict[str, Any]:
    """
    Generate per-document summaries and a unified cross-document synthesis.

    Single source  → comprehensive 3-4 paragraph summary.
    Multiple sources → short per-doc summary + structured synthesis covering
                       Overview · Common Themes · Complementary Contributions ·
                       Contradictions · Key Takeaways.
    """
    _MAX_PER_DOC = 6_000

    errors: List[str] = list(state.get("errors", []))
    completed: List[str] = list(state.get("completed_steps", []))
    settings = state.get("settings", {})
    sources = state.get("sources", [])
    chunks = state.get("chunks", [])

    if not sources:
        errors.append("Summarization skipped: no sources in notebook.")
        return {
            "errors": errors,
            "current_step": "summarize",
            "progress_pct": _progress(2),
            "completed_steps": completed + ["summarize"],
        }

    llm = _make_llm(settings, temperature=0.3, num_predict=2048)

    by_doc: Dict[str, List[str]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.get("doc_id", "")].append(chunk.get("text", ""))

    id_to_name = {s["doc_id"]: s["filename"] for s in sources}
    per_doc_summaries: Dict[str, str] = {}

    if len(sources) == 1:
        doc_id = sources[0]["doc_id"]
        name = id_to_name.get(doc_id, "document")
        text = " ".join(by_doc.get(doc_id, []))[:_MAX_PER_DOC]
        system = (
            "You are a research analyst. Summarize this document in 3–4 paragraphs covering: "
            "main argument or purpose, methodology (if applicable), key findings or conclusions, "
            "and implications. Be specific — refer to the document by name."
        )
        try:
            per_doc_summaries[name] = _invoke(llm, system, f"Document: {name}\n\n{text}")
        except Exception as exc:
            errors.append(f"Summarization of '{name}' failed: {exc}")
            per_doc_summaries[name] = ""
        cross_summary = per_doc_summaries.get(name, "")

    else:
        for src in sources:
            doc_id = src["doc_id"]
            name = id_to_name.get(doc_id, doc_id)
            text = " ".join(by_doc.get(doc_id, []))[:_MAX_PER_DOC]
            system = (
                "Summarize this document in 3–4 sentences covering its "
                "main argument, methods, and key finding."
            )
            try:
                per_doc_summaries[name] = _invoke(llm, system, f"Document: {name}\n\n{text}")
            except Exception as exc:
                errors.append(f"Summarization of '{name}' failed: {exc}")
                per_doc_summaries[name] = ""

        context = _sources_context_from_state(state, max_chars_per_doc=_MAX_PER_DOC)
        system = (
            "You are a research analyst synthesising multiple sources. Write a structured synthesis:\n\n"
            "**Overview** — what these sources collectively address\n"
            "**Common Themes** — shared ideas, methods, or conclusions\n"
            "**Complementary Contributions** — how sources build on each other\n"
            "**Contradictions / Tensions** — where sources diverge or conflict\n"
            "**Key Takeaways** — the most important insights across all sources\n\n"
            "Cite source filenames when making specific claims."
        )
        try:
            cross_summary = _invoke(llm, system, f"Sources:\n\n{context}")
        except Exception as exc:
            errors.append(f"Cross-document synthesis failed: {exc}")
            cross_summary = "\n\n".join(
                f"**{n}:** {s}" for n, s in per_doc_summaries.items()
            )

    logger.info(
        "summarization_node: %d per-doc summaries, cross_summary %d chars",
        len(per_doc_summaries),
        len(cross_summary),
    )
    return {
        "per_doc_summaries": per_doc_summaries,
        "cross_summary": cross_summary,
        "current_step": "summarize",
        "completed_steps": completed + ["summarize"],
        "progress_pct": _progress(2),
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_node(state: NotebookPipelineState) -> Dict[str, Any]:
    """
    Build (or reuse) a HybridStore and retrieve the most relevant chunks for
    the pipeline's focus query using dense FAISS + sparse BM25 fused with RRF.

    Falls back to simple keyword matching when the embedding model is unavailable.
    The session key `pipeline_<notebook_id>` keeps this store separate from the
    interactive Q&A store (`notebook_<notebook_id>`).
    """
    errors: List[str] = list(state.get("errors", []))
    completed: List[str] = list(state.get("completed_steps", []))
    settings = state.get("settings", {})
    sources = state.get("sources", [])
    chunks = state.get("chunks", [])
    notebook_id = state.get("notebook_id", "")
    query = state.get("query") or "key concepts main findings methodology results"
    top_k = settings.get("top_k", 8)

    if not chunks:
        errors.append("Retrieval skipped: no chunks available.")
        return {
            "retrieved_chunks": [],
            "retrieval_mode": "empty",
            "errors": errors,
            "current_step": "retrieve",
            "progress_pct": _progress(3),
            "completed_steps": completed + ["retrieve"],
        }

    retrieved: List[Dict] = []
    mode = "fallback"
    rag_meta: Dict[str, Any] = {}

    try:
        from agents.notebook_nodes import _rebuild_docs_from_chunks
        from tools.hybrid_store import get_or_create_store

        rebuilt = _rebuild_docs_from_chunks({"sources": sources, "chunks": chunks})
        store = get_or_create_store(
            session_id=f"pipeline_{notebook_id}",
            embed_model=settings.get("embed_model", "nomic-embed-text"),
            ollama_base_url=cfg.ollama_base_url,
            persist_dir=cfg.chroma_persist_dir,
        )
        if not store.is_indexed():
            store.add_documents(rebuilt, warning_callback=logger.warning)

        from agents.self_reflective_rag import self_reflective_retrieve
        retrieved, rag_meta = self_reflective_retrieve(
            store, query, top_k,
            model_name=settings.get("model", cfg.ollama_model),
            num_ctx=settings.get("num_ctx", cfg.num_ctx),
        )
        mode = "self_reflective" if store.embedder_available() else "fallback"
        logger.info("retrieval_node: %d chunks via %s", len(retrieved), mode)

    except Exception as exc:
        errors.append(f"HybridStore retrieval failed ({exc}); using keyword fallback.")
        words = set(query.lower().split())
        retrieved = (
            [c for c in chunks if any(w in c.get("text", "").lower() for w in words)][:top_k]
            or chunks[:top_k]
        )
        mode = "fallback"

    return {
        "retrieved_chunks": retrieved,
        "retrieval_mode": mode,
        "rag_reflection_info": rag_meta,
        "current_step": "retrieve",
        "completed_steps": completed + ["retrieve"],
        "progress_pct": _progress(3),
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — Citation Verification
# ─────────────────────────────────────────────────────────────────────────────

def citation_verification_node(state: NotebookPipelineState) -> Dict[str, Any]:
    """
    Identify 5–8 specific factual claims in the cross-document summary and
    verify each against the raw source material.

    Confidence ratings:
      HIGH   — direct supporting quote found in the sources
      MEDIUM — claim implied or clearly paraphrased by a source
      LOW    — claim not clearly supported by any available source

    Returns a structured markdown verification table.
    """
    errors: List[str] = list(state.get("errors", []))
    completed: List[str] = list(state.get("completed_steps", []))
    settings = state.get("settings", {})
    cross_summary = state.get("cross_summary", "")
    chunks = state.get("chunks", [])

    if not cross_summary or not chunks:
        errors.append("Citation verification skipped: no summary or chunks available.")
        return {
            "verified_citations": [],
            "citation_report": "No summary available for verification.",
            "errors": errors,
            "current_step": "verify_citations",
            "progress_pct": _progress(4),
            "completed_steps": completed + ["verify_citations"],
        }

    context = _sources_context_from_state(state, max_chars_per_doc=5_000)
    llm = _make_llm(settings, temperature=0.1, num_predict=1_500)

    system = (
        "You are a research fact-checker. Identify 5–8 specific verifiable factual claims "
        "from the summary, then verify each against the provided source material.\n\n"
        "Return ONLY a JSON array — no markdown, no preamble:\n"
        "[\n"
        "  {\n"
        '    "claim": "one-sentence factual claim from the summary",\n'
        '    "source_name": "filename of the supporting source",\n'
        '    "confidence": "HIGH|MEDIUM|LOW",\n'
        '    "supporting_text": "short quote or paraphrase from the source"\n'
        "  }\n"
        "]\n\n"
        "Confidence:\n"
        "  HIGH   = direct quote present in source material\n"
        "  MEDIUM = claim implied or paraphrased by source\n"
        "  LOW    = claim not clearly supported by any source"
    )
    human = (
        f"SOURCE MATERIAL:\n{context}\n\n"
        f"SUMMARY TO VERIFY:\n{cross_summary[:2000]}"
    )

    data: List[Dict[str, Any]] = []
    try:
        raw = _invoke(llm, system, human)
        parsed = _parse_json_array(raw)
        data = parsed if isinstance(parsed, list) else []
    except Exception as exc:
        errors.append(f"Citation verification LLM call failed: {exc}")

    # Build markdown table report
    if data:
        _badge = {"HIGH": "✅", "MEDIUM": "🟡", "LOW": "❌"}
        high = sum(1 for c in data if c.get("confidence") == "HIGH")
        medium = sum(1 for c in data if c.get("confidence") == "MEDIUM")
        low = sum(1 for c in data if c.get("confidence") == "LOW")

        rows = [
            "## Citation Verification Report\n",
            f"Verified **{len(data)} claims** — "
            f"{high} ✅ HIGH · {medium} 🟡 MEDIUM · {low} ❌ LOW\n",
            "| Claim | Source | Confidence | Evidence |",
            "|-------|--------|-----------|----------|",
        ]
        for item in data:
            claim = str(item.get("claim", ""))[:80]
            source = item.get("source_name", "—")
            conf = item.get("confidence", "")
            icon = _badge.get(conf, "")
            snippet = str(item.get("supporting_text", ""))[:60]
            rows.append(f"| {claim} | {source} | {icon} {conf} | *{snippet}…* |")

        citation_report = "\n".join(rows)
    else:
        citation_report = "Citation verification produced no results."

    logger.info("citation_verification_node: %d claims verified", len(data))
    return {
        "verified_citations": data,
        "citation_report": citation_report,
        "current_step": "verify_citations",
        "completed_steps": completed + ["verify_citations"],
        "progress_pct": _progress(4),
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 — Knowledge Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_graph_node(state: NotebookPipelineState) -> Dict[str, Any]:
    """
    Extract entities and relationships from all sources and render them as a
    Graphviz DOT knowledge graph.

    Reuses _knowledge_graph_to_dot() and _parse_json_object_from_llm() from
    notebook_advanced — same LLM prompt, same DOT output format.

    Node types: concept · method · dataset · author · institution
    """
    errors: List[str] = list(state.get("errors", []))
    completed: List[str] = list(state.get("completed_steps", []))
    settings = state.get("settings", {})
    chunks = state.get("chunks", [])

    if not chunks:
        errors.append("Knowledge graph skipped: no source material.")
        return {
            "knowledge_graph_dot": "",
            "kg_data": {},
            "errors": errors,
            "current_step": "build_kg",
            "progress_pct": _progress(5),
            "completed_steps": completed + ["build_kg"],
        }

    from agents.notebook_advanced import (
        _knowledge_graph_to_dot,
        _parse_json_object_from_llm,
    )

    context = _sources_context_from_state(state, max_chars_per_doc=3_000)
    llm = _make_llm(settings, temperature=0.2, num_predict=1_500)

    system = (
        "Extract a knowledge graph from the provided research sources.\n"
        "Return ONLY valid JSON — no markdown fences, no explanation:\n"
        '{"nodes": [{"id": "1", "label": "...", '
        '"type": "concept|method|dataset|author|institution"}],\n'
        ' "edges": [{"from": "1", "to": "2", "label": "relationship verb"}]}\n\n'
        "Rules:\n"
        "• 15–20 nodes, 15–25 edges\n"
        "• Every edge 'from'/'to' must reference a valid node 'id'\n"
        "• Use precise domain-specific labels\n"
        "• 'concept' = ideas/theories, 'method' = techniques/algorithms, "
        "'dataset' = data sources, 'author' = people, 'institution' = organisations"
    )

    kg_data: Dict[str, Any] = {}
    dot = ""
    try:
        raw = _invoke(llm, system, f"Source material:\n\n{context}")
        kg_data = _parse_json_object_from_llm(raw)
        dot = _knowledge_graph_to_dot(kg_data)
        logger.info(
            "knowledge_graph_node: %d nodes, %d edges",
            len(kg_data.get("nodes", [])),
            len(kg_data.get("edges", [])),
        )
    except Exception as exc:
        errors.append(f"Knowledge graph construction failed: {exc}")

    return {
        "knowledge_graph_dot": dot,
        "kg_data": kg_data,
        "current_step": "build_kg",
        "completed_steps": completed + ["build_kg"],
        "progress_pct": _progress(5),
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 6 — Study Guide Generation
# ─────────────────────────────────────────────────────────────────────────────

def study_guide_node(state: NotebookPipelineState) -> Dict[str, Any]:
    """
    Generate a structured study guide from the summaries and raw source material.

    Sections:
      ## Key Concepts     8–12 bullet-point definitions
      ## Glossary         term | definition | source table (8–12 terms)
      ## Review Questions 6–8 Q&A pairs grounded in the sources
      ## Quick Summary    2–3 paragraph synthesis
    """
    _MAX_TOTAL = 20_000

    errors: List[str] = list(state.get("errors", []))
    completed: List[str] = list(state.get("completed_steps", []))
    settings = state.get("settings", {})
    cross_summary = state.get("cross_summary", "")
    per_doc_summaries = state.get("per_doc_summaries", {})
    chunks = state.get("chunks", [])

    if not cross_summary and not chunks:
        errors.append("Study guide skipped: no source material.")
        return {
            "study_guide": "",
            "errors": errors,
            "current_step": "study_guide",
            "progress_pct": _progress(6),
            "completed_steps": completed + ["study_guide"],
        }

    # Build the best available context (summaries > raw chunks)
    context_parts: List[str] = []
    if cross_summary:
        context_parts.append(f"SYNTHESIS:\n{cross_summary}")
    for name, summary in per_doc_summaries.items():
        if summary:
            context_parts.append(f"SOURCE '{name}':\n{summary}")
    if not context_parts:
        context_parts.append(_sources_context_from_state(state))

    combined = "\n\n".join(context_parts)[:_MAX_TOTAL]

    llm = _make_llm(settings, temperature=0.3, num_predict=2048)
    system = (
        "You are an expert tutor creating a comprehensive study guide from research material.\n\n"
        "Generate a well-structured study guide with ALL four sections below. "
        "Use Markdown. Cite source filenames where appropriate.\n\n"
        "## Key Concepts\n"
        "List 8–12 of the most important concepts as bullet points. "
        "Format: **Concept Name** — one-sentence definition.\n\n"
        "## Glossary\n"
        "A Markdown table with columns: | Term | Definition | Source |\n"
        "Include 8–12 domain-specific terms.\n\n"
        "## Review Questions\n"
        "Write 6–8 questions with detailed answers grounded in the sources.\n"
        "Format each as:\n**Q:** ...\n**A:** ...\n\n"
        "## Quick Summary\n"
        "2–3 paragraphs synthesising the most important insights across all sources."
    )

    study_guide = ""
    try:
        study_guide = _invoke(llm, system, f"Research material:\n\n{combined}")
        logger.info("study_guide_node: %d chars generated", len(study_guide))
    except Exception as exc:
        errors.append(f"Study guide generation failed: {exc}")

    return {
        "study_guide": study_guide,
        "current_step": "study_guide",
        "completed_steps": completed + ["study_guide"],
        "progress_pct": _progress(6),
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 7 — Podcast Script Generation
# ─────────────────────────────────────────────────────────────────────────────

def podcast_script_node(state: NotebookPipelineState) -> Dict[str, Any]:
    """
    Generate an engaging two-speaker podcast episode based on the research sources.

    Speakers:
      HOST (Alex)          — curious interviewer; hooks the listener, asks great
                             questions, provides smooth transitions between topics
      EXPERT (Dr. Jordan)  — knowledgeable researcher; explains clearly, never
                             uses jargon without defining it first

    Script length: 450–600 words.  Pure dialogue — no stage directions.
    """
    errors: List[str] = list(state.get("errors", []))
    completed: List[str] = list(state.get("completed_steps", []))
    settings = state.get("settings", {})
    cross_summary = state.get("cross_summary", "")
    study_guide = state.get("study_guide", "")
    sources = state.get("sources", [])

    if not cross_summary and not study_guide:
        errors.append("Podcast script skipped: no source material.")
        return {
            "podcast_script": "",
            "errors": errors,
            "current_step": "podcast",
            "progress_pct": _progress(7),
            "completed_steps": completed + ["podcast"],
        }

    source_list = (
        ", ".join(s["filename"] for s in sources) if sources else "the provided documents"
    )

    # Build focused context: cross-summary + Key Concepts section from study guide
    ctx_parts = [f"KEY INSIGHTS:\n{cross_summary[:5000]}"]
    if study_guide:
        m = re.search(r"## Key Concepts\n(.*?)(?=\n##|\Z)", study_guide, re.DOTALL)
        if m:
            ctx_parts.append(f"KEY CONCEPTS:\n{m.group(1)[:2500]}")
    context = "\n\n".join(ctx_parts)

    llm = _make_llm(settings, temperature=0.7, num_predict=2048)
    system = (
        "You are a podcast script writer. Create an engaging educational podcast transcript.\n\n"
        "Speakers:\n"
        "  HOST (Alex)         — curious, asks great questions, smooth transitions\n"
        "  EXPERT (Dr. Jordan) — knowledgeable, explains clearly, no unexplained jargon\n\n"
        "Format EVERY line exactly as one of:\n"
        "HOST: [dialogue]\n"
        "EXPERT: [dialogue]\n\n"
        "Requirements:\n"
        "• 450–600 words total\n"
        "• Natural conversational tone throughout\n"
        "• Cover: what these sources are about, the key findings, and why it matters\n"
        "• Open with a compelling hook from HOST that immediately draws the listener in\n"
        "• Close with HOST naming 2–3 concrete takeaways and signing off warmly\n"
        "• Pure dialogue only — no stage directions, no music cues, no parentheticals"
    )
    human = (
        f"Create a podcast episode based on these research sources: {source_list}\n\n"
        f"{context}"
    )

    podcast_script = ""
    try:
        podcast_script = _invoke(llm, system, human)
        logger.info("podcast_script_node: %d chars generated", len(podcast_script))
    except Exception as exc:
        errors.append(f"Podcast script generation failed: {exc}")

    return {
        "podcast_script": podcast_script,
        "current_step": "podcast",
        "completed_steps": completed + ["podcast"],
        "progress_pct": _progress(7),
        "errors": errors,
    }
