"""
agents/notebook_pipeline_nodes.py
Seven LangGraph nodes for the Research Notebook pipeline.
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

logger = logging.getLogger(__name__)
cfg = get_settings()

_TOTAL_STEPS = 7

_memory: NotebookMemory | None = None


def _get_memory() -> NotebookMemory:
    global _memory
    if _memory is None:
        _memory = NotebookMemory()
    return _memory


def _make_llm(settings: dict, temperature: float = 0.3, num_predict: int = 2048):
    import httpx
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.get("model", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=temperature,
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
    from agents.notebook_advanced import _sources_context
    return _sources_context(
        {"sources": state.get("sources", []), "chunks": state.get("chunks", [])},
        max_chars_per_doc=max_chars_per_doc,
    )


def _parse_json_array(raw: str) -> Any:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return json.loads(raw)


def _progress(step: int) -> int:
    return min(100, round(step / _TOTAL_STEPS * 100))


def ingestion_node(state: NotebookPipelineState) -> Dict[str, Any]:
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


def summarization_node(state: NotebookPipelineState) -> Dict[str, Any]:
    _MAX_PER_DOC = 3_500

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

    llm = _make_llm(settings, temperature=0.3, num_predict=1024)

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

    return {
        "per_doc_summaries": per_doc_summaries,
        "cross_summary": cross_summary,
        "current_step": "summarize",
        "completed_steps": completed + ["summarize"],
        "progress_pct": _progress(2),
        "errors": errors,
    }


def retrieval_node(state: NotebookPipelineState) -> Dict[str, Any]:
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


def citation_verification_node(state: NotebookPipelineState) -> Dict[str, Any]:
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

    context = _sources_context_from_state(state, max_chars_per_doc=3_500)
    llm = _make_llm(settings, temperature=0.1, num_predict=1024)

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
        "Confidence: HIGH = direct quote, MEDIUM = implied/paraphrased, LOW = not clearly supported"
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

    return {
        "verified_citations": data,
        "citation_report": citation_report,
        "current_step": "verify_citations",
        "completed_steps": completed + ["verify_citations"],
        "progress_pct": _progress(4),
        "errors": errors,
    }


def knowledge_graph_node(state: NotebookPipelineState) -> Dict[str, Any]:
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

    context = _sources_context_from_state(state, max_chars_per_doc=1_500)
    llm = _make_llm(settings, temperature=0.2, num_predict=1024)

    system = (
        "Extract a knowledge graph from the provided research sources.\n"
        "Return ONLY valid JSON — no markdown fences, no explanation:\n"
        '{"nodes": [{"id": "1", "label": "...", '
        '"type": "concept|method|dataset|author|institution"}],\n'
        ' "edges": [{"from": "1", "to": "2", "label": "relationship verb"}]}\n\n'
        "Rules:\n"
        "• 15–20 nodes, 15–25 edges\n"
        "• Every edge 'from'/'to' must reference a valid node 'id'\n"
        "• Use precise domain-specific labels"
    )

    kg_data: Dict[str, Any] = {}
    dot = ""
    try:
        raw = _invoke(llm, system, f"Source material:\n\n{context}")
        kg_data = _parse_json_object_from_llm(raw)
        dot = _knowledge_graph_to_dot(kg_data)
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


def study_guide_node(state: NotebookPipelineState) -> Dict[str, Any]:
    _MAX_TOTAL = 14_000

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
    except Exception as exc:
        errors.append(f"Study guide generation failed: {exc}")

    return {
        "study_guide": study_guide,
        "current_step": "study_guide",
        "completed_steps": completed + ["study_guide"],
        "progress_pct": _progress(6),
        "errors": errors,
    }


def podcast_script_node(state: NotebookPipelineState) -> Dict[str, Any]:
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

    ctx_parts = [f"KEY INSIGHTS:\n{cross_summary[:3000]}"]
    if study_guide:
        m = re.search(r"## Key Concepts\n(.*?)(?=\n##|\Z)", study_guide, re.DOTALL)
        if m:
            ctx_parts.append(f"KEY CONCEPTS:\n{m.group(1)[:1500]}")
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
        "• Open with a compelling hook from HOST\n"
        "• Close with HOST naming 2–3 concrete takeaways and signing off warmly\n"
        "• Pure dialogue only — no stage directions, no music cues"
    )
    human = (
        f"Create a podcast episode based on these research sources: {source_list}\n\n"
        f"{context}"
    )

    podcast_script = ""
    try:
        podcast_script = _invoke(llm, system, human)
    except Exception as exc:
        errors.append(f"Podcast script generation failed: {exc}")

    return {
        "podcast_script": podcast_script,
        "current_step": "podcast",
        "completed_steps": completed + ["podcast"],
        "progress_pct": _progress(7),
        "errors": errors,
    }
