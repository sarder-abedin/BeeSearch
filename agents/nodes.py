"""
agents/nodes.py
───────────────
Each function here is one node in the LangGraph research workflow.
Every node:
  1. Reads relevant fields from `state`
  2. Performs its specific task (LLM call / tool call / data transform)
  3. Returns a dict of updated state fields (partial update pattern)

TUTORIAL NOTE — Hybrid RAG
───────────────────────────
This workflow uses **Hybrid RAG**: dense vector search (Ollama embeddings +
FAISS) combined with sparse keyword search (BM25), fused via Reciprocal
Rank Fusion (RRF).

Pipeline:
  Document → chunks → embed (Ollama) → FAISS + ChromaDB cache
                    → BM25 keyword index
  At query time: FAISS top-k ∪ BM25 top-k → RRF → top-k chunks → LLM

Benefits over pure vectorless injection:
  • Retrieves only the most relevant chunks — keeps context window efficient
  • BM25 catches exact-term matches that dense search can miss
  • RRF fusion is score-normalisation-free and empirically robust
  • ChromaDB persists embeddings so the same doc is never re-embedded

Fallback: if the embedding model is not pulled, the node falls back to
direct raw-text injection (vectorless mode) and sets state["rag_fallback"].

Node execution order (defined in graph.py):
  document_ingestion → query_generation → academic_search
  → [web_search] → document_analysis → reference_compilation
  → report_generation
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import ResearchState
from config.settings import get_settings
from tools.search_tools import AcademicSearcher, Paper, WebSearcher
from tools.hybrid_store import get_or_create_store, HybridStore
from agents.self_reflective_rag import react_retrieve

logger = logging.getLogger(__name__)
cfg = get_settings()

# ── Shared singletons (created once, reused across nodes) ─────────────────────

_academic: AcademicSearcher | None = None
_web: WebSearcher | None = None


def _get_academic() -> AcademicSearcher:
    global _academic
    if _academic is None:
        _academic = AcademicSearcher()
    return _academic


def _get_web() -> WebSearcher:
    global _web
    if _web is None:
        _web = WebSearcher()
    return _web


def _llm(state: ResearchState) -> ChatOllama:
    """Return a ChatOllama instance configured for the current session."""
    import httpx
    return ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=0.3,
        num_predict=4096,
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        # 3-minute hard timeout — prevents the background thread hanging forever
        # if Ollama is under load or the model needs to be swapped in from disk.
        sync_client_kwargs={"timeout": httpx.Timeout(180.0)},
    )


def _llm_fast(state: ResearchState) -> ChatOllama:
    """Lightweight LLM for short structured outputs (query generation, JSON lists).

    Caps context at 8192 to avoid loading a massive KV cache for a few tokens,
    while still accommodating goals that include a few hundred words of context.
    """
    import httpx
    return ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=0.2,
        num_predict=768,
        num_ctx=min(state.get("num_ctx", cfg.num_ctx), 8192),
        sync_client_kwargs={"timeout": httpx.Timeout(90.0)},
    )


def _invoke(llm: ChatOllama, system: str, human: str) -> str:
    """Helper: call the LLM with a system + human message pair."""
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    response = llm.invoke(messages)
    return response.content.strip()


def _style_block(state: ResearchState) -> str:
    """Return a style injection block if a profile is active, else empty string."""
    profile = state.get("style_profile")
    if not profile:
        return ""
    injection = profile.get("injection_prompt", "").strip()
    if not injection:
        return ""
    name = profile.get("name", "custom")
    return f"\n\nUSER WRITING STYLE — '{name}' (follow these guidelines precisely):\n{injection}"


def _clarification_context(state: ResearchState) -> str:
    """Return formatted user clarifications if provided, else empty string."""
    clarifications = state.get("clarifications") or {}
    if not clarifications:
        return ""
    lines = [
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in clarifications.items()
        if v and str(v).strip()
    ]
    if not lines:
        return ""
    return "\n\nUSER REQUIREMENTS CLARIFICATION (follow these precisely):\n" + "\n".join(lines)


# ── Node 1: Document Ingestion ────────────────────────────────────────────────

def document_ingestion_node(state: ResearchState) -> Dict[str, Any]:
    """
    Parse uploaded documents and index them in the Hybrid RAG store.

    Steps:
      1. Validate that documents contain extractable text
      2. Build the HybridStore for this session (FAISS + BM25 + ChromaDB)
      3. Embed all chunks via Ollama and index them
      4. If embedding fails (model not pulled), set rag_fallback=True

    The store is cached in a module-level dict keyed by session_id so that
    subsequent nodes can retrieve it without re-indexing.
    """
    logger.info("[Node 1] Document Ingestion + Hybrid RAG Indexing")
    errors = list(state.get("errors", []))
    uploaded_docs = state.get("uploaded_docs", [])
    session_id = state.get("session_id", "default")
    embed_model = state.get("embed_model", cfg.embedding_model)
    rag_fallback = False

    for doc in uploaded_docs:
        chars = len(doc.raw_text)
        logger.info("  ✓ %s — %d pages, %d chunks, %s chars",
                    doc.filename, doc.total_pages, doc.total_chunks, f"{chars:,}")
        if chars == 0:
            errors.append(f"'{doc.filename}' produced no extractable text.")

    # Index documents in the hybrid store (only when docs are present)
    if uploaded_docs:
        store = get_or_create_store(
            session_id=session_id,
            embed_model=embed_model,
            ollama_base_url=cfg.ollama_base_url,
            persist_dir=cfg.chroma_persist_dir,
        )
        try:
            store.add_documents(uploaded_docs, warning_callback=logger.warning)
            logger.info("  Hybrid RAG index built: %d chunks.", sum(len(d.chunks) for d in uploaded_docs))
            # If embedding failed inside add_documents, FAISS is absent but BM25 is built.
            if store._faiss_index is None and store.is_indexed():
                errors.append(
                    f"Full hybrid RAG unavailable — using BM25 keyword search. "
                    f"Run `ollama pull {embed_model}` for dense vector search."
                )
        except Exception as e:
            # Both embedding and BM25 completely failed — fall back to direct text injection.
            logger.warning("  Document indexing failed (%s). Using direct text injection.", e)
            errors.append(
                f"Document indexing failed: {e}. Using direct text injection this session."
            )
            rag_fallback = True

    if not uploaded_docs:
        _status_detail = "No documents — search-only mode"
    elif rag_fallback:
        _n = sum(len(d.chunks) for d in uploaded_docs)
        _status_detail = f"{_n} chunks · text injection (embedding unavailable)"
    elif store.is_indexed() and store._faiss_index is None:
        _n = sum(len(d.chunks) for d in uploaded_docs)
        _status_detail = f"{_n} chunks indexed · BM25 keyword search"
    else:
        _n = sum(len(d.chunks) for d in uploaded_docs)
        _status_detail = f"{_n} chunks indexed · FAISS + BM25 + RRF"

    return {
        "rag_fallback": rag_fallback,
        "current_step": "document_ingestion",
        "completed_steps": state.get("completed_steps", []) + ["document_ingestion"],
        "errors": errors,
        "progress_pct": 15,
        "status_detail": _status_detail,
    }


# ── Node 2: Query Generation ──────────────────────────────────────────────────

def query_generation_node(state: ResearchState) -> Dict[str, Any]:
    """
    Use the LLM to decompose the user's goal into focused search sub-queries.

    Why decompose?
    ──────────────
    A single broad query like "climate change impacts" returns generic results.
    Breaking it into 3-5 specific sub-questions lets us retrieve diverse,
    targeted papers and document chunks.
    """
    logger.info("[Node 2] Query Generation")
    llm = _llm_fast(state)  # short output — use light config to avoid loading delay

    system_prompt = """You are an academic research librarian expert in database search strategy.
Generate 4–6 distinct search queries for arXiv and Semantic Scholar.

Rules:
- Each query MUST target a different angle: e.g. mechanisms, applications, methodology, comparisons, evidence base
- Use precise academic/scientific terminology from the relevant field
- Vary specificity: some broad (2-3 key terms), some narrow (4-6 specific terms)
- Include synonyms and closely related concepts across queries, not within a single query
- Avoid redundancy — each query should retrieve different papers
- Do NOT include the phrase "research on" or "study of" — go straight to the topic terms

Return ONLY a valid JSON array of strings. No explanation, no commentary."""

    human_prompt = f"""Research goal:
{state['goal']}{_clarification_context(state)}

Generate 4–6 focused academic search queries. Return a JSON array only."""

    try:
        raw = _invoke(llm, system_prompt, human_prompt)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else []
        # Filter out non-string items (e.g. nested lists from malformed LLM output)
        queries = [q for q in parsed if isinstance(q, str)]
        if state["goal"] not in queries:
            queries.insert(0, state["goal"])
        queries = queries[:6]
        if not queries:
            queries = [state["goal"]]
    except Exception as e:
        logger.warning("Query generation failed (%s) — using raw goal.", e)
        queries = [state["goal"]]

    logger.info("  Generated %d queries: %s", len(queries), queries)

    return {
        "search_queries": queries,
        "current_step": "query_generation",
        "completed_steps": state.get("completed_steps", []) + ["query_generation"],
        "progress_pct": 25,
        "status_detail": f"Generated {len(queries)} search {'query' if len(queries) == 1 else 'queries'}",
    }


# ── Node 3: Academic Search ────────────────────────────────────────────────────

def academic_search_node(state: ResearchState) -> Dict[str, Any]:
    """
    Search arXiv and Semantic Scholar using a ReAct loop.

    Phase 1 — batch: all queries from query_generation_node are searched and
    results are deduplicated across queries.

    Phase 2 — ReAct gap-fill: papers are graded for relevance to the research
    goal; if fewer than 5 relevant papers are found, the LLM reasons about what
    topic angle is missing and fires a targeted follow-up search (up to 3 rounds).
    Only newly-added papers are graded per round.
    """
    from agents.self_reflective_rag import react_paper_search

    logger.info("[Node 3] Academic Search (ReAct)")

    all_papers, react_meta = react_paper_search(
        searcher=_get_academic(),
        initial_queries=state.get("search_queries", [state.get("goal", "")]),
        goal=state.get("goal", ""),
        model_name=state.get("model_name", cfg.ollama_model),
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        max_per_source=cfg.max_search_results // 2,
    )

    logger.info(
        "  [ReAct] %d unique papers | %d relevant | %d gap queries fired.",
        len(all_papers),
        react_meta.get("total_relevant", 0),
        len(react_meta.get("gap_queries", [])),
    )

    _gap_n = len(react_meta.get("gap_queries", []))
    _react_detail = f" · {_gap_n} ReAct follow-up {'query' if _gap_n == 1 else 'queries'}" if _gap_n else ""
    _acad_detail = f"{len(all_papers)} papers via arXiv · Semantic Scholar · CrossRef{_react_detail}"

    return {
        "academic_papers": all_papers,
        "rag_reflection_info": [{"type": "react_paper_search", **react_meta}],
        "current_step": "academic_search",
        "completed_steps": state.get("completed_steps", []) + ["academic_search"],
        "progress_pct": 45,
        "status_detail": _acad_detail,
    }


# ── Node 4: Web Search (optional) ─────────────────────────────────────────────

def web_search_node(state: ResearchState) -> Dict[str, Any]:
    """
    Supplement academic results with DuckDuckGo web search.

    This node only runs if `include_web_search` is True in the state.
    It's useful for topics with limited academic coverage (e.g. current events,
    industry reports, technical documentation).
    """
    logger.info("[Node 4] Web Search")

    if not state.get("include_web_search", False):
        logger.info("  Web search disabled — skipping.")
        return {
            "web_results": [],
            "current_step": "web_search",
            "completed_steps": state.get("completed_steps", []) + ["web_search"],
            "progress_pct": 50,
            "status_detail": "Web search not enabled",
        }

    web = _get_web()
    all_results = []
    # Use only the first 2 queries to avoid DuckDuckGo rate limits
    for query in state.get("search_queries", [])[:2]:
        try:
            results = web.search(query, max_results=4)
            all_results.extend(results)
        except Exception as e:
            logger.warning("Web search failed for '%s': %s", query[:40], e)

    logger.info("  Found %d web results.", len(all_results))

    return {
        "web_results": all_results,
        "current_step": "web_search",
        "completed_steps": state.get("completed_steps", []) + ["web_search"],
        "progress_pct": 55,
        "status_detail": f"DuckDuckGo: {len(all_results)} {'result' if len(all_results) == 1 else 'results'}",
    }


# ── Node 5: Document Analysis (Hybrid RAG) ────────────────────────────────────

def document_analysis_node(state: ResearchState) -> Dict[str, Any]:
    """
    Core analysis node: retrieves the most relevant document chunks via
    Hybrid RAG (dense FAISS + sparse BM25 → RRF) and injects them into
    the LLM context window.

    Retrieval strategy
    ──────────────────
    For each search query generated in Node 2, the hybrid store is queried
    and the results are deduplicated by chunk_id.  This ensures the context
    block contains diverse, query-relevant chunks rather than repetitive hits.

    Fallback
    ────────
    If the embedding model was unavailable (rag_fallback=True from Node 1),
    the node falls back to direct raw-text injection (vectorless mode).
    If no documents were uploaded (search-only mode), academic abstracts
    are used instead.
    """
    logger.info("[Node 5] Document Analysis (Hybrid RAG)")
    import httpx as _httpx_da
    llm = ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=0.3,
        num_predict=6000,
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": _httpx_da.Timeout(180.0)},
    )
    uploaded_docs = state.get("uploaded_docs", [])
    session_id = state.get("session_id", "default")
    num_ctx = state.get("num_ctx", cfg.num_ctx)
    rag_fallback = state.get("rag_fallback", False)
    top_k = cfg.hybrid_top_k

    doc_context: List[Dict] = []
    retrieved_chunks: List[Dict] = []
    # Preserve ReAct metadata from academic_search_node (if any); append chunk-level info below.
    rag_reflection_info: List[Dict] = list(state.get("rag_reflection_info") or [])

    if uploaded_docs:
        if not rag_fallback:
            # ── Hybrid RAG retrieval ──────────────────────────────────────
            store = get_or_create_store(
                session_id=session_id,
                embed_model=state.get("embed_model", cfg.embedding_model),
                ollama_base_url=cfg.ollama_base_url,
                persist_dir=cfg.chroma_persist_dir,
            )
            queries = state.get("search_queries", [state.get("goal", "")])
            seen_ids: set[str] = set()

            for query in queries:
                try:
                    chunks, meta = react_retrieve(
                        store, query, top_k,
                        model_name=state.get("model_name", cfg.ollama_model),
                        num_ctx=num_ctx,
                    )
                    rag_reflection_info.append({"query": query, **meta})
                    for chunk in chunks:
                        if chunk["chunk_id"] not in seen_ids:
                            seen_ids.add(chunk["chunk_id"])
                            retrieved_chunks.append(chunk)
                except Exception as e:
                    logger.warning("Hybrid retrieval failed for query '%s': %s", query[:50], e)

            # Cap total context to ~50% of context window (chars)
            max_total_chars = int(num_ctx * 0.5 * 3)
            accumulated = 0
            capped_chunks = []
            for chunk in retrieved_chunks:
                chunk_len = len(chunk["text"])
                if accumulated + chunk_len > max_total_chars:
                    break
                capped_chunks.append(chunk)
                accumulated += chunk_len

            retrieved_chunks = capped_chunks

            for chunk in retrieved_chunks:
                doc_context.append({
                    "text": chunk["text"],
                    "metadata": {
                        "source": chunk["doc_name"],
                        "page": chunk["page_num"],
                        "chunk": chunk["chunk_index"],
                    },
                    "score": chunk.get("rrf_score", 0.0),
                })

            logger.info(
                "  Hybrid RAG: %d unique chunks retrieved (%d chars).",
                len(doc_context), accumulated,
            )

            context_block = "\n\n---\n".join(
                f"[{c['metadata']['source']} | page {c['metadata']['page']} | "
                f"chunk {c['metadata']['chunk']} | score {c['score']:.3f}]\n{c['text']}"
                for c in doc_context
            )

        else:
            # ── Vectorless fallback ───────────────────────────────────────
            logger.info("  Using vectorless fallback (embed model unavailable).")
            total_char_budget = int(num_ctx * 0.6 * 3)
            chars_per_doc = min(20000, total_char_budget // len(uploaded_docs))
            for doc in uploaded_docs:
                excerpt = doc.raw_text[:chars_per_doc]
                doc_context.append({
                    "text": excerpt,
                    "metadata": {"source": doc.filename, "page": "full"},
                    "score": 1.0,
                })
            context_block = "\n\n---\n".join(
                f"[Document: {c['metadata']['source']}]\n{c['text']}"
                for c in doc_context
            )

        source_label = "UPLOADED DOCUMENTS"
    else:
        # Fall back to academic abstracts (search-only mode)
        papers = state.get("academic_papers", [])[:8]
        context_block = "\n\n---\n".join(
            f"[{p.citation_key}] {p.title}\n{p.abstract}" for p in papers
        )
        source_label = "ACADEMIC LITERATURE"

    system_prompt = (
        f"You are a senior research scientist writing a structured analytical report.\n"
        f"Source type: {source_label}.\n\n"
        "Standards:\n"
        "- Every analytical claim must be grounded in a specific passage, author, or result "
        "from the provided sources — do not make assertions without textual support\n"
        "- Use precise academic vocabulary; define specialist terms on first use\n"
        "- Write in flowing prose paragraphs (no bullet lists in the main body)\n"
        "- Target 1000–1500 words across 7–10 paragraphs\n"
        "- Be thorough and comprehensive. Cover each point with sufficient depth and supporting evidence.\n"
        "- Do NOT start with 'This analysis…', 'This document…', or 'In this report…'\n"
        "- End with a short 'Critical Assessment' paragraph noting the strength of evidence "
        "and what the most important remaining questions are"
        + _style_block(state)
        + _clarification_context(state)
    )

    human_prompt = f"""RESEARCH GOAL:
{state['goal']}

{source_label}:
{context_block}

Write a detailed analytical report covering:
1. Main findings most relevant to the research goal (with specific evidence from sources)
2. Key themes, patterns, and convergences across the material
3. Contradictions, debates, or tensions in the evidence
4. Methodological considerations or quality of evidence
5. Critical Assessment: strength of evidence, most important open questions

Cite evidence specifically — name authors, reference paper titles, or quote brief passages."""

    try:
        analysis = _invoke(llm, system_prompt, human_prompt)
    except Exception as e:
        analysis = f"[Analysis error: {e}]"
        logger.error("Document analysis failed: %s", e)

    # ── Extract bullet-point key findings ─────────────────────
    findings_prompt = f"""From this analysis, extract 5–8 concise key findings.
Each finding must be a single sentence starting with an action verb.
Return ONLY a JSON array of strings.

Analysis:
{analysis[:3000]}"""

    try:
        raw_findings = _invoke(
            llm,
            "You are a research summariser. Return only valid JSON arrays.",
            findings_prompt,
        )
        match = re.search(r"\[.*\]", raw_findings, re.DOTALL)
        key_findings = json.loads(match.group(0)) if match else []
    except Exception as e:
        logger.warning("Key findings extraction failed (%s) — returning empty list", e)
        key_findings = []

    # ── Per-document analysis (only when multiple docs are uploaded) ──────
    uploaded_docs = state.get("uploaded_docs", [])
    per_doc_analyses: Dict[str, str] = {}
    multi_doc_synthesis: str = ""

    if len(uploaded_docs) > 1:
        per_doc_system = """You are a research analyst. Analyse the provided document
excerpt against the research goal. Write a focused 400–600 word analysis identifying:
1. The document's main contribution relevant to the goal
2. Key methods, findings, or arguments
3. Strengths or limitations worth noting
Be specific and evidence-based."""

        for doc in uploaded_docs:
            # Use retrieved chunks for this doc, fall back to raw text if none
            doc_chunks = [
                c for c in retrieved_chunks if c.get("doc_name") == doc.filename
            ]
            if doc_chunks:
                doc_text = "\n\n".join(c["text"] for c in doc_chunks[:6])
            else:
                doc_text = doc.raw_text[:5000]

            per_doc_human = f"""RESEARCH GOAL: {state.get('goal', '')}

DOCUMENT: {doc.filename}
RELEVANT CONTENT:
{doc_text}

Write a focused 400–600 word analysis of this document's contribution to the research goal."""
            try:
                per_doc_analyses[doc.filename] = _invoke(llm, per_doc_system, per_doc_human)
                logger.info("  ✓ Per-doc analysis: %s", doc.filename)
            except Exception as e:
                per_doc_analyses[doc.filename] = f"[Analysis error: {e}]"
                logger.warning("Per-doc analysis failed for %s: %s", doc.filename, e)

        # Cross-document synthesis
        doc_summaries = "\n\n---\n".join(
            f"DOCUMENT: {fn}\n{analysis_text[:800]}"
            for fn, analysis_text in per_doc_analyses.items()
        )
        synth_system = """You are a research analyst synthesising findings across multiple documents.
Identify: (1) common themes shared across documents, (2) contradictions or tensions,
(3) how the documents complement each other, (4) collective gaps or open questions.
Write 400–600 words covering common themes, contradictions, complementary contributions, and collective gaps in detail."""
        synth_human = f"""RESEARCH GOAL: {state.get('goal', '')}

DOCUMENT ANALYSES:
{doc_summaries}

Write a cross-document synthesis paragraph."""
        try:
            multi_doc_synthesis = _invoke(llm, synth_system, synth_human)
            logger.info("  ✓ Cross-document synthesis complete")
        except Exception as e:
            multi_doc_synthesis = f"[Synthesis error: {e}]"
            logger.warning("Cross-doc synthesis failed: %s", e)

    if uploaded_docs:
        _da_n_chunks = len(retrieved_chunks)
        _da_n_docs = len({c["doc_name"] for c in retrieved_chunks}) if retrieved_chunks else len(uploaded_docs)
        _da_mode = "text injection" if rag_fallback else "FAISS + BM25 + RRF"
        _da_detail = f"{_da_n_chunks} chunks from {_da_n_docs} source(s) · {_da_mode}"
    else:
        _da_detail = f"Using {len(state.get('academic_papers', [])[:8])} academic abstracts"

    return {
        "doc_context": doc_context,
        "retrieved_chunks": retrieved_chunks,
        "analysis": analysis,
        "key_findings": key_findings,
        "per_doc_analyses": per_doc_analyses,
        "multi_doc_synthesis": multi_doc_synthesis,
        "rag_reflection_info": rag_reflection_info,
        "current_step": "document_analysis",
        "completed_steps": state.get("completed_steps", []) + ["document_analysis"],
        "progress_pct": 70,
        "status_detail": _da_detail,
    }


# ── Node 6: Reference Compilation ─────────────────────────────────────────────

def reference_compilation_node(state: ResearchState) -> Dict[str, Any]:
    """
    Build a curated, deduplicated bibliography from:
      • Academic papers found by the searcher
      • Citations extracted from uploaded documents (if any)

    Each reference is mapped to the sections of the analysis that cite it,
    enabling proper in-text citation linking in the final report.
    """
    logger.info("[Node 6] Reference Compilation")
    llm = _llm(state)

    papers = state.get("academic_papers", [])
    analysis = state.get("analysis", "")

    # ── Ask LLM to select the most relevant papers ────────────
    if papers:
        paper_list = "\n".join(
            f"{i+1}. [{p.citation_key}] {p.title} ({p.year or 'n.d.'}) — {p.source}"
            for i, p in enumerate(papers[:20])
        )

        selection_prompt = f"""Given this research analysis and list of papers,
select the paper numbers that are MOST RELEVANT to the analysis.
Return ONLY a JSON array of integers. Example: [1, 3, 5, 7]

Analysis excerpt:
{analysis[:3000]}

Papers:
{paper_list}"""

        try:
            raw_sel = _invoke(
                llm,
                "You are a reference curator. Return only valid JSON arrays of integers.",
                selection_prompt,
            )
            match = re.search(r"\[[\d,\s]+\]", raw_sel)
            selected_idx = json.loads(match.group(0)) if match else list(range(1, len(papers) + 1))
        except Exception as e:
            logger.warning("Reference selection failed (%s) — defaulting to first 10 papers", e)
            selected_idx = list(range(1, min(len(papers) + 1, 11)))

        selected_papers = [
            papers[i - 1] for i in selected_idx
            if 1 <= i <= len(papers)
        ]
    else:
        selected_papers = []

    # ── Build structured reference dicts ──────────────────────
    references = []
    for i, p in enumerate(selected_papers[:15]):  # max 15 references
        references.append({
            "ref_num": i + 1,
            "citation_key": p.citation_key,
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "journal": p.journal or p.venue or "Preprint",
            "doi": p.doi,
            "url": p.url,
            "citation_count": p.citation_count,
            "source": p.source,
            "abstract_snippet": p.abstract[:200],
            "apa": p.to_apa(),
        })

    logger.info("  Compiled %d references.", len(references))

    return {
        "references": references,
        "current_step": "reference_compilation",
        "completed_steps": state.get("completed_steps", []) + ["reference_compilation"],
        "progress_pct": 85,
        "status_detail": f"Compiled {len(references)} references",
    }


# ── Node 7: Report Generation ─────────────────────────────────────────────────

def report_generation_node(state: ResearchState) -> Dict[str, Any]:
    """
    Synthesise everything into a well-structured Markdown research report.

    Report structure:
    ─────────────────
    # Research Report: <Goal>
    ## Executive Summary
    ## Key Findings
    ## Detailed Analysis
    ## Methodology Note
    ## References
    """
    logger.info("[Node 7] Report Generation")
    import httpx as _httpx
    llm = ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=0.3,
        num_predict=8000,
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": _httpx.Timeout(180.0)},
    )

    goal = state.get("goal", "")
    analysis = state.get("analysis", "")
    key_findings = state.get("key_findings", [])
    references = state.get("references", [])
    docs = state.get("uploaded_docs", [])
    mode = state.get("mode", "hybrid")
    per_doc_analyses = state.get("per_doc_analyses", {})
    multi_doc_synthesis = state.get("multi_doc_synthesis", "")

    # ── Format references block ───────────────────────────────
    # Use citation_key format so reference list matches inline citations in the analysis.
    # The analysis LLM cites papers as [Smith et al., 2022]; the reference list must use
    # the same keys — not numeric [1],[2] — or inline citations become orphaned.
    ref_block = "\n".join(
        f"[{r['citation_key']}] {r['apa']}" for r in references
    ) if references else "_No external references found._"

    # ── Format findings block ─────────────────────────────────
    findings_block = (
        "\n".join(f"- {f}" for f in key_findings)
        if key_findings
        else "_See detailed analysis below._"
    )

    # ── Document sources summary ──────────────────────────────
    doc_summary = ""
    if docs:
        doc_summary = "**Uploaded Documents Analysed:**\n" + "\n".join(
            f"- {d.filename} ({d.total_pages} pages, {d.total_chunks} chunks)"
            for d in docs
        )

    # ── Web results block ─────────────────────────────────────
    web_block = ""
    web_results = state.get("web_results", [])
    if web_results:
        web_block = "\n## Additional Web Sources\n" + "\n".join(
            f"- [{r.title}]({r.url}): {r.snippet[:150]}…"
            for r in web_results[:5]
        )

    # ── Ask LLM to write the executive summary ────────────────
    summary_prompt = f"""Write a comprehensive executive summary (2–3 paragraphs, 200–300 words) covering the main topic, key discoveries, and significance of the findings for a research report.

Research Goal: {goal}

Key Findings:
{findings_block}

Be specific, objective, and informative. Do NOT use filler phrases."""

    try:
        exec_summary = _invoke(
            llm,
            "You are a scientific writer. Be thorough and precise."
            + _style_block(state)
            + _clarification_context(state),
            summary_prompt,
        )
    except Exception as e:
        exec_summary = f"_Summary generation failed: {e}_"

    # ── Assemble the full Markdown report ─────────────────────
    # ── Per-document breakdown section (multi-doc only) ───────────────────
    per_doc_section = ""
    if per_doc_analyses:
        parts_list = [f"## Per-Document Breakdown\n"]
        for filename, doc_analysis in per_doc_analyses.items():
            parts_list.append(f"### {filename}\n\n{doc_analysis}\n")
        if multi_doc_synthesis:
            parts_list.append(f"### Cross-Document Synthesis\n\n{multi_doc_synthesis}\n")
        per_doc_section = "\n".join(parts_list)

    report_parts = [
        f"# Research Report\n\n**Goal:** {goal}\n",
        f"## Executive Summary\n\n{exec_summary}\n",
        f"## Key Findings\n\n{findings_block}\n",
    ]

    if per_doc_section:
        report_parts.append(per_doc_section)

    report_parts.append(f"## Detailed Analysis\n\n{analysis}\n")

    if doc_summary:
        report_parts.append(f"## Source Documents\n\n{doc_summary}\n")

    if web_block:
        report_parts.append(web_block)

    # Methodology note — important for transparency
    paper_count = len(state.get("academic_papers", []))
    num_ctx = state.get("num_ctx", cfg.num_ctx)
    total_chars_injected = sum(len(c["text"]) for c in state.get("doc_context", []))
    mode_desc = {
        "document": "uploaded document analysis (vectorless RAG)",
        "search": "academic literature search",
        "hybrid": "uploaded document analysis + academic literature search (vectorless RAG)",
    }.get(mode, mode)

    methodology = (
        f"## Methodology Note\n\n"
        f"This report was generated using an agentic AI workflow powered by "
        f"**{state.get('model_name', 'local LLM')}** (local, open-source).\n\n"
        f"- **Mode:** {mode_desc}\n"
        f"- **Documents processed:** {len(docs)}\n"
        f"- **Text injected (vectorless RAG):** {total_chars_injected:,} chars\n"
        f"- **Context window:** {num_ctx:,} tokens\n"
        f"- **Academic papers searched:** {paper_count}\n"
        f"- **References cited:** {len(references)}\n"
        f"- **Search queries used:** {len(state.get('search_queries', []))}\n"
    )
    report_parts.append(methodology)

    ref_section = f"## References\n\n{ref_block}\n"
    report_parts.append(ref_section)

    report = "\n---\n".join(report_parts)

    logger.info("[Node 7] Report generation complete (%d chars).", len(report))

    return {
        "report": report,
        "current_step": "complete",
        "completed_steps": state.get("completed_steps", []) + ["report_generation"],
        "progress_pct": 100,
        "status_detail": f"Report ready · {len(report):,} chars · {len(state.get('references', []))} references",
    }
