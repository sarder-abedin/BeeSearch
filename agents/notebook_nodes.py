"""
agents/notebook_nodes.py
─────────────────────────
The three nodes that form the Research Notebook (Mode 8) workflow.

  START → retrieve → answer → save → END

Node responsibilities
─────────────────────
  retrieve : Load the notebook, (re)build the HybridStore, and run hybrid
             retrieval (dense FAISS + sparse BM25 fused with RRF) for the query.
             Also loads recent conversation history.
  answer   : Generate an answer grounded ONLY in the retrieved chunks, with
             inline [n] citations, plus 2–3 follow-up questions.
  save     : Persist the user question and the cited assistant answer to memory.

GROUNDING CONTRACT
──────────────────
The answer node is given numbered source excerpts and is instructed to answer
strictly from them, cite every claim with [n], and explicitly say when the
sources do not contain the answer. This is what makes the notebook a
"closed-book over your sources" assistant rather than a general chatbot.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.notebook_memory import NotebookMemory
from agents.notebook_state import NotebookState
from config.settings import get_settings
from tools.hybrid_store import get_or_create_store

logger = logging.getLogger(__name__)
cfg = get_settings()

# Lazy singleton — not created at import time so tests can inject a different instance.
_memory: NotebookMemory | None = None


def _get_memory() -> NotebookMemory:
    global _memory
    if _memory is None:
        _memory = NotebookMemory()
    return _memory


# ── Minimal ProcessedDocument rebuild (for re-indexing from stored chunks) ──────

@dataclass
class _RebuiltChunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_num: int
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class _RebuiltDoc:
    """A lightweight stand-in for ProcessedDocument with just what HybridStore needs."""
    doc_id: str
    filename: str
    content_md5: str
    chunks: List[_RebuiltChunk]


def _rebuild_docs_from_chunks(notebook: Dict[str, Any]) -> List[_RebuiltDoc]:
    """
    Reconstruct per-document chunk lists from the notebook's stored chunks so the
    HybridStore can rebuild its FAISS + BM25 indexes. Embeddings are read from the
    ChromaDB cache (keyed by stable chunk_id) — no re-embedding occurs.
    """
    md5_by_doc = {s["doc_id"]: s.get("content_md5", "") for s in notebook.get("sources", [])}
    name_by_doc = {s["doc_id"]: s.get("filename", "source") for s in notebook.get("sources", [])}

    grouped: Dict[str, List[_RebuiltChunk]] = {}
    for c in notebook.get("chunks", []):
        grouped.setdefault(c["doc_id"], []).append(
            _RebuiltChunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                doc_name=c.get("doc_name", name_by_doc.get(c["doc_id"], "source")),
                page_num=c.get("page_num", 0),
                chunk_index=c.get("chunk_index", 0),
                text=c.get("text", ""),
            )
        )

    docs: List[_RebuiltDoc] = []
    for doc_id, chunks in grouped.items():
        docs.append(_RebuiltDoc(
            doc_id=doc_id,
            filename=name_by_doc.get(doc_id, "source"),
            content_md5=md5_by_doc.get(doc_id, ""),
            chunks=chunks,
        ))
    return docs


# ── Node 1: Retrieve ───────────────────────────────────────────────────────────

def retrieve_node(state: NotebookState) -> Dict[str, Any]:
    """
    Load the notebook, ensure the hybrid index is built, and retrieve the most
    relevant chunks for the user's question.

    Falls back to the first chunks of the notebook (in document order) if the
    embedding model is unavailable, so the mode still works without a pulled
    embedding model — just without semantic ranking.
    """
    logger.info("[Notebook Node 1] Retrieve")
    notebook_id = state.get("notebook_id", "")
    errors = list(state.get("errors", []))

    notebook = _get_memory().load(notebook_id) if notebook_id else None
    if not notebook:
        return {
            "retrieved_chunks": [],
            "conversation_history": [],
            "source_count": 0,
            "retrieval_mode": "empty",
            "errors": errors + [f"Notebook '{notebook_id}' not found."],
            "current_step": "retrieve",
            "completed_steps": state.get("completed_steps", []) + ["retrieve"],
            "progress_pct": 30,
        }

    history = notebook.get("conversation", [])[-8:]
    stored_chunks = notebook.get("chunks", [])
    source_count = len(notebook.get("sources", []))

    query = state.get("user_message", "")
    top_k = state.get("top_k", cfg.hybrid_top_k)
    embed_model = state.get("embed_model", cfg.embedding_model)

    retrieved: List[Dict[str, Any]] = []
    retrieval_mode = "empty"
    rag_meta: Dict[str, Any] = {}

    if not stored_chunks:
        # No uploaded sources. If web search is also off, return early with nothing to retrieve.
        if not state.get("include_web_search", False):
            return {
                "retrieved_chunks": [],
                "conversation_history": history,
                "source_count": source_count,
                "retrieval_mode": "empty",
                "current_step": "retrieve",
                "completed_steps": state.get("completed_steps", []) + ["retrieve"],
                "progress_pct": 30,
            }
        # Web search is enabled — skip RAG and fall through to the web search block below.
    else:
        # There are stored chunks — run hybrid RAG retrieval.
        retrieval_mode = "fallback"
        try:
            store = get_or_create_store(
                session_id=f"notebook_{notebook_id}",
                embed_model=embed_model,
                ollama_base_url=cfg.ollama_base_url,
                persist_dir=cfg.chroma_persist_dir,
            )
            if not store.is_indexed():
                docs = _rebuild_docs_from_chunks(notebook)
                # add_documents auto-falls-back to BM25-only if embedding is unavailable.
                store.add_documents(docs, warning_callback=logger.warning)
            from agents.self_reflective_rag import react_retrieve
            retrieved, rag_meta = react_retrieve(
                store, query, top_k,
                model_name=state.get("model_name", cfg.embedding_model),
                num_ctx=state.get("num_ctx", cfg.num_ctx),
            )
            if retrieved:
                retrieval_mode = "react" if store._faiss_index else "bm25"
            else:
                logger.warning("Retrieve returned empty — using fallback chunks.")
                errors.append(
                    f"Semantic retrieval unavailable. "
                    f"Run `ollama pull {embed_model}` for full hybrid search. "
                    "Falling back to the first sections of your sources."
                )
                retrieved = stored_chunks[:top_k]
                retrieval_mode = "fallback"
        except Exception as e:
            # Catches RuntimeError from embedder AND any import/memory error from FAISS.
            logger.warning("Retrieval error (%s) — using BM25 keyword fallback.", e)
            errors.append(f"Retrieval degraded to keyword search: {e}")
            # Build BM25-only index as last resort (no FAISS, no Ollama)
            try:
                store = get_or_create_store(
                    session_id=f"notebook_{notebook_id}_bm25",
                    embed_model=embed_model,
                    ollama_base_url=cfg.ollama_base_url,
                    persist_dir=cfg.chroma_persist_dir,
                )
                if not store.is_indexed():
                    docs = _rebuild_docs_from_chunks(notebook)
                    store.add_documents_bm25_only(docs)
                retrieved = store.search_hybrid(query, top_k) or stored_chunks[:top_k]
            except Exception:
                retrieved = stored_chunks[:top_k]
            retrieval_mode = "fallback"

    # Optional automatic Google search — supplements notebook sources.
    if state.get("include_web_search", False):
        try:
            from tools.search_tools import WebSearcher
            web_results = WebSearcher().search(query, max_results=5)
            for i, wr in enumerate(web_results):
                retrieved.append({
                    "chunk_id": f"web_{i}",
                    "doc_id": f"web_{wr.url}",
                    "doc_name": (wr.title or "").strip()[:80] or wr.url,
                    "page_num": -1,
                    "chunk_index": i,
                    "text": wr.snippet or wr.title,
                    "metadata": {"url": wr.url, "source": "google"},
                })
            if web_results:
                retrieval_mode = retrieval_mode + "+web" if retrieval_mode != "empty" else "web"
            else:
                # Web search ran but returned nothing — mark so answer_node can give a better message.
                retrieval_mode = retrieval_mode + "+web_empty" if retrieval_mode != "empty" else "web_empty"
        except Exception as exc:
            logger.warning("Notebook auto web search failed: %s", exc)
            errors.append(f"Web search unavailable: {exc}")
            retrieval_mode = retrieval_mode + "+web_error" if retrieval_mode != "empty" else "web_error"

    return {
        "retrieved_chunks": retrieved,
        "conversation_history": history,
        "source_count": source_count,
        "retrieval_mode": retrieval_mode,
        "rag_reflection_info": rag_meta,
        "errors": errors,
        "current_step": "retrieve",
        "completed_steps": state.get("completed_steps", []) + ["retrieve"],
        "progress_pct": 35,
    }


# ── Node 2: Answer ───────────────────────────────────────────────────────────

def _max_predict(state: NotebookState) -> int:
    """Reserve 25% of context for the prompt; use the rest for output (min 4096)."""
    return max(4096, int(state.get("num_ctx", cfg.num_ctx) * 0.75))


def _llm(state: NotebookState, temperature: float = 0.3) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=_max_predict(state),
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(180.0)},
    )


def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Number each retrieved chunk as a citable source with doc name + page."""
    lines = []
    for i, ch in enumerate(chunks, 1):
        page = ch.get("page_num", 0)
        page_label = f"p.{page}" if isinstance(page, int) and page >= 0 else "n/a"
        lines.append(
            f"[{i}] (source: {ch.get('doc_name', 'unknown')}, {page_label})\n"
            f"{ch.get('text', '').strip()}"
        )
    return "\n\n".join(lines)


def _format_history(history: List[Dict]) -> str:
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('content', '')[:500]}")
    return "\n\nPREVIOUS CONVERSATION:\n" + "\n\n".join(lines)


def answer_node(state: NotebookState) -> Dict[str, Any]:
    """
    Generate an answer grounded strictly in the retrieved chunks, with inline
    [n] citations and a trailing suggested_questions JSON block.
    """
    logger.info("[Notebook Node 2] Answer")
    chunks = state.get("retrieved_chunks", [])
    source_count = state.get("source_count", 0)

    # No sources and no web results → ask the user to add some.
    if source_count == 0 and not chunks:
        retrieval_mode = state.get("retrieval_mode", "")
        if "web_error" in retrieval_mode:
            msg = (
                "Auto web search is enabled but the search failed. "
                "Check your internet connection or upload a document "
                "(PDF, DOCX, TXT, or Markdown) to use as a source."
            )
        elif "web" in retrieval_mode:
            # web search ran and returned empty
            msg = (
                "Auto web search ran but found no usable results for this query. "
                "Try rephrasing your question or upload a relevant document as a source."
            )
        else:
            msg = (
                "This notebook has no sources yet. Upload a document (PDF, DOCX, TXT, "
                "or Markdown), add a web page on the left, or enable **Auto web search** "
                "to let the agent search Google automatically."
            )
        return {
            "assistant_response": msg,
            "citations": [],
            "suggested_questions": [],
            "current_step": "answer",
            "completed_steps": state.get("completed_steps", []) + ["answer"],
            "progress_pct": 80,
        }

    # Sources exist but nothing relevant retrieved.
    if not chunks:
        msg = (
            "I couldn't find anything in this notebook's sources that addresses "
            "your question. Try rephrasing, or add a source that covers this topic."
        )
        return {
            "assistant_response": msg,
            "citations": [],
            "suggested_questions": [],
            "current_step": "answer",
            "completed_steps": state.get("completed_steps", []) + ["answer"],
            "progress_pct": 80,
        }

    context_block = _build_context_block(chunks)
    history_block = _format_history(state.get("conversation_history", []))

    system = """You are a Research Notebook assistant. You answer questions using \
ONLY the numbered source excerpts provided — never your own outside knowledge.

STRICT RULES:
1. Base every statement on the provided sources. Do NOT use prior knowledge.
2. Cite each claim inline with the bracketed source number, e.g. "...reduces error [2]."
   You may cite multiple sources like [1][3].
3. If the sources do not contain enough information to answer, say so plainly:
   "The sources in this notebook don't cover that." Do not guess or invent facts.
4. Give comprehensive, detailed answers. Cover all relevant aspects the sources support. Do not truncate your response — write until the answer is complete.
5. Quote short phrases verbatim when precision matters.
6. Never cite a source number that was not provided.
7. At the very end, append EXACTLY this JSON on its own line and nothing after it:
   {"suggested_questions": ["Question 1?", "Question 2?", "Question 3?"]}
   The questions must be answerable from the same sources."""

    human = f"""SOURCE EXCERPTS:
{context_block}
{history_block}

QUESTION: {state.get('user_message', '')}

Answer using only the sources above, with inline [n] citations. End with the suggested_questions JSON."""

    try:
        resp = _llm(state).invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = resp.content.strip()
    except Exception as e:
        logger.error("Notebook answer LLM call failed: %s", e)
        return {
            "assistant_response": f"[Error generating answer: {e}]",
            "citations": [],
            "suggested_questions": [],
            "errors": state.get("errors", []) + [str(e)],
            "current_step": "answer",
            "completed_steps": state.get("completed_steps", []) + ["answer"],
            "progress_pct": 80,
        }

    main_response, suggested_questions = _split_suggested_questions(raw)
    citations = _extract_citations(main_response, chunks)

    return {
        "assistant_response": main_response,
        "citations": citations,
        "suggested_questions": suggested_questions,
        "current_step": "answer",
        "completed_steps": state.get("completed_steps", []) + ["answer"],
        "progress_pct": 80,
    }


_SQ_PATTERNS = [
    # Full JSON object  {"suggested_questions": [...]}
    re.compile(r'\{[^{}]*"suggested_questions"\s*:\s*(\[.*?\])\s*\}', re.DOTALL),
    # Quoted key without outer braces  "suggested_questions": [...]
    re.compile(r'"suggested_questions"\s*:\s*(\[.*?\])', re.DOTALL),
    # Bare key  suggested_questions: [...] or suggested_questions = [...]
    re.compile(r'suggested_questions\s*[:=]\s*(\[.*?\])', re.DOTALL | re.IGNORECASE),
]


def _split_suggested_questions(raw: str) -> tuple[str, List[str]]:
    """Extract suggested_questions from any format the LLM may emit, strip from body."""
    for pat in _SQ_PATTERNS:
        m = pat.search(raw)
        if m:
            try:
                candidates = json.loads(m.group(1))
                if isinstance(candidates, list) and candidates:
                    questions = [str(q) for q in candidates if q][:3]
                    body = raw[:m.start()].strip()
                    return body, questions
            except Exception:
                continue
    logger.warning("Could not parse suggested_questions — raw tail: %s", raw[-120:])
    return raw, []


def _extract_citations(answer: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a citation list for the [n] markers that actually appear in the answer.
    Falls back to all retrieved chunks if the model cited nothing explicitly.
    """
    cited_nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    citations: List[Dict[str, Any]] = []
    for n in cited_nums:
        if 1 <= n <= len(chunks):
            ch = chunks[n - 1]
            entry: Dict[str, Any] = {
                "n": n,
                "doc_name": ch.get("doc_name", "unknown"),
                "page": ch.get("page_num", 0),
                "snippet": ch.get("text", "")[:240].strip(),
            }
            url = (ch.get("metadata") or {}).get("url", "")
            if url:
                entry["url"] = url
            citations.append(entry)
    return citations


# ── Node 3: Save ───────────────────────────────────────────────────────────────

def save_node(state: NotebookState) -> Dict[str, Any]:
    """Persist the user question and cited assistant answer to the notebook."""
    logger.info("[Notebook Node 3] Save")
    notebook_id = state.get("notebook_id", "")
    if not notebook_id:
        return {
            "current_step": "save",
            "completed_steps": state.get("completed_steps", []) + ["save"],
            "progress_pct": 100,
        }

    mem = _get_memory()
    mem.add_turn(notebook_id, role="user", content=state.get("user_message", ""))
    mem.add_turn(
        notebook_id,
        role="assistant",
        content=state.get("assistant_response", ""),
        citations=state.get("citations", []),
        suggested_questions=state.get("suggested_questions", []),
    )

    return {
        "current_step": "save",
        "completed_steps": state.get("completed_steps", []) + ["save"],
        "progress_pct": 100,
    }
