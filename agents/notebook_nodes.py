"""
agents/notebook_nodes.py
Three nodes for the Research Notebook: retrieve → answer → save
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

_memory: NotebookMemory | None = None


def _get_memory() -> NotebookMemory:
    global _memory
    if _memory is None:
        _memory = NotebookMemory()
    return _memory


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
    doc_id: str
    filename: str
    content_md5: str
    chunks: List[_RebuiltChunk]


def _rebuild_docs_from_chunks(notebook: Dict[str, Any]) -> List[_RebuiltDoc]:
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


def retrieve_node(state: NotebookState) -> Dict[str, Any]:
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
    else:
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
            logger.warning("Retrieval error (%s) — using BM25 keyword fallback.", e)
            errors.append(f"Retrieval degraded to keyword search: {e}")
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
                    "metadata": {"url": wr.url, "source": "web"},
                })
            if web_results:
                retrieval_mode = retrieval_mode + "+web" if retrieval_mode != "empty" else "web"
            else:
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


def _llm(state: NotebookState, temperature: float = 0.3) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=4096,
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(180.0)},
    )


def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
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
    logger.info("[Notebook Node 2] Answer")
    chunks = state.get("retrieved_chunks", [])
    source_count = state.get("source_count", 0)

    if source_count == 0 and not chunks:
        retrieval_mode = state.get("retrieval_mode", "")
        if "web_error" in retrieval_mode:
            msg = (
                "Auto web search is enabled but the search failed. "
                "Check your internet connection or upload a document "
                "(PDF, DOCX, TXT, or Markdown) to use as a source."
            )
        elif "web" in retrieval_mode:
            msg = (
                "Auto web search ran but found no usable results for this query. "
                "Try rephrasing your question or upload a relevant document as a source."
            )
        else:
            msg = (
                "This notebook has no sources yet. Upload a document (PDF, DOCX, TXT, "
                "or Markdown), add a web page on the left, or enable **Auto web search** "
                "to let the agent search the web automatically."
            )
        return {
            "assistant_response": msg,
            "citations": [],
            "suggested_questions": [],
            "current_step": "answer",
            "completed_steps": state.get("completed_steps", []) + ["answer"],
            "progress_pct": 80,
        }

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
4. Give comprehensive, detailed answers. When the sources contain sufficient information, write 5–8 paragraphs covering all relevant aspects.
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


def _split_suggested_questions(raw: str) -> tuple[str, List[str]]:
    suggested: List[str] = []
    body = raw
    marker = '{"suggested_questions"'
    if marker in raw:
        idx = raw.rfind(marker)
        body = raw[:idx].strip()
        fragment = raw[idx:]
        try:
            suggested = json.loads(fragment).get("suggested_questions", [])[:3]
        except Exception:
            m = re.search(r'"suggested_questions"\s*:\s*(\[.*?\])', fragment, re.DOTALL)
            if m:
                try:
                    suggested = json.loads(m.group(1))[:3]
                except Exception:
                    logger.warning("Could not parse suggested_questions tail: %s", fragment[-80:])
    return body, [q for q in suggested if isinstance(q, str)]


def _extract_citations(answer: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def save_node(state: NotebookState) -> Dict[str, Any]:
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
