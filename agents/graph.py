"""
agents/graph.py
───────────────
Research Report workflow for the Research Notebook mode.

Public entry point:
    run_research(state, stream_callback=None) → final_state

Steps
─────
  document_ingestion    Build a text context block from notebook sources
  query_generation      LLM generates focused search queries from the goal
  academic_search       arXiv + Semantic Scholar (skipped for mode="document")
  web_search            DuckDuckGo (only when include_web_search=True)
  document_analysis     Merge all context blocks for the report step
  reference_compilation Build a reference list from the academic papers found
  report_generation     LLM generates a structured Markdown research report
  research_eval         LLM self-evaluates report quality (1-5 per dimension)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

_MAX_CHARS_PER_DOC = 5_000
_MAX_DOC_CONTEXT = 15_000
_MAX_ACADEMIC_CONTEXT = 5_000
_MAX_WEB_CONTEXT = 2_000


# ── LLM helpers ────────────────────────────────────────────────────────────────

def _llm(state: dict, temperature: float = 0.3, num_predict: int = 4096) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=state.get("model_name") or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(300.0)},
    )


def _invoke(llm: ChatOllama, system: str, human: str) -> str:
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content.strip()


# ── Context builders ────────────────────────────────────────────────────────────

def _build_doc_context(uploaded_docs: list) -> str:
    parts: list[str] = []
    total = 0
    for i, doc in enumerate(uploaded_docs, 1):
        chunks = doc.chunks if hasattr(doc, "chunks") else []
        text = " ".join(c.text for c in chunks)[:_MAX_CHARS_PER_DOC]
        parts.append(f"[Source {i}: {doc.filename}]\n{text}")
        total += len(text)
        if total >= _MAX_DOC_CONTEXT:
            break
    return "\n\n".join(parts)


def _build_academic_context(papers: list) -> str:
    parts: list[str] = []
    total = 0
    for i, p in enumerate(papers, 1):
        title = getattr(p, "title", "Untitled")
        authors = ", ".join((getattr(p, "authors", None) or [])[:3])
        year = getattr(p, "year", "") or ""
        abstract = (getattr(p, "abstract", "") or "")[:400]
        source = getattr(p, "source", "") or ""
        snippet = (
            f"[Paper {i}: {title} ({year}, {source})]\n"
            f"Authors: {authors}\nAbstract: {abstract}"
        )
        parts.append(snippet)
        total += len(snippet)
        if total >= _MAX_ACADEMIC_CONTEXT:
            break
    return "\n\n".join(parts)


def _build_web_context(web_results: list) -> str:
    parts: list[str] = []
    for i, r in enumerate(web_results[:5], 1):
        title = getattr(r, "title", "")
        snippet = getattr(r, "snippet", "")
        url = getattr(r, "url", "")
        parts.append(f"[Web {i}: {title}]\n{snippet}\n{url}")
    return "\n\n".join(parts)


# ── Workflow steps ──────────────────────────────────────────────────────────────

def _step_document_ingestion(state: dict) -> dict:
    docs = state.get("uploaded_docs") or []
    state["_doc_context"] = _build_doc_context(docs)
    logger.info("[Research Report] Ingested %d document(s)", len(docs))
    return state


def _step_query_generation(state: dict) -> dict:
    goal = state["goal"]
    if state["mode"] == "document" and not state.get("include_web_search"):
        state["search_queries"] = [goal]
        return state

    system = (
        "You are a research assistant. Given a research goal, output exactly 3 focused "
        "search queries suitable for arXiv and Semantic Scholar. "
        "One query per line. No numbering, no bullets, no extra text."
    )
    try:
        raw = _invoke(_llm(state, temperature=0.2, num_predict=200), system,
                      f"Research goal: {goal}")
        queries = [q.strip().lstrip("•-*0123456789.) ") for q in raw.splitlines() if q.strip()]
        state["search_queries"] = queries[:3] or [goal]
    except Exception as e:
        logger.warning("[Research Report] Query generation failed: %s", e)
        state["search_queries"] = [goal]
    return state


def _step_academic_search(state: dict) -> dict:
    if state["mode"] == "document":
        state["academic_papers"] = []
        return state

    from tools.search_tools import AcademicSearcher
    searcher = AcademicSearcher()
    papers: list = []
    seen: set[str] = set()

    for query in (state.get("search_queries") or [state["goal"]])[:2]:
        try:
            for p in searcher.search(query, max_per_source=4):
                key = re.sub(r"\W+", "", (getattr(p, "title", "") or "").lower())[:60]
                if key and key not in seen:
                    seen.add(key)
                    papers.append(p)
        except Exception as e:
            logger.warning("[Research Report] Academic search failed for '%s': %s", query[:40], e)

    state["academic_papers"] = papers[:12]
    logger.info("[Research Report] %d unique academic paper(s) found", len(papers))
    return state


def _step_web_search(state: dict) -> dict:
    if not state.get("include_web_search"):
        state["web_results"] = []
        return state

    from tools.search_tools import WebSearcher
    searcher = WebSearcher()
    results: list = []
    for query in (state.get("search_queries") or [state["goal"]])[:2]:
        try:
            results.extend(searcher.search(query, max_results=3))
        except Exception as e:
            logger.warning("[Research Report] Web search failed: %s", e)
    state["web_results"] = results[:6]
    return state


def _step_document_analysis(state: dict) -> dict:
    state["_academic_context"] = _build_academic_context(state.get("academic_papers") or [])
    state["_web_context"] = _build_web_context(state.get("web_results") or [])
    return state


def _step_reference_compilation(state: dict) -> dict:
    refs: list[dict] = []
    for i, p in enumerate(state.get("academic_papers") or [], 1):
        authors = getattr(p, "authors", None) or []
        year = getattr(p, "year", "") or ""
        title = getattr(p, "title", "") or ""
        journal = getattr(p, "journal", "") or ""
        doi = getattr(p, "doi", "") or ""
        url = getattr(p, "url", "") or ""
        source = getattr(p, "source", "") or ""
        abstract = (getattr(p, "abstract", "") or "")[:250]
        citation_count = getattr(p, "citation_count", None)

        auth_str = "; ".join(authors[:3]) + ("…" if len(authors) > 3 else "")
        apa = f"{auth_str} ({year}). {title}. {journal}."
        if doi:
            apa += f" https://doi.org/{doi}"
        elif url:
            apa += f" {url}"

        refs.append({
            "ref_num": i,
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "doi": doi,
            "url": url,
            "abstract_snippet": abstract,
            "source": source,
            "citation_count": citation_count,
            "apa": apa,
        })
    state["references"] = refs
    return state


def _step_report_generation(state: dict) -> dict:
    goal = state["goal"]
    doc_ctx = state.get("_doc_context", "")
    acad_ctx = state.get("_academic_context", "")
    web_ctx = state.get("_web_context", "")

    context_parts: list[str] = []
    if doc_ctx:
        context_parts.append("## Notebook Sources\n" + doc_ctx)
    if acad_ctx:
        context_parts.append("## Academic Literature\n" + acad_ctx)
    if web_ctx:
        context_parts.append("## Web Sources\n" + web_ctx)
    context = "\n\n".join(context_parts) or "(No sources provided — draw from general knowledge.)"

    system = (
        "You are an expert research analyst. Write a comprehensive, well-structured "
        "Markdown research report grounded in the provided sources. "
        "Use this exact section structure:\n\n"
        "# [Descriptive Report Title]\n"
        "## Executive Summary\n"
        "## Background and Context\n"
        "## Key Findings\n"
        "## Analysis and Discussion\n"
        "## Conclusions\n"
        "## Limitations\n\n"
        "Rules:\n"
        "- Cite sources inline as [Source N] or [Paper N] where relevant.\n"
        "- Under Key Findings, use a numbered list (1. finding text).\n"
        "- Be analytical and critical, not just descriptive.\n"
        "- Do not fabricate information not present in the sources."
    )

    try:
        report = _invoke(_llm(state, temperature=0.3, num_predict=3000),
                         system, f"Research goal: {goal}\n\n{context}")
        state["report"] = report

        findings: list[str] = []
        in_findings = False
        for line in report.splitlines():
            if "## Key Findings" in line:
                in_findings = True
                continue
            if in_findings:
                if line.startswith("## "):
                    break
                m = re.match(r"^\s*\d+\.\s+(.+)", line)
                if m:
                    findings.append(m.group(1).strip())
        state["key_findings"] = findings[:10]
    except Exception as e:
        logger.error("[Research Report] Report generation failed: %s", e)
        state["report"] = f"*Report generation failed: {e}*"
        state.setdefault("errors", []).append(str(e))
    return state


def _step_research_eval(state: dict) -> dict:
    report = state.get("report", "")
    if not report or report.startswith("*Report generation failed"):
        state["eval_result"] = {}
        return state

    system = (
        "Score this research report on five dimensions (1–5 scale). "
        "Reply with ONLY a JSON object — no extra text:\n"
        '{"relevance": N, "depth": N, "clarity": N, "evidence": N, '
        '"overall": N, "summary": "one sentence"}'
    )
    try:
        raw = _invoke(_llm(state, temperature=0.0, num_predict=200),
                      system, f"Goal: {state['goal']}\n\nReport:\n{report[:2000]}")
        m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if m:
            state["eval_result"] = json.loads(m.group())
    except Exception as e:
        logger.warning("[Research Report] Eval failed: %s", e)
        state["eval_result"] = {}
    return state


# ── Public entry point ──────────────────────────────────────────────────────────

def run_research(
    state: dict,
    stream_callback: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """Run the research report workflow and return the final state.

    Calls ``stream_callback(node_name, state)`` after each step so the UI
    can update a progress bar.
    """
    _steps = [
        ("document_ingestion",    15, _step_document_ingestion),
        ("query_generation",      30, _step_query_generation),
        ("academic_search",       50, _step_academic_search),
        ("web_search",            60, _step_web_search),
        ("document_analysis",     75, _step_document_analysis),
        ("reference_compilation", 82, _step_reference_compilation),
        ("report_generation",     95, _step_report_generation),
        ("research_eval",        100, _step_research_eval),
    ]

    for node_name, pct, fn in _steps:
        try:
            state = fn(state)
        except Exception as exc:
            logger.exception("[Research Report] Step '%s' failed", node_name)
            state.setdefault("errors", []).append(f"{node_name}: {exc}")

        state["progress_pct"] = pct
        if stream_callback:
            try:
                stream_callback(node_name, state)
            except Exception:
                pass

    return state
