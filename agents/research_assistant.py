"""
agents/research_assistant.py
────────────────────────────────
AI Research Assistant — free-form question answering over PUBLISHED literature.

The standalone counterpart to the two structured pipelines: unlike the
Systematic Review (which needs a PICO question + inclusion/exclusion criteria
and runs a full PRISMA workflow) and the Research Notebook (which is scoped to
documents the user personally uploaded), this answers an arbitrary research
question by searching published literature in general — Google Scholar, arXiv,
Semantic Scholar (+ optional CrossRef) and the web — then grounds an LLM answer
in whatever it found. Think Elicit / Perplexity / Consensus, local-first.

Citation grounding follows the same pattern as the rest of BeeSearch
(`notebook_advanced._build_numbered_excerpts`, `story_nodes.build_numbered_doc_context`):
number every retrieved source with a real tag, bake the tags into the context
string handed to the LLM, then after generation regex-rebuild the citations
list in code from whichever [n] markers the model actually cited — never trust
the LLM's own self-written References section.

Stateless, like the SR pipeline: no SQLite persistence, no graph — a single
`run_research_assistant()` call does search → ground → answer → rebuild.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import get_settings
from tools.temperature_levels import DEFAULT_TEMPERATURE_LEVEL, apply_temperature_level

logger = logging.getLogger(__name__)
cfg = get_settings()

# Same defensive backstop pattern used by story_nodes / notebook_advanced: if the
# model writes its own References/Sources section despite being told not to, cut it.
_REF_HEADING_RE = re.compile(
    r"\n+(?:#{1,4}\s*(?:references|sources|citations|bibliography)\b.*"
    r"|\*\*(?:references|sources|citations|bibliography)\*\*:?.*)",
    re.IGNORECASE | re.DOTALL,
)


def _make_llm(settings: Dict[str, Any], temperature: float = 0.3, num_predict: int = 2048) -> ChatOllama:
    """Build a ChatOllama client whose temperature is adjusted by the user's response-tuning level.

    Mirrors the `_llm`/`_make_llm` factories in the other pipelines — reads
    ``model``/``num_ctx``/``temperature_level`` from the sidebar settings dict,
    falling back to config defaults.
    """
    import httpx
    from config.observability import get_langfuse_callbacks
    level = settings.get("temperature_level", DEFAULT_TEMPERATURE_LEVEL)
    return ChatOllama(
        model=settings.get("model") or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=apply_temperature_level(temperature, level),
        num_predict=num_predict,
        num_ctx=settings.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(300.0)},
        callbacks=get_langfuse_callbacks(),
    )


def _call(llm: ChatOllama, system: str, human: str) -> str:
    """Invoke the LLM with a system/human message pair and return the stripped text content."""
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content.strip()


def _g(obj: Any, attr: str, default: Any = None) -> Any:
    """Read ``attr`` from either a dataclass/object (getattr) or a dict (get).

    Lets the search/grounding helpers accept both `Paper`/`WebResult` dataclasses
    (as returned by tools.search_tools) and plain dicts (as used in tests) without
    branching at every field access.
    """
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ── Search ────────────────────────────────────────────────────────────────────

def search_literature(
    question: str,
    settings: Dict[str, Any],
    max_per_source: int = 4,
    include_web: bool = True,
    include_crossref: bool = True,
) -> Dict[str, List[Any]]:
    """Search published literature (+ optionally the web) for a free-form question.

    Returns ``{"papers": [...], "web_results": [...]}``. Each searcher fails soft:
    a search backend that errors is logged and contributes nothing rather than
    aborting the whole answer — the same resilience the SR literature_search node
    relies on.
    """
    papers: List[Any] = []
    web_results: List[Any] = []

    try:
        from tools.search_tools import AcademicSearcher
        papers = AcademicSearcher().search(
            question, max_per_source=max_per_source, include_crossref=include_crossref
        )
    except Exception as e:
        logger.warning("Research Assistant academic search failed: %s", e)

    if include_web:
        try:
            from tools.search_tools import WebSearcher
            web_results = WebSearcher().search(question, max_results=4)
        except Exception as e:
            logger.warning("Research Assistant web search failed: %s", e)

    return {"papers": papers, "web_results": web_results}


# ── Citation grounding ─────────────────────────────────────────────────────────

def build_numbered_sources(
    papers: List[Any],
    web_results: List[Any],
    max_chars: int = 8000,
    max_chars_per_source: int = 600,
) -> tuple[str, Dict[int, Dict[str, Any]]]:
    """Number academic papers then web results in ONE ``[n]`` namespace.

    Academic papers are numbered first (1..N) followed by web results, so a
    single inline ``[n]`` marker unambiguously identifies any source. Returns
    ``(context_str, source_map)`` where source_map maps each ``n`` to
    ``{kind, title, authors, year, url, snippet, apa}`` for later citation
    rebuilding and the UI's source-expander. Respects an overall character
    budget so a long source list can't blow the prompt.
    """
    lines: List[str] = []
    source_map: Dict[int, Dict[str, Any]] = {}
    total = 0
    n = 0

    for p in papers:
        snippet = (_g(p, "abstract", "") or "")[:max_chars_per_source]
        authors = _g(p, "authors", []) or []
        author_str = "; ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        year = _g(p, "year")
        title = _g(p, "title", "") or "Untitled"
        apa = p.to_apa() if hasattr(p, "to_apa") else f"{author_str} ({year or 'n.d.'}). {title}."
        entry = f"[{n + 1}] ({title}; {author_str or 'Unknown'}, {year or 'n.d.'}) {snippet}"
        if total + len(entry) > max_chars and source_map:
            break
        n += 1
        total += len(entry)
        lines.append(entry)
        source_map[n] = {
            "kind": "academic",
            "title": title,
            "authors": authors,
            "year": year,
            "url": _g(p, "url", "") or "",
            "snippet": snippet,
            "apa": apa,
            "source": _g(p, "source", "") or "",
        }

    for r in web_results:
        snippet = (_g(r, "snippet", "") or "")[:max_chars_per_source]
        title = _g(r, "title", "") or "Untitled page"
        url = _g(r, "url", "") or ""
        entry = f"[{n + 1}] ({title}; web) {snippet}"
        if total + len(entry) > max_chars and source_map:
            break
        n += 1
        total += len(entry)
        lines.append(entry)
        source_map[n] = {
            "kind": "web",
            "title": title,
            "authors": [],
            "year": None,
            "url": url,
            "snippet": snippet,
            "apa": f"{title}. Retrieved from {url}",
            "source": "web",
        }

    return "\n\n".join(lines), source_map


def _strip_llm_references_section(body: str) -> str:
    """Cut off any References/Sources section the model wrote on its own.

    The system prompt tells it not to (we rebuild citations in code), so this is
    a defensive backstop — identical in spirit to the same-named helpers in
    story_nodes.py and notebook_advanced.py.
    """
    match = _REF_HEADING_RE.search(body)
    return body[: match.start()].rstrip() if match else body.rstrip()


def build_citations(body: str, source_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rebuild the citations list from whichever ``[n]`` markers actually appear in *body*.

    Returns one dict per cited source (``{n, kind, title, authors, year, url,
    snippet, apa}``), in ascending citation order. A hallucinated number with no
    backing source is silently ignored. Returns ``[]`` when nothing was cited —
    a general-knowledge answer with no grounded sources is legitimate.
    """
    citations: List[Dict[str, Any]] = []
    for n in sorted({int(x) for x in re.findall(r"\[(\d+)\]", body)}):
        src = source_map.get(n)
        if not src:
            continue
        citations.append({"n": n, **src})
    return citations


# ── Orchestration ──────────────────────────────────────────────────────────────

def run_research_assistant(
    question: str,
    settings: Dict[str, Any],
    stream_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    include_web: bool = True,
) -> Dict[str, Any]:
    """Answer a free-form research question from published literature, with citations.

    Pipeline: search literature (+web) → number sources into a grounded context →
    LLM answer citing ``[n]`` → rebuild the citations list in code from the markers
    actually used. ``stream_callback(stage, info)`` is invoked (best-effort) at
    "searching"/"reading"/"answering"/"done" for UI/CLI progress.

    Returns a dict with: ``question``, ``answer`` (References section stripped),
    ``citations`` (code-rebuilt), ``sources`` (every numbered source, cited or
    not), ``academic_count``, ``web_count``, ``suggested_questions``, ``grounded``
    (False when no sources were found — the answer then carries an explicit
    "general knowledge" caveat and ``citations`` is empty).
    """
    def _emit(stage: str, **info: Any) -> None:
        if stream_callback:
            try:
                stream_callback(stage, info)
            except Exception as e:  # never let UI progress break the answer
                logger.debug("research assistant stream_callback error: %s", e)

    _emit("searching", question=question)
    found = search_literature(
        question, settings, include_web=include_web,
        include_crossref=settings.get("include_crossref", True),
    )
    papers = found["papers"]
    web_results = found["web_results"]
    _emit("reading", academic_count=len(papers), web_count=len(web_results))

    context, source_map = build_numbered_sources(papers, web_results)
    grounded = bool(source_map)

    llm = _make_llm(settings, temperature=0.3, num_predict=settings.get("num_predict", 2048))

    if grounded:
        system = (
            "You are a careful research assistant answering a question using ONLY the numbered "
            "sources provided. Write a clear, well-structured answer in formal prose.\n"
            "- Cite sources inline with their bracketed number, e.g. [1] or [2], immediately after "
            "each claim they support.\n"
            "- Only cite numbers that appear in the sources. Do not invent citations.\n"
            "- If the sources disagree, say so and cite both sides.\n"
            "- If the sources do not cover part of the question, say so plainly.\n"
            "- Do NOT write your own References, Sources, or Bibliography section — it is generated "
            "separately from the numbers you cite."
        )
        human = f"QUESTION: {question}\n\nNUMBERED SOURCES:\n{context}"
    else:
        system = (
            "You are a careful research assistant. No published sources could be retrieved for this "
            "question, so answer from general knowledge. Begin by stating clearly that the answer is "
            "not grounded in retrieved literature and should be verified. Do not fabricate citations "
            "or a References section."
        )
        human = f"QUESTION: {question}"

    _emit("answering", grounded=grounded)
    try:
        raw = _call(llm, system, human)
    except Exception as e:
        logger.warning("Research Assistant generation failed: %s", e)
        raw = f"The research assistant could not generate an answer ({e})."

    answer = _strip_llm_references_section(raw)
    citations = build_citations(answer, source_map) if grounded else []
    suggested = _suggest_followups(question, answer, llm)

    _emit("done", citations=len(citations))
    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "sources": [{"n": n, **src} for n, src in sorted(source_map.items())],
        "academic_count": len(papers),
        "web_count": len(web_results),
        "suggested_questions": suggested,
        "grounded": grounded,
    }


def _suggest_followups(question: str, answer: str, llm: ChatOllama) -> List[str]:
    """Best-effort: ask the LLM for 3 follow-up questions; return [] on any failure.

    Kept non-fatal (its own try/except) so a parsing hiccup never costs the user
    the primary answer already produced.
    """
    import json
    try:
        raw = _call(
            llm,
            'Return ONLY a JSON array of exactly 3 short follow-up research questions. No other text.',
            f"Original question: {question}\n\nAnswer given:\n{answer[:1200]}",
        )
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        items = json.loads(match.group(0)) if match else []
        return [str(q) for q in items][:3] if isinstance(items, list) else []
    except Exception:
        return []
