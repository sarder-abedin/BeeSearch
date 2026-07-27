"""
agents/story_nodes.py
──────────────────────
The six nodes that form the Research Partner (Storytelling) workflow.

  START → context_loader → repetition_tracker → source_router → storyteller
        → concept_visualizer → memory_saver → END

Node responsibilities
─────────────────────
  context_loader      : Load conversation history + document context from memory
  repetition_tracker  : Detect a repeated/rephrased question and pick a different style
  source_router       : LLM scores doc coverage (0-10); fetches online results if < 6
  storyteller         : Generate an explanation in the requested style + follow-up questions
  concept_visualizer  : On a detected repeat, render an interactive concept map
  memory_saver        : Persist the new user + assistant turns back to memory

TUTORIAL NOTE — Temperature choice
────────────────────────────────────
Research nodes use temperature=0.3 for factual precision.
The storyteller uses temperature=0.7 because creative explanations
(analogies, walkthroughs, debates) benefit from more varied language.
Too low → dry recitation; too high → hallucinations. 0.7 is the sweet spot
for science communication.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.story_memory import StorytellerMemory
from agents.story_state import StoryState
from config.settings import get_settings
from tools.temperature_levels import DEFAULT_TEMPERATURE_LEVEL, apply_temperature_level
from tools.text_parsing import extract_suggested_questions, format_page_label
from tools.writing_style import ANTI_AI_TELL_NARRATIVE_INSTRUCTION

logger = logging.getLogger(__name__)
cfg = get_settings()

# Lazy singleton — not created at import time so tests can inject a different instance.
_memory: StorytellerMemory | None = None


def _get_memory() -> StorytellerMemory:
    """Return the module-level lazy StorytellerMemory singleton, creating it on first use."""
    global _memory
    if _memory is None:
        _memory = StorytellerMemory()
    return _memory


def _llm(state: StoryState, temperature: float = 0.7) -> ChatOllama:
    """Build a ChatOllama client whose temperature is adjusted by the user's response-tuning level."""
    import httpx
    from config.observability import get_langfuse_callbacks
    level = state.get("temperature_level", DEFAULT_TEMPERATURE_LEVEL)
    return ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=apply_temperature_level(temperature, level),
        num_predict=4096,
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(180.0)},
        callbacks=get_langfuse_callbacks(),
    )


def _call(llm: ChatOllama, system: str, human: str) -> str:
    """Invoke the LLM with a system/human message pair and return the stripped text content."""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return response.content.strip()


def _clarification_context(state: StoryState) -> str:
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
    return "\n\nUSER CONTEXT (tailor your explanations accordingly):\n" + "\n".join(lines)


# ── Node 1: Context Loader ─────────────────────────────────────────────────────

def context_loader_node(state: StoryState) -> Dict[str, Any]:
    """
    Load conversation history and document context from the JSON memory file.

    Keeps the last 8 turns (≤ ~4,000 chars) to stay within context limits
    without truncating the most recent exchanges.

    Note: This node deliberately does NOT use ChromaDB/VectorStoreManager.
    Document context is stored as plain text in the session JSON to avoid
    contaminating the shared 'research_docs' vector collection.
    """
    logger.info("[Story Node 1] Context Loader")
    session_id = state.get("session_id", "")

    if not session_id:
        return {
            "conversation_history": [],
            "document_context": "",
            "document_names": [],
            "concepts_covered": [],
            "current_step": "context_loader",
            "completed_steps": state.get("completed_steps", []) + ["context_loader"],
            "progress_pct": 20,
        }

    session = _get_memory().load(session_id)
    if not session:
        return {
            "conversation_history": [],
            "document_context": "",
            "document_names": [],
            "concepts_covered": [],
            "errors": state.get("errors", []) + [f"Session {session_id} not found."],
            "current_step": "context_loader",
            "completed_steps": state.get("completed_steps", []) + ["context_loader"],
            "progress_pct": 20,
        }

    # Limit history to last 8 turns and cap total chars at 4,000
    conversation = session.get("conversation", [])
    recent = conversation[-8:]
    total_chars = 0
    truncated = []
    for turn in recent:
        content_len = len(turn.get("content", ""))
        if total_chars + content_len > 4000:
            break
        truncated.append(turn)
        total_chars += content_len

    return {
        "conversation_history": truncated,
        "document_context": session.get("document_context", ""),
        "document_names": session.get("document_names", []),
        "concepts_covered": session.get("concepts_covered", []),
        "topic": session.get("topic", state.get("topic", "")),
        "current_step": "context_loader",
        "completed_steps": state.get("completed_steps", []) + ["context_loader"],
        "progress_pct": 15,
    }


# ── Node 2: Repetition Tracker ────────────────────────────────────────────────

# Framing words ("what", "how", "explain") get dropped before comparing two
# questions — keeping them would make almost any two questions look similar,
# since framing words recur constantly while the topic words ("backpropagation",
# "gradient") are what actually makes two questions the same underlying ask.
_SIMILARITY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "how", "why", "when", "where", "who", "which",
    "does", "do", "did", "doesn't", "dont", "don't",
    "can", "could", "would", "should", "will",
    "you", "i", "me", "my", "we", "us", "it", "its",
    "this", "that", "these", "those", "of", "to", "in", "on", "for", "with",
    "and", "or", "but", "so", "please", "again",
}

_QUESTION_SIMILARITY_THRESHOLD = 0.4  # Fraction of shared meaningful tokens (Jaccard)

_CONFUSION_PHRASES = (
    "i don't understand", "i dont understand", "i'm confused", "im confused",
    "doesn't make sense", "doesnt make sense", "don't get it", "dont get it",
    "what do you mean", "still confused", "still don't", "still dont",
    "i'm lost", "im lost", "still lost", "not clicking", "doesn't click", "can you clarify",
    "could you clarify", "explain differently", "explain it differently",
    "simpler terms", "in other words", "rephrase", "explain again",
    "say that again", "not clear", "unclear",
)

# Rotation the storyteller cycles through on a detected repeat. Reuses the
# user-facing styles from _STYLE_DESCRIPTIONS below — "explain it differently"
# then needs no new prompt vocabulary, just a different entry from this list.
_STYLE_ROTATION = ["simple", "analogy", "walkthrough", "debate"]


def _tokenize_for_similarity(text: str) -> set:
    """Lowercase, alphanumeric-only tokens with framing stopwords removed."""
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _SIMILARITY_STOPWORDS and len(w) > 1
    }


def _question_similarity(a: str, b: str) -> float:
    """Jaccard overlap of meaningful tokens between two questions.

    Word-overlap is more forgiving of reordering and rephrasing than a raw
    character/sequence comparison — two paraphrases of the same question
    often share few consecutive characters but the same handful of topic words.
    """
    tokens_a, tokens_b = _tokenize_for_similarity(a), _tokenize_for_similarity(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _is_confusion_phrase(message: str) -> bool:
    """Whether the message explicitly signals the user is still confused."""
    lowered = message.strip().lower()
    return any(phrase in lowered for phrase in _CONFUSION_PHRASES)


def _similar_to_recent_question(message: str, history: List[Dict]) -> str:
    """Return the most recent prior user question similar enough to *message*
    to count as a repeat/rephrase, or "" if none matches."""
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        prior = turn.get("content", "")
        if prior and _question_similarity(message, prior) >= _QUESTION_SIMILARITY_THRESHOLD:
            return prior
    return ""


def _next_explanation_strategy(requested: str, last_used: str) -> str:
    """Pick a style different from last_used.

    Prefers the user's currently selected style unless that's exactly the one
    that already failed to land, in which case it rotates to the next style
    in _STYLE_ROTATION.
    """
    if requested != last_used:
        return requested
    idx = _STYLE_ROTATION.index(last_used) if last_used in _STYLE_ROTATION else -1
    return _STYLE_ROTATION[(idx + 1) % len(_STYLE_ROTATION)]


def repetition_tracker_node(state: StoryState) -> Dict[str, Any]:
    """
    Detect whether this question repeats or re-asks for clarification on
    something already answered this session, and if so override
    explanation_style to something different from the immediately preceding answer.

    "Explain it differently" only works if the style actually changes — without
    this override, storyteller_node would run again with the same user-selected
    style and likely reproduce an explanation similar to the one that already
    didn't land. Detection requires at least one prior assistant turn: confusion
    language or topical overlap on a user's very first message in a session has
    nothing to be "a repeat of" yet.
    """
    logger.info("[Story Node 2] Repetition Tracker")
    message = state.get("user_message", "")
    history = state.get("conversation_history", [])

    has_prior_assistant_turn = any(t.get("role") == "assistant" for t in history)
    matched_prior = _similar_to_recent_question(message, history) if has_prior_assistant_turn else ""
    is_repeat = has_prior_assistant_turn and (bool(matched_prior) or _is_confusion_phrase(message))

    effective_style = state.get("explanation_style", "simple")
    if is_repeat:
        last_assistant_style = ""
        for turn in reversed(history):
            if turn.get("role") == "assistant":
                last_assistant_style = turn.get("explanation_style") or ""
                break
        # If we don't know what style was used last time, we can't guarantee
        # a different one — keep the user's current selection rather than guess.
        if last_assistant_style:
            effective_style = _next_explanation_strategy(effective_style, last_assistant_style)
        logger.info("  Repeat clarification detected — style: %s", effective_style)

    return {
        "is_repeat_clarification": is_repeat,
        "repeated_question": matched_prior,
        "explanation_style": effective_style,
        "current_step": "repetition_tracker",
        "completed_steps": state.get("completed_steps", []) + ["repetition_tracker"],
        "progress_pct": 25,
    }


# ── Node 3: Source Router ─────────────────────────────────────────────────────

_COVERAGE_THRESHOLD = 6  # Score below this triggers online search (0–10 scale)


def _build_web_query(question: str, doc_context: str, model_name: str, num_ctx: int) -> str:
    """Rewrite a conversational question into a self-contained search query.

    Questions like "What's the state of the art compared to this approach?"
    mean nothing to a web/academic search engine — it has no idea what "this
    approach" refers to, so the search comes back irrelevant even though it
    technically succeeds. Grounding the rewrite in a snippet of the uploaded
    document context lets the LLM name the real topic before the query
    reaches DuckDuckGo/arXiv/Semantic Scholar. Falls back to the raw question
    when there's no document context to ground it in, or if the rewrite call
    itself fails.
    """
    if not doc_context.strip():
        return question
    try:
        import httpx
        from config.observability import get_langfuse_callbacks
        llm = ChatOllama(
            model=model_name or cfg.ollama_model,
            base_url=cfg.ollama_base_url,
            temperature=0.0,
            num_predict=40,
            num_ctx=min(num_ctx, 4096),
            sync_client_kwargs={"timeout": httpx.Timeout(30.0)},
            callbacks=get_langfuse_callbacks(),
        )
        system = (
            "Rewrite the question as a short, self-contained search query "
            "(max 12 words). Replace vague references like 'this work'/'this "
            "approach'/'this problem' with the actual topic, inferred from the "
            "context. Output ONLY the query — no quotes, no explanation."
        )
        human = f"CONTEXT:\n{doc_context[:300]}\n\nQUESTION: {question}"
        rewritten = _call(llm, system, human)
        return rewritten.strip("'\"“”‘’") or question
    except Exception as e:
        logger.warning("Web query rewrite failed in router, using raw question: %s", e)
        return question


def source_router_node(state: StoryState) -> Dict[str, Any]:
    """
    Assess how well the uploaded documents cover the user's question.

    Uses a fast LLM call (temperature=0, small token budget) to score coverage
    0-10.  If the score is below _COVERAGE_THRESHOLD, runs an academic search
    (arXiv + Semantic Scholar + Google Scholar) and a web search (DuckDuckGo)
    and stores the results in state for the storyteller to cite.
    """
    logger.info("[Story Node 2] Source Router")

    question = state.get("user_message", "")
    doc_context = state.get("document_context", "")

    # No documents at all — skip LLM assessment and go straight to online search
    if not doc_context.strip():
        coverage_score = 0
        reason = "No documents uploaded — searching online for context."
    else:
        import httpx
        from config.observability import get_langfuse_callbacks
        router_llm = ChatOllama(
            model=state.get("model_name", cfg.ollama_model),
            base_url=cfg.ollama_base_url,
            temperature=0.0,
            num_predict=128,
            num_ctx=min(state.get("num_ctx", cfg.num_ctx), 4096),
            sync_client_kwargs={"timeout": httpx.Timeout(60.0)},
            callbacks=get_langfuse_callbacks(),
        )
        system = (
            "You are a document coverage assessor. Score how well the document context "
            "covers the question. Return ONLY valid JSON: "
            '{"score": <0-10>, "reason": "<one sentence>"}\n\n'
            "Scoring guide:\n"
            "0-3: context has almost nothing relevant\n"
            "4-5: partial/tangential — online search would significantly help\n"
            "6-7: covers the topic reasonably well\n"
            "8-10: directly and thoroughly answers the question"
        )
        human = (
            f"QUESTION: {question}\n\n"
            f"DOCUMENT CONTEXT (first 1500 chars):\n{doc_context[:1500]}\n\n"
            "Score how well the document context covers this question. Return only JSON."
        )
        try:
            raw = _call(router_llm, system, human)
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                coverage_score = max(0, min(10, int(parsed.get("score", 7))))
                reason = parsed.get("reason", "")
            else:
                coverage_score = 7
                reason = "Coverage assessment inconclusive — defaulting to document-only."
        except Exception as e:
            logger.warning("Source router LLM call failed: %s", e)
            coverage_score = 7
            reason = f"Coverage assessment failed ({e}) — defaulting to document-only."

    online_results: List[Dict[str, Any]] = []
    sources_searched: List[str] = []

    if coverage_score < _COVERAGE_THRESHOLD:
        logger.info("  Coverage score %d/10 — triggering online search", coverage_score)

        from tools.search_tools import AcademicSearcher, WebSearcher

        search_query = _build_web_query(
            question, doc_context, state.get("model_name", cfg.ollama_model), state.get("num_ctx", cfg.num_ctx)
        )

        # Academic search (arXiv + Semantic Scholar + Google Scholar)
        try:
            papers = AcademicSearcher().search(search_query, max_per_source=3)[:5]
            for p in papers:
                if p.title:
                    online_results.append({
                        "type": "academic",
                        "title": p.title,
                        "authors": p.citation_key,
                        "url": p.url or (f"https://doi.org/{p.doi}" if p.doi else ""),
                        "snippet": (p.abstract or "")[:400],
                        "source": p.source,
                        "year": p.year,
                        "apa": p.to_apa(),
                    })
            if papers:
                sources_searched.append("academic")
        except Exception as e:
            logger.warning("Academic search failed in router: %s", e)

        # Web search (DuckDuckGo — white papers, blogs, tutorials, etc.)
        try:
            web_hits = WebSearcher().search(search_query, max_results=4)
            for w in web_hits:
                if w.url and w.title:
                    online_results.append({
                        "type": "web",
                        "title": w.title,
                        "authors": "",
                        "url": w.url,
                        "snippet": w.snippet,
                        "source": "web",
                        "year": None,
                        "apa": f"{w.title}. Retrieved from {w.url}",
                    })
            if web_hits:
                sources_searched.append("web")
        except Exception as e:
            logger.warning("Web search failed in router: %s", e)

    source_decision = {
        "coverage_score": coverage_score,
        "used_docs": bool(doc_context.strip()),
        "used_online": len(online_results) > 0,
        # True whenever low coverage triggered a search — independent of whether it
        # found anything. Without this, "search ran and found nothing" is
        # indistinguishable from "coverage was fine, search never ran" downstream.
        "search_attempted": coverage_score < _COVERAGE_THRESHOLD,
        "reason": reason,
        "sources_searched": sources_searched,
        "online_count": len(online_results),
    }

    logger.info(
        "  Source decision: score=%d, online=%s (%d results)",
        coverage_score, source_decision["used_online"], len(online_results),
    )

    return {
        "online_results": online_results,
        "source_decision": source_decision,
        "current_step": "source_router",
        "completed_steps": state.get("completed_steps", []) + ["source_router"],
        "progress_pct": 40,
    }


# ── Node 4: Storyteller ────────────────────────────────────────────────────────

_STYLE_DESCRIPTIONS = {
    "simple": (
        "Explain the concept as simply as possible. Use everyday language. "
        "Assume the reader knows nothing about this topic. Build up from basic "
        "first principles. Use short sentences."
    ),
    "analogy": (
        "Use one extended analogy or metaphor throughout your entire response. "
        "Pick something familiar (cooking, sports, music, architecture) and "
        "consistently map every aspect of the concept onto it. Don't mix analogies."
    ),
    "walkthrough": (
        "Give a numbered step-by-step walkthrough. Walk the reader through the "
        "concept as if guiding them through a process in real time. "
        "Number each step clearly. Show what happens at each stage."
    ),
    "debate": (
        "Present both the strongest arguments FOR and AGAINST this concept, "
        "approach, or claim. Structure it as a clear for-and-against debate. "
        "Then give your balanced assessment."
    ),
}

_LEVEL_DESCRIPTIONS = {
    "novice": (
        "Write for someone with NO background in this field. Define every "
        "technical term the moment you introduce it, in everyday words. Lean on "
        "familiar real-world comparisons. Favor the big picture and why it "
        "matters over mechanism details."
    ),
    "intermediate": (
        "Write for someone with general science/research literacy (e.g. an "
        "undergraduate or informed generalist). Standard field terminology is "
        "fine, but briefly gloss any less-common terms. Go one layer deeper "
        "into mechanisms and nuance than you would for a complete beginner."
    ),
    "expert": (
        "Write for a researcher or practitioner already familiar with this "
        "field. Use precise technical and disciplinary terminology without "
        "hand-holding definitions. Emphasize methodological nuance, caveats, "
        "open questions, and connections to the broader literature."
    ),
}


# ── Citation grounding for document excerpts ─────────────────────────────────
# Mirrors notebook_advanced.py's _build_numbered_excerpts / _build_references_section
# / _strip_llm_references_section for Literature Review. Duplicated rather than
# imported — each pipeline keeps its own helpers (see CLAUDE.md's per-pipeline
# isolation convention).

_DOC_EXCERPT_BLOCK_RE = re.compile(
    r"\[(\d+)\]\s*\(source:\s*([^,]+?),\s*p\.\s*(\d+)\)(?:\s*\[[^\]]+\])?\n(.*?)(?=\n\n\[\d+\]\s*\(source:|\Z)",
    re.DOTALL,
)


def build_numbered_doc_context(
    notebook: Dict[str, Any],
    max_chars: int = 2000,
    max_chars_per_chunk: int = 500,
) -> str:
    """
    Build the Explain tab's document_context as numbered, page-tagged excerpts
    — one tag per chunk, e.g. "[1] (source: paper.pdf, p. 1)" — instead of an
    untagged blob of joined snippets.

    Without per-chunk tags the storyteller has nothing concrete to cite when it
    draws on the uploaded documents — the same gap that let Literature
    Review's inline citations drift from its References list. The tags are
    baked directly into the string StorytellerMemory persists as
    document_context, rather than stored as a separate field, so
    _parse_doc_excerpts can recover them again on every later turn without a
    story-session schema change.
    """
    sources = notebook.get("sources", [])
    chunks = notebook.get("chunks", [])

    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chunks:
        by_doc.setdefault(ch["doc_id"], []).append(ch)

    lines: List[str] = []
    total_chars = 0
    for src in sources:
        if total_chars >= max_chars:
            break
        doc_chunks = sorted(by_doc.get(src["doc_id"], []), key=lambda c: c.get("chunk_index", 0))
        for ch in doc_chunks:
            content_type = ch.get("content_type", "text")
            if content_type == "table" and ch.get("table_md"):
                type_tag = " [TABLE]"
                text = ch["table_md"].strip()[:max_chars_per_chunk]
            elif content_type == "figure":
                type_tag = " [FIGURE]"
                text = ch.get("text", "").strip()[:max_chars_per_chunk]
            else:
                type_tag = ""
                text = ch.get("text", "").strip()[:max_chars_per_chunk]
            if not text:
                continue
            if total_chars + len(text) > max_chars:
                break
            n = len(lines) + 1
            page_label = format_page_label(ch.get("page_num"))
            doc_name = ch.get("doc_name") or src.get("filename", "unknown")
            lines.append(f"[{n}] (source: {doc_name}, {page_label}){type_tag}\n{text}")
            total_chars += len(text)

    return "\n\n".join(lines)


def _parse_doc_excerpts(document_context: str) -> Dict[int, Dict[str, Any]]:
    """
    Recover the {excerpt_number: {"doc_name", "page_num", "snippet"}} map
    baked into document_context by build_numbered_doc_context, so the
    citations list can be rebuilt from whichever numbers the storyteller
    actually cited.

    page_num is recovered as the raw 0-based value: the tag itself shows the
    1-based page a user would see in their PDF (via format_page_label), so
    the displayed digit is converted back by -1 here — matching the rest of
    the codebase's "raw stays raw until final display" convention and
    feeding the same raw value the PDF jump-navigation button needs.

    Sessions created before this tagging existed have no matching tags —
    parsing returns an empty map and the storyteller falls back to its
    pre-existing untagged behavior rather than crashing.
    """
    excerpts: Dict[int, Dict[str, Any]] = {}
    for m in _DOC_EXCERPT_BLOCK_RE.finditer(document_context):
        excerpts[int(m.group(1))] = {
            "doc_name": m.group(2).strip(),
            "page_num": int(m.group(3)) - 1,
            "snippet": m.group(4).strip(),
        }
    return excerpts


_REF_HEADING_RE = re.compile(
    r"\n+(?:#{1,4}\s*References\b.*|\*\*References\*\*:?.*)",
    re.IGNORECASE | re.DOTALL,
)


def _strip_llm_references_section(body: str) -> str:
    """Cut off any References section the storyteller wrote on its own.

    The system prompt says not to (see storyteller_node) — this is a
    defensive backstop, identical in spirit to notebook_advanced.py's
    same-named helper for Literature Review.
    """
    match = _REF_HEADING_RE.search(body)
    return body[: match.start()].rstrip() if match else body.rstrip()


def _build_citations_list(
    body: str,
    doc_excerpts: Dict[int, Dict[str, Any]],
    online_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Rebuild the citations list from whichever [n] document-excerpt numbers
    and [Source n] online-result numbers were actually cited in *body* —
    instead of trusting the model's own self-written list.

    Structured output (matching ui/tabs/notebook.py::_render_citations'
    shared {"n", "doc_name", "page", "snippet", "url"} shape) for the
    snippet-expander UI, mirroring notebook_advanced.py's
    _build_references_list for Literature Review — but as data the caller
    renders/persists separately, rather than text baked into the body.
    Document citations get an int "n" matching their inline [n] marker;
    online citations get a "Source N" string "n" so the two independent
    numbering schemes never collide when shown in one combined list.

    Returns [] when nothing was cited. Unlike Literature Review (which is
    always about its sources), a conversational Explain-tab turn may
    legitimately cite nothing — e.g. a follow-up answered from general
    knowledge — so an empty list is correct here rather than falling back to
    a source list.
    """
    citations: List[Dict[str, Any]] = []
    for n in sorted({int(x) for x in re.findall(r"\[(\d+)\]", body)}):
        ex = doc_excerpts.get(n)
        if ex:
            citations.append({
                "n": n,
                "doc_name": ex.get("doc_name", "unknown"),
                "page": ex.get("page_num"),
                "snippet": ex.get("snippet", ""),
            })
    for n in sorted({int(x) for x in re.findall(r"\[Source (\d+)\]", body)}):
        if 1 <= n <= len(online_results):
            r = online_results[n - 1]
            citations.append({
                "n": f"Source {n}",
                "doc_name": r.get("title", "unknown"),
                "snippet": r.get("snippet", ""),
                "url": r.get("url", ""),
            })
    return citations


def storyteller_node(state: StoryState) -> Dict[str, Any]:
    """
    Generate a research explanation in the requested style.

    The response ends with a JSON block containing 3 suggested follow-up questions.
    A second micro LLM call extracts concept names newly introduced in this turn.
    """
    logger.info("[Story Node 2] Storyteller")
    llm = _llm(state, temperature=0.7)

    style = state.get("explanation_style", "simple")
    style_instruction = _STYLE_DESCRIPTIONS.get(style, _STYLE_DESCRIPTIONS["simple"])

    level = state.get("explanation_level", "intermediate")
    level_instruction = _LEVEL_DESCRIPTIONS.get(level, _LEVEL_DESCRIPTIONS["intermediate"])

    topic = state.get("topic", "the research topic")
    concepts_covered = state.get("concepts_covered", [])
    doc_context = state.get("document_context", "")
    doc_excerpts = _parse_doc_excerpts(doc_context)
    history = state.get("conversation_history", [])

    # Format conversation history for the prompt
    history_block = ""
    if history:
        history_lines = []
        for turn in history:
            role = "User" if turn["role"] == "user" else "Research Partner"
            # Truncate long turns in the history for prompt efficiency
            content = turn.get("content", "")[:600]
            history_lines.append(f"{role}: {content}")
        history_block = "\n\nPREVIOUS CONVERSATION:\n" + "\n\n".join(history_lines)

    # Document context block
    doc_block = ""
    if doc_context:
        if doc_excerpts:
            doc_block = (
                "\n\nDOCUMENT CONTEXT — numbered excerpts tagged with source filename "
                "and page. Cite inline with its number (e.g. [2]) whenever you draw on "
                f"one; never invent a number not listed here:\n{doc_context}"
            )
        else:
            doc_block = f"\n\nDOCUMENT CONTEXT (quote short passages when relevant):\n{doc_context}"

    # Online results block (when source router fetched supplementary material)
    online_results: List[Dict[str, Any]] = state.get("online_results", [])
    source_decision: Dict[str, Any] = state.get("source_decision", {})
    online_block = ""
    attribution_format = ""
    if online_results:
        lines = []
        for i, r in enumerate(online_results, 1):
            src_label = "Academic" if r.get("type") == "academic" else "Web"
            authors = f" — {r['authors']}" if r.get("authors") else ""
            year = f" ({r['year']})" if r.get("year") else ""
            lines.append(
                f"[Source {i}] [{src_label}] {r['title']}{authors}{year}\n"
                f"URL: {r.get('url', '')}\n"
                f"Excerpt: {r.get('snippet', '')[:350]}"
            )
        online_block = (
            "\n\nONLINE SOURCES (fetched because document coverage was insufficient):\n"
            + "\n\n".join(lines)
        )
        coverage_score = source_decision.get("coverage_score", 5)
        gap_reason = source_decision.get("reason", "the documents do not fully cover this topic")
        doc_citation_instruction = (
            "Cite document excerpts inline with their [n] number whenever you draw on one."
            if doc_excerpts else
            "Quote brief passages where useful."
        )
        attribution_format = f"""

CRITICAL — PER-SECTION SOURCE ATTRIBUTION REQUIRED:
The uploaded documents scored only {coverage_score}/10 coverage for this question ({gap_reason}).
You MUST structure your response using these exact labelled sections in this order:

**From your documents:**
Explain what the uploaded documents actually say about this question. {doc_citation_instruction}
If the documents say very little, keep this section short and honest — do not pad it.

**Why online search was needed:**
1–2 sentences only. What specific gap did the documents leave? What could not be answered from them alone?

**From online sources:**
Fill the gap using the online sources provided. Apply the chosen style and audience level here too.
Cite every claim from an online source with [Source N] placed immediately after it.
Only use sources that are genuinely relevant — skip irrelevant ones.

Do NOT write your own References section — one listing every [n] and [Source N] you
actually cited above is generated automatically. After your last content section,
end with the suggested_questions JSON and nothing else."""

    # Concepts already covered
    covered_block = ""
    if concepts_covered:
        covered_block = (
            f"\n\nCONCEPTS ALREADY EXPLAINED (do not re-explain from scratch): "
            f"{', '.join(concepts_covered[:20])}"
        )

    repeat_instruction = ""
    if state.get("is_repeat_clarification"):
        repeat_instruction = (
            "\n10. This question repeats or re-asks for clarification on something "
            "already discussed — the previous explanation did not land. Do NOT just "
            "reword the same explanation. Use a genuinely different angle (a "
            "different analogy, a concrete worked example, or a different entry "
            "point into the idea) and briefly acknowledge you're taking a different "
            "approach before diving in."
        )

    online_note = (
        " When online sources are provided, follow the attribution format below."
        if online_results else ""
    )
    system = f"""You are a Research Partner — a gifted science communicator who excels at making
complex research concepts genuinely understandable.{attribution_format}
{ANTI_AI_TELL_NARRATIVE_INSTRUCTION}
CORE RULES:
1. Never use unexplained jargon relative to the target audience below — define any term that audience wouldn't already know.
2. STYLE — {style_instruction}
3. AUDIENCE LEVEL — {level_instruction}
4. {"Follow the per-section attribution format above." if online_results else "Write 3–6 paragraphs only — no lengthy essays. Be concise and memorable."}
5. Build on the previous conversation — reference and connect to what was discussed before.
6. {"Each section should be written in the chosen style and at the chosen audience level." if online_results else ("Cite document excerpts inline with their [n] number whenever you draw on one; never invent a number not listed in DOCUMENT CONTEXT. Do not write your own References section — one is generated automatically from whichever numbers you cite." if doc_excerpts else "Quote short passages from the provided document context when they are directly relevant.")}
7. At the very end of your response, append EXACTLY this JSON (no other text after it):
   {{"suggested_questions": ["Question 1?", "Question 2?", "Question 3?"]}}
   The questions should be natural follow-ups a curious reader would want to ask next.
8. Do NOT start your response with "Certainly!" or "Of course!" or similar filler phrases.
9. The topic being explored is: {topic}{_clarification_context(state)}{repeat_instruction}"""

    human = f"""USER QUESTION: {state.get('user_message', '')}
{history_block}{doc_block}{online_block}{covered_block}

Respond in the "{style}" style, calibrated for a "{level}"-level audience.
Remember to end with the suggested_questions JSON."""

    try:
        raw_response = _call(llm, system, human)
    except Exception as e:
        logger.error("Storyteller LLM call failed: %s", e)
        return {
            "assistant_response": f"[Error generating response: {e}]",
            "suggested_questions": [],
            "new_concepts": [],
            "citations": [],
            "errors": state.get("errors", []) + [str(e)],
            "current_step": "storyteller",
            "completed_steps": state.get("completed_steps", []) + ["storyteller"],
            "progress_pct": 65,
        }

    # Parse the trailing suggested_questions block (handles markdown-bolded
    # keys, smart quotes, and trailing commas — see tools/text_parsing).
    main_response, suggested_questions = extract_suggested_questions(raw_response)

    if not suggested_questions:
        logger.warning("suggested_questions: no parseable block found — raw tail: %s",
                       raw_response[-120:])

    # Drop any References section the model wrote on its own (it's told not
    # to, but instructions aren't guarantees). A structured citations list is
    # rebuilt separately from whichever [n]/[Source N] numbers it actually
    # cited, for the snippet-expander UI — never trust the model's own list.
    main_response = _strip_llm_references_section(main_response)
    citations = _build_citations_list(main_response, doc_excerpts, online_results)

    # Second micro LLM call: extract newly explained concept names
    new_concepts: List[str] = []
    concepts_prompt = f"""From this explanation, list the names of scientific or technical concepts
that were EXPLAINED (not just mentioned). Return ONLY a JSON array of short concept names.
Example: ["attention mechanism", "softmax function"]

EXPLANATION:
{main_response[:1500]}"""

    try:
        import httpx
        from config.observability import get_langfuse_callbacks
        micro_llm = ChatOllama(
            model=state.get("model_name", cfg.ollama_model),
            base_url=cfg.ollama_base_url,
            temperature=0.1,
            num_predict=256,
            num_ctx=min(state.get("num_ctx", cfg.num_ctx), 4096),
            sync_client_kwargs={"timeout": httpx.Timeout(60.0)},
            callbacks=get_langfuse_callbacks(),
        )
        raw_concepts = _call(
            micro_llm,
            "You extract concept names from explanations. Return only valid JSON arrays.",
            concepts_prompt,
        )
        match = re.search(r"\[.*\]", raw_concepts, re.DOTALL)
        if match:
            new_concepts = json.loads(match.group(0))[:10]
    except Exception as e:
        logger.warning("Concept extraction micro-call failed (%s) — skipping", e)

    return {
        "assistant_response": main_response,
        "suggested_questions": suggested_questions,
        "new_concepts": new_concepts,
        "citations": citations,
        "current_step": "storyteller",
        "completed_steps": state.get("completed_steps", []) + ["storyteller"],
        "progress_pct": 65,
    }


# ── Node 5: Concept Visualizer ─────────────────────────────────────────────────
# Only runs its LLM extraction + Pyvis render when is_repeat_clarification is
# True — most turns skip it entirely, since the extraction call has a real cost
# and a second explanation of something the user already understood doesn't
# need a diagram. Any failure (LLM, JSON parse, or pyvis missing) is a safe
# no-op: it never blocks the primary explanation already produced upstream.

def _safe_label(text: Any, maxlen: int = 60) -> str:
    """Trim/clean an LLM-extracted label for safe use as a pyvis node label."""
    return re.sub(r"\s+", " ", str(text)).strip()[:maxlen]


def _extract_concept_graph_data(user_message: str, explanation_text: str, state: StoryState) -> Dict[str, Any]:
    """
    Ask the LLM for a small hub-and-spoke breakdown of the concept just
    explained: one central concept plus 3-6 directly related ideas.

    Uses a fixed low temperature via a directly-constructed ChatOllama rather
    than _llm()/apply_temperature_level — mirrors the same choice already made
    for source_router's coverage scoring and storyteller's concept-extraction
    micro-call: structured-JSON extraction needs reliable parsing, not
    response-tuning variety.
    """
    import httpx
    from config.observability import get_langfuse_callbacks
    llm = ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=0.2,
        num_predict=512,
        num_ctx=min(state.get("num_ctx", cfg.num_ctx), 4096),
        sync_client_kwargs={"timeout": httpx.Timeout(60.0)},
        callbacks=get_langfuse_callbacks(),
    )
    system = (
        "You are a concept-map extractor. From the explanation, identify the "
        "ONE central concept and 3-6 directly related ideas.\n"
        "Output ONLY valid JSON, no code fences:\n"
        '{"central": "Concept Name", "related": [\n'
        '  {"label": "Related idea", "relation": "short verb phrase"}\n'
        "]}\n\n"
        "Rules:\n"
        "- central: 2-5 words.\n"
        "- 3 to 6 related items, each label 2-5 words.\n"
        "- relation: a short verb phrase describing how it connects to the "
        "central concept (e.g. 'enables', 'depends on', 'is a type of').\n"
        "- Output ONLY the JSON object."
    )
    human = f"QUESTION: {user_message}\n\nEXPLANATION:\n{explanation_text[:1500]}"
    raw = _call(llm, system, human)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in concept-graph extraction response")
    return json.loads(match.group(0))


def _concept_graph_to_pyvis_html(data: Dict[str, Any]) -> str:
    """Convert the LLM's {"central", "related": [...]}  JSON into an
    interactive Pyvis HTML string — a hub-and-spoke map of one concept,
    deliberately simpler than notebook_advanced.py's general knowledge-graph
    extraction since Explain is about clarifying a single recurring question,
    not mapping a whole document's entities."""
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pip install pyvis")

    central = _safe_label(data.get("central") or "Concept")
    related = [item for item in (data.get("related") or [])[:6] if isinstance(item, dict)]

    net = Network(height="420px", width="100%", directed=True, bgcolor="#0F172A", font_color="white")
    net.barnes_hut(spring_length=140)
    net.add_node(central, label=central, color="#3B82F6", size=28, title="Central concept")

    seen = {central}
    for item in related:
        label = _safe_label(item.get("label", ""))
        if not label or label in seen:
            continue
        seen.add(label)
        relation = _safe_label(item.get("relation", ""), maxlen=40)
        net.add_node(label, label=label, color="#10B981", size=18, title=relation)
        net.add_edge(central, label, label=relation, color="#6B7280", arrows="to")

    return net.generate_html()


def concept_visualizer_node(state: StoryState) -> Dict[str, Any]:
    """
    On a detected repeat, render an interactive concept map as a second,
    visual modality of explanation — a different writing style alone may
    still not land, but a different modality (diagram vs. prose) often will.
    """
    logger.info("[Story Node 5] Concept Visualizer")
    base = {
        "current_step": "concept_visualizer",
        "completed_steps": state.get("completed_steps", []) + ["concept_visualizer"],
        "progress_pct": 80,
    }
    if not state.get("is_repeat_clarification"):
        return {**base, "concept_visual_html": ""}

    try:
        data = _extract_concept_graph_data(
            state.get("user_message", ""),
            state.get("assistant_response", ""),
            state,
        )
        html = _concept_graph_to_pyvis_html(data)
    except Exception as e:
        logger.warning("Concept visualization skipped (%s)", e)
        return {**base, "concept_visual_html": ""}

    return {**base, "concept_visual_html": html}


# ── Node 6: Memory Saver ───────────────────────────────────────────────────────

def memory_saver_node(state: StoryState) -> Dict[str, Any]:
    """
    Persist the current user + assistant turns to the session JSON file.

    Appends two turns: the user message and the assistant response.
    Also updates the list of concepts covered in this session.
    """
    logger.info("[Story Node 6] Memory Saver")
    session_id = state.get("session_id", "")

    if not session_id:
        return {
            "current_step": "memory_saver",
            "completed_steps": state.get("completed_steps", []) + ["memory_saver"],
            "progress_pct": 100,
        }

    # Save user turn
    _get_memory().add_turn(
        session_id,
        role="user",
        content=state.get("user_message", ""),
    )

    # Save assistant turn with suggested questions. explanation_style records
    # whichever style was actually used (possibly overridden by
    # repetition_tracker_node), so the next repeat can rotate to a different one.
    _get_memory().add_turn(
        session_id,
        role="assistant",
        content=state.get("assistant_response", ""),
        suggested_questions=state.get("suggested_questions", []),
        explanation_style=state.get("explanation_style", ""),
        citations=state.get("citations", []),
    )

    # Update concepts covered
    new_concepts = state.get("new_concepts", [])
    if new_concepts:
        _get_memory().add_concepts(session_id, new_concepts)

    logger.info(
        "  Memory saved for session %s — %d new concept(s)",
        session_id, len(new_concepts),
    )

    return {
        "current_step": "memory_saver",
        "completed_steps": state.get("completed_steps", []) + ["memory_saver"],
        "progress_pct": 100,
    }
