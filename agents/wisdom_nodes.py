"""
agents/wisdom_nodes.py
───────────────────────
Six nodes that form the Wisdom Mode (Mode 6) workflow.

Graph structure
───────────────
  START → context_loader → [route_from_context]
      ├── "clarification" → clarification_node → [route_after_clarification]
      │       ├── "knowledge_search" → knowledge_search_node
      │       │       → wisdom_synthesis_node
      │       │       → wisdom_validator_node
      │       │       → memory_saver_node → END
      │       └── "memory_saver"  → memory_saver_node → END   (still asking Qs)
      └── "wisdom_followup"       → wisdom_followup_node
                                  → memory_saver_node → END   (post-wisdom Q&A)

Design notes
────────────
• temperature=0.5 for synthesis (creative but grounded), 0.2 for validation
  (precision matters), 0.6 for clarification (conversational warmth)
• Related sessions are injected silently into synthesis — agent never names them
• Max 3 clarifying questions; after 3 the agent is forced to proceed
• Validation uses self-critique: LLM rates its own claims against the source list
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.wisdom_memory import WisdomMemory
from agents.wisdom_state import WisdomState
from config.settings import get_settings
from tools.search_tools import AcademicSearcher, WebSearcher

logger = logging.getLogger(__name__)
cfg = get_settings()

# Lazy singletons — not created at import time so tests can inject test instances.
_memory: WisdomMemory | None = None
_academic: AcademicSearcher | None = None
_web: WebSearcher | None = None


def _get_memory() -> WisdomMemory:
    global _memory
    if _memory is None:
        _memory = WisdomMemory()
    return _memory


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


def _llm(state: WisdomState, temperature: float = 0.5, num_predict: int = 4096) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(180.0)},
    )


def _call(llm: ChatOllama, system: str, human: str) -> str:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return response.content.strip()


def _format_history(history: List[Dict], max_chars: int = 3000) -> str:
    if not history:
        return ""
    lines = []
    total = 0
    for turn in history:
        role = "User" if turn["role"] == "user" else "Wisdom Oracle"
        content = turn.get("content", "")[:500]
        line = f"{role}: {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n\nCONVERSATION SO FAR:\n" + "\n\n".join(lines)


def _clarification_context(state: WisdomState) -> str:
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
    return "\n\nUSER CONTEXT (factor into your wisdom):\n" + "\n".join(lines)


# ── Routing functions ──────────────────────────────────────────────────────────

def route_from_context(state: WisdomState) -> str:
    """Route to follow-up if wisdom already generated, else to clarification."""
    return "wisdom_followup" if state.get("phase") == "done" else "clarification"


def route_after_clarification(state: WisdomState) -> str:
    """Route to knowledge search if agent decided it has enough context."""
    return "knowledge_search" if state.get("phase") == "ready_to_generate" else "memory_saver"


# ── Node 1: Context Loader ─────────────────────────────────────────────────────

def context_loader_node(state: WisdomState) -> Dict[str, Any]:
    """
    Load conversation history, document context, and passive cross-session
    knowledge from the JSON memory file.

    Related sessions are fetched by topic-tag overlap and silently passed to
    the synthesis node — the agent never names them explicitly in its output.
    """
    logger.info("[Wisdom Node 1] Context Loader")
    session_id = state.get("session_id", "")

    if not session_id:
        return {
            "conversation_history": [], "document_context": "",
            "document_names": [], "related_sessions": [],
            "phase": "clarifying", "clarification_count": 0,
            "current_step": "context_loader",
            "completed_steps": state.get("completed_steps", []) + ["context_loader"],
            "progress_pct": 10,
        }

    session = _get_memory().load(session_id)
    if not session:
        return {
            "conversation_history": [], "document_context": "",
            "document_names": [], "related_sessions": [],
            "phase": "clarifying", "clarification_count": 0,
            "errors": state.get("errors", []) + [f"Session {session_id} not found."],
            "current_step": "context_loader",
            "completed_steps": state.get("completed_steps", []) + ["context_loader"],
            "progress_pct": 10,
        }

    conversation = session.get("conversation", [])

    # Count assistant clarifying-question turns (for max-question enforcement)
    clarification_count = sum(
        1 for t in conversation
        if t.get("role") == "assistant"
        and t.get("is_question", False)
    )

    # Recent history (last 8 turns, cap at 3000 chars total)
    recent = conversation[-8:]

    # Passive cross-session context
    topic_tags = session.get("topic_tags", [])
    related: List[Dict] = []
    if topic_tags:
        related = _get_memory().find_related_sessions(topic_tags, session_id, limit=3)

    # Wisdom output fields — needed so wisdom_followup_node has full context
    wo = session.get("wisdom_output", {})
    val = wo.get("validation", {})

    return {
        "conversation_history": recent,
        "document_context": session.get("document_context", ""),
        "document_names": session.get("document_names", []),
        "phase": session.get("phase", "clarifying"),
        "clarification_count": clarification_count,
        "topic": session.get("topic", state.get("topic", "")),
        "scenario": session.get("scenario", state.get("scenario", "")),
        "topic_tags": topic_tags,
        "related_sessions": related,
        # Pre-load wisdom for follow-up context
        "deep_understanding": wo.get("deep_understanding", ""),
        "simple_explanation": wo.get("simple_explanation", ""),
        "actionable_takeaways": wo.get("actionable_takeaways", []),
        "wisdom_claims": val.get("claims", []),
        "devils_advocate": val.get("devils_advocate", ""),
        "overall_confidence": val.get("overall_confidence", ""),
        "current_step": "context_loader",
        "completed_steps": state.get("completed_steps", []) + ["context_loader"],
        "progress_pct": 10,
    }


# ── Node 2: Clarification ──────────────────────────────────────────────────────

def clarification_node(state: WisdomState) -> Dict[str, Any]:
    """
    Socratic clarification loop.

    Asks one focused question per turn to understand:
      - What specific outcome the user needs wisdom for
      - What they already know or have tried
      - Any constraints that matter (time, resources, etc.)

    After 3 questions (or earlier if context is sufficient), the node signals
    "ready_to_generate" — the graph then routes to knowledge_search.

    The transition is signalled by starting the response with PROCEED_TO_WISDOM.
    """
    logger.info("[Wisdom Node 2] Clarification")
    llm = _llm(state, temperature=0.6)

    clarification_count = state.get("clarification_count", 0)
    history = state.get("conversation_history", [])
    topic = state.get("topic", "")
    scenario = state.get("scenario", "")
    doc_context = state.get("document_context", "")

    history_block = _format_history(history)
    doc_block = f"\n\nDOCUMENT PROVIDED BY USER:\n{doc_context[:800]}" if doc_context else ""
    force_proceed = clarification_count >= 3

    system = f"""You are the Wisdom Oracle — a Socratic guide who transforms knowledge into wisdom.

CLARIFICATION PHASE:
Your job is to ask ONE focused question per turn to understand the user's situation.
Focus on uncovering: (1) the specific outcome or decision they need wisdom for,
(2) what they already know or tried, (3) any constraints that matter.

After at most 3 questions total you MUST proceed — even with partial context.
You have asked {clarification_count} question(s) so far.

WHEN READY TO PROCEED (or when forced after 3 questions):
- Start your response with exactly: PROCEED_TO_WISDOM
- Follow it with ONE sentence telling the user you're researching now.
- Example: "PROCEED_TO_WISDOM\nI have what I need — searching the scientific literature now..."

WHEN STILL CLARIFYING:
- Ask exactly ONE clear, focused question.
- Do NOT explain your reasoning or add preambles like "Great question!"
- Do NOT ask multiple things at once."""

    human = f"""TOPIC: {topic}
SCENARIO: {scenario}
{history_block}{doc_block}

USER'S LATEST MESSAGE: {state.get("user_message", "")}

{"IMPORTANT: You have asked the maximum number of questions. You MUST start your response with PROCEED_TO_WISDOM." if force_proceed else "Ask one focused question, or proceed if you have sufficient context."}"""

    try:
        response = _call(llm, system, human)
    except Exception as e:
        logger.error("Clarification LLM call failed: %s", e)
        return {
            "phase": "ready_to_generate",
            "assistant_response": "Let me search the scientific literature and craft your wisdom now...",
            "errors": state.get("errors", []) + [str(e)],
            "current_step": "clarification",
            "completed_steps": state.get("completed_steps", []) + ["clarification"],
            "progress_pct": 20,
        }

    if re.search(r"PROCEED_TO_WISDOM", response, re.IGNORECASE):
        transition = re.sub(
            r"PROCEED_TO_WISDOM\s*\n?", "", response, count=1, flags=re.IGNORECASE
        ).strip()
        if not transition:
            transition = (
                "I have enough context. Searching the scientific literature and crafting your wisdom now..."
            )
        return {
            "phase": "ready_to_generate",
            "assistant_response": transition,
            "current_step": "clarification",
            "completed_steps": state.get("completed_steps", []) + ["clarification"],
            "progress_pct": 25,
        }
    else:
        return {
            "phase": "clarifying",
            "assistant_response": response,
            "clarification_count": clarification_count + 1,
            "current_step": "clarification",
            "completed_steps": state.get("completed_steps", []) + ["clarification"],
            "progress_pct": 15 + clarification_count * 3,
        }


# ── Node 3: Knowledge Search ───────────────────────────────────────────────────

def knowledge_search_node(state: WisdomState) -> Dict[str, Any]:
    """
    Search arXiv + Semantic Scholar for scientific evidence.

    An LLM call first generates focused academic sub-queries from the
    conversation context. Google Search (FastAPI service) is also called for
    supplementary context; the service gracefully returns [] if not running.
    """
    logger.info("[Wisdom Node 3] Knowledge Search")
    llm = _llm(state, temperature=0.2)

    topic = state.get("topic", "")
    scenario = state.get("scenario", "")
    history = state.get("conversation_history", [])

    # Build a compact context summary for query generation
    context_lines = [f"Topic: {topic}", f"Scenario: {scenario[:300]}"]
    for turn in history[-4:]:
        content = turn.get("content", "")[:200]
        context_lines.append(f"{turn['role'].upper()}: {content}")
    context_summary = "\n".join(context_lines)

    # Generate academic search queries
    try:
        raw_queries = _call(
            llm,
            """You are an academic research librarian expert in evidence-based search strategy.
Generate 3–5 distinct search queries to find the best scientific evidence for a topic.

Rules:
- Cover different angles: underlying mechanisms, empirical studies, clinical evidence, theoretical models
- Use precise scientific terminology — no colloquial phrasing
- Vary specificity: include both broad (2-3 key terms) and narrow (4-6 specific terms) queries
- Include relevant synonyms across queries (not within a single query)
- Each query should retrieve DIFFERENT papers — avoid redundant overlapping queries
- Focus on empirical and peer-reviewed sources

Return ONLY a valid JSON array of strings. No explanation, no commentary.""",
            f"Generate queries to find scientific evidence for:\n{context_summary}",
        )
        match = re.search(r"\[.*\]", raw_queries, re.DOTALL)
        if match:
            queries = json.loads(match.group(0))
        else:
            logger.warning(
                "No JSON array in query generation response — using topic as fallback. "
                "Raw tail: %s", raw_queries[-80:],
            )
            queries = [topic]
        queries = queries[:5]
    except Exception as e:
        logger.warning("Query generation LLM call failed (%s) — using default queries", e)
        queries = [topic, f"{topic} scientific study", f"{topic} mechanism research"]

    # Academic search
    searcher = _get_academic()
    all_papers = []
    seen: set[str] = set()

    for query in queries:
        try:
            papers = searcher.search(query, max_per_source=4)
            for p in papers:
                key = re.sub(r"\W+", "", p.title.lower())[:60]
                if key not in seen:
                    seen.add(key)
                    all_papers.append(p)
        except Exception as e:
            logger.warning("Academic search failed for '%s': %s", query[:50], e)

    # Sort by citation count; use year as secondary key for papers without citation data
    all_papers.sort(key=lambda p: (
        0 if p.citation_count is not None else 1,
        -(p.citation_count or 0),
        -(p.year or 0),
    ))
    all_papers = all_papers[:15]

    # Grade papers for relevance before building context
    papers_before_grading = len(all_papers)
    if all_papers:
        from agents.self_reflective_rag import grade_papers
        paper_dicts = [{"title": p.title, "abstract": p.abstract or ""} for p in all_papers]
        grades = grade_papers(
            paper_dicts,
            query=state.get("user_message", "") or state.get("scenario", ""),
            model_name=state.get("model_name", cfg.ollama_model),
            num_ctx=state.get("num_ctx", cfg.num_ctx),
        )
        filtered = [p for p, g in zip(all_papers, grades) if g]
        all_papers = filtered or all_papers  # fallback: keep all if none pass

    papers_data = [
        {
            "title": p.title,
            "authors": p.authors[:3],
            "year": p.year,
            "abstract": p.abstract[:600],
            "url": p.url,
            "source": p.source,
            "citation_key": p.citation_key,
            "citation_count": p.citation_count,
        }
        for p in all_papers
    ]

    # Supplementary web context (passive — not cited, just for extra grounding)
    web_context: List[Dict] = []
    try:
        results = _get_web().search(f"{topic} scientific evidence", max_results=4)
        web_context = [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]
    except Exception:
        pass

    logger.info(
        "  Wisdom search: %d papers, %d web results for topic '%s'",
        len(papers_data), len(web_context), topic[:50],
    )

    return {
        "search_queries": queries,
        "academic_papers": papers_data,
        "web_context": web_context,
        "rag_reflection_info": {"papers_retrieved": papers_before_grading,
                                "papers_after_grading": len(all_papers)},
        "current_step": "knowledge_search",
        "completed_steps": state.get("completed_steps", []) + ["knowledge_search"],
        "progress_pct": 45,
    }


# ── Node 4: Wisdom Synthesis ───────────────────────────────────────────────────

def wisdom_synthesis_node(state: WisdomState) -> Dict[str, Any]:
    """
    Generate two complementary explanations + actionable takeaways.

    Related sessions from memory are silently injected as background context —
    the agent does NOT reference them by name in the output.
    """
    logger.info("[Wisdom Node 4] Wisdom Synthesis")
    llm = _llm(state, temperature=0.5)

    topic = state.get("topic", "")
    scenario = state.get("scenario", "")
    papers = state.get("academic_papers", [])
    doc_context = state.get("document_context", "")
    related_sessions = state.get("related_sessions", [])
    history = state.get("conversation_history", [])
    web_context = state.get("web_context", [])

    # Format academic knowledge
    paper_block = "\n\n".join(
        f"[{p['citation_key']}] {p['title']} ({p['year'] or 'n.d.'})\n"
        f"Authors: {'; '.join(p['authors'][:3])}\nCitations: {p.get('citation_count', '?')}\n"
        f"Abstract: {p['abstract']}"
        for p in papers[:10]
    )

    # Clarification context from conversation
    clarification_summary = "\n".join(
        f"Q: {h['content'][:200]}" if h["role"] == "assistant" else f"A: {h['content'][:200]}"
        for h in history[-6:]
    )

    # Passive cross-session context (silently enriches the prompt)
    passive_context = ""
    if related_sessions:
        snippets = []
        for s in related_sessions:
            if s.get("wisdom_snippet"):
                snippets.append(f"[Background] {s['wisdom_snippet'][:350]}")
            if s.get("actionable_snippet"):
                snippets.append(f"[Related actions] {s['actionable_snippet'][:200]}")
        if snippets:
            passive_context = "\n\nBACKGROUND KNOWLEDGE (use to enrich; do NOT mention source):\n" + "\n".join(snippets)

    doc_block = f"\n\nUSER'S DOCUMENT:\n{doc_context[:1500]}" if doc_context else ""

    web_block = ""
    if web_context:
        web_lines = [f"- {w['title']}: {w['snippet'][:200]}" for w in web_context[:3]]
        web_block = "\n\nADDITIONAL CONTEXT:\n" + "\n".join(web_lines)

    # ── Small LLMs cannot reliably embed 500-word essays inside JSON strings.
    # Split into three focused calls: scientific prose, simple prose, compact JSON.
    # This is the same pattern that fixed the identical bug in PRISMA synthesis.

    literature_context = f"""TOPIC: {topic}
SCENARIO: {scenario}
{_clarification_context(state)}
CLARIFICATION DIALOGUE:
{clarification_summary}

SCIENTIFIC LITERATURE:
{paper_block}
{doc_block}{web_block}{passive_context}"""

    # ── Call 1: Deep (scientific) explanation as plain text ───────────────
    deep_system = """You are a wise scientist explaining knowledge to a thoughtful colleague.
Write 6-8 paragraphs (800-1200 words) covering:
- The underlying mechanisms and WHY things work this way
- What the scientific evidence shows — cite papers as [citation_key]
- Key theoretical frameworks
- Where the evidence is strong, where it is mixed or uncertain
Rules: define every technical term inline; write flowing prose (no bullet points);
do NOT say 'In conclusion' or 'In summary'; ground every claim in the provided papers.
Be thorough and comprehensive. Each paragraph should develop its point fully with examples and evidence."""

    try:
        deep_understanding = _call(llm, deep_system, literature_context)
    except Exception as e:
        logger.error("Wisdom deep synthesis failed: %s", e)
        deep_understanding = f"[Scientific explanation could not be generated: {e}]"

    # ── Call 2: Simple explanation as plain text ───────────────────────────
    llm_simple = _llm(state, temperature=0.6)
    simple_system = """You explain science to anyone, regardless of background.
Write 4-6 paragraphs (500-700 words) covering the same core insight as the scientific explanation,
but translated into everyday language:
- Use one vivid concrete analogy (cooking, sport, building, nature) and stick with it
- No unexplained jargon — define every term in plain English
- Make it memorable and relatable
- Do NOT use the phrase 'In conclusion' or 'In summary'"""

    try:
        simple_explanation = _call(llm_simple, simple_system, literature_context)
    except Exception as e:
        logger.warning("Wisdom simple synthesis failed: %s", e)
        simple_explanation = deep_understanding  # fall back to the scientific version

    # ── Call 3: Compact structured fields as JSON ──────────────────────────
    llm_structured = _llm(state, temperature=0.1, num_predict=768)
    structured_system = (
        "You are a research assistant. Return ONLY a valid JSON object — no explanation, "
        "no preamble, no markdown fences. The object must have exactly two keys:\n"
        "  actionable_takeaways: array of 3-5 concrete action steps drawn from the research. "
        "Each step must start with a verb (e.g. 'Take', 'Reduce', 'Schedule') and be "
        "specific and grounded in the science.\n"
        "  topic_tags: array of 5-8 short concept tags (e.g. 'cortisol', 'sleep deprivation').\n\n"
        "Example of the required output shape (use real content, not these placeholders):\n"
        '{"actionable_takeaways":["Limit caffeine intake after 2 pm to protect sleep quality",'
        '"Schedule 20-minute naps before 3 pm to restore alertness"],'
        '"topic_tags":["sleep deprivation","adenosine","circadian rhythm"]}'
    )

    try:
        raw_structured = _call(
            llm_structured,
            structured_system,
            f"Topic: {topic}\nScenario: {scenario[:300]}\nKey papers: {paper_block[:600]}",
        )
        match = re.search(r"\{.*\}", raw_structured, re.DOTALL)
        structured = json.loads(match.group(0)) if match else {}
    except Exception as e:
        logger.warning("Wisdom structured fields failed: %s", e)
        structured = {}
        errors = list(state.get("errors") or []) + [f"Structured fields extraction failed: {e}"]
    else:
        errors = list(state.get("errors") or [])

    # Reject any items that look like un-replaced prompt placeholders
    # (model echoed the instruction text instead of generating real content).
    _placeholder_fragments = ("starting with a verb", "short concept tags", "each grounded")

    def _is_real(items: list) -> list:
        return [
            s for s in items
            if isinstance(s, str) and not any(f in s for f in _placeholder_fragments)
        ]

    return {
        "deep_understanding": deep_understanding,
        "simple_explanation": simple_explanation,
        "actionable_takeaways": _is_real(structured.get("actionable_takeaways", [])),
        "topic_tags": _is_real(structured.get("topic_tags", [])),
        "errors": errors,
        "current_step": "wisdom_synthesis",
        "completed_steps": state.get("completed_steps", []) + ["wisdom_synthesis"],
        "progress_pct": 70,
    }


# ── Node 5: Wisdom Validator ───────────────────────────────────────────────────

def wisdom_validator_node(state: WisdomState) -> Dict[str, Any]:
    """
    Self-critique node: rates the generated wisdom claims against the source list.

    Returns:
      • Per-claim confidence (High / Medium / Low) with source attribution
      • Consensus label (Scientific consensus / Emerging / Debated / Minority view)
      • Devil's advocate: the strongest caveat or counter-argument
      • Overall confidence rating
    """
    logger.info("[Wisdom Node 5] Validation")
    llm = _llm(state, temperature=0.2)

    deep_understanding = state.get("deep_understanding", "")
    papers = state.get("academic_papers", [])

    paper_list = "\n".join(
        f"[{p['citation_key']}] {p['title']} ({p.get('year','?')}) "
        f"— {p.get('citation_count','?')} citations — {p['source']}"
        for p in papers[:12]
    )

    system = """You are a rigorous scientific validator and critical thinker.

Analyse the wisdom text below and return a JSON object:
{
  "claims": [
    {
      "claim": "The specific factual assertion (one sentence)",
      "confidence": "High|Medium|Low",
      "consensus": "Scientific consensus|Emerging evidence|Debated|Minority view",
      "source": "citation_key or 'General scientific understanding'"
    }
  ],
  "devils_advocate": "2-3 sentences: the single strongest caveat, limitation, or counter-argument to this wisdom. Be intellectually honest.",
  "overall_confidence": "High|Medium|Low"
}

Confidence guide:
  High   — multiple well-cited peer-reviewed studies converge on this
  Medium — some empirical support, but results are mixed or samples are small
  Low    — limited evidence; theoretical or based on few/weak studies

Extract 3–5 key factual claims. Return ONLY valid JSON."""

    human = f"""WISDOM TO VALIDATE:
{deep_understanding[:3000]}

AVAILABLE SOURCES:
{paper_list}

Rate each claim and write the devil's advocate. Return JSON only."""

    try:
        raw = _call(llm, system, human)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON in validation response")
        validation = json.loads(match.group(0))
    except Exception as e:
        logger.warning("Validation failed (%s) — using defaults", e)
        validation = {
            "claims": [],
            "devils_advocate": "Automated validation could not be completed. Treat this wisdom as a starting point and consult primary sources before making important decisions.",
            "overall_confidence": "Medium",
        }

    conf = validation.get("overall_confidence", "Medium")
    conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "🟡")
    n_papers = len(papers)

    assistant_response = (
        f"Your wisdom has been synthesised and validated. "
        f"Overall confidence: {conf_emoji} **{conf}** — based on **{n_papers}** academic source(s). "
        f"Explore the tabs below for the scientific explanation, the simple version, "
        f"action steps, and the full validation breakdown."
    )

    return {
        "wisdom_claims": validation.get("claims", []),
        "devils_advocate": validation.get("devils_advocate", ""),
        "overall_confidence": conf,
        "assistant_response": assistant_response,
        "phase": "done",
        "current_step": "wisdom_validator",
        "completed_steps": state.get("completed_steps", []) + ["wisdom_validator"],
        "progress_pct": 90,
    }


# ── Node 6: Wisdom Follow-up ───────────────────────────────────────────────────

def wisdom_followup_node(state: WisdomState) -> Dict[str, Any]:
    """
    Handles user questions after wisdom has been generated.

    Has access to the full wisdom output via context_loader_node.
    Answers with depth, references specific sections, and suggests
    what the user might want to explore next.
    """
    logger.info("[Wisdom Node 6] Follow-up")
    llm = _llm(state, temperature=0.6)

    deep_understanding = state.get("deep_understanding", "")
    simple_explanation = state.get("simple_explanation", "")
    actionable_takeaways = state.get("actionable_takeaways", [])
    devils_advocate = state.get("devils_advocate", "")
    history = state.get("conversation_history", [])

    steps_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(actionable_takeaways))
    history_block = _format_history(history[-6:])

    system = f"""You are the Wisdom Oracle. You have already generated wisdom on this topic.

WISDOM YOU PRODUCED:
═══ SCIENTIFIC EXPLANATION ═══
{deep_understanding[:2500]}

═══ SIMPLE EXPLANATION ═══
{simple_explanation[:1200]}

═══ ACTION STEPS ═══
{steps_text}

═══ KEY CAVEAT ═══
{devils_advocate}

Answer the user's follow-up with care and depth. Reference specific parts of the wisdom.
Write 5–7 detailed paragraphs — comprehensive, well-evidenced, and actionable.
End with one short suggested question the user might find valuable to explore next."""

    human = f"""{history_block}

USER'S FOLLOW-UP: {state.get("user_message", "")}"""

    try:
        response = _call(llm, system, human)
    except Exception as e:
        logger.error("Follow-up LLM call failed: %s", e)
        response = f"[Error generating follow-up: {e}]"

    return {
        "assistant_response": response,
        "current_step": "wisdom_followup",
        "completed_steps": state.get("completed_steps", []) + ["wisdom_followup"],
        "progress_pct": 95,
    }


# ── Node 7: Memory Saver ───────────────────────────────────────────────────────

def memory_saver_node(state: WisdomState) -> Dict[str, Any]:
    """
    Persist the current user + assistant turns to the session JSON.

    If the phase just transitioned to "done" (wisdom was generated this turn),
    also calls WisdomMemory.save_wisdom() to store the full wisdom output.
    """
    logger.info("[Wisdom Node 7] Memory Saver")
    session_id = state.get("session_id", "")

    if not session_id:
        return {
            "current_step": "memory_saver",
            "completed_steps": state.get("completed_steps", []) + ["memory_saver"],
            "progress_pct": 100,
        }

    phase = state.get("phase", "clarifying")
    user_msg = state.get("user_message", "")
    assistant_msg = state.get("assistant_response", "")

    # Save user turn
    _get_memory().add_turn(session_id, "user", user_msg)

    # Save assistant turn (with metadata flags)
    is_wisdom_turn = phase == "done" and bool(state.get("deep_understanding"))
    is_question = phase == "clarifying"

    metadata: Dict[str, Any] = {}
    if is_question:
        metadata["is_question"] = True
    if is_wisdom_turn:
        metadata["has_wisdom"] = True

    _get_memory().add_turn(session_id, "assistant", assistant_msg, metadata=metadata)

    # Persist full wisdom output when generation just completed
    if is_wisdom_turn:
        _get_memory().save_wisdom(
            session_id=session_id,
            deep_understanding=state.get("deep_understanding", ""),
            simple_explanation=state.get("simple_explanation", ""),
            actionable_takeaways=state.get("actionable_takeaways", []),
            validation={
                "claims": state.get("wisdom_claims", []),
                "devils_advocate": state.get("devils_advocate", ""),
                "overall_confidence": state.get("overall_confidence", "Medium"),
            },
            papers=state.get("academic_papers", [])[:10],
            queries=state.get("search_queries", []),
            topic_tags=state.get("topic_tags", []),
        )
        logger.info("  Wisdom output saved for session %s", session_id)
    else:
        # Just update phase in JSON
        _get_memory().update_phase(session_id, phase)

    return {
        "current_step": "memory_saver",
        "completed_steps": state.get("completed_steps", []) + ["memory_saver"],
        "progress_pct": 100,
    }
