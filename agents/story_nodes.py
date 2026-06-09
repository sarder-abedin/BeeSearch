"""
agents/story_nodes.py
──────────────────────
Three nodes that form the Research Partner (Storytelling) workflow.

  START → context_loader → storyteller → memory_saver → END

Node responsibilities
─────────────────────
  context_loader  : Load conversation history + document context from memory
  storyteller     : Generate an explanation in the requested style + follow-up questions
  memory_saver    : Persist the new user + assistant turns back to memory

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

logger = logging.getLogger(__name__)
cfg = get_settings()

# Lazy singleton — not created at import time so tests can inject a different instance.
_memory: StorytellerMemory | None = None


def _get_memory() -> StorytellerMemory:
    global _memory
    if _memory is None:
        _memory = StorytellerMemory()
    return _memory


def _llm(state: StoryState, temperature: float = 0.7) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=4096,
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(180.0)},
    )


def _call(llm: ChatOllama, system: str, human: str) -> str:
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
        "progress_pct": 20,
    }


# ── Node 1.5: Source Router ───────────────────────────────────────────────────

_COVERAGE_THRESHOLD = 6  # Score below this triggers online search (0–10 scale)


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
        router_llm = ChatOllama(
            model=state.get("model_name", cfg.ollama_model),
            base_url=cfg.ollama_base_url,
            temperature=0.0,
            num_predict=128,
            num_ctx=min(state.get("num_ctx", cfg.num_ctx), 4096),
            sync_client_kwargs={"timeout": httpx.Timeout(60.0)},
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

        # Academic search (arXiv + Semantic Scholar + Google Scholar)
        try:
            papers = AcademicSearcher().search(question, max_per_source=3)[:5]
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
            web_hits = WebSearcher().search(question, max_results=4)
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


# ── Node 2: Storyteller ────────────────────────────────────────────────────────

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
        attribution_format = f"""

CRITICAL — PER-SECTION SOURCE ATTRIBUTION REQUIRED:
The uploaded documents scored only {coverage_score}/10 coverage for this question ({gap_reason}).
You MUST structure your response using these exact labelled sections in this order:

**From your documents:**
Explain what the uploaded documents actually say about this question. Quote brief passages where useful.
If the documents say very little, keep this section short and honest — do not pad it.

**Why online search was needed:**
1–2 sentences only. What specific gap did the documents leave? What could not be answered from them alone?

**From online sources:**
Fill the gap using the online sources provided. Apply the chosen style and audience level here too.
Cite every claim from an online source with [Source N] placed immediately after it.
Only use sources that are genuinely relevant — skip irrelevant ones.

**References:**
List only the online sources you actually cited, in this format:
[Source N] Title — URL (Authors, Year if available)

After the References section, end with the suggested_questions JSON and nothing else."""

    # Concepts already covered
    covered_block = ""
    if concepts_covered:
        covered_block = (
            f"\n\nCONCEPTS ALREADY EXPLAINED (do not re-explain from scratch): "
            f"{', '.join(concepts_covered[:20])}"
        )

    online_note = (
        " When online sources are provided, follow the attribution format below."
        if online_results else ""
    )
    system = f"""You are a Research Partner — a gifted science communicator who excels at making
complex research concepts genuinely understandable.{attribution_format}

CORE RULES:
1. Never use unexplained jargon relative to the target audience below — define any term that audience wouldn't already know.
2. STYLE — {style_instruction}
3. AUDIENCE LEVEL — {level_instruction}
4. {"Follow the per-section attribution format above." if online_results else "Write 3–6 paragraphs only — no lengthy essays. Be concise and memorable."}
5. Build on the previous conversation — reference and connect to what was discussed before.
6. {"Each section should be written in the chosen style and at the chosen audience level." if online_results else "Quote short passages from the provided document context when they are directly relevant."}
7. At the very end of your response (after any References section), append EXACTLY this JSON (no other text after it):
   {{"suggested_questions": ["Question 1?", "Question 2?", "Question 3?"]}}
   The questions should be natural follow-ups a curious reader would want to ask next.
8. Do NOT start your response with "Certainly!" or "Of course!" or similar filler phrases.
9. The topic being explored is: {topic}{_clarification_context(state)}"""

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
            "errors": state.get("errors", []) + [str(e)],
            "current_step": "storyteller",
            "completed_steps": state.get("completed_steps", []) + ["storyteller"],
            "progress_pct": 70,
        }

    # Parse suggested_questions from the response.
    # LLMs emit this in several formats — try each in order of specificity:
    #   1. {"suggested_questions": [...]}   (ideal full JSON object)
    #   2. "suggested_questions": [...]     (JSON key without outer braces)
    #   3. suggested_questions: [...]       (bare key, no quotes, no braces)
    # In every case strip the matched fragment from main_response so it never
    # shows as raw text in the chat bubble.
    suggested_questions: List[str] = []
    main_response = raw_response

    _SQ_PATTERNS = [
        # Full JSON object  {"suggested_questions": [...]}
        re.compile(r'\{[^{}]*"suggested_questions"\s*:\s*(\[.*?\])\s*\}', re.DOTALL),
        # Quoted key without outer braces  "suggested_questions": [...]
        re.compile(r'"suggested_questions"\s*:\s*(\[.*?\])', re.DOTALL),
        # Bare key  suggested_questions: [...] or suggested_questions = [...]
        re.compile(r'suggested_questions\s*[:=]\s*(\[.*?\])', re.DOTALL | re.IGNORECASE),
    ]

    for pat in _SQ_PATTERNS:
        m = pat.search(raw_response)
        if m:
            try:
                candidates = json.loads(m.group(1))
                if isinstance(candidates, list) and candidates:
                    suggested_questions = [str(q) for q in candidates if q][:3]
                    # Remove everything from the start of the match to end of response
                    main_response = raw_response[:m.start()].strip()
                    break
            except Exception:
                continue

    if not suggested_questions:
        logger.warning("suggested_questions: no parseable block found — raw tail: %s",
                       raw_response[-120:])

    # Second micro LLM call: extract newly explained concept names
    new_concepts: List[str] = []
    concepts_prompt = f"""From this explanation, list the names of scientific or technical concepts
that were EXPLAINED (not just mentioned). Return ONLY a JSON array of short concept names.
Example: ["attention mechanism", "softmax function"]

EXPLANATION:
{main_response[:1500]}"""

    try:
        import httpx
        micro_llm = ChatOllama(
            model=state.get("model_name", cfg.ollama_model),
            base_url=cfg.ollama_base_url,
            temperature=0.1,
            num_predict=256,
            num_ctx=min(state.get("num_ctx", cfg.num_ctx), 4096),
            sync_client_kwargs={"timeout": httpx.Timeout(60.0)},
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
        "current_step": "storyteller",
        "completed_steps": state.get("completed_steps", []) + ["storyteller"],
        "progress_pct": 70,
    }


# ── Node 3: Memory Saver ───────────────────────────────────────────────────────

def memory_saver_node(state: StoryState) -> Dict[str, Any]:
    """
    Persist the current user + assistant turns to the session JSON file.

    Appends two turns: the user message and the assistant response.
    Also updates the list of concepts covered in this session.
    """
    logger.info("[Story Node 3] Memory Saver")
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

    # Save assistant turn with suggested questions
    _get_memory().add_turn(
        session_id,
        role="assistant",
        content=state.get("assistant_response", ""),
        suggested_questions=state.get("suggested_questions", []),
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
