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

    # Concepts already covered
    covered_block = ""
    if concepts_covered:
        covered_block = (
            f"\n\nCONCEPTS ALREADY EXPLAINED (do not re-explain from scratch): "
            f"{', '.join(concepts_covered[:20])}"
        )

    system = f"""You are a Research Partner — a gifted science communicator who excels at making
complex research concepts genuinely understandable.

CORE RULES:
1. Never use unexplained jargon. Every technical term you introduce must be defined inline in plain language.
2. {style_instruction}
3. Write 3–6 paragraphs only — no lengthy essays. Be concise and memorable.
4. Build on the previous conversation — reference and connect to what was discussed before.
5. Quote short passages from the provided document context when they are directly relevant.
6. At the very end of your response, append EXACTLY this JSON (no other text after it):
   {{"suggested_questions": ["Question 1?", "Question 2?", "Question 3?"]}}
   The questions should be natural follow-ups a curious reader would want to ask next.
7. Do NOT start your response with "Certainly!" or "Of course!" or similar filler phrases.
8. The topic being explored is: {topic}{_clarification_context(state)}"""

    human = f"""USER QUESTION: {state.get('user_message', '')}
{history_block}{doc_block}{covered_block}

Respond in the "{style}" style. Remember to end with the suggested_questions JSON."""

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

    # Parse out the suggested_questions JSON from the end of the response
    suggested_questions: List[str] = []
    main_response = raw_response

    json_marker = '{"suggested_questions":'
    if json_marker in raw_response:
        split_idx = raw_response.rfind(json_marker)
        main_response = raw_response[:split_idx].strip()
        json_fragment = raw_response[split_idx:]
        try:
            parsed = json.loads(json_fragment)
            suggested_questions = parsed.get("suggested_questions", [])[:3]
        except Exception:
            # Try extracting with regex if JSON is malformed
            match = re.search(r'"suggested_questions"\s*:\s*(\[.*\])', json_fragment, re.DOTALL)
            if match:
                try:
                    suggested_questions = json.loads(match.group(1))[:3]
                except Exception as inner_e:
                    logger.warning(
                        "suggested_questions: regex fallback also failed (%s) — "
                        "returning []. Raw tail: %s",
                        inner_e, json_fragment[-80:],
                    )
            else:
                logger.warning(
                    "suggested_questions: JSON marker present but no valid array "
                    "extracted — returning []. Raw tail: %s",
                    json_fragment[-80:],
                )

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
