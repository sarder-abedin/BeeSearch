"""
agents/section_summary.py
──────────────────────────
Section-by-section document breakdown for the Research Notebook.

Detects document sections using a hybrid approach:
  1. Heuristic: scan chunk text for heading signatures — short, no trailing
     sentence punctuation, matches known academic section names or numbered-
     heading patterns.
  2. LLM fallback: if < 2 sections found heuristically, ask the LLM to
     identify section boundaries from numbered chunk excerpts.

Each detected section is then summarised in plain language calibrated to the
user-chosen level (novice / intermediate / expert).  A scoped Q&A function
lets users ask follow-up questions grounded strictly in that section's chunks.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()


# ── Audience-level descriptions (mirrors story_nodes._LEVEL_DESCRIPTIONS) ─────

_LEVEL_DESCRIPTIONS: Dict[str, str] = {
    "novice": (
        "Write for someone with NO background in this field. Define every "
        "technical term the moment you introduce it, in everyday words. "
        "Lean on familiar real-world comparisons. Focus on the big picture "
        "and why it matters — not on mechanism details."
    ),
    "intermediate": (
        "Write for someone with general science/research literacy (e.g. an "
        "undergraduate or informed generalist). Standard field terminology is "
        "fine, but briefly gloss any less-common terms. Go one layer deeper "
        "into mechanisms and nuance than for a complete beginner."
    ),
    "expert": (
        "Write for a researcher already familiar with this field. Use precise "
        "technical and disciplinary terminology without hand-holding definitions. "
        "Emphasise methodological nuance, caveats, open questions, and "
        "connections to the broader literature."
    ),
}

# ── Known academic section headings for heuristic detection ──────────────────

_KNOWN_HEADINGS = {
    "abstract", "introduction", "background", "related work", "related works",
    "literature review", "prior work", "motivation", "problem statement",
    "problem formulation", "research questions", "objectives", "contributions",
    "methodology", "methods", "method", "materials and methods", "approach",
    "experimental setup", "experimental design", "experiments", "implementation",
    "proposed method", "proposed approach", "framework", "architecture", "model",
    "results", "findings", "evaluation", "performance", "analysis", "experiments",
    "discussion", "limitations", "threats to validity", "future work",
    "future directions", "conclusion", "conclusions", "summary",
    "acknowledgments", "acknowledgements", "references", "appendix",
    "data", "dataset", "datasets", "overview", "preliminaries",
}


# ── LLM factory ───────────────────────────────────────────────────────────────

def _make_llm(
    model_name: str,
    num_ctx: int,
    temperature: float = 0.3,
    num_predict: int = 1024,
) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=num_ctx or cfg.num_ctx,
        sync_client_kwargs={"timeout": httpx.Timeout(300.0)},
    )


def _invoke(llm: ChatOllama, system: str, human: str) -> str:
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return resp.content.strip()


# ── Section detection — heuristic ─────────────────────────────────────────────

def _is_heading_heuristic(text: str) -> bool:
    """Return True if the chunk text looks like a section heading."""
    t = text.strip()
    if not t or len(t) > 120:
        return False
    # Headings don't end with sentence-final punctuation (colon is allowed — "Methods:")
    if t[-1] in ".,;!?":
        return False
    lower = t.lower()
    # Direct match against known headings (strip trailing 's' for plurals)
    if lower in _KNOWN_HEADINGS or lower.rstrip("s") in _KNOWN_HEADINGS:
        return True
    # Numbered heading: "1.", "1.1", "1.1.1 Title", "2.3 Methods"
    if re.match(r"^\d+(\.\d+)*\.?\s+\w", t):
        return True
    # "Chapter N", "Section N", "Part N"
    if re.match(r"^(chapter|section|part)\s+\d+", t, re.IGNORECASE):
        return True
    # Short ALL-CAPS phrase (common in some PDF/DOCX styles)
    if t.isupper() and 2 <= len(t.split()) <= 6:
        return True
    return False


def detect_sections_heuristic(
    chunks: List[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Group chunks into sections by scanning for heading-like chunks.

    Returns [(section_title, [chunk, ...]), ...].
    Returns an empty list (< 2 sections found) to signal caller to fall back
    to LLM-based detection.
    """
    sections: List[Tuple[str, List[Dict[str, Any]]]] = []
    current_title = ""
    current_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if _is_heading_heuristic(text):
            if current_chunks:
                sections.append((current_title or "Preamble", current_chunks))
            current_title = text
            current_chunks = []
        else:
            current_chunks.append(chunk)

    if current_chunks:
        sections.append((current_title or "Full Document", current_chunks))

    return sections if len(sections) >= 2 else []


# ── Section detection — LLM fallback ─────────────────────────────────────────

def detect_sections_llm(
    chunks: List[Dict[str, Any]],
    model_name: str,
    num_ctx: int,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Ask the LLM to identify section boundaries from numbered chunk excerpts.

    Returns [(section_title, [chunk, ...])].  Falls back to a single
    "Full Document" section if the LLM call fails or yields no useful output.
    """
    numbered = "\n".join(
        f"[{i}] {chunk.get('text', '')[:200].strip()}"
        for i, chunk in enumerate(chunks)
    )

    system = (
        "You are a document structure analyser. "
        "Given numbered text excerpts from a document, identify the start of "
        "each major section. Return ONLY a JSON array with no markdown fences:\n"
        '[{"title": "Section Name", "start": <chunk_index>}, ...]'
        "\nOrder by start index. Include every major section."
    )
    human = (
        f"DOCUMENT EXCERPTS (first 200 chars each, {len(chunks)} total):\n"
        f"{numbered}\n\n"
        "Return the section boundary JSON array."
    )

    try:
        llm = _make_llm(model_name, num_ctx, temperature=0.0, num_predict=512)
        raw = _invoke(llm, system, human)
        raw = re.sub(r"```[a-zA-Z0-9]*\n?", "", raw).strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            raise ValueError("No JSON array in LLM response")
        boundaries: List[Dict[str, Any]] = json.loads(m.group(0))
        boundaries = sorted(
            [b for b in boundaries if isinstance(b.get("start"), (int, float))],
            key=lambda b: b["start"],
        )
        if not boundaries:
            raise ValueError("Empty boundary list")

        sections: List[Tuple[str, List[Dict[str, Any]]]] = []
        for i, boundary in enumerate(boundaries):
            start = int(boundary["start"])
            end = int(boundaries[i + 1]["start"]) if i + 1 < len(boundaries) else len(chunks)
            title = str(boundary.get("title", f"Section {i + 1}")).strip()
            section_chunks = chunks[start:end]
            if section_chunks:
                sections.append((title, section_chunks))

        return sections if sections else [("Full Document", chunks)]

    except Exception as exc:
        logger.warning("detect_sections_llm failed (%s) — single-section fallback", exc)
        return [("Full Document", chunks)]


# ── Section detection — hybrid ────────────────────────────────────────────────

def detect_sections_hybrid(
    chunks: List[Dict[str, Any]],
    model_name: str,
    num_ctx: int,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Hybrid section detection: heuristic first, LLM fallback.

    The heuristic scans chunk text for heading signatures — fast and reliable
    for well-formatted PDFs and DOCX files.  If it finds fewer than 2 sections
    (e.g. unformatted text, scanned PDFs), the LLM analyses the document
    structure from chunk text excerpts.
    """
    sections = detect_sections_heuristic(chunks)
    if sections:
        logger.info("[Sections] Heuristic found %d section(s)", len(sections))
        return sections
    logger.info("[Sections] Heuristic insufficient — falling back to LLM detection")
    return detect_sections_llm(chunks, model_name, num_ctx)


# ── Section summarisation ─────────────────────────────────────────────────────

def summarize_section(
    title: str,
    section_chunks: List[Dict[str, Any]],
    level: str,
    model_name: str,
    num_ctx: int,
) -> str:
    """
    Generate a plain-language summary of one document section, calibrated to
    the chosen explanation level (novice / intermediate / expert).
    """
    level_instruction = _LEVEL_DESCRIPTIONS.get(level, _LEVEL_DESCRIPTIONS["intermediate"])
    text = " ".join(c.get("text", "") for c in section_chunks)
    if len(text) > 8_000:
        text = text[:8_000] + "…"

    system = f"""You are a science communicator explaining one section of a research document.

AUDIENCE LEVEL — {level_instruction}

RULES:
1. Summarise what this section covers, its key points, and why they matter.
2. Write 3–5 short, clear paragraphs.  Be concise — do not pad.
3. Rephrase in simpler terms appropriate for the audience level above.
4. Do NOT add information absent from the section text.
5. Do NOT repeat the section title as the first line."""

    human = (
        f"SECTION TITLE: {title}\n\n"
        f"SECTION CONTENT:\n{text}\n\n"
        "Write the plain-language summary now."
    )

    try:
        llm = _make_llm(model_name, num_ctx, temperature=0.3, num_predict=1024)
        return _invoke(llm, system, human)
    except Exception as exc:
        logger.error("summarize_section failed for '%s': %s", title, exc)
        return f"[Could not summarise this section: {exc}]"


# ── Section Q&A ───────────────────────────────────────────────────────────────

def answer_section_question(
    title: str,
    section_chunks: List[Dict[str, Any]],
    question: str,
    level: str,
    history: List[Dict[str, str]],
    model_name: str,
    num_ctx: int,
) -> str:
    """
    Answer a question grounded strictly in the given section's chunks,
    calibrated to the chosen explanation level.  Never raises.
    """
    level_instruction = _LEVEL_DESCRIPTIONS.get(level, _LEVEL_DESCRIPTIONS["intermediate"])
    context = " ".join(c.get("text", "") for c in section_chunks)
    if len(context) > 6_000:
        context = context[:6_000] + "…"

    history_block = ""
    if history:
        lines = []
        for turn in history[-6:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {turn.get('content', '')[:300]}")
        history_block = "\n\nPREVIOUS CONVERSATION:\n" + "\n\n".join(lines)

    system = f"""You are a Research Notebook assistant answering a question \
about ONE specific section of a document.

SECTION: {title}
AUDIENCE LEVEL — {level_instruction}

STRICT RULES:
1. Use ONLY the section content provided. Do not add outside knowledge.
2. Calibrate your language and depth to the audience level stated above.
3. If the section does not contain enough information to answer, say so plainly.
4. Be concise — 2–4 paragraphs maximum."""

    human = (
        f"SECTION CONTENT:\n{context}"
        f"{history_block}\n\n"
        f"QUESTION: {question}"
    )

    try:
        llm = _make_llm(model_name, num_ctx, temperature=0.3, num_predict=1024)
        return _invoke(llm, system, human)
    except Exception as exc:
        logger.error("answer_section_question failed: %s", exc)
        return f"[Could not answer: {exc}]"


# ── Expert section review ─────────────────────────────────────────────────────

def review_section(
    title: str,
    section_chunks: List[Dict[str, Any]],
    model_name: str,
    num_ctx: int,
) -> Dict[str, str]:
    """
    Produce a structured expert-level critical review of one document section,
    mimicking the evaluation style of a top-tier journal or conference reviewer.

    Returns a dict with keys: strengths, weaknesses, limitations, improvements.
    On failure, each value contains an error message so the UI can still render.
    """
    context = " ".join(c.get("text", "") for c in section_chunks)
    if len(context) > 8_000:
        context = context[:8_000] + "…"

    system = """You are a rigorous expert reviewer for a top-tier academic journal \
or conference (Nature, Science, NeurIPS, ICML, ACL, etc.).

Your task is to critically review ONE section of a research document and produce \
structured, actionable feedback — the same quality you would give in a formal peer review.

Return ONLY a JSON object with exactly these four keys (no markdown fences):
{
  "strengths": "<2–4 specific strengths of this section>",
  "weaknesses": "<2–4 specific weaknesses or gaps>",
  "limitations": "<acknowledged or unacknowledged limitations that affect this section>",
  "improvements": "<concrete, prioritised suggestions for how the authors could improve this section>"
}

Be specific and technical. Reference actual content from the section. \
Do not be vague or generic. Write as a domain expert."""

    human = (
        f"SECTION TITLE: {title}\n\n"
        f"SECTION CONTENT:\n{context}\n\n"
        "Provide your expert review in the JSON format specified."
    )

    fallback = {
        "strengths": "",
        "weaknesses": "",
        "limitations": "",
        "improvements": "",
    }

    try:
        llm = _make_llm(model_name, num_ctx, temperature=0.1, num_predict=1024)
        raw = _invoke(llm, system, human)
        raw = re.sub(r"```[a-zA-Z0-9]*\n?", "", raw).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("No JSON object in response")
        data = json.loads(m.group(0))
        # If the LLM wrapped the result in an outer object, dig one level deeper.
        _keys = {"strengths", "weaknesses", "limitations", "improvements"}
        if not _keys.intersection(data.keys()):
            for v in data.values():
                if isinstance(v, dict) and _keys.intersection(v.keys()):
                    data = v
                    break
        return {
            "strengths":    str(data.get("strengths", "")).strip(),
            "weaknesses":   str(data.get("weaknesses", "")).strip(),
            "limitations":  str(data.get("limitations", "")).strip(),
            "improvements": str(data.get("improvements", "")).strip(),
        }
    except Exception as exc:
        logger.error("review_section failed for '%s': %s", title, exc)
        err = f"[Review failed: {exc}]"
        return {k: err for k in fallback}


# ── Claim-based questions ─────────────────────────────────────────────────────

def generate_section_claim_questions(
    title: str,
    section_chunks: List[Dict[str, Any]],
    model_name: str,
    num_ctx: int,
) -> List[str]:
    """
    Identify unclear, weakly supported, or questionable claims in a section
    and return 3–5 critical questions a careful reader should ask about them.

    Returns a list of question strings.  Returns an empty list on failure.
    """
    context = " ".join(c.get("text", "") for c in section_chunks)
    if len(context) > 6_000:
        context = context[:6_000] + "…"

    system = (
        "You are a critical reader reviewing a section of a research document. "
        "Identify claims that are unclear, lack supporting evidence, make strong "
        "assumptions, or a careful reader would want to interrogate further. "
        "For each such claim, write one pointed question that challenges or seeks "
        "clarification on it.\n"
        'Return ONLY a JSON array of question strings: ["Question 1?", "Question 2?", ...]'
        "\nGenerate 3–5 questions. No markdown fences."
    )
    human = (
        f"SECTION TITLE: {title}\n\n"
        f"SECTION CONTENT:\n{context}\n\n"
        "Return the JSON array of critical questions about unclear or unsupported claims."
    )

    try:
        llm = _make_llm(model_name, num_ctx, temperature=0.2, num_predict=512)
        raw = _invoke(llm, system, human)
        raw = re.sub(r"```[a-zA-Z0-9]*\n?", "", raw).strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            raise ValueError("No JSON array in response")
        parsed = json.loads(m.group(0))
        # Flatten if the LLM wrapped questions in objects (e.g. [{"question": "..."}])
        questions: List[str] = []
        for item in parsed:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        questions.append(v)
                        break
        return [q.strip() for q in questions if q.strip()][:5]
    except Exception as exc:
        logger.warning("generate_section_claim_questions failed for '%s': %s", title, exc)
        return []


# ── Utility ───────────────────────────────────────────────────────────────────

def get_doc_chunks(
    notebook: Dict[str, Any],
    doc_id: str,
) -> List[Dict[str, Any]]:
    """Return chunks for a specific document, sorted by chunk_index."""
    chunks = [
        c for c in notebook.get("chunks", [])
        if c.get("doc_id") == doc_id
    ]
    return sorted(chunks, key=lambda c: c.get("chunk_index", 0))
