"""
tools/style_profiler.py
───────────────────────
Analyses writing style from uploaded documents using an LLM and produces:
  1. A structured analysis dict covering four dimensions:
       • Tone & formality
       • Structure & format
       • Vocabulary & complexity
       • Citation & evidence style
  2. A compact injection_prompt (≤ 280 words) ready to be appended to any
     LLM system prompt to reproduce the user's writing style.

No new dependencies — uses langchain_ollama (already in requirements.txt).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Maximum characters taken from each document for style analysis.
# Keeps the analysis focused and context-window friendly.
_MAX_CHARS_PER_DOC = 3000
_MAX_TOTAL_CHARS = 8000


def _truncate_docs(documents: list) -> str:
    """Combine document excerpts into a single analysis corpus."""
    parts: List[str] = []
    total = 0
    for doc in documents:
        text = getattr(doc, "raw_text", "") or ""
        excerpt = text[:_MAX_CHARS_PER_DOC]
        if not excerpt.strip():
            continue
        fname = getattr(doc, "filename", "document")
        parts.append(f"[Document: {fname}]\n{excerpt}")
        total += len(excerpt)
        if total >= _MAX_TOTAL_CHARS:
            break
    return "\n\n---\n\n".join(parts)


def analyse_writing_style(
    documents: list,
    model_name: str = "llama3.1:8b",
    ollama_base_url: str = "http://localhost:11434",
    num_ctx: int = 32768,
) -> Dict[str, Any]:
    """
    Analyse writing style from a list of ProcessedDocument objects.

    Returns
    -------
    dict with keys:
      "analysis"         : structured dict of style characteristics
      "injection_prompt" : compact instruction block for LLM system prompts
    """
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    corpus = _truncate_docs(documents)
    if not corpus.strip():
        logger.warning("No extractable text found in documents for style analysis.")
        return {"analysis": {}, "injection_prompt": ""}

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_base_url,
        temperature=0.2,
        num_predict=2048,
        num_ctx=num_ctx,
    )

    # ── Step 1: Extract structured style characteristics ──────────────────
    analysis_system = """You are a linguistic style analyst. Your task is to analyse
writing samples and extract precise, actionable style characteristics.
Return ONLY valid JSON — no prose, no explanation, no markdown fences."""

    analysis_human = f"""Analyse the writing style of the following document(s) across
four dimensions. Be specific and concrete — general descriptions are useless.

DOCUMENTS:
{corpus}

Return ONLY this JSON structure (fill each field with precise observations):
{{
  "tone_formality": {{
    "register": "formal | semi-formal | informal",
    "person": "first person | third person | mixed",
    "hedging": "describe frequency and exact hedge words used (e.g. 'suggests', 'appears to', 'evidence indicates')",
    "assertiveness": "describe how confidently claims are stated"
  }},
  "structure_format": {{
    "paragraph_length": "average sentence count and length pattern (e.g. '4–6 sentences, topic sentence always first')",
    "sentence_complexity": "describe typical sentence structure (e.g. 'complex with subordinate clauses, avg ~28 words')",
    "transitions": "list the exact transition phrases used (e.g. 'Furthermore', 'In contrast', 'Building on this')",
    "use_of_lists": "describe when bullet points or numbered lists are used vs prose",
    "section_ordering": "describe how sections or arguments are typically sequenced"
  }},
  "vocabulary_complexity": {{
    "technical_density": "low | medium | high — and how technical terms are handled",
    "preferred_terms": "list 3–5 distinctive vocabulary choices or recurring terms",
    "readability": "describe target audience and complexity level",
    "avoids": "list specific words or constructions this writer avoids"
  }},
  "citation_evidence": {{
    "citation_density": "describe how often claims are cited (e.g. 'nearly every factual claim', 'paragraph-level')",
    "citation_placement": "where citations appear (end of sentence, inline after claim, etc.)",
    "evidence_hierarchy": "how the writer ranks evidence types (peer-reviewed > preprint > grey literature, etc.)",
    "confidence_language": "how uncertainty is expressed around cited claims"
  }}
}}"""

    try:
        raw = llm.invoke([
            SystemMessage(content=analysis_system),
            HumanMessage(content=analysis_human),
        ])
        text = raw.content.strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        analysis = json.loads(match.group(0)) if match else {}
    except Exception as e:
        logger.warning("Style analysis LLM call failed: %s", e)
        analysis = {}

    # ── Step 2: Generate the compact injection prompt ─────────────────────
    injection_system = """You write concise system prompt instructions. Be direct and specific.
Do NOT use any headers or bullet points — write flowing prose instructions only."""

    injection_human = f"""Based on this writing style analysis, write a compact instruction block
(≤ 280 words) that tells an AI assistant exactly how to reproduce this writing style.

STYLE ANALYSIS:
{json.dumps(analysis, indent=2)}

Write the instructions in second person ("Write in..."), covering:
1. Tone and register (person, formality, hedging words to use)
2. Sentence and paragraph structure (length, transitions, complexity)
3. Vocabulary choices (technical terms, preferred words, what to avoid)
4. Citation and evidence style (frequency, placement, confidence language)

Be prescriptive and specific. A reader should be able to follow these instructions
without having seen the original documents. Return only the instruction text."""

    try:
        raw2 = llm.invoke([
            SystemMessage(content=injection_system),
            HumanMessage(content=injection_human),
        ])
        injection_prompt = raw2.content.strip()
    except Exception as e:
        logger.warning("Injection prompt generation failed: %s", e)
        injection_prompt = _fallback_injection(analysis)

    return {
        "analysis": analysis,
        "injection_prompt": injection_prompt,
    }


def _fallback_injection(analysis: dict) -> str:
    """Generate a basic injection prompt from the analysis dict if the LLM call fails."""
    parts = []
    tone = analysis.get("tone_formality", {})
    if tone.get("register"):
        parts.append(f"Write in a {tone['register']} register.")
    if tone.get("person"):
        parts.append(f"Use {tone['person']}.")
    if tone.get("hedging"):
        parts.append(f"Hedging: {tone['hedging']}.")

    struct = analysis.get("structure_format", {})
    if struct.get("paragraph_length"):
        parts.append(f"Paragraph structure: {struct['paragraph_length']}.")
    if struct.get("transitions"):
        parts.append(f"Transitions: {struct['transitions']}.")

    vocab = analysis.get("vocabulary_complexity", {})
    if vocab.get("avoids"):
        parts.append(f"Avoid: {vocab['avoids']}.")

    cit = analysis.get("citation_evidence", {})
    if cit.get("citation_density"):
        parts.append(f"Citations: {cit['citation_density']}.")

    return " ".join(parts) if parts else ""
