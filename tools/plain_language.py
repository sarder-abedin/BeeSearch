"""
tools/plain_language.py
───────────────────────
Convert a completed systematic review into three lay-audience summary formats.

  patient  — 8th-grade reading level, no jargon, "what this means for you"
  policy   — 1-page executive brief with numbered recommendations
  press    — Press release (inverted-pyramid, headline + lede)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()


def _make_llm(model_name: str, num_ctx: int) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=0.4,
        num_predict=1024,
        num_ctx=num_ctx,
        sync_client_kwargs={"timeout": httpx.Timeout(180.0)},
    )


def _call(llm: ChatOllama, system: str, human: str) -> str:
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content.strip()


def _evidence_context(state: Dict[str, Any]) -> str:
    parts = []
    narrative = state.get("narrative_synthesis", "")
    themes = state.get("key_themes", [])
    conclusion = state.get("conclusion", "")
    evidence_table = state.get("evidence_table", [])

    if narrative:
        parts.append(f"Synthesis: {narrative[:800]}")
    if themes:
        parts.append(f"Key themes: {'; '.join(themes[:5])}")
    if conclusion:
        parts.append(f"Conclusion: {conclusion[:400]}")
    high_q = [e for e in evidence_table if e.get("quality") == "High"]
    if high_q:
        top = high_q[0]
        parts.append(f"Top finding: {top.get('key_finding', '')} [{top.get('citation_key', '')}]")
    return "\n\n".join(parts)


def generate_patient_summary(
    state: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
) -> str:
    """Patient-facing summary: 8th-grade reading level, ~350 words, plain paragraphs."""
    rq = state.get("research_question", "")
    n = len(state.get("included_papers", []))
    return _call(
        _make_llm(model_name, num_ctx),
        """Write a patient-friendly plain-language summary of scientific research.

Rules:
- 8th-grade reading level: simple words, short sentences, no jargon
- If a technical term is necessary, explain it in parentheses immediately after
- Structure (4 paragraphs, no headers, no bullet points):
    1. What did researchers study and why does it matter?
    2. What did they find? (most important results)
    3. What does this mean for you as a patient or member of the public?
    4. What are the limits of this evidence, and should you talk to a doctor?
- ~350 words total
- Active voice; avoid passive constructions
- End with a reminder that findings do not replace professional medical advice""",
        f"Research question: {rq}\nStudies reviewed: {n}\n\n{_evidence_context(state)}",
    )


def generate_policy_brief(
    state: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
) -> str:
    """Policy brief: ~500 words, structured Markdown for decision-makers."""
    rq = state.get("research_question", "")
    n = len(state.get("included_papers", []))
    gaps = state.get("research_gaps", [])
    return _call(
        _make_llm(model_name, num_ctx),
        """Write a concise policy brief for government and institutional decision-makers.

Use these exact Markdown section headers:
## The Issue
## What the Evidence Shows
## Key Findings
## Policy Recommendations
## Evidence Gaps and Uncertainties
## Suggested Next Steps

Rules:
- ~500 words total
- Lead with the most newsworthy or actionable finding
- Key Findings: 3-5 bullet points with specific numbers where available
- Policy Recommendations: 3-4 numbered, specific, actionable recommendations
- Acknowledge uncertainty honestly
- No jargon; define acronyms on first use""",
        (
            f"Research question: {rq}\n"
            f"Studies reviewed: {n}\n"
            f"Research gaps: {'; '.join(gaps[:4]) if gaps else 'None identified'}\n\n"
            f"{_evidence_context(state)}"
        ),
    )


def generate_press_release(
    state: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
) -> str:
    """Press release: inverted-pyramid, ~400 words, AP style."""
    rq = state.get("research_question", "")
    n = len(state.get("included_papers", []))
    date_str = datetime.today().strftime("%B %d, %Y")
    return _call(
        _make_llm(model_name, num_ctx),
        f"""Write a press release for a scientific journal.

Follow this exact structure:
FOR IMMEDIATE RELEASE
{date_str}

**HEADLINE** (max 15 words, active voice, no jargon)

**SUBHEADLINE** (max 20 words, adds one key detail)

[City, {date_str}] — [Lede: 1-2 sentences covering who, what, why it matters]

[Body paragraph 1: most important finding with a specific number or statistic]

[Body paragraph 2: brief methodology — how was the research done?]

[Body paragraph 3: second key finding or real-world implication]

[Researcher quote: a plausible quote in quotation marks, attributed to "the lead researcher"]

[Body paragraph 4: limitations and what comes next]

**About this Review:** [1 sentence]

**Media Contact:** press@beesearch.org

Rules:
- Inverted pyramid: most newsworthy information first
- Active voice and present tense where possible
- No jargon; explain every technical term
- ~400 words""",
        f"Research question: {rq}\nStudies reviewed: {n}\n\n{_evidence_context(state)}",
    )


def generate_all_summaries(
    state: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
) -> Dict[str, str]:
    """Generate all three plain-language formats in one call."""
    return {
        "patient": generate_patient_summary(state, model_name, num_ctx),
        "policy": generate_policy_brief(state, model_name, num_ctx),
        "press": generate_press_release(state, model_name, num_ctx),
    }
