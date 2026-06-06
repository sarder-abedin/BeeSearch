"""
tools/cli_recommender.py
─────────────────────────
Smart CLI recommendation engine — pure stdlib, no new dependencies.

Analyses the user's inputs and workflow results to produce contextual,
actionable tips for all six research modes.

Each public function returns a list of tip strings. Call
`print_recommendations()` to render them as a Rich panel.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ── Keyword sets for pre-run goal analysis ─────────────────────────────────

_KW_NEEDS_DOCS = frozenset({
    "this paper", "this file", "this document", "my paper", "my document",
    "my report", "my thesis", "my notes", "the pdf", "the document",
    "uploaded", "extract from", "summarise the", "summarize the",
    "in the attached", "in the file",
})

_KW_NEEDS_SEARCH = frozenset({
    "literature", "survey", "review", "state of the art", "recent advances",
    "recent research", "studies on", "papers on", "journal articles",
    "published research", "academic papers", "scholarly",
    "what has been done", "related work",
})

_KW_BENEFITS_WEB = frozenset({
    "recent", "2024", "2025", "current", "latest", "new development",
    "news", "trending", "cutting-edge", "breakthrough",
})

_KW_BROAD = frozenset({
    "overview", "introduction", "what is", "explain", "define",
    "beginner", "basics", "fundamentals", "history of",
})


# ── Helpers ────────────────────────────────────────────────────────────────

def _contains(text: str, keywords: frozenset) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def _render_tip(tip: str, console=None) -> None:
    if console:
        console.print(f"  {tip}")


# ── Public recommendation functions ───────────────────────────────────────

def recommend_mode(
    goal: str,
    files: list,
    current_mode: str,
    web: bool = False,
) -> List[str]:
    """
    Pre-run recommendations for research Modes 1–3.

    Analyses the goal text and file list to suggest mode changes,
    flag additions, or context-window adjustments.
    """
    tips: List[str] = []
    has_files = bool(files)

    # Mode consistency
    if current_mode == "document" and not has_files:
        tips.append(
            "⚠  --mode document requires at least one --files argument.\n"
            "   Fix: add --files paper.pdf  OR  switch to --mode search"
        )
    if current_mode == "search" and has_files:
        tips.append(
            "💡 You provided files but chose --mode search (files are ignored in search mode).\n"
            "   Tip: use --mode hybrid to analyse your documents AND search the literature."
        )

    # Goal → mode mismatch
    if _contains(goal, _KW_NEEDS_SEARCH) and current_mode == "document":
        tips.append(
            "💡 Your goal sounds like a literature survey.\n"
            "   Tip: switch to --mode hybrid or --mode search for academic paper retrieval."
        )
    if _contains(goal, _KW_NEEDS_DOCS) and not has_files:
        tips.append(
            "💡 Your goal references a specific document, but no --files were given.\n"
            "   Tip: add --files <path.pdf> to enable document analysis."
        )

    # Web search hints
    if _contains(goal, _KW_BENEFITS_WEB) and not web:
        tips.append(
            "💡 Your goal mentions recent or current work.\n"
            "   Tip: add --web to supplement academic results with Google search."
        )

    # Multi-file context window
    if len(files) > 3:
        tips.append(
            f"💡 {len(files)} files detected — the default context window may be tight.\n"
            "   Tip: add --num-ctx 65536 (or higher) if you have a large enough GPU/RAM."
        )
    elif len(files) == 1 and not _contains(goal, _KW_NEEDS_SEARCH):
        if current_mode == "hybrid":
            tips.append(
                "💡 Single file + hybrid mode: --mode document is faster when you only need "
                "this one file analysed without literature search."
            )

    # Broad overview → retrieval tuning
    if _contains(goal, _KW_BROAD) and current_mode != "search":
        tips.append(
            "💡 Broad overview goal detected.\n"
            "   Tip: --top-k 12 retrieves more context chunks for wide-coverage reports."
        )

    return tips


def recommend_post_research(
    final_state: Dict[str, Any],
    mode: str,
    elapsed: float,
    web: bool,
) -> List[str]:
    """
    Post-run recommendations after a research workflow (Modes 1–3) completes.

    Inspects the final state and suggests follow-up actions.
    """
    tips: List[str] = []

    papers       = final_state.get("academic_papers", [])
    refs         = final_state.get("references", [])
    findings     = final_state.get("key_findings", [])
    errors       = final_state.get("errors", [])
    report       = final_state.get("report", "")
    session_id   = final_state.get("session_id", "")
    per_doc      = final_state.get("per_doc_analyses", {})

    if len(papers) < 3 and not web:
        tips.append(
            "💡 Only a few academic papers found.\n"
            "   Tip: re-run with --web to add Google search results."
        )
    elif len(papers) < 3 and web:
        tips.append(
            "💡 Few papers found even with web search. Try rephrasing:\n"
            "   --goal \"<more specific phrasing>\" or --mode search"
        )

    if not refs:
        tips.append(
            "⚠  No references were cited in the final report.\n"
            "   Try --mode hybrid or --mode search to pull academic sources."
        )
    elif len(refs) < 3:
        tips.append(
            "💡 Only a few references. For a richer literature section:\n"
            "   add --web, or try --mode search with a broader --goal."
        )

    if len(findings) < 2:
        tips.append(
            "💡 Few key findings extracted — the goal may be too broad or no matching papers were found.\n"
            "   Try narrowing the --goal, or upload a focused document with --files."
        )

    if len(per_doc) > 1:
        tips.append(
            f"💡 Per-document breakdown generated for {len(per_doc)} file(s).\n"
            "   Open the Streamlit UI for the full visual breakdown across tabs."
        )

    if errors:
        tips.append(
            f"⚠  {len(errors)} issue(s) during the run (e.g. search timeouts).\n"
            "   Re-run with --verbose to see details."
        )

    if elapsed > 120:
        tips.append(
            "💡 Long run time detected.\n"
            "   Tip: --top-k 4 reduces retrieval overhead; --num-ctx 16384 speeds up LLM calls."
        )

    if report and len(report) < 600:
        tips.append(
            "💡 Short report generated. Add document files with --files for richer analysis."
        )

    if session_id:
        tips.append(
            f"💾 Session saved (ID: {session_id}).\n"
            "   Restore instantly from the Streamlit UI sidebar → 'Recent Research Sessions'."
        )

    return tips


def recommend_proposal_pre(goal: str, instructions: str) -> List[str]:
    """Pre-run recommendations before a proposal is generated (Mode 4)."""
    tips: List[str] = []

    if not instructions:
        tips.append(
            "💡 No --instructions given. Specify tone, length, and focus:\n"
            "   --instructions 'Target 2500 words. Academic tone. Focus on computational methods.'"
        )

    if len(goal.split()) < 6:
        tips.append(
            "💡 Short goal detected — a detailed goal improves all proposal sections.\n"
            "   Example: --goal 'Machine learning for early Alzheimer's detection using MRI data'"
        )

    if not _contains(goal, _KW_NEEDS_SEARCH) and not _contains(goal, _KW_BROAD):
        tips.append(
            "💡 Upload a reference paper with --files to give the proposal writer\n"
            "   domain-specific terminology and context from your own literature."
        )

    return tips


def recommend_proposal_post(final_state: Dict[str, Any]) -> List[str]:
    """Post-run recommendations after a proposal is written (Mode 4)."""
    tips: List[str] = []

    wc          = final_state.get("word_counts", {})
    total_words = sum(wc.values()) if wc else 0
    refs        = final_state.get("selected_references", [])
    session_id  = final_state.get("session_id", "")

    if total_words < 800:
        tips.append(
            "💡 Proposal is short. Revise with a length instruction:\n"
            f"   python main.py --revise {session_id} "
            "--revision 'Expand every section to at least 300 words each.'"
        )
    elif total_words < 1500:
        tips.append(
            "💡 Proposal is medium-length. To expand:\n"
            f"   python main.py --revise {session_id} "
            "--revision 'Add more detail to Methodology and Literature Review.'"
        )

    if len(refs) < 5:
        tips.append(
            "💡 Few references found. For a more grounded proposal:\n"
            "   re-run with a more specific --goal or add --files <reference_paper.pdf>."
        )

    if session_id:
        tips.append(
            f"✏️  To revise: python main.py --revise {session_id} "
            "--revision 'your change instruction'"
        )
        tips.append(
            f"📄 To export: python main.py --export-proposal {session_id}"
        )

    return tips


def recommend_story_turn(
    final_state: Dict[str, Any],
    turn_count: int,
    current_style: str,
    session_id: str,
) -> List[str]:
    """Per-turn recommendations inside the Research Partner chat (Mode 5)."""
    tips: List[str] = []

    concepts = final_state.get("concepts_covered", [])
    questions = final_state.get("suggested_questions", [])
    errors    = final_state.get("errors", [])

    if errors:
        tips.append(f"⚠  {errors[0]}")

    if len(concepts) >= 8:
        tips.append(
            "💡 Many concepts covered. Ask:\n"
            "   'Can you summarise everything we've covered so far?' to consolidate."
        )

    if turn_count >= 5 and current_style == "simple":
        tips.append(
            "💡 You've had several exchanges in Simple mode.\n"
            "   Try restarting with --style walkthrough for a structured step-by-step deep dive."
        )

    if turn_count == 3:
        tips.append(
            "💡 After a few more exchanges, consider:\n"
            "   'How does this connect to [related concept]?' to explore connections."
        )

    if questions:
        tips.append(
            "💡 Type [bold]1[/bold], [bold]2[/bold], or [bold]3[/bold] to ask a suggested question."
        )

    if turn_count >= 10:
        tips.append(
            f"💡 Long session. Return any time with:\n"
            f"   python main.py --story-session {session_id}"
        )

    return tips


def recommend_wisdom_turn(
    final_state: Dict[str, Any],
    phase: str,
    clarification_count: int,
    session_id: str,
) -> List[str]:
    """Per-turn recommendations inside the Wisdom Mode chat (Mode 6)."""
    tips: List[str] = []

    errors = final_state.get("errors", [])

    if errors:
        tips.append(f"⚠  {errors[0]}")

    if phase == "clarifying":
        remaining = max(0, 3 - clarification_count)
        if remaining > 0:
            tips.append(
                f"💡 {remaining} clarification round(s) remaining before wisdom is generated.\n"
                "   Detailed, specific answers produce richer, more personalised wisdom."
            )
        if clarification_count >= 2:
            tips.append(
                "💡 Next response will trigger the wisdom generation pipeline.\n"
                "   Include any additional context (timeframe, constraints, goals) now."
            )

    if phase == "done":
        tips.append(
            "💡 Wisdom generated. You can now ask follow-up questions to go deeper.\n"
            f"   Return any time: python main.py --wisdom-session {session_id}"
        )
        topic_tags = final_state.get("topic_tags", [])
        if topic_tags:
            tag_str = ", ".join(topic_tags[:4])
            tips.append(
                f"💡 Topic tags: [{tag_str}].\n"
                "   Future sessions on overlapping topics will silently benefit from this wisdom."
            )

    return tips


def recommend_startup(goal: str = "", mode: str = "") -> List[str]:
    """
    Shown when no valid mode is selected. Helps the user pick the right command.
    """
    tips: List[str] = []

    if goal:
        if _contains(goal, _KW_NEEDS_DOCS):
            tips.append("💡 Your goal looks like document analysis → python main.py --goal \"...\" --files paper.pdf")
        elif _contains(goal, _KW_NEEDS_SEARCH):
            tips.append("💡 Your goal looks like a literature survey → python main.py --goal \"...\" --mode search")
        else:
            tips.append("💡 Hybrid mode recommended → python main.py --goal \"...\" --mode hybrid")
    else:
        tips.append("💡 Not sure which mode? Run: python main.py --check-system  to see model recommendations.")

    return tips


def print_recommendations(tips: List[str], console, title: str = "💡 Recommendations") -> None:
    """Render a list of tip strings as a Rich panel. No-op if list is empty."""
    if not tips:
        return
    from rich.panel import Panel
    from rich.text import Text

    body = "\n\n".join(tips)
    console.print(
        Panel(body, title=title, border_style="yellow", padding=(0, 1))
    )
