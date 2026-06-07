"""
ui/glossary.py
──────────────
Plain-language definitions of research / RAG / statistics jargon.

Used two ways:
  1. `term_help(term)` — feeds Streamlit's `help=` tooltip parameter so any
     widget or header can carry an inline "(?)" hint without changing its
     visible label (experts see the same UI; newcomers get an explanation
     on hover).
  2. `render_glossary_expander(...)` — an optional "Jargon buster" expander
     that lists curated definitions for a whole tab in plain language.

Keep entries short (1–2 sentences): they render inside small tooltip
popovers as well as the expander.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

GLOSSARY: Dict[str, str] = {
    "PRISMA": (
        "A standard checklist and flow-diagram format for systematic reviews. "
        "It records exactly how many papers were found, screened, included and "
        "excluded — and why — so the review is transparent and repeatable."
    ),
    "Systematic review": (
        "A structured literature review that follows an explicit, repeatable "
        "recipe (search → screen → extract → synthesise) instead of informally "
        "reading whatever turns up."
    ),
    "Inclusion / exclusion criteria": (
        "The rules that decide which papers make it into your review "
        "(e.g. 'human studies only') and which get filtered out "
        "(e.g. 'animal studies', 'conference abstracts only')."
    ),
    "RAG (Retrieval-Augmented Generation)": (
        "The AI first retrieves the most relevant passages from your documents, "
        "then writes its answer grounded in those passages — instead of relying "
        "only on what it memorised during training. This is how ResearchBuddy "
        "keeps answers tied to your sources."
    ),
    "Hybrid retrieval": (
        "Combines two search styles — meaning-based vector search (FAISS) and "
        "exact-keyword search (BM25) — then merges the results, so you get both "
        "conceptual matches and precise term hits."
    ),
    "Embedding model": (
        "Converts text into a list of numbers that captures its meaning, so "
        "passages with similar meaning end up with similar numbers and can be "
        "found through semantic search."
    ),
    "Context window": (
        "How much text (measured in tokens) the AI can consider at once. A "
        "larger window lets it weigh more of your documents per answer, but "
        "needs more memory and runs slower."
    ),
    "Quality score": (
        "A 1–5 rating the AI gives an answer or paper, judging things like "
        "relevance and evidence support — a quick sanity check, not a "
        "replacement for your own judgement."
    ),
    "Faithfulness": (
        "The share of claims in an AI-written answer that are actually backed "
        "up by the retrieved source passages — a guard against the model "
        "stating things that aren't really in your documents."
    ),
    "Effect size": (
        "A single number capturing how big a difference or relationship a "
        "study found (e.g. how much a treatment improved an outcome), which "
        "lets results from different studies be compared and combined."
    ),
    "Heterogeneity (I²)": (
        "A 0–100% score for how much studies' results disagree beyond what "
        "chance alone would explain. Rough guide: under 25% = studies broadly "
        "agree, over 75% = they vary a lot and pooled results need extra care."
    ),
    "Forest plot": (
        "The standard meta-analysis chart: each study's effect size is drawn "
        "as a square (sized by how heavily it's weighted) with a line showing "
        "its confidence interval, and a diamond at the bottom shows the "
        "combined 'pooled' result across all studies."
    ),
    "Pooled effect / meta-analysis": (
        "Statistically combining results from several studies into one overall "
        "estimate, weighting more precise / larger studies more heavily."
    ),
    "Fixed-effect vs. random-effects model": (
        "Two ways to combine studies. Fixed-effect assumes every study "
        "estimates the same true effect (best when studies are very alike); "
        "random-effects assumes the true effect varies study to study — the "
        "safer default when heterogeneity is high."
    ),
    "Citation network": (
        "A graph of how the included papers cite one another — useful for "
        "spotting influential works and clusters of closely related research."
    ),
    "Concept drift": (
        "How a field's dominant topics and vocabulary shift over time — "
        "useful for checking whether older 'foundational' papers still use "
        "today's terminology."
    ),
    "Evidence map": (
        "A bubble chart that plots included studies by theme, population or "
        "outcome, so you can see at a glance where evidence is dense and where "
        "it's thin."
    ),
    "Chunking": (
        "Documents are split into overlapping passages ('chunks') before "
        "indexing, so retrieval can return the precise passage that answers "
        "your question rather than an entire file."
    ),
    "Docling / OCR": (
        "Docling is the advanced parser that preserves layout and tables from "
        "PDFs/Office files; OCR additionally reads text out of scanned, "
        "image-only pages. Both are slower and more memory-hungry than plain "
        "text extraction, which is why large PDFs can auto-switch to a faster "
        "parser."
    ),
    "Hardware tier": (
        "ResearchBuddy's auto-detected performance profile (Low / Standard / "
        "High / Maximum) based on your RAM, used to pre-select sensible "
        "defaults for model size, context window, and document chunking."
    ),
}


def term_help(term: str) -> Optional[str]:
    """Return the plain-language definition for `term`, for use as `help=`."""
    return GLOSSARY.get(term)


def render_glossary_expander(
    terms: Optional[List[str]] = None,
    *,
    title: str = "Jargon buster — what do these terms mean?",
    expanded: bool = False,
) -> None:
    """
    Render an expander of plain-language glossary definitions.

    Pass `terms` to show a curated subset in that order (e.g. only the terms
    relevant to the current tab); omit it to list every known term A–Z.
    Unknown terms are silently skipped so callers can pass optimistic lists.
    """
    selected = [t for t in terms if t in GLOSSARY] if terms is not None else sorted(GLOSSARY)
    if not selected:
        return
    with st.expander(title, expanded=expanded):
        st.caption("Plain-language explanations — hover the (?) icons next to labels for quick hints too.")
        for term in selected:
            st.markdown(f"**{term}** — {GLOSSARY[term]}")
