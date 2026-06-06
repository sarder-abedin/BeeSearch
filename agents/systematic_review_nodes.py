"""
agents/systematic_review_nodes.py
───────────────────────────────────
Nodes for Mode 7 — Simplified PRISMA Systematic Review.

Graph structure:
  START → query_generation → literature_search → screening →
          evidence_extraction → synthesis → evaluation → END

Node roles:
  query_generation   — Generate 3–5 academic search queries from the research question
  literature_search  — Run AcademicSearcher across all queries, deduplicate
  screening          — Screen papers by title/abstract against inclusion/exclusion criteria
  evidence_extraction— For each included paper extract study design, key finding, quality score
  synthesis          — PRISMA flow table, narrative synthesis, themes, gaps, conclusion
  evaluation         — Quality self-evaluation via eval_nodes pattern
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.systematic_review_state import SystematicReviewState
from config.settings import get_settings
from tools.search_tools import AcademicSearcher

logger = logging.getLogger(__name__)
cfg = get_settings()

_searcher: AcademicSearcher | None = None


def _get_searcher() -> AcademicSearcher:
    global _searcher
    if _searcher is None:
        _searcher = AcademicSearcher()
    return _searcher


def _llm(state: SystematicReviewState, temperature: float = 0.2, num_predict: int = 4096) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=state.get("model_name", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=state.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(300.0)},
    )


def _call(llm: ChatOllama, system: str, human: str) -> str:
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content.strip()


# ── Node 1: Query Generation ───────────────────────────────────────────────────

def query_generation_node(state: SystematicReviewState) -> Dict[str, Any]:
    logger.info("[SR Node 1] Query Generation")
    llm = _llm(state, temperature=0.1, num_predict=512)
    rq = state.get("research_question", "")
    inclusion = state.get("inclusion_criteria", [])
    exclusion = state.get("exclusion_criteria", [])

    criteria_block = ""
    if inclusion:
        criteria_block += f"\nInclusion: {'; '.join(inclusion)}"
    if exclusion:
        criteria_block += f"\nExclusion: {'; '.join(exclusion)}"

    raw = _call(
        llm,
        """You are an academic librarian expert in systematic review search strategy (PRISMA).
Generate 4–6 distinct search queries optimised for arXiv and Semantic Scholar.

Rules:
- Cover different facets: population/phenomenon, intervention/exposure, outcome, methodology, theory
- Use MeSH-style or field-specific controlled vocabulary where applicable
- Vary specificity: some broad (2-3 key terms), some narrow (4-6 specific terms)
- Include synonyms across queries (not within one query) to maximise recall
- Add "systematic review" or "meta-analysis" to at least one query to find prior reviews
- Respect the inclusion/exclusion criteria when selecting terms

Return ONLY a valid JSON array of strings. No explanation.""",
        f"Research question: {rq}{criteria_block}",
    )
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        queries = json.loads(match.group(0)) if match else [rq]
    except Exception:
        queries = [rq, f"{rq} systematic review", f"{rq} meta-analysis"]

    return {
        "search_queries": queries[:6],
        "current_step": "query_generation",
        "completed_steps": state.get("completed_steps", []) + ["query_generation"],
        "progress_pct": 10,
        "status_detail": f"Generated {len(queries[:6])} search queries",
    }


# ── Node 2: Literature Search ──────────────────────────────────────────────────

def literature_search_node(state: SystematicReviewState) -> Dict[str, Any]:
    logger.info("[SR Node 2] Literature Search")
    queries = state.get("search_queries", [state.get("research_question", "")])
    searcher = _get_searcher()

    all_papers: List[Dict] = []
    seen_titles: set = set()

    for query in queries:
        try:
            papers = searcher.search(query, max_per_source=8)
            for p in papers:
                key = re.sub(r"\W+", "", p.title.lower())[:60]
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    all_papers.append({
                        "title": p.title,
                        "authors": p.authors[:4],
                        "year": p.year,
                        "abstract": p.abstract[:600],
                        "url": p.url,
                        "doi": p.doi,
                        "journal": p.journal or p.venue,
                        "source": p.source,
                        "citation_key": p.citation_key,
                        "citation_count": p.citation_count,
                    })
        except Exception as e:
            logger.warning("Search failed for '%s': %s", query[:50], e)

    all_papers.sort(key=lambda p: -(p.get("citation_count") or 0))

    # Pre-screening relevance filter — reduces screening_node workload
    papers_before_grading = len(all_papers)
    if all_papers:
        from agents.self_reflective_rag import grade_papers
        paper_dicts = [{"title": p.get("title", ""), "abstract": p.get("abstract", "")} for p in all_papers]
        grades = grade_papers(
            paper_dicts,
            query=state.get("research_question", ""),
            model_name=state.get("model_name", ""),
            num_ctx=state.get("num_ctx", 4096),
        )
        filtered = [p for p, g in zip(all_papers, grades) if g]
        all_papers = filtered or all_papers  # fallback: keep all if none pass

    # Deduplicate citation keys — two "Smith et al., 2022" → "Smith et al., 2022a", "...2022b"
    seen_ckeys: dict[str, int] = {}
    for paper in all_papers:
        base_ck = paper["citation_key"]
        count = seen_ckeys.get(base_ck, 0)
        seen_ckeys[base_ck] = count + 1
        if count > 0:
            paper["citation_key"] = f"{base_ck}{chr(ord('a') + count - 1)}"

    logger.info("[SR] Found %d unique papers across %d queries", len(all_papers), len(queries))

    # Abstract screening — score all papers before LLM criteria screening
    screener_scores: List[Dict] = []
    try:
        from tools.abstract_screener import screen_abstracts
        rq = state.get("research_question", "")
        screener_scores = screen_abstracts(
            papers=all_papers,
            research_question=rq,
            inclusion_criteria=state.get("inclusion_criteria", []),
            exclusion_criteria=state.get("exclusion_criteria", []),
            model_name=state.get("model_name", cfg.ollama_model),
            num_ctx=state.get("num_ctx", 4096),
        )
        # Re-order all_papers to match screener ranking (highest score first)
        score_map = {r["paper"].get("title", ""): r["score"] for r in screener_scores}
        all_papers.sort(key=lambda p: -score_map.get(p.get("title", ""), 50))
    except Exception as e:
        logger.warning("Abstract screener failed (continuing without scores): %s", e)

    return {
        "raw_papers": all_papers,
        "screener_scores": screener_scores,
        "rag_reflection_info": {
            "papers_retrieved": papers_before_grading,
            "papers_after_grading": len(all_papers),
        },
        "current_step": "literature_search",
        "completed_steps": state.get("completed_steps", []) + ["literature_search"],
        "progress_pct": 30,
        "status_detail": (
            f"Found {len(all_papers)} papers via Google Scholar · arXiv · Semantic Scholar · CrossRef"
        ),
    }


# ── Node 3: Screening ──────────────────────────────────────────────────────────

def screening_node(state: SystematicReviewState) -> Dict[str, Any]:
    """Screen papers by title/abstract against inclusion and exclusion criteria."""
    logger.info("[SR Node 3] Screening")
    llm = _llm(state, temperature=0.1, num_predict=128)

    raw_papers = state.get("raw_papers", [])
    rq = state.get("research_question", "")
    inclusion = state.get("inclusion_criteria", [])
    exclusion = state.get("exclusion_criteria", [])

    if not raw_papers:
        return {
            "screened_papers": [], "included_papers": [], "excluded_papers": [],
            "current_step": "screening",
            "completed_steps": state.get("completed_steps", []) + ["screening"],
            "progress_pct": 50,
            "status_detail": "No papers to screen",
        }

    inc_block = "; ".join(inclusion) if inclusion else "Relevant to the research question"
    exc_block = "; ".join(exclusion) if exclusion else "Clearly off-topic papers"

    # Screen in batches of 10
    screened: List[Dict] = []
    excluded: List[Dict] = []

    for paper in raw_papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")[:600]
        raw = _call(
            llm,
            f"""You screen papers for a systematic review.
Research question: {rq}
Inclusion criteria: {inc_block}
Exclusion criteria: {exc_block}

For each paper, decide: INCLUDE or EXCLUDE.
Return JSON: {{"decision": "INCLUDE"|"EXCLUDE", "reason": "one sentence"}}
Return ONLY valid JSON.""",
            f"Title: {title}\nAbstract: {abstract}",
        )
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(match.group(0)) if match else {"decision": "INCLUDE", "reason": ""}
        except Exception:
            result = {"decision": "INCLUDE", "reason": ""}

        if result.get("decision", "INCLUDE").upper() == "INCLUDE":
            screened.append(paper)
        else:
            excluded.append({**paper, "exclusion_reason": result.get("reason", "")})

    logger.info("[SR] Screened: %d included, %d excluded from %d total", len(screened), len(excluded), len(raw_papers))

    return {
        "screened_papers": screened,
        "included_papers": screened,  # same set after this phase; evidence_extraction may further filter
        "excluded_papers": excluded,
        "current_step": "screening",
        "completed_steps": state.get("completed_steps", []) + ["screening"],
        "progress_pct": 55,
        "status_detail": f"Screened {len(raw_papers)} → {len(screened)} included · {len(excluded)} excluded",
    }


# ── Node 4: Evidence Extraction ────────────────────────────────────────────────

def evidence_extraction_node(state: SystematicReviewState) -> Dict[str, Any]:
    """Extract structured evidence from each included paper."""
    logger.info("[SR Node 4] Evidence Extraction")
    llm = _llm(state, temperature=0.1, num_predict=256)

    included = state.get("included_papers", [])
    rq = state.get("research_question", "")
    evidence_table: List[Dict] = []

    for paper in included[:20]:  # cap at 20 papers
        raw = _call(
            llm,
            f"""Extract evidence from this paper for a systematic review on: {rq}
Return JSON with EXACTLY these keys:
{{"study_design": "e.g. RCT|Cohort|Cross-sectional|Meta-analysis|Review|Other", "sample_size": "N or unknown", "key_finding": "one sentence", "quality": "High|Medium|Low", "relevance_score": 1-5}}
Return ONLY valid JSON.""",
            f"Title: {paper.get('title','')}\nAbstract: {paper.get('abstract','')[:500]}",
        )
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            evidence = json.loads(match.group(0)) if match else {}
        except Exception as e:
            logger.warning("Evidence extraction failed for '%s': %s", paper.get("title","")[:50], e)
            evidence = {}

        evidence_table.append({
            "title": paper.get("title", ""),
            "authors": paper.get("authors", [])[:2],
            "year": paper.get("year"),
            "citation_key": paper.get("citation_key", ""),
            "url": paper.get("url", ""),
            "doi": paper.get("doi"),
            "journal": paper.get("journal"),
            "study_design": evidence.get("study_design", "Unknown"),
            "sample_size": evidence.get("sample_size", "Unknown"),
            "key_finding": evidence.get("key_finding", ""),
            "quality": evidence.get("quality", "Medium"),
            "relevance_score": evidence.get("relevance_score", 3),
        })

    # Sort by quality and relevance
    quality_rank = {"High": 0, "Medium": 1, "Low": 2}
    evidence_table.sort(key=lambda x: (quality_rank.get(x.get("quality","Medium"), 1), -x.get("relevance_score", 3)))

    return {
        "evidence_table": evidence_table,
        "current_step": "evidence_extraction",
        "completed_steps": state.get("completed_steps", []) + ["evidence_extraction"],
        "progress_pct": 70,
        "status_detail": f"Extracted evidence from {len(evidence_table)} papers",
    }


# ── Node 5: Synthesis ──────────────────────────────────────────────────────────

def synthesis_node(state: SystematicReviewState) -> Dict[str, Any]:
    """Generate PRISMA flow, narrative synthesis, themes, gaps, and conclusion."""
    logger.info("[SR Node 5] Synthesis")
    llm = _llm(state, temperature=0.3, num_predict=6000)

    rq = state.get("research_question", "")
    evidence_table = state.get("evidence_table", [])
    raw_papers = state.get("raw_papers", [])
    screened_papers = state.get("screened_papers", [])
    included_papers = state.get("included_papers", [])
    excluded_papers = state.get("excluded_papers", [])

    prisma_flow = {
        "identified": len(raw_papers),
        "screened": len(raw_papers),
        "eligibility": len(screened_papers),
        "included": len(included_papers),
        "excluded": len(excluded_papers),
    }

    if not evidence_table:
        # Build a diagnostic summary of why papers were excluded
        exclusion_reasons: dict = {}
        for p in excluded_papers[:10]:
            reason = p.get("exclusion_reason", "no reason recorded")
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        reason_summary = "; ".join(f"{r} (×{n})" for r, n in sorted(exclusion_reasons.items(), key=lambda x: -x[1])[:5])

        return {
            "prisma_flow": prisma_flow,
            "narrative_synthesis": (
                f"No papers met the inclusion criteria after screening "
                f"({len(raw_papers)} identified, {len(excluded_papers)} excluded). "
                f"Top exclusion reasons: {reason_summary or 'none recorded'}. "
                f"Consider: (1) broadening the research question, "
                f"(2) relaxing inclusion criteria (e.g. publication date range or study type), "
                f"or (3) adding alternative search terms."
            ),
            "key_themes": [],
            "research_gaps": [
                "No eligible studies found — the search may need broader scope.",
                "Consider alternative databases (PubMed, Cochrane) for clinical topics.",
            ],
            "limitations": (
                f"This review identified {len(raw_papers)} papers but none passed screening. "
                "Results should be interpreted as a search scope problem, not evidence of absence."
            ),
            "conclusion": (
                "The systematic review could not synthesise findings because no studies "
                "passed the inclusion/exclusion screening. "
                f"Top exclusion reasons were: {reason_summary or 'not recorded'}. "
                "Revise the search terms or criteria and retry."
            ),
            "current_step": "synthesis",
            "completed_steps": state.get("completed_steps", []) + ["synthesis"],
            "progress_pct": 90,
            "status_detail": f"Synthesised {len(state.get('included_papers', []))} papers into narrative review",
        }

    evidence_text = "\n".join(
        f"[{e['citation_key']}] {e['title']} ({e.get('year','n.d.')}) — {e.get('study_design','')} — "
        f"Quality: {e.get('quality','')} — Finding: {e.get('key_finding','')}"
        for e in evidence_table[:15]
    )

    # ── Call 1: narrative synthesis as plain text (avoids JSON-in-JSON issues) ──
    # Small LLMs consistently fail when asked to embed 600-word essays inside JSON.
    # Generating plain text first, then structured fields separately, is far more robust.
    narrative_synthesis = _call(
        llm,
        f"""You are writing the narrative synthesis section of a systematic review.
Research question: {rq}

Write 6-8 paragraphs (700-1100 words) synthesising the findings across the included papers.
- Use inline citations like [citation_key] after each claim.
- Discuss convergence and contradictions in the evidence.
- Comment on strength and quality of evidence.
- Write in formal academic prose. No bullet points, no headings.""",
        f"EVIDENCE SUMMARY:\n{evidence_text}",
    )

    # ── Call 2: structured fields as compact JSON ──────────────────────────────
    llm_structured = _llm(state, temperature=0.1, num_predict=1024)
    raw_structured = _call(
        llm_structured,
        """Return a JSON object with EXACTLY these keys (no other text):
{"key_themes":["theme1","theme2"],"research_gaps":["gap1","gap2"],"limitations":"3-4 sentences","conclusion":"4-6 sentences"}""",
        f"Research question: {rq}\n\nEvidence summary:\n{evidence_text[:1800]}",
    )

    try:
        match = re.search(r"\{.*\}", raw_structured, re.DOTALL)
        structured = json.loads(match.group(0)) if match else {}
    except Exception as e:
        logger.warning("Structured fields JSON parse failed: %s", e)
        structured = {}

    return {
        "prisma_flow": prisma_flow,
        "narrative_synthesis": narrative_synthesis,
        "key_themes": structured.get("key_themes", []),
        "research_gaps": structured.get("research_gaps", []),
        "limitations": structured.get("limitations", ""),
        "conclusion": structured.get("conclusion", ""),
        "current_step": "synthesis",
        "completed_steps": state.get("completed_steps", []) + ["synthesis"],
        "progress_pct": 90,
        "status_detail": f"Synthesised {len(state.get('included_papers', []))} papers into narrative review",
    }


# ── Node 6: Evaluation ─────────────────────────────────────────────────────────

def sr_eval_node(state: SystematicReviewState) -> Dict[str, Any]:
    """Quality self-evaluation for the systematic review."""
    logger.info("[SR Node 6] Evaluation")
    llm = _llm(state, temperature=0.1, num_predict=512)

    synthesis = state.get("narrative_synthesis", "")
    n_included = len(state.get("included_papers", []))
    n_raw = len(state.get("raw_papers", []))
    n_excluded = len(state.get("excluded_papers", []))
    rq = state.get("research_question", "")
    evidence_table = state.get("evidence_table", [])
    key_themes = state.get("key_themes", [])
    research_gaps = state.get("research_gaps", [])

    # Summarise evidence quality distribution for the evaluator
    quality_dist: Dict[str, int] = {}
    for e in evidence_table:
        q = e.get("quality", "Unknown")
        quality_dist[q] = quality_dist.get(q, 0) + 1

    study_designs = list({e.get("study_design", "Unknown") for e in evidence_table})

    eval_context = (
        f"Research question: {rq}\n"
        f"PRISMA flow: {n_raw} identified → {n_included} included, {n_excluded} excluded\n"
        f"Evidence quality distribution: {quality_dist}\n"
        f"Study designs represented: {', '.join(study_designs[:8]) or 'none'}\n"
        f"Key themes identified: {len(key_themes)}\n"
        f"Research gaps identified: {len(research_gaps)}\n"
        f"Synthesis excerpt (first 2000 chars):\n{synthesis[:2000]}"
    )

    raw = _call(
        llm,
        """Rate this systematic review on 5 dimensions (1–5 integer scale).
Score honestly based on the evidence provided — do not default to mid-range scores.
Return ONLY valid JSON:
{"search_comprehensiveness": <1-5>, "screening_rigor": <1-5>, "evidence_quality": <1-5>, "synthesis_depth": <1-5>, "gap_identification": <1-5>, "summary": "one sentence overall assessment"}""",
        eval_context,
    )

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        eval_data = json.loads(match.group(0)) if match else {}
    except Exception:
        eval_data = {}

    return {
        "eval_result": eval_data,
        "current_step": "sr_eval",
        "completed_steps": state.get("completed_steps", []) + ["sr_eval"],
        "progress_pct": 100,
        "status_detail": f"Review quality scored · {len(state.get('included_papers', []))} papers included",
    }
