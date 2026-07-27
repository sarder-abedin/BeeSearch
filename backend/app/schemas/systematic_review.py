"""backend/app/schemas/systematic_review.py
───────────────────────────────────────────────
Pydantic request/response shapes for Mode 1 (Systematic Literature Review).

Mirrors ``agents.systematic_review_state.create_systematic_review_state``'s
input contract (the same knobs the CLI ``--systematic-review`` flags and the
Streamlit inputs in ``ui/tabs/systematic_review.py`` already drive) and
``agents.systematic_review_graph.run_systematic_review``'s output contract
(the same ``final_state`` fields ``ui/tabs/systematic_review.py`` already
renders across its Synthesis/Evidence/Explore/Write-up & Export tabs).

Fields with a fixed shape confirmed directly from ``agents/systematic_review_
nodes.py`` (papers, evidence rows, PRISMA counts) get strongly-typed
sub-models. Quality-assessment fields (``rob_table``, ``grade_results``,
``contradictions``, ``screener_scores``, ``eval_result``, and the Explore
tools' own outputs) stay as loose ``Dict[str, Any]``/``List[Dict[str, Any]]``
on purpose: the underlying Python code itself has no fixed schema for these
(RoB domain keys vary RCT vs. observational; the UI renders them via generic
``.items()``), so a strict schema here would be false precision that breaks
the moment a tool module's output shape shifts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .jobs import JobStatusBase

ExploreTool = Literal[
    "citation_network",
    "citation_context",
    "reference_checking",
    "preprint_status",
    "research_trends",
    "evidence_map",
    "meta_analysis",
    "concept_drift",
]


class SRRequest(BaseModel):
    research_question: str = Field(..., description="PICO-style research question for the review.")
    inclusion_criteria: List[str] = Field(default_factory=list, description="One criterion per item.")
    exclusion_criteria: List[str] = Field(default_factory=list, description="One criterion per item.")
    model: Optional[str] = Field(None, description="Ollama model override; omit to use the server's configured default.")
    num_ctx: Optional[int] = Field(None, gt=0, description="Context window override; omit to use the server's configured default.")
    max_results: Optional[int] = Field(None, gt=0, description="Max papers per source per query; omit to use the pipeline default (8).")
    include_crossref: Optional[bool] = Field(None, description="Also search CrossRef; omit to use the pipeline default (True).")

    @field_validator("research_question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Please enter a research question.")
        return v


class PaperLite(BaseModel):
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    url: str = ""
    doi: Optional[str] = None
    journal: Optional[str] = None
    source: str = ""
    citation_key: str = ""
    citation_count: Optional[int] = None


class ExcludedPaper(PaperLite):
    exclusion_reason: str = ""


class EvidenceRow(BaseModel):
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    citation_key: str = ""
    url: str = ""
    doi: Optional[str] = None
    journal: Optional[str] = None
    abstract: str = ""
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcome: str = ""
    study_design: str = "Unknown"
    sample_size: str = "Unknown"
    key_finding: str = ""
    quality: str = "Medium"
    relevance_score: int = 3

    @field_validator("sample_size", mode="before")
    @classmethod
    def _stringify_sample_size(cls, v: Any) -> str:
        # The LLM is asked for a string ("N or unknown") but occasionally
        # returns a bare JSON number -- coerce rather than 500 the request.
        return "Unknown" if v is None else str(v)

    @field_validator("relevance_score", mode="before")
    @classmethod
    def _coerce_relevance_score(cls, v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 3


class PrismaFlow(BaseModel):
    identified: int = 0
    screened: int = 0
    eligibility: int = 0
    included: int = 0
    excluded: int = 0


class SRResult(BaseModel):
    session_id: str = ""
    research_question: str = ""
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    model_name: str = ""
    num_ctx: int = 0

    raw_papers: List[PaperLite] = Field(default_factory=list)
    screened_papers: List[PaperLite] = Field(default_factory=list)
    included_papers: List[PaperLite] = Field(default_factory=list)
    excluded_papers: List[ExcludedPaper] = Field(default_factory=list)

    prisma_flow: PrismaFlow = Field(default_factory=PrismaFlow)
    evidence_table: List[EvidenceRow] = Field(default_factory=list)
    narrative_synthesis: str = ""
    key_themes: List[str] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)
    limitations: str = ""
    conclusion: str = ""

    eval_result: Dict[str, Any] = Field(default_factory=dict)
    rag_reflection_info: Dict[str, Any] = Field(default_factory=dict)

    rob_table: List[Dict[str, Any]] = Field(default_factory=list)
    grade_results: Dict[str, Any] = Field(default_factory=dict)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)

    screener_scores: List[Dict[str, Any]] = Field(default_factory=list)
    preprint_tracking: List[Dict[str, Any]] = Field(default_factory=list)
    citation_graph_html: str = ""

    trend_data: Dict[str, Any] = Field(default_factory=dict)
    evidence_map_data: Dict[str, Any] = Field(default_factory=dict)
    concept_drift_data: Dict[str, Any] = Field(default_factory=dict)

    errors: List[str] = Field(default_factory=list)
    progress_pct: int = 0


class SRJobStatus(JobStatusBase):
    result: Optional[SRResult] = None


class ToolJobStatus(JobStatusBase):
    """Generic poll response shared by all 8 Explore-tool background jobs."""
    result: Optional[Dict[str, Any]] = None


class ExploreToolRequest(BaseModel):
    """Body for ``POST /jobs/{job_id}/explore/{tool}``. ``options`` is a generic
    bag of per-tool knobs (e.g. ``{"classify_stances": true}`` for citation_network)
    -- kept loose rather than 8 bespoke request models, matching the single generic
    trigger endpoint shared by all Explore tools."""
    options: Dict[str, Any] = Field(default_factory=dict)


class EvidenceMapResponse(BaseModel):
    map_data: Dict[str, Any]
    html: Optional[str] = None


class MetaAnalysisRow(BaseModel):
    citation_key: str = ""
    label: str = ""
    effect: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n: Optional[int] = None


class MetaAnalysisSeedResponse(BaseModel):
    rows: List[MetaAnalysisRow]
    measure_labels: Dict[str, str]


class MetaAnalysisPoolRequest(BaseModel):
    rows: List[MetaAnalysisRow]
    measure: str = "OR"
    model: Literal["fixed", "random"] = "random"


class MetaAnalysisPoolResponse(BaseModel):
    result: Dict[str, Any]
    forest_html: Optional[str] = None


class ExportRequest(BaseModel):
    author: str = ""
    institution: str = ""


class MetaAnalysisDraftRequest(BaseModel):
    """Body for ``POST /jobs/{job_id}/meta-analysis/draft`` -- the LLM-assisted
    "draft effect sizes from abstracts" step, run as a background job since
    it calls the LLM once per row."""
    rows: List[MetaAnalysisRow]
    measure: str = "OR"
    model: Optional[str] = None
    num_ctx: Optional[int] = None


class PlainLanguageSummaryRequest(BaseModel):
    format: Literal["patient", "policy", "press", "all"] = "all"
    model: Optional[str] = None
    num_ctx: Optional[int] = None


class PlainLanguageSummaryResponse(BaseModel):
    """Shape of the background job's ``result`` once a plain-language-summary
    job completes -- keys are a subset of {"patient", "policy", "press"}
    depending on the requested ``format``."""
    summaries: Dict[str, str] = Field(default_factory=dict)


class SRTemplate(BaseModel):
    """One guided-template preset (see ``tools/sr_templates.py``)."""
    key: str
    label: str
    description: str
    research_question: str
    inclusion: List[str] = Field(default_factory=list)
    exclusion: List[str] = Field(default_factory=list)
    note: str = ""


class GrammarCheckRequest(BaseModel):
    text: str
    context_hint: str = ""
    model: Optional[str] = None
    num_ctx: Optional[int] = None


class GrammarCheckResponse(BaseModel):
    original: str
    corrected: str
    changed: bool
