"""backend/app/schemas/notebook_report.py
───────────────────────────────────────────
Pydantic request/response shapes for Mode 2 Phase E: the Research Report
workflow (``agents/graph.py`` + ``agents/state.py``).

``ReportResult`` mirrors the "outputs" section of the state dict built by
``agents.state.create_initial_state`` (``report``, ``key_findings``,
``references``, ``eval_result``, ``errors``, ``progress_pct``) plus
``web_search_status``, which -- although populated alongside the
"intermediate" fields -- is what the Streamlit tab actually reads to decide
whether to show its "web search found nothing / failed" warning. The
purely-intermediate fields (``search_queries``, ``academic_papers``,
``web_results``) are never rendered by the Streamlit tab either -- their
information already reaches the response fully formed via ``references``
-- so, matching existing behaviour exactly, this schema omits them rather
than inventing new surface area.

``mode`` ("document" | "hybrid" | "search") is never a *request* field: the
Streamlit tab derives it itself from the notebook's current sources plus
the academic-search toggle (see ``ui/tabs/notebook.py::_tab_research_report``),
re-checked fresh on every run rather than trusted from client input. The
service layer mirrors that derivation; ``mode`` is echoed back on
``ReportResult`` purely for transparency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .jobs import JobStatusBase


class ReportRequest(BaseModel):
    notebook_id: str = Field(..., description="Target notebook id, returned by /notebooks (POST).")
    goal: str = Field(..., description="The research goal or question to write the report about.")
    include_academic: bool = Field(
        True, description="Search arXiv + Semantic Scholar for peer-reviewed papers."
    )
    include_web: bool = Field(False, description="Also search the web via DuckDuckGo.")
    model: Optional[str] = Field(None, description="Ollama model override; omit to use the server's configured default.")
    num_ctx: Optional[int] = Field(None, gt=0, description="Context window override (tokens).")
    embed_model: Optional[str] = Field(
        None, description="Embedding model override (reserved for future use); omit to use the server default."
    )

    @field_validator("notebook_id")
    @classmethod
    def notebook_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("notebook_id is required.")
        return v

    @field_validator("goal")
    @classmethod
    def goal_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Please enter a research goal.")
        return v


class ReportReference(BaseModel):
    ref_num: int
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    journal: str = ""
    year: str = ""
    doi: str = ""
    url: str = ""
    abstract_snippet: str = ""
    source: str = ""
    citation_count: Optional[int] = None
    apa: str = ""


class ReportResult(BaseModel):
    notebook_id: str = ""
    goal: str = ""
    mode: str = ""
    report: str = ""
    key_findings: List[str] = Field(default_factory=list)
    references: List[ReportReference] = Field(default_factory=list)
    web_search_status: str = "disabled"
    eval_result: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    progress_pct: int = 100


class ReportJobStatus(JobStatusBase):
    result: Optional[ReportResult] = None
