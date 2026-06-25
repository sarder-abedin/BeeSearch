"""backend/app/schemas/research_assistant.py
───────────────────────────────────────────────
Pydantic request/response shapes for Mode 3 (AI Research Assistant).

Mirrors ``agents.research_assistant.run_research_assistant``'s exact
input/output contract: the same settings-dict knobs the CLI (``--ask``) and
Streamlit sidebar already drive (model, num_ctx, temperature_level,
include_crossref), and the same result-dict shape the Streamlit tab already
renders (question, answer, citations, sources, academic_count, web_count,
suggested_questions, grounded).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .jobs import JobStatusBase

TemperatureLevel = Literal["precise", "focused", "balanced", "creative"]


class AskRequest(BaseModel):
    question: str = Field(
        ..., description="Free-form research question to answer from published literature."
    )
    include_web: bool = Field(
        True,
        description='Also search the web (DuckDuckGo) in addition to academic sources -- '
        'mirrors the Streamlit "Also search the web" checkbox, default checked.',
    )
    include_crossref: bool = Field(
        True, description="Include CrossRef alongside Google Scholar, arXiv, and Semantic Scholar."
    )
    model: Optional[str] = Field(
        None, description="Ollama model override; omit to use the server's configured default model."
    )
    num_ctx: Optional[int] = Field(
        None, gt=0, description="Context window override (tokens); omit to use the server's configured default."
    )
    temperature_level: Optional[TemperatureLevel] = Field(
        None, description="Response tuning level; omit to use the module default ('focused')."
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Please enter a research question.")
        return v


class SourceItem(BaseModel):
    n: int
    kind: Literal["academic", "web"]
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    url: str = ""
    snippet: str = ""
    apa: str = ""
    source: str = ""


class AskResult(BaseModel):
    question: str
    answer: str
    citations: List[SourceItem]
    sources: List[SourceItem]
    academic_count: int
    web_count: int
    suggested_questions: List[str]
    grounded: bool


class AskJobStatus(JobStatusBase):
    result: Optional[AskResult] = None
