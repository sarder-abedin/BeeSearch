"""backend/app/schemas/notebook_explain.py
─────────────────────────────────────────────
Pydantic request/response shapes for Mode 2 Phase D: the Explain tab
(agents/story_*.py's storyteller pipeline, internal "Mode 5").

Mirrors backend/app/schemas/notebook.py's chat shapes, with two deliberate
divergences driven by StoryState's own shape (agents/story_state.py):

  - ExplainCitationItem.n is a *str*, not an int -- agents/story_nodes.py::
    _build_citations_list emits document-excerpt citations with an int n
    and online-source citations with a string "Source N" n into the same
    list, so a single shared field has to be the (stringified) union of
    both -- the frontend only ever displays "[{n}]", never does arithmetic
    on it.
  - ExplainTurn carries an explanation_style (StorytellerMemory.add_turn's
    own per-turn field) that ConversationTurn has no equivalent for, plus
    an online-sources block (source_decision / online_results) Phase A's
    chat has no equivalent of.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .jobs import JobStatusBase

TemperatureLevel = Literal["precise", "focused", "balanced", "creative"]
ExplanationStyle = Literal["simple", "analogy", "walkthrough", "debate"]
ExplanationLevel = Literal["novice", "intermediate", "expert"]


class ExplainCitationItem(BaseModel):
    n: str
    doc_name: str = "unknown"
    page: Optional[int] = None
    page_label: str = "n/a"
    snippet: str = ""
    url: str = ""


class OnlineResultItem(BaseModel):
    type: str
    title: str = ""
    authors: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""
    year: Optional[int] = None
    apa: str = ""


class SourceDecision(BaseModel):
    coverage_score: int = 0
    used_docs: bool = False
    used_online: bool = False
    search_attempted: bool = False
    reason: str = ""
    sources_searched: List[str] = Field(default_factory=list)
    online_count: int = 0


class ExplainTurn(BaseModel):
    role: str
    content: str
    timestamp: str = ""
    citations: Optional[List[ExplainCitationItem]] = None
    suggested_questions: Optional[List[str]] = None
    explanation_style: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Turn (background job + polling, same pattern as Mode 1 / Mode 3 / Phase A)
# ─────────────────────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    notebook_id: str = Field(..., description="Target notebook id, returned by /notebooks (POST).")
    message: str = Field(..., description="The user's question for this turn.")
    explanation_style: ExplanationStyle = Field(
        "simple",
        description="Requested explanation style; overridden if this turn is a detected "
        "repeat of the last question and this style matches the one just used.",
    )
    explanation_level: ExplanationLevel = Field("intermediate", description="Target audience level.")
    model: Optional[str] = Field(None, description="Ollama model override; omit to use the server's configured default.")
    num_ctx: Optional[int] = Field(None, gt=0, description="Context window override (tokens).")
    temperature_level: Optional[TemperatureLevel] = Field(
        None, description="Response tuning level; omit to use the module default ('focused')."
    )

    @field_validator("notebook_id")
    @classmethod
    def notebook_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("notebook_id is required.")
        return v

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Please enter a question.")
        return v


class ExplainResult(BaseModel):
    notebook_id: str
    user_message: str = ""
    assistant_response: str = ""
    explanation_style: str = ""
    citations: List[ExplainCitationItem] = Field(default_factory=list)
    suggested_questions: List[str] = Field(default_factory=list)
    is_repeat_clarification: bool = False
    repeated_question: str = ""
    new_concepts: List[str] = Field(default_factory=list)
    concept_visual_html: str = ""
    source_decision: Optional[SourceDecision] = None
    online_results: List[OnlineResultItem] = Field(default_factory=list)
    eval_result: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    progress_pct: int = 100


class ExplainJobStatus(JobStatusBase):
    result: Optional[ExplainResult] = None
