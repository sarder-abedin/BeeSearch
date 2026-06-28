"""backend/app/schemas/notebook_pipeline.py
─────────────────────────────────────────────
Pydantic request/response shapes for Mode 2 Phase B: the 7-agent Research
Notebook analysis pipeline (``agents/notebook_pipeline_*.py``).

``PipelineResult`` mirrors every output field of ``NotebookPipelineState``
(``agents/notebook_pipeline_state.py``) so the raw final-state dict returned
by ``run_notebook_pipeline()`` can be handed straight to the response model
and validated/coerced field-by-field -- the same pattern ``ChatResult`` uses
for ``NotebookState`` (see ``schemas/notebook.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .jobs import JobStatusBase

TemperatureLevel = Literal["precise", "focused", "balanced", "creative"]


class PipelineRequest(BaseModel):
    notebook_id: str = Field(..., description="Target notebook id, returned by /notebooks (POST).")
    query: str = Field(
        "",
        description="Optional focus query for Agent 3's retrieval step; "
        "blank uses the pipeline's generic default query.",
    )
    model: Optional[str] = Field(None, description="Ollama model override; omit to use the server's configured default.")
    num_ctx: Optional[int] = Field(None, gt=0, description="Context window override (tokens).")
    embed_model: Optional[str] = Field(
        None, description="Embedding model override for hybrid retrieval; omit to use the server default."
    )
    top_k: Optional[int] = Field(None, gt=0, description="Number of chunks Agent 3 retrieves; omit to use the server default.")
    temperature_level: Optional[TemperatureLevel] = Field(
        None, description="Response tuning level; omit to use the module default ('focused')."
    )

    @field_validator("notebook_id")
    @classmethod
    def notebook_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("notebook_id is required.")
        return v


class PipelineResult(BaseModel):
    notebook_id: str = ""
    doc_count: int = 0
    ingestion_summary: str = ""
    per_doc_summaries: Dict[str, str] = Field(default_factory=dict)
    cross_summary: str = ""
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_mode: str = "empty"
    verified_citations: List[Dict[str, Any]] = Field(default_factory=list)
    citation_report: str = ""
    knowledge_graph_dot: str = ""
    kg_data: Dict[str, Any] = Field(default_factory=dict)
    study_guide: str = ""
    podcast_script: str = ""
    errors: List[str] = Field(default_factory=list)
    completed_steps: List[str] = Field(default_factory=list)
    eval_result: Dict[str, Any] = Field(default_factory=dict)
    rag_reflection_info: Dict[str, Any] = Field(default_factory=dict)
    progress_pct: int = 100


class PipelineJobStatus(JobStatusBase):
    result: Optional[PipelineResult] = None
