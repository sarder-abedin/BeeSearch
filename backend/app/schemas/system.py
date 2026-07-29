"""backend/app/schemas/system.py
──────────────────────────────────
Pydantic shapes for the system/settings endpoint that powers the React
settings panel -- the REST equivalent of ``ui/sidebar.py``'s Hardware,
Model Recommendation, Recommended Configuration, LLM Model, Response
Tuning, Context Window, Hybrid RAG, and Search Settings sections.

Mirrors ``config/hardware.py``'s return shapes (``detect_hardware()``,
``recommend_config()``, ``get_recommended_tier()``) and
``tools/temperature_levels.py::temperature_level_options()`` field-for-field
so the service layer can pass those dicts straight into these models.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

TemperatureLevel = Literal["precise", "focused", "balanced", "creative"]


class HardwareInfo(BaseModel):
    os: str
    arch: str
    cpu: str
    ram_gb: float
    gpu_type: str
    is_apple_silicon: bool
    in_docker: bool
    is_docker_on_apple_silicon: bool


class TierInfo(BaseModel):
    tier: str
    label: str
    description: str
    num_ctx: int
    hybrid_top_k: int
    chunk_size: int
    chunk_overlap: int
    max_results: int
    large_doc_page_threshold: int


class SafeAlternative(BaseModel):
    name: str
    ram_gb: float


class ModelRecommendation(BaseModel):
    model: Optional[str] = None
    num_ctx: int
    reasoning: str
    hardware_note: str
    pull_command: Optional[str] = None
    can_run: bool
    tight_fit: bool = False
    safe_alternative: Optional[SafeAlternative] = None


class EmbedModelInfo(BaseModel):
    name: str
    dim: int = 0
    size_gb: float = 0.0
    note: str = ""
    pulled: bool


class ModelSuggestion(BaseModel):
    name: str
    num_ctx: int
    chunk_size: int
    chunk_overlap: int
    ram_gb: float
    label: str
    note: str
    quality: int


class TemperatureLevelOption(BaseModel):
    key: TemperatureLevel
    label: str
    description: str


class SystemStatusResponse(BaseModel):
    hardware: HardwareInfo
    tier: TierInfo
    recommendation: ModelRecommendation
    available_models: List[str] = Field(default_factory=list)
    embed_models: List[EmbedModelInfo] = Field(default_factory=list)
    model_suggestions: Dict[str, ModelSuggestion] = Field(default_factory=dict)
    temperature_levels: List[TemperatureLevelOption] = Field(default_factory=list)
    default_temperature_level: TemperatureLevel
    context_window_options: List[int] = Field(default_factory=list)
    vision_model: str = ""


class ShutdownResult(BaseModel):
    message: str
