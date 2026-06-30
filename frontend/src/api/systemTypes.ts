/**
 * TypeScript mirrors of backend/app/schemas/system.py.
 * Keep field names/optionality in sync with those Pydantic models.
 */

export type TemperatureLevel = "precise" | "focused" | "balanced" | "creative";

export interface HardwareInfo {
  os: string;
  arch: string;
  cpu: string;
  ram_gb: number;
  gpu_type: string;
  is_apple_silicon: boolean;
  in_docker: boolean;
  is_docker_on_apple_silicon: boolean;
}

export interface TierInfo {
  tier: string;
  label: string;
  description: string;
  num_ctx: number;
  hybrid_top_k: number;
  chunk_size: number;
  chunk_overlap: number;
  max_results: number;
  large_doc_page_threshold: number;
}

export interface SafeAlternative {
  name: string;
  ram_gb: number;
}

export interface ModelRecommendation {
  model: string | null;
  num_ctx: number;
  reasoning: string;
  hardware_note: string;
  pull_command: string | null;
  can_run: boolean;
  tight_fit: boolean;
  safe_alternative: SafeAlternative | null;
}

export interface EmbedModelInfo {
  name: string;
  dim: number;
  size_gb: number;
  note: string;
  pulled: boolean;
}

export interface TemperatureLevelOption {
  key: TemperatureLevel;
  label: string;
  description: string;
}

export interface SystemStatusResponse {
  hardware: HardwareInfo;
  tier: TierInfo;
  recommendation: ModelRecommendation;
  available_models: string[];
  embed_models: EmbedModelInfo[];
  temperature_levels: TemperatureLevelOption[];
  default_temperature_level: TemperatureLevel;
  context_window_options: number[];
}

export interface ShutdownResult {
  message: string;
}
