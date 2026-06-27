/**
 * TypeScript mirrors of backend/app/schemas/{jobs,research_assistant}.py.
 * Keep field names/optionality in sync with those Pydantic models.
 */

export type TemperatureLevel = "precise" | "focused" | "balanced" | "creative";

export interface AskRequest {
  question: string;
  include_web?: boolean;
  include_crossref?: boolean;
  model?: string | null;
  num_ctx?: number | null;
  temperature_level?: TemperatureLevel | null;
}

export type SourceKind = "academic" | "web";

export interface SourceItem {
  n: number;
  kind: SourceKind;
  title: string;
  authors: string[];
  year: number | null;
  url: string;
  snippet: string;
  apa: string;
  source: string;
}

export interface AskResult {
  question: string;
  answer: string;
  citations: SourceItem[];
  sources: SourceItem[];
  academic_count: number;
  web_count: number;
  suggested_questions: string[];
  grounded: boolean;
}

export type JobStatusValue = "queued" | "running" | "done" | "error";

export interface JobCreated {
  job_id: string;
}

export interface AskJobStatus {
  id: string;
  status: JobStatusValue;
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: AskResult | null;
}
