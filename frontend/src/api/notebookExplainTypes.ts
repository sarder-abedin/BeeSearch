/**
 * TypeScript mirrors of backend/app/schemas/notebook_explain.py.
 * Keep field names/optionality in sync with those Pydantic models.
 */
import type { JobStatusValue, TemperatureLevel } from "./notebookTypes";

export type ExplanationStyle = "simple" | "analogy" | "walkthrough" | "debate";
export type ExplanationLevel = "novice" | "intermediate" | "expert";

/** n is a *string* (not a number) -- agents/story_nodes.py emits document-excerpt
 * citations with an int n and online-source citations with a "Source N" string n
 * into the same list, so the frontend only ever displays "[{n}]", never does
 * arithmetic on it (see notebook_explain.py's own module docstring). */
export interface ExplainCitationItem {
  n: string;
  doc_name: string;
  page: number | null;
  page_label: string;
  snippet: string;
  url: string;
}

export interface OnlineResultItem {
  type: string;
  title: string;
  authors: string;
  url: string;
  snippet: string;
  source: string;
  year: number | null;
  apa: string;
}

export interface SourceDecision {
  coverage_score: number;
  used_docs: boolean;
  used_online: boolean;
  search_attempted: boolean;
  reason: string;
  sources_searched: string[];
  online_count: number;
}

export interface ExplainTurn {
  role: string;
  content: string;
  timestamp: string;
  citations: ExplainCitationItem[] | null;
  suggested_questions: string[] | null;
  explanation_style: string | null;
}

export interface ExplainRequest {
  notebook_id: string;
  message: string;
  explanation_style?: ExplanationStyle;
  explanation_level?: ExplanationLevel;
  model?: string | null;
  num_ctx?: number | null;
  temperature_level?: TemperatureLevel | null;
}

export interface ExplainResult {
  notebook_id: string;
  user_message: string;
  assistant_response: string;
  explanation_style: string;
  citations: ExplainCitationItem[];
  suggested_questions: string[];
  is_repeat_clarification: boolean;
  repeated_question: string;
  new_concepts: string[];
  concept_visual_html: string;
  source_decision: SourceDecision | null;
  online_results: OnlineResultItem[];
  eval_result: Record<string, unknown>;
  errors: string[];
  progress_pct: number;
}

export interface ExplainJobStatus {
  id: string;
  status: JobStatusValue;
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: ExplainResult | null;
}
