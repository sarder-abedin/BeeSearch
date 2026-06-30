/**
 * TypeScript mirrors of backend/app/schemas/notebook_report.py.
 * Keep field names/optionality in sync with those Pydantic models.
 */
import type { JobStatusValue } from "./notebookTypes";

export interface ReportRequest {
  notebook_id: string;
  goal: string;
  include_academic?: boolean;
  include_web?: boolean;
  model?: string | null;
  num_ctx?: number | null;
  embed_model?: string | null;
}

export interface ReportReference {
  ref_num: number;
  title: string;
  authors: string[];
  journal: string;
  year: string;
  doi: string;
  url: string;
  abstract_snippet: string;
  source: string;
  citation_count: number | null;
  apa: string;
}

export interface ReportResult {
  notebook_id: string;
  goal: string;
  mode: string;
  report: string;
  key_findings: string[];
  references: ReportReference[];
  web_search_status: string;
  eval_result: Record<string, unknown>;
  errors: string[];
  progress_pct: number;
}

export interface ReportJobStatus {
  id: string;
  status: JobStatusValue;
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: ReportResult | null;
}

export type ReportCitationFormat = "bibtex" | "ris";
