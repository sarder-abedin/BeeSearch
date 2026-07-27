/**
 * TypeScript mirrors of backend/app/schemas/systematic_review.py.
 * Keep field names/optionality in sync with those Pydantic models.
 */

import type { JobCreated, JobStatusValue } from "./types";

export type { JobCreated, JobStatusValue };

export type ExploreTool =
  | "citation_network"
  | "citation_context"
  | "reference_checking"
  | "preprint_status"
  | "research_trends"
  | "evidence_map"
  | "meta_analysis"
  | "concept_drift";

export interface SRRequest {
  research_question: string;
  inclusion_criteria?: string[];
  exclusion_criteria?: string[];
  model?: string | null;
  num_ctx?: number | null;
  max_results?: number | null;
  include_crossref?: boolean | null;
}

export interface PaperLite {
  title: string;
  authors: string[];
  year: number | null;
  abstract: string;
  url: string;
  doi: string | null;
  journal: string | null;
  source: string;
  citation_key: string;
  citation_count: number | null;
}

export interface ExcludedPaper extends PaperLite {
  exclusion_reason: string;
}

export interface EvidenceRow {
  title: string;
  authors: string[];
  year: number | null;
  citation_key: string;
  url: string;
  doi: string | null;
  journal: string | null;
  abstract: string;
  population: string;
  intervention: string;
  comparator: string;
  outcome: string;
  study_design: string;
  sample_size: string;
  key_finding: string;
  quality: string;
  relevance_score: number;
}

export interface PrismaFlow {
  identified: number;
  screened: number;
  eligibility: number;
  included: number;
  excluded: number;
}

export interface SRResult {
  session_id: string;
  research_question: string;
  inclusion_criteria: string[];
  exclusion_criteria: string[];
  search_queries: string[];
  model_name: string;
  num_ctx: number;

  raw_papers: PaperLite[];
  screened_papers: PaperLite[];
  included_papers: PaperLite[];
  excluded_papers: ExcludedPaper[];

  prisma_flow: PrismaFlow;
  evidence_table: EvidenceRow[];
  narrative_synthesis: string;
  key_themes: string[];
  research_gaps: string[];
  limitations: string;
  conclusion: string;

  eval_result: Record<string, unknown>;
  rag_reflection_info: Record<string, unknown>;

  rob_table: Record<string, unknown>[];
  grade_results: Record<string, unknown>;
  contradictions: Record<string, unknown>[];

  screener_scores: Record<string, unknown>[];
  preprint_tracking: Record<string, unknown>[];
  citation_graph_html: string;

  trend_data: Record<string, unknown>;
  evidence_map_data: Record<string, unknown>;
  concept_drift_data: Record<string, unknown>;

  errors: string[];
  progress_pct: number;
}

export interface SRJobStatus {
  id: string;
  status: JobStatusValue;
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: SRResult | null;
}

export interface ToolJobStatus {
  id: string;
  status: JobStatusValue;
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: Record<string, unknown> | null;
}

export interface ExploreToolRequest {
  options?: Record<string, unknown>;
}

export interface EvidenceMapResponse {
  map_data: Record<string, unknown>;
  html: string | null;
}

export interface MetaAnalysisRow {
  citation_key: string;
  label: string;
  effect: number | null;
  ci_low: number | null;
  ci_high: number | null;
  n: number | null;
}

export interface MetaAnalysisSeedResponse {
  rows: MetaAnalysisRow[];
  measure_labels: Record<string, string>;
}

export type PoolingModel = "fixed" | "random";

export interface MetaAnalysisPoolRequest {
  rows: MetaAnalysisRow[];
  measure?: string;
  model?: PoolingModel;
}

export interface MetaAnalysisPoolResponse {
  result: Record<string, unknown>;
  forest_html: string | null;
}

export interface ExportRequest {
  author?: string;
  institution?: string;
}

export interface MetaAnalysisDraftRequest {
  rows: MetaAnalysisRow[];
  measure?: string;
  model?: string | null;
  num_ctx?: number | null;
}

export type PlainLanguageFormat = "patient" | "policy" | "press" | "all";

export interface PlainLanguageSummaryRequest {
  format?: PlainLanguageFormat;
  model?: string | null;
  num_ctx?: number | null;
}

export interface PlainLanguageSummaryResponse {
  summaries: Record<string, string>;
}

export interface SRTemplate {
  key: string;
  label: string;
  description: string;
  research_question: string;
  inclusion: string[];
  exclusion: string[];
  note: string;
}

export interface GrammarCheckRequest {
  text: string;
  context_hint?: string;
  model?: string | null;
  num_ctx?: number | null;
}

export interface GrammarCheckResponse {
  original: string;
  corrected: string;
  changed: boolean;
}
