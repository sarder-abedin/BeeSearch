/**
 * TypeScript mirrors of backend/app/schemas/notebook_advanced.py.
 * Keep field names/optionality in sync with those Pydantic models.
 */
import type { JobStatusValue, TemperatureLevel } from "./notebookTypes";

interface BaseAdvancedRequest {
  notebook_id: string;
  model?: string | null;
  num_ctx?: number | null;
  temperature_level?: TemperatureLevel | null;
}

export type CrossDocumentSummaryRequest = BaseAdvancedRequest;

export interface FaqRequest extends BaseAdvancedRequest {
  n_questions?: number;
}

export type LiteratureReviewRequest = BaseAdvancedRequest;
export type MindmapRequest = BaseAdvancedRequest;
export type AudioSummaryRequest = BaseAdvancedRequest;

export interface CompareSourcesRequest extends BaseAdvancedRequest {
  doc_id_a: string;
  doc_id_b: string;
}

export type KnowledgeGraphRequest = BaseAdvancedRequest;

export interface CitationTimelineRequest extends BaseAdvancedRequest {
  enrich_with_abstracts?: boolean;
}

export type StudyComparisonRequest = BaseAdvancedRequest;

export interface PaperReviewRequest extends BaseAdvancedRequest {
  doc_id: string;
}

export interface ReviewChatItem {
  role: "user" | "assistant";
  content: string;
}

export interface ReviewChatRequest extends BaseAdvancedRequest {
  doc_id: string;
  review_text: string;
  chat_history: ReviewChatItem[];
  user_message: string;
  external_refs: ExternalReference[];
}

export interface FaqItem {
  question: string;
  answer: string;
  sources: number[];
}

export interface ReferenceItem {
  n: number | null;
  doc_name: string;
  page: number | null;
  snippet: string;
  doc_id: string;
}

export interface CitationTimelineItem {
  year: string;
  title: string;
  authors: string;
  gist: string;
  source: number;
  url: string;
}

export interface ExternalReference {
  ref_num: string;
  title: string;
  authors: string[];
  year: number | null;
  url: string;
  source: string;
  abstract_snippet: string;
}

export interface AdvancedResult {
  notebook_id: string;
  summary: string;
  faqs: FaqItem[];
  review: string;
  references: ReferenceItem[];
  mindmap_dot: string;
  audio_script: string;
  comparison: string;
  knowledge_graph_dot: string;
  timeline: CitationTimelineItem[];
  study_comparison: string;
  paper_review: string;
  paper_review_refs: ExternalReference[];
  reviewer_chat_response: string;
}

export interface AdvancedJobStatus {
  id: string;
  status: JobStatusValue;
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: AdvancedResult | null;
}

export type TextArtifact = "summary" | "review" | "audio-script" | "comparison" | "study-comparison";
export type DocumentArtifact = "summary" | "review" | "study-comparison";
export type DotArtifact = "mindmap" | "knowledge-graph";
export type DocumentFormat = "docx" | "pdf";
export type DotFormat = "png" | "svg";
