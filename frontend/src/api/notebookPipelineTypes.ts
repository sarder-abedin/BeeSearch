/**
 * TypeScript mirrors of backend/app/schemas/notebook_pipeline.py.
 * Keep field names/optionality in sync with those Pydantic models.
 */
import type { JobStatusValue, TemperatureLevel } from "./notebookTypes";

export interface PipelineRequest {
  notebook_id: string;
  query?: string;
  model?: string | null;
  num_ctx?: number | null;
  embed_model?: string | null;
  top_k?: number | null;
  temperature_level?: TemperatureLevel | null;
}

export interface PipelineChunk {
  chunk_id: string;
  doc_name: string;
  page_num: number;
  text: string;
}

export interface VerifiedCitation {
  claim: string;
  source_name: string;
  confidence: "HIGH" | "MEDIUM" | "LOW";
  supporting_text: string;
}

export interface KnowledgeGraphNode {
  id: string;
  label: string;
  [key: string]: unknown;
}

export interface KnowledgeGraphEdge {
  from: string;
  to: string;
  label: string;
}

export interface KnowledgeGraphData {
  nodes?: KnowledgeGraphNode[];
  edges?: KnowledgeGraphEdge[];
}

export interface PipelineResult {
  notebook_id: string;
  doc_count: number;
  ingestion_summary: string;
  per_doc_summaries: Record<string, string>;
  cross_summary: string;
  retrieved_chunks: PipelineChunk[];
  retrieval_mode: string;
  verified_citations: VerifiedCitation[];
  citation_report: string;
  knowledge_graph_dot: string;
  kg_data: KnowledgeGraphData;
  study_guide: string;
  podcast_script: string;
  errors: string[];
  completed_steps: string[];
  eval_result: Record<string, unknown>;
  rag_reflection_info: Record<string, unknown>;
  progress_pct: number;
}

export interface PipelineJobStatus {
  id: string;
  status: JobStatusValue;
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: PipelineResult | null;
}

export type PipelineTextArtifact = "summary" | "citations" | "study-guide" | "podcast";
export type StudyGuideFormat = "docx" | "pdf";
export type KnowledgeGraphFormat = "png" | "svg";
