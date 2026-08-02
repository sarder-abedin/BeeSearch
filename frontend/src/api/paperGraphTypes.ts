// Types for the paper discovery API (Feature 1: Similarity Graph,
// Feature 2: Discovery Network).  Mirror backend/app/schemas/paper_graph.py.

export interface PaperNode {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  venue: string | null;
  abstract: string | null;   // null displayed as "Abstract unavailable"
  citation_count: number | null;
  url: string | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  edge_type: "similarity" | "reference" | "citation" | "recommendation" | "co_author";
}

export interface GraphData {
  nodes: PaperNode[];
  edges: GraphEdge[];
  partial: boolean;
  notice: string;
}

// ── Feature 1 ────────────────────────────────────────────────────────────────

export interface SimilarityGraphRequest {
  paper_id: string;
  top_n?: number;
  bc_weight?: number;
  cc_weight?: number;
}

export interface PaperGraphJobResult {
  graph: GraphData;
}

export interface PaperGraphJobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: PaperGraphJobResult | null;
}

// ── Feature 2 ────────────────────────────────────────────────────────────────

export interface CreateCollectionRequest {
  seed_paper_ids: string[];
}

export type ExpandRelationship = "earlier" | "later" | "similar" | "authors";

export interface ExpandCollectionRequest {
  node_id: string;
  relationship: ExpandRelationship;
}

export interface CollectionResponse {
  collection_id: string;
  graph: GraphData;
}

export interface ExpandJobResult {
  collection_id: string;
  graph: GraphData;
}

export interface ExpandJobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: ExpandJobResult | null;
}
