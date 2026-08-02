import { apiFetch } from "./client";
import type {
  CollectionResponse,
  CreateCollectionRequest,
  ExpandCollectionRequest,
  ExpandJobStatus,
  PaperGraphJobStatus,
  PaperNode,
  SimilarityGraphRequest,
} from "./paperGraphTypes";

const BASE = "/api/paper-graph";
const POLL_MS = 700;

function pollUntilTerminal<T extends { status: string }>(
  fetchStatus: () => Promise<T>,
  onUpdate: (s: T) => void,
  signal?: AbortSignal,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let tid: ReturnType<typeof setTimeout> | undefined;
    const onAbort = () => {
      if (tid !== undefined) clearTimeout(tid);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort);
    const cleanup = () => signal?.removeEventListener("abort", onAbort);

    const tick = async () => {
      if (signal?.aborted) return;
      try {
        const s = await fetchStatus();
        if (signal?.aborted) return;
        onUpdate(s);
        if (s.status === "done" || s.status === "error") {
          cleanup();
          resolve(s);
          return;
        }
        tid = setTimeout(tick, POLL_MS);
      } catch (err) {
        cleanup();
        reject(err);
      }
    };
    void tick();
  });
}

// ── Feature 1 ────────────────────────────────────────────────────────────────

export function runSimilarityGraph(req: SimilarityGraphRequest): Promise<{ job_id: string }> {
  return apiFetch<{ job_id: string }>(`${BASE}/similarity-graph`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getSimilarityGraphJob(jobId: string): Promise<PaperGraphJobStatus> {
  return apiFetch<PaperGraphJobStatus>(`${BASE}/jobs/${jobId}`);
}

export function pollSimilarityGraphJob(
  jobId: string,
  onUpdate: (s: PaperGraphJobStatus) => void,
  signal?: AbortSignal,
): Promise<PaperGraphJobStatus> {
  return pollUntilTerminal(() => getSimilarityGraphJob(jobId), onUpdate, signal);
}

// ── Building blocks ───────────────────────────────────────────────────────────

export function getPaper(paperId: string): Promise<PaperNode> {
  return apiFetch<PaperNode>(`${BASE}/papers/${paperId}`);
}

// ── Feature 2 ────────────────────────────────────────────────────────────────

export function createCollection(req: CreateCollectionRequest): Promise<CollectionResponse> {
  return apiFetch<CollectionResponse>(`${BASE}/collections`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getCollection(collectionId: string): Promise<CollectionResponse> {
  return apiFetch<CollectionResponse>(`${BASE}/collections/${collectionId}`);
}

export function expandCollection(
  collectionId: string,
  req: ExpandCollectionRequest,
): Promise<{ job_id: string }> {
  return apiFetch<{ job_id: string }>(`${BASE}/collections/${collectionId}/expand`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getExpandJob(collectionId: string, jobId: string): Promise<ExpandJobStatus> {
  return apiFetch<ExpandJobStatus>(`${BASE}/collections/${collectionId}/jobs/${jobId}`);
}

export function pollExpandJob(
  collectionId: string,
  jobId: string,
  onUpdate: (s: ExpandJobStatus) => void,
  signal?: AbortSignal,
): Promise<ExpandJobStatus> {
  return pollUntilTerminal(() => getExpandJob(collectionId, jobId), onUpdate, signal);
}
