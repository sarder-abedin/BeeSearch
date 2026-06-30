import { apiFetch, apiFetchBlob } from "./client";
import type { JobCreated } from "./notebookTypes";
import type {
  KnowledgeGraphFormat,
  PipelineJobStatus,
  PipelineRequest,
  StudyGuideFormat,
} from "./notebookPipelineTypes";

const POLL_INTERVAL_MS = 700;
const BASE = "/api/notebook/pipeline";

function pollUntilTerminal<T extends { status: string }>(
  fetchStatus: () => Promise<T>,
  onUpdate: (status: T) => void,
  signal?: AbortSignal,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    const onAbort = () => {
      if (timeoutId !== undefined) clearTimeout(timeoutId);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort);
    const cleanup = () => signal?.removeEventListener("abort", onAbort);

    const tick = async () => {
      if (signal?.aborted) return;
      try {
        const status = await fetchStatus();
        if (signal?.aborted) return;
        onUpdate(status);
        if (status.status === "done" || status.status === "error") {
          cleanup();
          resolve(status);
          return;
        }
        timeoutId = setTimeout(tick, POLL_INTERVAL_MS);
      } catch (err) {
        cleanup();
        reject(err);
      }
    };

    void tick();
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline run (background job + polling, same pattern as notebook chat)
// ─────────────────────────────────────────────────────────────────────────────

export function runPipeline(req: PipelineRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/run`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getPipelineJobStatus(jobId: string): Promise<PipelineJobStatus> {
  return apiFetch<PipelineJobStatus>(`${BASE}/jobs/${jobId}`);
}

/** Poll a pipeline job until it reaches a terminal status ("done" | "error"). */
export function pollPipelineJob(
  jobId: string,
  onUpdate: (status: PipelineJobStatus) => void,
  signal?: AbortSignal,
): Promise<PipelineJobStatus> {
  return pollUntilTerminal(() => getPipelineJobStatus(jobId), onUpdate, signal);
}

// ─────────────────────────────────────────────────────────────────────────────
// Export: binary artifacts (DOCX/PDF/PNG/SVG -- all GET, no request body).
// Plain-text artifacts (summary/citations/study-guide/podcast) are rendered
// straight from the in-memory PipelineResult instead of refetched here.
// ─────────────────────────────────────────────────────────────────────────────

export function exportStudyGuide(jobId: string, fmt: StudyGuideFormat): Promise<Blob> {
  return apiFetchBlob(`${BASE}/jobs/${jobId}/export/study-guide/${fmt}`);
}

export function exportKnowledgeGraph(jobId: string, fmt: KnowledgeGraphFormat): Promise<Blob> {
  return apiFetchBlob(`${BASE}/jobs/${jobId}/export/knowledge-graph/${fmt}`);
}
