import { apiFetch } from "./client";
import type { JobCreated } from "./notebookTypes";
import type { ExplainJobStatus, ExplainRequest, ExplainTurn } from "./notebookExplainTypes";

const POLL_INTERVAL_MS = 700;
const BASE = "/api/notebook/explain";

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
// Turn (background job + polling, same pattern as Mode 1 / Mode 3 / Phase A)
// ─────────────────────────────────────────────────────────────────────────────

export function runExplainTurn(req: ExplainRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/turn`, { method: "POST", body: JSON.stringify(req) });
}

export function getExplainJobStatus(jobId: string): Promise<ExplainJobStatus> {
  return apiFetch<ExplainJobStatus>(`${BASE}/jobs/${jobId}`);
}

/** Poll an Explain turn job until it reaches a terminal status ("done" | "error"). */
export function pollExplainJob(
  jobId: string,
  onUpdate: (status: ExplainJobStatus) => void,
  signal?: AbortSignal,
): Promise<ExplainJobStatus> {
  return pollUntilTerminal(() => getExplainJobStatus(jobId), onUpdate, signal);
}

export function getExplainHistory(notebookId: string): Promise<ExplainTurn[]> {
  return apiFetch<ExplainTurn[]>(`${BASE}/${notebookId}/history`);
}
