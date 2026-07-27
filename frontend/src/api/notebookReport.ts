import { apiFetch, apiFetchText } from "./client";
import type { JobCreated } from "./notebookTypes";
import type { ReportCitationFormat, ReportJobStatus, ReportRequest } from "./notebookReportTypes";

const POLL_INTERVAL_MS = 700;
const BASE = "/api/notebook/report";

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
// Report run (background job + polling, same pattern as the other pipelines)
// ─────────────────────────────────────────────────────────────────────────────

export function runReport(req: ReportRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/run`, { method: "POST", body: JSON.stringify(req) });
}

export function getReportJobStatus(jobId: string): Promise<ReportJobStatus> {
  return apiFetch<ReportJobStatus>(`${BASE}/jobs/${jobId}`);
}

/** Poll a report job until it reaches a terminal status ("done" | "error"). */
export function pollReportJob(
  jobId: string,
  onUpdate: (status: ReportJobStatus) => void,
  signal?: AbortSignal,
): Promise<ReportJobStatus> {
  return pollUntilTerminal(() => getReportJobStatus(jobId), onUpdate, signal);
}

// ─────────────────────────────────────────────────────────────────────────────
// Export: BibTeX / RIS citations (text/plain -- no client-side equivalent for
// the conversion itself, so this stays a server round-trip). The Markdown
// report needs no endpoint: it's rendered straight from the in-memory
// ReportResult.report, mirroring ui/helpers.py::render_report.
// ─────────────────────────────────────────────────────────────────────────────

export function exportReportCitations(jobId: string, fmt: ReportCitationFormat): Promise<string> {
  return apiFetchText(`${BASE}/jobs/${jobId}/export/citations/${fmt}`);
}
