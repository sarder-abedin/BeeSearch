import { apiFetch } from "./client";
import type { AskJobStatus, AskRequest, JobCreated } from "./types";

const POLL_INTERVAL_MS = 700;

export function askResearchAssistant(req: AskRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>("/api/research-assistant/ask", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getAskJobStatus(jobId: string): Promise<AskJobStatus> {
  return apiFetch<AskJobStatus>(`/api/research-assistant/jobs/${jobId}`);
}

/**
 * Poll a job until it reaches a terminal status ("done" | "error"), invoking
 * `onUpdate` after every poll so the caller can render live stage progress.
 * Resolves with the terminal status; rejects with an AbortError if `signal`
 * fires first.
 */
export function pollAskJob(
  jobId: string,
  onUpdate: (status: AskJobStatus) => void,
  signal?: AbortSignal,
): Promise<AskJobStatus> {
  return new Promise<AskJobStatus>((resolve, reject) => {
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
        const status = await getAskJobStatus(jobId);
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
