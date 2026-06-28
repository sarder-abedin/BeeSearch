import { useCallback, useRef, useState } from "react";
import { ApiError } from "../../../api/client";
import { pollAdvancedJob } from "../../../api/notebookAdvanced";
import type { AdvancedResult } from "../../../api/notebookAdvancedTypes";
import type { JobCreated } from "../../../api/notebookTypes";

export type AdvancedJobState = "idle" | "running" | "done" | "error";

export interface AdvancedToolJob {
  state: AdvancedJobState;
  jobId: string | null;
  result: AdvancedResult | null;
  error: string | null;
  run: (trigger: () => Promise<JobCreated>) => void;
  clear: () => void;
}

/** Shared trigger+poll boilerplate for the 9 Advanced Tools panels. Mirrors
 * sr/explore/useExploreToolJob.ts's shape, adapted for Phase C's per-tool
 * trigger endpoints (`POST /api/notebook/advanced/{feature}`, one per tool)
 * sharing a single `GET /api/notebook/advanced/jobs/{id}` poll endpoint --
 * unlike Explore's one generic `POST /explore/{tool}` dispatch. Also surfaces
 * `jobId` (which Explore's hook doesn't), since several of these tools have
 * job-id-keyed server export endpoints (docx/pdf/dot/wav) that Explore's
 * tools have no equivalent of. */
export function useAdvancedToolJob(): AdvancedToolJob {
  const [state, setState] = useState<AdvancedJobState>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<AdvancedResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback((trigger: () => Promise<JobCreated>) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setState("running");
    setError(null);

    void (async () => {
      try {
        const { job_id } = await trigger();
        setJobId(job_id);
        const final = await pollAdvancedJob(job_id, () => {}, controller.signal);
        if (final.status === "done" && final.result) {
          setResult(final.result);
          setState("done");
        } else {
          setError(final.error ?? "Unknown error.");
          setState("error");
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        const message = err instanceof ApiError ? err.detail : (err as Error).message;
        setError(message);
        setState("error");
      }
    })();
  }, []);

  const clear = useCallback(() => {
    abortRef.current?.abort();
    setState("idle");
    setJobId(null);
    setResult(null);
    setError(null);
  }, []);

  return { state, jobId, result, error, run, clear };
}
