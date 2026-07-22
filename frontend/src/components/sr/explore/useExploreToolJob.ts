import { useCallback, useRef, useState } from "react";
import { ApiError } from "../../../api/client";
import { pollToolJob, triggerExploreTool } from "../../../api/systematicReview";
import type { ExploreTool } from "../../../api/systematicReviewTypes";

export type ExploreJobState = "idle" | "running" | "done" | "error";

export interface ExploreToolJob {
  state: ExploreJobState;
  result: Record<string, unknown> | null;
  error: string | null;
  run: (options?: Record<string, unknown>) => void;
}

/** Shared trigger+poll boilerplate for the Explore tools that use the generic
 * `POST /jobs/{id}/explore/{tool}` + `GET /tool-jobs/{id}` dispatch (everything
 * except Evidence Map and Meta-Analysis, which have their own dedicated
 * endpoints and manage their own state). */
export function useExploreToolJob(jobId: string, tool: ExploreTool): ExploreToolJob {
  const [state, setState] = useState<ExploreJobState>("idle");
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(
    (options: Record<string, unknown> = {}) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState("running");
      setError(null);

      void (async () => {
        try {
          const { job_id } = await triggerExploreTool(jobId, tool, options);
          const final = await pollToolJob(job_id, () => {}, controller.signal);
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
    },
    [jobId, tool],
  );

  return { state, result, error, run };
}
