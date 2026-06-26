import { apiFetch, apiFetchBlob, apiFetchText } from "./client";
import type {
  EvidenceMapResponse,
  ExploreTool,
  GrammarCheckRequest,
  GrammarCheckResponse,
  JobCreated,
  MetaAnalysisDraftRequest,
  MetaAnalysisPoolRequest,
  MetaAnalysisPoolResponse,
  MetaAnalysisSeedResponse,
  PlainLanguageSummaryRequest,
  SRJobStatus,
  SRRequest,
  SRTemplate,
  ToolJobStatus,
} from "./systematicReviewTypes";

const POLL_INTERVAL_MS = 700;
const BASE = "/api/systematic-review";

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
// Main pipeline run
// ─────────────────────────────────────────────────────────────────────────────

export function runSystematicReview(req: SRRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/run`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getSRJobStatus(jobId: string): Promise<SRJobStatus> {
  return apiFetch<SRJobStatus>(`${BASE}/jobs/${jobId}`);
}

/** Poll the main SR run job until it reaches a terminal status ("done" | "error"). */
export function pollSRJob(
  jobId: string,
  onUpdate: (status: SRJobStatus) => void,
  signal?: AbortSignal,
): Promise<SRJobStatus> {
  return pollUntilTerminal(() => getSRJobStatus(jobId), onUpdate, signal);
}

// ─────────────────────────────────────────────────────────────────────────────
// Explore tools (generic background-job dispatch, shared by 6 of the 8 tools --
// Evidence Map and Meta-Analysis have their own dedicated endpoints below)
// ─────────────────────────────────────────────────────────────────────────────

export function triggerExploreTool(
  jobId: string,
  tool: ExploreTool,
  options: Record<string, unknown> = {},
): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/jobs/${jobId}/explore/${tool}`, {
    method: "POST",
    body: JSON.stringify({ options }),
  });
}

export function getToolJobStatus(jobId: string): Promise<ToolJobStatus> {
  return apiFetch<ToolJobStatus>(`${BASE}/tool-jobs/${jobId}`);
}

/** Poll any background tool job (Explore tools, meta-analysis draft, plain-language summary). */
export function pollToolJob(
  jobId: string,
  onUpdate: (status: ToolJobStatus) => void,
  signal?: AbortSignal,
): Promise<ToolJobStatus> {
  return pollUntilTerminal(() => getToolJobStatus(jobId), onUpdate, signal);
}

// ─────────────────────────────────────────────────────────────────────────────
// Evidence Map (sync -- no LLM call)
// ─────────────────────────────────────────────────────────────────────────────

export function getEvidenceMap(jobId: string): Promise<EvidenceMapResponse> {
  return apiFetch<EvidenceMapResponse>(`${BASE}/jobs/${jobId}/evidence-map`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Meta-Analysis: seed (sync) -> draft (background job, LLM) -> pool (sync)
// ─────────────────────────────────────────────────────────────────────────────

export function seedMetaAnalysis(jobId: string): Promise<MetaAnalysisSeedResponse> {
  return apiFetch<MetaAnalysisSeedResponse>(`${BASE}/jobs/${jobId}/meta-analysis/seed`);
}

export function draftMetaAnalysis(jobId: string, req: MetaAnalysisDraftRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/jobs/${jobId}/meta-analysis/draft`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function poolMetaAnalysis(req: MetaAnalysisPoolRequest): Promise<MetaAnalysisPoolResponse> {
  return apiFetch<MetaAnalysisPoolResponse>(`${BASE}/meta-analysis/pool`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Export: Markdown / DOCX / PDF
// ─────────────────────────────────────────────────────────────────────────────

export function exportMarkdown(jobId: string): Promise<string> {
  return apiFetchText(`${BASE}/jobs/${jobId}/export/markdown`);
}

export function exportDocx(jobId: string, author: string, institution: string): Promise<Blob> {
  return apiFetchBlob(`${BASE}/jobs/${jobId}/export/docx`, {
    method: "POST",
    body: JSON.stringify({ author, institution }),
  });
}

export function exportPdf(jobId: string, author: string, institution: string): Promise<Blob> {
  return apiFetchBlob(`${BASE}/jobs/${jobId}/export/pdf`, {
    method: "POST",
    body: JSON.stringify({ author, institution }),
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Plain-language summaries (background job -- calls the LLM)
// ─────────────────────────────────────────────────────────────────────────────

export function triggerPlainLanguageSummary(
  jobId: string,
  req: PlainLanguageSummaryRequest,
): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/jobs/${jobId}/plain-language-summary`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Guided templates + grammar-check gate
// ─────────────────────────────────────────────────────────────────────────────

export function listTemplates(): Promise<SRTemplate[]> {
  return apiFetch<SRTemplate[]>(`${BASE}/templates`);
}

export function checkGrammar(req: GrammarCheckRequest): Promise<GrammarCheckResponse> {
  return apiFetch<GrammarCheckResponse>(`${BASE}/grammar-check`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}
