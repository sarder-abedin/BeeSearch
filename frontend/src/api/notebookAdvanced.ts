import { apiFetch, apiFetchBlob, apiFetchText } from "./client";
import type { JobCreated } from "./notebookTypes";
import type {
  AdvancedJobStatus,
  AudioSummaryRequest,
  CitationTimelineRequest,
  CompareSourcesRequest,
  CrossDocumentSummaryRequest,
  DocumentArtifact,
  DocumentFormat,
  DotArtifact,
  DotFormat,
  FaqRequest,
  KnowledgeGraphRequest,
  LiteratureReviewRequest,
  MindmapRequest,
  PaperReviewRequest,
  ReviewChatRequest,
  StudyComparisonRequest,
  TextArtifact,
} from "./notebookAdvancedTypes";

const POLL_INTERVAL_MS = 700;
const BASE = "/api/notebook/advanced";

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
// Run (one trigger endpoint per feature, background job + polling)
// ─────────────────────────────────────────────────────────────────────────────

export function runCrossDocumentSummary(req: CrossDocumentSummaryRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/cross-document-summary`, { method: "POST", body: JSON.stringify(req) });
}

export function runFaq(req: FaqRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/faq`, { method: "POST", body: JSON.stringify(req) });
}

export function runLiteratureReview(req: LiteratureReviewRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/literature-review`, { method: "POST", body: JSON.stringify(req) });
}

export function runMindmap(req: MindmapRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/mindmap`, { method: "POST", body: JSON.stringify(req) });
}

export function runAudioSummary(req: AudioSummaryRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/audio-summary`, { method: "POST", body: JSON.stringify(req) });
}

export function runCompareSources(req: CompareSourcesRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/compare-sources`, { method: "POST", body: JSON.stringify(req) });
}

export function runKnowledgeGraph(req: KnowledgeGraphRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/knowledge-graph`, { method: "POST", body: JSON.stringify(req) });
}

export function runCitationTimeline(req: CitationTimelineRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/citation-timeline`, { method: "POST", body: JSON.stringify(req) });
}

export function runStudyComparison(req: StudyComparisonRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/study-comparison`, { method: "POST", body: JSON.stringify(req) });
}

export function runPaperReview(req: PaperReviewRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/paper-review`, { method: "POST", body: JSON.stringify(req) });
}

export function runReviewerChat(req: ReviewChatRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/reviewer-chat`, { method: "POST", body: JSON.stringify(req) });
}

export function getAdvancedJobStatus(jobId: string): Promise<AdvancedJobStatus> {
  return apiFetch<AdvancedJobStatus>(`${BASE}/jobs/${jobId}`);
}

/** Poll an advanced-tools job until it reaches a terminal status ("done" | "error"). */
export function pollAdvancedJob(
  jobId: string,
  onUpdate: (status: AdvancedJobStatus) => void,
  signal?: AbortSignal,
): Promise<AdvancedJobStatus> {
  return pollUntilTerminal(() => getAdvancedJobStatus(jobId), onUpdate, signal);
}

// ─────────────────────────────────────────────────────────────────────────────
// Export: text / DOCX / PDF / PNG / SVG / WAV (all GET, no request body).
// Most plain-text artifacts are rendered straight from the in-memory
// AdvancedResult instead of refetched here -- "review" is the one exception
// that needs the server's body+references composition (see
// routers/notebook_advanced.py::_resolve_text); FAQ and Citation Timeline
// have no text export endpoint at all and are composed client-side.
// ─────────────────────────────────────────────────────────────────────────────

export function exportText(jobId: string, artifact: TextArtifact): Promise<string> {
  return apiFetchText(`${BASE}/jobs/${jobId}/export/text/${artifact}`);
}

export function exportDocument(jobId: string, artifact: DocumentArtifact, fmt: DocumentFormat): Promise<Blob> {
  return apiFetchBlob(`${BASE}/jobs/${jobId}/export/document/${artifact}/${fmt}`);
}

export function exportDot(jobId: string, artifact: DotArtifact, fmt: DotFormat): Promise<Blob> {
  return apiFetchBlob(`${BASE}/jobs/${jobId}/export/dot/${artifact}/${fmt}`);
}

export function exportAudioWav(jobId: string): Promise<Blob> {
  return apiFetchBlob(`${BASE}/jobs/${jobId}/export/audio/wav`);
}
