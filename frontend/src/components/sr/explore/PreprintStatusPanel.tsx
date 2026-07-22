import type { SRResult } from "../../../api/systematicReviewTypes";
import CollapsibleCard from "../CollapsibleCard";
import { useExploreToolJob } from "./useExploreToolJob";
import { arr, nullableStr, num, obj, str } from "./parse";
import "../sr-common.css";

interface PreprintStatusPanelProps {
  jobId: string;
  result: SRResult;
}

interface PreprintRow {
  paper: { title: string };
  preprint_status: string | null;
  published_doi: string | null;
  published_venue: string | null;
  note: string;
}

function parsePreprintRow(raw: unknown): PreprintRow {
  const r = obj(raw);
  const paper = obj(r.paper);
  return {
    paper: { title: str(paper.title) },
    preprint_status: nullableStr(r.preprint_status),
    published_doi: nullableStr(r.published_doi),
    published_venue: nullableStr(r.published_venue),
    note: str(r.note),
  };
}

interface Summary {
  journal: number;
  published: number;
  preprint: number;
  retracted: number;
}

function parseSummary(raw: unknown): Summary {
  const s = obj(raw);
  return {
    journal: num(s.journal),
    published: num(s.published),
    preprint: num(s.preprint),
    retracted: num(s.retracted),
  };
}

function summarizeTracking(tracking: PreprintRow[]): Summary {
  const out: Summary = { journal: 0, published: 0, preprint: 0, retracted: 0 };
  for (const r of tracking) {
    const status = r.preprint_status ?? "journal";
    if (status in out) out[status as keyof Summary] += 1;
  }
  return out;
}

const STATUS_BADGE: Record<string, string> = {
  journal: "[JOURNAL]",
  published: "[PUBLISHED]",
  preprint: "[PREPRINT]",
  retracted: "[RETRACTED]",
};

interface PreprintTrackingViewProps {
  tracking: PreprintRow[];
  summary?: Summary;
}

function PreprintTrackingView({ tracking, summary }: PreprintTrackingViewProps) {
  const s = summary ?? summarizeTracking(tracking);
  return (
    <>
      <div className="sr-metric-row">
        <div className="sr-metric">
          <span className="sr-metric__label">Journal</span>
          <span className="sr-metric__value">{s.journal}</span>
        </div>
        <div className="sr-metric">
          <span className="sr-metric__label">Published (was preprint)</span>
          <span className="sr-metric__value">{s.published}</span>
        </div>
        <div className="sr-metric">
          <span className="sr-metric__label">Preprint only</span>
          <span className="sr-metric__value">{s.preprint}</span>
        </div>
        <div className="sr-metric">
          <span className="sr-metric__label">Retracted</span>
          <span className="sr-metric__value">{s.retracted}</span>
        </div>
      </div>
      <hr />
      {tracking.map((r, i) => {
        const status = r.preprint_status ?? "unknown";
        const badge = STATUS_BADGE[status] ?? "[UNKNOWN]";
        return (
          <CollapsibleCard
            key={i}
            header={
              <span>
                {badge} {r.paper.title.slice(0, 70)} — {status.toUpperCase()}
              </span>
            }
          >
            <p>
              <strong>Status:</strong> {status}
            </p>
            <p>
              <strong>Note:</strong> {r.note}
            </p>
            {r.published_doi && (
              <p>
                <strong>Published DOI:</strong>{" "}
                <a href={`https://doi.org/${r.published_doi}`} target="_blank" rel="noreferrer">
                  {r.published_doi}
                </a>
              </p>
            )}
            {r.published_venue && (
              <p>
                <strong>Journal:</strong> {r.published_venue}
              </p>
            )}
          </CollapsibleCard>
        );
      })}
    </>
  );
}

function PreprintStatusPanel({ jobId, result }: PreprintStatusPanelProps) {
  const job = useExploreToolJob(jobId, "preprint_status");
  const included = result.included_papers;
  const existingTracking = result.preprint_tracking.map(parsePreprintRow);

  return (
    <div className="sr-explore-panel">
      <h3>Preprint Status</h3>
      <p>
        Checks each included paper against CrossRef to identify unverified preprints and flag any
        retractions. Requires CrossRef API calls — click below to check (~0.25s per paper).
      </p>

      {included.length === 0 ? (
        <p className="sr-info">No included papers.</p>
      ) : existingTracking.length > 0 ? (
        <PreprintTrackingView tracking={existingTracking} />
      ) : (
        <>
          <div className="sr-button-row">
            <button
              type="button"
              className="sr-button"
              disabled={job.state === "running"}
              onClick={() => job.run()}
            >
              Check Preprint Status
            </button>
          </div>

          {job.state === "running" && (
            <p className="sr-spinner-text">Querying CrossRef for publication status…</p>
          )}
          {job.state === "error" && <p className="sr-error">Preprint tracking failed: {job.error}</p>}

          {job.state === "done" && job.result && (
            <PreprintTrackingView
              tracking={arr(job.result.results).map(parsePreprintRow)}
              summary={parseSummary(job.result.summary)}
            />
          )}
        </>
      )}
    </div>
  );
}

export default PreprintStatusPanel;
