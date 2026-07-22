import { useState } from "react";
import type { SRResult } from "../../api/systematicReviewTypes";
import CollapsibleCard from "./CollapsibleCard";
import "./sr-common.css";
import "./EvidenceTab.css";

interface EvidenceTabProps {
  result: SRResult;
}

interface ScreenerRow {
  paper: { title: string; abstract: string };
  score: number;
  verdict: string;
  rationale: string;
}

function parseScreenerRow(raw: Record<string, unknown>): ScreenerRow {
  const paperRaw = raw.paper;
  const paper = typeof paperRaw === "object" && paperRaw !== null ? (paperRaw as Record<string, unknown>) : {};
  return {
    paper: {
      title: typeof paper.title === "string" ? paper.title : "",
      abstract: typeof paper.abstract === "string" ? paper.abstract : "",
    },
    score: typeof raw.score === "number" ? raw.score : 0,
    verdict: typeof raw.verdict === "string" ? raw.verdict : "",
    rationale: typeof raw.rationale === "string" ? raw.rationale : "",
  };
}

function screenerSummary(rows: ScreenerRow[]) {
  const total = rows.length;
  const include = rows.filter((r) => r.verdict === "include").length;
  const uncertain = rows.filter((r) => r.verdict === "uncertain").length;
  const exclude = rows.filter((r) => r.verdict === "exclude").length;
  const meanScore = total > 0 ? Math.round((rows.reduce((sum, r) => sum + r.score, 0) / total) * 10) / 10 : 0;
  return { total, include, uncertain, exclude, meanScore };
}

const VERDICT_BADGE: Record<string, string> = {
  include: "[INCLUDE]",
  uncertain: "[UNCERTAIN]",
  exclude: "[EXCLUDE]",
};

type VerdictFilter = "all" | "include" | "uncertain" | "exclude";

function EvidenceTab({ result }: EvidenceTabProps) {
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>("all");

  const nInc = result.included_papers.length;
  const nExc = result.excluded_papers.length;
  const screenerRows = result.screener_scores.map(parseScreenerRow);
  const summary = screenerSummary(screenerRows);
  const filteredScreenerRows =
    verdictFilter === "all" ? screenerRows : screenerRows.filter((r) => r.verdict === verdictFilter);

  return (
    <div className="sr-evidence">
      <h3>Evidence Table ({nInc} included papers)</h3>
      {result.evidence_table.length === 0 ? (
        <p className="sr-info">No papers were included in the review.</p>
      ) : (
        result.evidence_table.map((row) => (
          <CollapsibleCard
            key={row.citation_key}
            header={
              <span>
                [{row.citation_key}] {row.title.slice(0, 70)} ({row.year ?? "n.d."}) — {row.quality}
              </span>
            }
          >
            <div className="evidence-card__columns">
              <div>
                <p>
                  <strong>Authors:</strong> {row.authors.slice(0, 3).join("; ")}
                  {row.authors.length > 3 ? "…" : ""}
                </p>
                <p>
                  <strong>Journal:</strong> {row.journal || "N/A"}
                </p>
                {row.doi ? (
                  <p>
                    <strong>DOI:</strong>{" "}
                    <a href={`https://doi.org/${row.doi}`} target="_blank" rel="noreferrer">
                      {row.doi}
                    </a>
                  </p>
                ) : row.url ? (
                  <p>
                    <a href={row.url} target="_blank" rel="noreferrer">
                      View paper
                    </a>
                  </p>
                ) : null}
              </div>
              <div className="evidence-card__metrics">
                <div className="sr-metric">
                  <span className="sr-metric__label">Study Design</span>
                  <span className="sr-metric__value">{row.study_design}</span>
                </div>
                <div className="sr-metric">
                  <span className="sr-metric__label">Sample Size</span>
                  <span className="sr-metric__value">{row.sample_size}</span>
                </div>
                <div className="sr-metric">
                  <span className="sr-metric__label">Relevance</span>
                  <span className="sr-metric__value">{row.relevance_score}/5</span>
                </div>
              </div>
            </div>
            {row.key_finding && (
              <p>
                <strong>Key finding:</strong> {row.key_finding}
              </p>
            )}
          </CollapsibleCard>
        ))
      )}

      {nExc > 0 && (
        <details className="sr-evidence__excluded">
          <summary>Excluded Papers ({nExc})</summary>
          <ul>
            {result.excluded_papers.map((p, i) => (
              <li key={`${p.title}-${i}`}>
                <strong>{p.title.slice(0, 70)}</strong> ({p.year ?? "n.d."}) — <em>{p.exclusion_reason}</em>
              </li>
            ))}
          </ul>
        </details>
      )}

      <hr />

      <h3>Abstract Screener</h3>
      <p>
        LLM relevance scores (0–100) for every paper retrieved before the inclusion/exclusion screening
        decision was made. Scores above 80 = clearly include; 50–79 = uncertain; below 50 = likely exclude.
      </p>

      {screenerRows.length === 0 ? (
        <p className="sr-info">
          Abstract screener scores will appear here after running the systematic review.
        </p>
      ) : (
        <>
          <div className="sr-evidence__summary">
            <div className="sr-metric">
              <span className="sr-metric__label">Total Scored</span>
              <span className="sr-metric__value">{summary.total}</span>
            </div>
            <div className="sr-metric">
              <span className="sr-metric__label">Include</span>
              <span className="sr-metric__value">{summary.include}</span>
            </div>
            <div className="sr-metric">
              <span className="sr-metric__label">Uncertain</span>
              <span className="sr-metric__value">{summary.uncertain}</span>
            </div>
            <div className="sr-metric">
              <span className="sr-metric__label">Exclude</span>
              <span className="sr-metric__value">{summary.exclude}</span>
            </div>
          </div>
          <p className="sr-evidence__caption">Mean relevance score: {summary.meanScore}/100</p>
          <hr />

          <label htmlFor="sr-screener-filter">Show verdicts:</label>
          <select
            id="sr-screener-filter"
            value={verdictFilter}
            onChange={(e) => setVerdictFilter(e.target.value as VerdictFilter)}
          >
            <option value="all">all</option>
            <option value="include">include</option>
            <option value="uncertain">uncertain</option>
            <option value="exclude">exclude</option>
          </select>

          {filteredScreenerRows.map((r, i) => (
            <CollapsibleCard
              key={`${r.paper.title}-${i}`}
              header={
                <span>
                  {VERDICT_BADGE[r.verdict] ?? ""} [{r.score}/100] {r.paper.title.slice(0, 70)}
                </span>
              }
            >
              <p>
                <strong>Verdict:</strong> {r.verdict.toUpperCase()} &nbsp;|&nbsp; <strong>Score:</strong> {r.score}
                /100
              </p>
              <p>
                <strong>Rationale:</strong> {r.rationale}
              </p>
              {r.paper.abstract && (
                <p>
                  <strong>Abstract:</strong> {r.paper.abstract.slice(0, 300)}…
                </p>
              )}
            </CollapsibleCard>
          ))}
        </>
      )}
    </div>
  );
}

export default EvidenceTab;
