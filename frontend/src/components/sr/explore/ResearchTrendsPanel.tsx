import type { SRResult } from "../../../api/systematicReviewTypes";
import HtmlFrame from "./HtmlFrame";
import { useExploreToolJob } from "./useExploreToolJob";
import { nullableNum, nullableStr, num, obj, str } from "./parse";
import "../sr-common.css";

interface ResearchTrendsPanelProps {
  jobId: string;
  result: SRResult;
}

interface TrendData {
  trend: string;
  peak_year: number | null;
  total_field: number;
}

function parseTrendData(raw: unknown): TrendData {
  const t = obj(raw);
  return {
    trend: str(t.trend, "unknown"),
    peak_year: nullableNum(t.peak_year),
    total_field: num(t.total_field),
  };
}

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

function ResearchTrendsPanel({ jobId }: ResearchTrendsPanelProps) {
  const job = useExploreToolJob(jobId, "research_trends");

  return (
    <div className="sr-explore-panel">
      <h3>Research Trend Forecaster</h3>
      <p>
        Publication volume by year for this research area, sourced from CrossRef (field-wide) and
        compared to the papers retrieved in this SR run. Requires CrossRef API calls — click below
        to fetch them (a few seconds).
      </p>

      <div className="sr-button-row">
        <button
          type="button"
          className="sr-button"
          disabled={job.state === "running"}
          onClick={() => job.run()}
        >
          Analyze Trends
        </button>
      </div>

      {job.state === "running" && (
        <p className="sr-spinner-text">Querying CrossRef for field-wide year counts…</p>
      )}
      {job.state === "error" && <p className="sr-error">Trend analysis failed: {job.error}</p>}

      {job.state === "done" && job.result && (() => {
        const td = parseTrendData(job.result.trend_data);
        const html = nullableStr(job.result.html);

        return (
          <>
            <div className="sr-metric-row">
              <div className="sr-metric">
                <span className="sr-metric__label">Field Trend</span>
                <span className="sr-metric__value">{capitalize(td.trend)}</span>
              </div>
              <div className="sr-metric">
                <span className="sr-metric__label">Peak Year</span>
                <span className="sr-metric__value">{td.peak_year ?? "N/A"}</span>
              </div>
              <div className="sr-metric">
                <span className="sr-metric__label">Total Publications (CrossRef)</span>
                <span className="sr-metric__value">{td.total_field.toLocaleString()}</span>
              </div>
            </div>

            {html ? (
              <HtmlFrame html={html} height={420} title="Research trend chart" />
            ) : (
              <p className="sr-info">Install plotly (pip install plotly) to see the trend chart.</p>
            )}
          </>
        );
      })()}
    </div>
  );
}

export default ResearchTrendsPanel;
