import type { SRResult } from "../../../api/systematicReviewTypes";
import CollapsibleCard from "../CollapsibleCard";
import { useExploreToolJob } from "./useExploreToolJob";
import { arr, nullableNum, obj, str, strArr } from "./parse";
import "../sr-common.css";

interface ReferenceCheckingPanelProps {
  jobId: string;
  result: SRResult;
}

interface RobRow {
  citation_key: string;
  title: string;
  tool: string;
  overall: string;
  justification: string;
  domains: [string, string][];
}

const ROB_KNOWN_KEYS = new Set(["citation_key", "title", "tool", "overall", "justification"]);

function titleCase(key: string): string {
  return key
    .replace(/_/g, " ")
    .split(" ")
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

function parseRobRow(raw: unknown): RobRow {
  const r = obj(raw);
  const domains: [string, string][] = [];
  for (const [key, val] of Object.entries(r)) {
    if (ROB_KNOWN_KEYS.has(key)) continue;
    domains.push([key, typeof val === "string" ? val : JSON.stringify(val)]);
  }
  return {
    citation_key: str(r.citation_key),
    title: str(r.title),
    tool: str(r.tool),
    overall: str(r.overall, "Some concerns"),
    justification: str(r.justification),
    domains,
  };
}

interface GradeResults {
  overall_grade: string;
  certainty_statement: string;
  summary: string;
  domains: [string, string][];
}

function parseGradeResults(raw: unknown): GradeResults {
  const g = obj(raw);
  const domainsObj = obj(g.domains);
  return {
    overall_grade: str(g.overall_grade, "n/a"),
    certainty_statement: str(g.certainty_statement),
    summary: str(g.summary),
    domains: Object.entries(domainsObj).map(([k, v]) => [k, str(v)]),
  };
}

interface Position {
  description: string;
  papers: string[];
}

function parsePosition(raw: unknown): Position {
  const p = obj(raw);
  return { description: str(p.description), papers: strArr(p.papers) };
}

interface Contradiction {
  claim: string;
  consensus_score: number | null;
  position_a: Position;
  position_b: Position;
  explanation: string;
}

function parseContradiction(raw: unknown): Contradiction {
  const c = obj(raw);
  return {
    claim: str(c.claim),
    consensus_score: nullableNum(c.consensus_score),
    position_a: parsePosition(c.position_a),
    position_b: parsePosition(c.position_b),
    explanation: str(c.explanation),
  };
}

const GRADE_ICON: Record<string, string> = { High: "🟢", Moderate: "🟡", Low: "🟠", "Very low": "🔴" };
const ROB_ICON: Record<string, string> = { Low: "🟢", "Some concerns": "🟡", High: "🔴" };

interface ReferenceCheckingResultsProps {
  robTable: unknown[];
  grade: unknown;
  contradictions: unknown[];
}

function ReferenceCheckingResults({ robTable, grade, contradictions }: ReferenceCheckingResultsProps) {
  const robRows = robTable.map(parseRobRow);
  const gradeObj = obj(grade);
  const hasGrade = Object.keys(gradeObj).length > 0;
  const gradeParsed = hasGrade ? parseGradeResults(gradeObj) : null;
  const contraRows = contradictions.map(parseContradiction);

  return (
    <>
      {gradeParsed && (
        <>
          <p>
            <strong>Certainty of Evidence (GRADE):</strong> {GRADE_ICON[gradeParsed.overall_grade] ?? "⚪"}{" "}
            <strong>{gradeParsed.overall_grade}</strong>
          </p>
          {gradeParsed.certainty_statement && <p className="sr-info">{gradeParsed.certainty_statement}</p>}
          {gradeParsed.summary && <p>{gradeParsed.summary}</p>}
          {gradeParsed.domains.length > 0 && (
            <div className="sr-metric-row">
              {gradeParsed.domains.map(([dom, rating]) => (
                <div className="sr-metric" key={dom}>
                  <span className="sr-metric__label">{titleCase(dom)}</span>
                  <span className="sr-metric__value">{rating}</span>
                </div>
              ))}
            </div>
          )}
          <hr />
        </>
      )}

      {robRows.length > 0 && (
        <>
          <p>
            <strong>Risk of Bias (per paper):</strong>
          </p>
          {robRows.map((r, i) => (
            <CollapsibleCard
              key={`${r.citation_key}-${i}`}
              header={
                <span>
                  {ROB_ICON[r.overall] ?? "⚪"} {r.citation_key} — {r.title.slice(0, 60)} ({r.tool}: {r.overall})
                </span>
              }
            >
              <ul>
                {r.domains.map(([key, val]) => (
                  <li key={key}>
                    <strong>{titleCase(key)}:</strong> {val}
                  </li>
                ))}
              </ul>
              {r.justification && <p className="sr-caption">{r.justification}</p>}
            </CollapsibleCard>
          ))}
          <hr />
        </>
      )}

      {contraRows.length > 0 ? (
        <>
          <p>
            <strong>Conflicting Findings ({contraRows.length}):</strong>
          </p>
          {contraRows.map((c, i) => (
            <CollapsibleCard
              key={i}
              header={
                <span>
                  ⚠️ {c.claim.slice(0, 80)} — consensus {c.consensus_score ?? "?"}/100
                </span>
              }
            >
              <p>
                <strong>Position A:</strong> {c.position_a.description}{" "}
                <em>(papers: {c.position_a.papers.join(", ")})</em>
              </p>
              <p>
                <strong>Position B:</strong> {c.position_b.description}{" "}
                <em>(papers: {c.position_b.papers.join(", ")})</em>
              </p>
              {c.explanation && <p>{c.explanation}</p>}
            </CollapsibleCard>
          ))}
        </>
      ) : (
        (robRows.length > 0 || hasGrade) && (
          <p className="sr-success">No material contradictions detected across the included papers.</p>
        )
      )}
    </>
  );
}

function ReferenceCheckingPanel({ jobId, result }: ReferenceCheckingPanelProps) {
  const job = useExploreToolJob(jobId, "reference_checking");

  const hasExisting =
    result.rob_table.length > 0 ||
    Object.keys(result.grade_results ?? {}).length > 0 ||
    result.contradictions.length > 0;

  return (
    <div className="sr-explore-panel">
      <h3>Risk of Bias · GRADE · Contradictions</h3>
      <p>
        Per-paper <strong>risk of bias</strong> (RoB 2 for trials, ROBINS-I for observational studies), an
        overall <strong>GRADE</strong> certainty-of-evidence rating, and detected <strong>contradictions</strong>{" "}
        across the included papers. Computed automatically during the review — recompute below if needed.
      </p>

      {result.evidence_table.length === 0 ? (
        <p className="sr-info">No extracted evidence to assess.</p>
      ) : hasExisting ? (
        <ReferenceCheckingResults
          robTable={result.rob_table}
          grade={result.grade_results}
          contradictions={result.contradictions}
        />
      ) : (
        <>
          <div className="sr-button-row">
            <button
              type="button"
              className="sr-button"
              disabled={job.state === "running"}
              onClick={() => job.run()}
            >
              Assess Risk of Bias, GRADE &amp; Contradictions
            </button>
          </div>

          {job.state === "running" && (
            <p className="sr-spinner-text">Assessing risk of bias, GRADE certainty, and contradictions…</p>
          )}
          {job.state === "error" && <p className="sr-error">Reference checking failed: {job.error}</p>}

          {job.state === "done" && job.result && (
            <ReferenceCheckingResults
              robTable={arr(job.result.rob_table)}
              grade={job.result.grade_results}
              contradictions={arr(job.result.contradictions)}
            />
          )}
        </>
      )}
    </div>
  );
}

export default ReferenceCheckingPanel;
