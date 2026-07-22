import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { SRResult } from "../../api/systematicReviewTypes";
import "./SynthesisTab.css";

const GRADE_COLOR: Record<string, string> = {
  High: "🟢",
  Moderate: "🟡",
  Low: "🟠",
  "Very low": "🔴",
};

interface SynthesisTabProps {
  result: SRResult;
}

function SynthesisTab({ result }: SynthesisTabProps) {
  const grade = result.grade_results;
  const overallGrade = typeof grade?.overall_grade === "string" ? grade.overall_grade : null;

  return (
    <div className="sr-synthesis">
      {result.key_themes.length > 0 && (
        <>
          <p>
            <strong>Key Themes:</strong>
          </p>
          <ul>
            {result.key_themes.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
          <hr />
        </>
      )}

      {overallGrade && (
        <>
          <p title="Overall GRADE rating across all included studies. See the Explore → Risk & Certainty tab for per-domain detail.">
            <strong>Certainty of evidence (GRADE):</strong> {GRADE_COLOR[overallGrade] ?? "⚪"} {overallGrade} —{" "}
            {typeof grade.certainty_statement === "string" ? grade.certainty_statement : ""}
          </p>
          {result.contradictions.length > 0 && (
            <p className="sr-synthesis__caption">
              ⚠️ {result.contradictions.length} conflicting finding(s) detected — see Explore → Risk & Certainty.
            </p>
          )}
          <hr />
        </>
      )}

      <h3>Narrative Synthesis</h3>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {result.narrative_synthesis || "*No synthesis generated.*"}
      </ReactMarkdown>
      <hr />

      {result.research_gaps.length > 0 && (
        <>
          <p>
            <strong>Research Gaps:</strong>
          </p>
          <ul>
            {result.research_gaps.map((g) => (
              <li key={g}>{g}</li>
            ))}
          </ul>
          <hr />
        </>
      )}

      {result.conclusion && (
        <>
          <h3>Conclusion</h3>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.conclusion}</ReactMarkdown>
        </>
      )}

      {result.limitations && (
        <>
          <h3>Limitations of this Review</h3>
          <p className="sr-synthesis__limitations">{result.limitations}</p>
        </>
      )}
    </div>
  );
}

export default SynthesisTab;
