import CollapsibleCard from "./sr/CollapsibleCard";
import "./EvalResultPanel.css";

interface EvalResultPanelProps {
  evalResult: Record<string, unknown> | null | undefined;
}

function titleCase(key: string): string {
  return key
    .replace(/_/g, " ")
    .split(" ")
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function EvalResultPanel({ evalResult }: EvalResultPanelProps) {
  const overall = Number(evalResult?.overall ?? 0);
  if (!evalResult || !overall) return null;

  const summary = String(evalResult.summary ?? "");
  const scoreLabel =
    overall >= 4
      ? `Quality Score: ${overall}/5 — Good`
      : overall >= 3
        ? `Quality Score: ${overall}/5 — Adequate`
        : `Quality Score: ${overall}/5 — Needs improvement`;

  const dimensionKeys = Object.keys(evalResult).filter(
    (k) => k !== "overall" && k !== "summary" && k !== "ragchecker_faithfulness",
  );

  const ragFaith = evalResult.ragchecker_faithfulness as Record<string, unknown> | undefined;
  const faithfulnessScore =
    ragFaith && !ragFaith.skipped && ragFaith.faithfulness_score != null
      ? Number(ragFaith.faithfulness_score)
      : null;

  return (
    <CollapsibleCard header={`${scoreLabel} — ${truncate(summary, 60)}`}>
      <div className="sr-metric-row">
        {dimensionKeys.map((key) => (
          <div className="sr-metric" key={key}>
            <span className="sr-metric__label">{titleCase(key)}</span>
            <span className="sr-metric__value">{String(evalResult[key])}/5</span>
          </div>
        ))}
        <div className="sr-metric">
          <span className="sr-metric__label">Overall</span>
          <span className="sr-metric__value">{overall}/5</span>
        </div>
      </div>
      {summary && <p className="sr-caption">{summary}</p>}
      {faithfulnessScore !== null && (
        <p className="eval-result-panel__faithfulness">
          <strong>RAGchecker Faithfulness: {Math.floor(faithfulnessScore * 100)}%</strong> —{" "}
          {Number(ragFaith?.supported_claims ?? 0)}/{Number(ragFaith?.checked_claims ?? 0)} claims supported by
          sources
        </p>
      )}
    </CollapsibleCard>
  );
}

export default EvalResultPanel;
