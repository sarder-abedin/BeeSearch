import type { ReactNode } from "react";
import CollapsibleCard from "./sr/CollapsibleCard";
import "./RagReflectionPanel.css";

type RagReflectionEntry = Record<string, unknown>;

interface RagReflectionPanelProps {
  ragReflectionInfo: RagReflectionEntry | RagReflectionEntry[] | null | undefined;
}

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function isEmpty(value: RagReflectionPanelProps["ragReflectionInfo"]): boolean {
  if (!value) return true;
  if (Array.isArray(value)) return value.length === 0;
  return Object.keys(value).length === 0;
}

function numField(entry: RagReflectionEntry, ...keys: string[]): number {
  for (const key of keys) {
    const v = entry[key];
    if (typeof v === "number") return v;
  }
  return 0;
}

function RagReflectionPanel({ ragReflectionInfo }: RagReflectionPanelProps) {
  if (isEmpty(ragReflectionInfo)) return null;

  const entries: RagReflectionEntry[] = Array.isArray(ragReflectionInfo)
    ? ragReflectionInfo
    : [ragReflectionInfo as RagReflectionEntry];

  const totalRetrieved = entries.reduce((sum, e) => sum + numField(e, "total_retrieved", "papers_retrieved"), 0);
  const totalRelevant = entries.reduce((sum, e) => sum + numField(e, "total_relevant", "papers_after_grading"), 0);
  if (totalRetrieved === 0) return null;

  const pct = Math.floor((100 * totalRelevant) / totalRetrieved);

  return (
    <CollapsibleCard header={`Self-Reflective RAG — ${totalRelevant}/${totalRetrieved} items passed grading (${pct}%)`}>
      <div className="sr-metric-row">
        <div className="sr-metric">
          <span className="sr-metric__label">Retrieved</span>
          <span className="sr-metric__value">{totalRetrieved}</span>
        </div>
        <div className="sr-metric">
          <span className="sr-metric__label">Relevant</span>
          <span className="sr-metric__value">{totalRelevant}</span>
        </div>
        <div className="sr-metric">
          <span className="sr-metric__label">Pass Rate</span>
          <span className="sr-metric__value">{pct}%</span>
        </div>
      </div>

      {entries.map((entry, i) => {
        const query = typeof entry.query === "string" ? entry.query : "";
        const cycles = typeof entry.cycles === "number" ? entry.cycles : null;
        const rewritten = Array.isArray(entry.rewritten_queries) ? (entry.rewritten_queries as string[]) : [];
        const skipped = Boolean(entry.grading_skipped);

        const details: ReactNode[] = [];
        if (query) {
          details.push(
            <span key="query">
              <strong>Query:</strong> {truncate(query, 120)}
            </span>,
          );
        }
        if (cycles !== null) {
          details.push(
            <span key="cycles">
              <strong>Cycles:</strong> {cycles}
            </span>,
          );
        }
        if (rewritten.length > 0) {
          details.push(
            <span key="rewritten">
              <strong>Query rewritten:</strong> {truncate(rewritten[0], 80)}
            </span>,
          );
        }
        if (skipped) {
          details.push(<em key="skipped">Grading skipped (all items returned true — silent LLM failure)</em>);
        }

        if (details.length === 0) return null;

        const multiple = entries.length > 1;

        return (
          <p className="rag-reflection-panel__entry" key={i}>
            {multiple && <strong>Query {i + 1}: </strong>}
            {details.map((detail, j) => (
              <span key={j}>
                {j > 0 && (multiple ? " · " : <br />)}
                {detail}
              </span>
            ))}
          </p>
        );
      })}
    </CollapsibleCard>
  );
}

export default RagReflectionPanel;
