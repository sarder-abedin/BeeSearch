import { useState } from "react";
import type { SourceItem } from "../api/types";
import { citationHeader } from "./citationFormat";

interface CitationCardProps {
  citation: SourceItem;
}

export default function CitationCard({ citation }: CitationCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="citation-card">
      <button
        type="button"
        className="citation-card__toggle"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span className="citation-card__chevron" aria-hidden="true">
          {expanded ? "▾" : "▸"}
        </span>
        {citationHeader(citation)}
      </button>
      {expanded && (
        <div className="citation-card__body">
          {citation.snippet && (
            <blockquote className="citation-card__snippet">{citation.snippet}</blockquote>
          )}
          {citation.apa && <p className="citation-card__apa">{citation.apa}</p>}
          {citation.url && (
            <p>
              <a href={citation.url} target="_blank" rel="noreferrer">
                Open source
              </a>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
