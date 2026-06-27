import { useState } from "react";
import type { CitationItem } from "../api/notebookTypes";

interface NotebookCitationsProps {
  citations: CitationItem[];
}

/** Mirrors ui/tabs/notebook.py::_render_citations's "Sources (N)" expander exactly
 * (minus the "View in PDF" jump, which has no Phase A REST endpoint yet). */
export default function NotebookCitations({ citations }: NotebookCitationsProps) {
  const [expanded, setExpanded] = useState(false);

  if (citations.length === 0) return null;

  return (
    <div className="notebook-citations">
      <button
        type="button"
        className="notebook-citations__toggle"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span className="notebook-citations__chevron" aria-hidden="true">
          {expanded ? "▾" : "▸"}
        </span>
        Sources ({citations.length})
      </button>
      {expanded && (
        <div className="notebook-citations__body">
          {citations.map((c) => (
            <div className="notebook-citations__item" key={c.n}>
              {c.url ? (
                <p>
                  <strong>[{c.n}]</strong>{" "}
                  <a href={c.url} target="_blank" rel="noreferrer">
                    {(c.doc_name || c.url).slice(0, 60)}
                  </a>
                </p>
              ) : (
                <p>
                  <strong>
                    [{c.n}] {c.doc_name}
                  </strong>{" "}
                  · {c.page_label}
                </p>
              )}
              {c.snippet && <blockquote className="notebook-citations__snippet">{c.snippet}</blockquote>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
