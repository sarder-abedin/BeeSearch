import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { runCitationTimeline } from "../../../api/notebookAdvanced";
import type { CitationTimelineItem } from "../../../api/notebookAdvancedTypes";
import { downloadText } from "../../../utils/download";
import { resolveSourceLabel } from "./format";
import { RunControls } from "./shared";
import { useAdvancedToolJob, useModelOverrides } from "./useAdvancedToolJob";

interface CitationTimelinePanelProps {
  notebookId: string;
  sourceNames: string[];
}

function escapePipes(s: string): string {
  return s.replace(/\|/g, "\\|");
}

/** Mirrors ui/tabs/notebook.py::_tab_timeline's markdown table algorithm
 * verbatim: header + separator row, then one row per item with title/authors/
 * gist pipe-escaped (gist's newlines collapsed to spaces first), title
 * optionally wrapped as a link when `url` is set, and source resolved to a
 * 20-char-truncated filename or "—" via resolveSourceLabel. Used for both the
 * on-screen render and the downloaded .md, same as the Streamlit tab. */
function composeCitationTimelineMarkdown(items: CitationTimelineItem[], sourceNames: string[]): string {
  const lines = ["| Year | Title | Authors | Key Idea | Source |", "|------|-------|---------|----------|--------|"];
  for (const item of items) {
    const title = escapePipes(item.title);
    const titleMd = item.url ? `[${title}](${item.url})` : title;
    const authors = escapePipes(item.authors);
    const gist = escapePipes(item.gist.replace(/\n/g, " "));
    const source = resolveSourceLabel(item.source, sourceNames, 20);
    lines.push(`| ${item.year} | ${titleMd} | ${authors} | ${gist} | ${source} |`);
  }
  return lines.join("\n");
}

/** Mirrors ui/tabs/notebook.py::_tab_timeline. No server text-export endpoint
 * for this tool -- the table is composed client-side from `result.timeline`
 * and that same string backs both the on-screen render and the download. */
function CitationTimelinePanel({ notebookId, sourceNames }: CitationTimelinePanelProps) {
  const job = useAdvancedToolJob();
  const overrides = useModelOverrides();
  const { state, result, error } = job;
  const [enrich, setEnrich] = useState(false);

  return (
    <div className="advanced-tools-tab__panel">
      <h3>Citation Timeline</h3>
      <p>
        Builds a citation timeline from the references/bibliography section of each source: when each
        cited work was published, who wrote it, and a one-line gist of its key idea.
      </p>

      <label
        className="sr-explore-panel__checkbox"
        title="Look up each cited work on Semantic Scholar and use its abstract/TL;DR for the gist instead of a title-only guess. Slower, and requires internet access."
      >
        <input
          type="checkbox"
          checked={enrich}
          disabled={state === "running"}
          onChange={(e) => setEnrich(e.target.checked)}
        />
        Enrich with abstracts (Semantic Scholar)
      </label>

      <RunControls
        state={state}
        runLabel="Extract Citation Timeline"
        rerunLabel="Regenerate Citation Timeline"
        spinnerText="Extracting citation timeline…"
        error={error}
        errorPrefix="Citation timeline extraction failed"
        onRun={() => job.run(() => runCitationTimeline({ notebook_id: notebookId, enrich_with_abstracts: enrich, ...overrides }))}
        onClear={job.clear}
      />

      {state === "done" &&
        result &&
        (result.timeline.length === 0 ? (
          <p className="sr-info">No citation timeline was generated for this run.</p>
        ) : (
          (() => {
            const tableMd = composeCitationTimelineMarkdown(result.timeline, sourceNames);
            return (
              <>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{tableMd}</ReactMarkdown>
                <div className="sr-button-row">
                  <button
                    type="button"
                    className="sr-button"
                    onClick={() => downloadText(tableMd, "citation_timeline.md", "text/markdown")}
                  >
                    Download (.md)
                  </button>
                </div>
              </>
            );
          })()
        ))}
    </div>
  );
}

export default CitationTimelinePanel;
