import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { runCrossDocumentSummary } from "../../../api/notebookAdvanced";
import { RunControls, TextExportButtons } from "./shared";
import { useAdvancedToolJob } from "./useAdvancedToolJob";

interface CrossDocumentSummaryPanelProps {
  notebookId: string;
}

/** Mirrors ui/tabs/notebook.py::_tab_cross_summary's top half (the
 * Section-by-Section Breakdown half has no REST endpoint yet). */
function CrossDocumentSummaryPanel({ notebookId }: CrossDocumentSummaryPanelProps) {
  const job = useAdvancedToolJob();
  const { state, jobId, result, error } = job;

  return (
    <div className="advanced-tools-tab__panel">
      <h3>Cross-Document Summary</h3>
      <p>
        Synthesizes all notebook sources into a unified markdown summary covering common themes,
        complementary contributions, contradictions, and key takeaways.
      </p>

      <RunControls
        state={state}
        runLabel="Generate Summary"
        rerunLabel="Regenerate Summary"
        spinnerText="Generating cross-document summary…"
        error={error}
        errorPrefix="Summary generation failed"
        onRun={() => job.run(() => runCrossDocumentSummary({ notebook_id: notebookId }))}
        onClear={job.clear}
      />

      {state === "done" &&
        result &&
        jobId &&
        (result.summary ? (
          <>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.summary}</ReactMarkdown>
            <TextExportButtons jobId={jobId} artifact="summary" filenameBase="summary" documentArtifact="summary" />
          </>
        ) : (
          <p className="sr-info">No summary was generated for this run.</p>
        ))}
    </div>
  );
}

export default CrossDocumentSummaryPanel;
