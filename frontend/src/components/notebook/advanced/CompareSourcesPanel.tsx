import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { runCompareSources } from "../../../api/notebookAdvanced";
import type { SourceMeta } from "../../../api/notebookTypes";
import { RunControls, TextExportButtons } from "./shared";
import { useAdvancedToolJob, useModelOverrides } from "./useAdvancedToolJob";

interface CompareSourcesPanelProps {
  notebookId: string;
  sources: SourceMeta[];
}

/** Mirrors ui/tabs/notebook.py::_tab_compare, including its "need >= 2
 * sources" guard and its Source B default of index min(1, len-1). */
function CompareSourcesPanel({ notebookId, sources }: CompareSourcesPanelProps) {
  const job = useAdvancedToolJob();
  const overrides = useModelOverrides();
  const { state, jobId, result, error } = job;
  const [docIdA, setDocIdA] = useState(sources[0]?.doc_id ?? "");
  const [docIdB, setDocIdB] = useState(sources[Math.min(1, sources.length - 1)]?.doc_id ?? "");

  if (sources.length < 2) {
    return (
      <div className="advanced-tools-tab__panel">
        <h3>Compare Sources</h3>
        <p className="sr-info">Add at least two sources to use source comparison.</p>
      </div>
    );
  }

  return (
    <div className="advanced-tools-tab__panel">
      <h3>Compare Sources</h3>
      <p>Select two sources to compare side-by-side.</p>

      <div className="sr-two-col">
        <div className="sr-field">
          <label htmlFor="compare-source-a">Source A</label>
          <select id="compare-source-a" value={docIdA} disabled={state === "running"} onChange={(e) => setDocIdA(e.target.value)}>
            {sources.map((s) => (
              <option key={s.doc_id} value={s.doc_id}>
                {s.filename}
              </option>
            ))}
          </select>
        </div>
        <div className="sr-field">
          <label htmlFor="compare-source-b">Source B</label>
          <select id="compare-source-b" value={docIdB} disabled={state === "running"} onChange={(e) => setDocIdB(e.target.value)}>
            {sources.map((s) => (
              <option key={s.doc_id} value={s.doc_id}>
                {s.filename}
              </option>
            ))}
          </select>
        </div>
      </div>

      <RunControls
        state={state}
        runLabel="Compare Sources"
        rerunLabel="Compare Again"
        spinnerText="Comparing sources…"
        error={error}
        errorPrefix="Source comparison failed"
        onRun={() => job.run(() => runCompareSources({ notebook_id: notebookId, doc_id_a: docIdA, doc_id_b: docIdB, ...overrides }))}
        onClear={job.clear}
      />

      {state === "done" &&
        result &&
        jobId &&
        (result.comparison ? (
          <>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.comparison}</ReactMarkdown>
            <TextExportButtons jobId={jobId} artifact="comparison" filenameBase="comparison" />
          </>
        ) : (
          <p className="sr-info">No comparison was generated for this run.</p>
        ))}
    </div>
  );
}

export default CompareSourcesPanel;
