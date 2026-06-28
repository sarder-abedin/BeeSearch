import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { runStudyComparison } from "../../../api/notebookAdvanced";
import { RunControls, TextExportButtons } from "./shared";
import { useAdvancedToolJob } from "./useAdvancedToolJob";

interface StudyComparisonPanelProps {
  notebookId: string;
}

/** Mirrors ui/tabs/notebook.py::_tab_study_comparison. */
function StudyComparisonPanel({ notebookId }: StudyComparisonPanelProps) {
  const job = useAdvancedToolJob();
  const { state, jobId, result, error } = job;

  return (
    <div className="advanced-tools-tab__panel">
      <h3>Study Comparison</h3>
      <p>Builds a structured comparison table of methodology, sample size, findings, and limitations across all sources.</p>

      <RunControls
        state={state}
        runLabel="Generate Study Comparison"
        rerunLabel="Regenerate Study Comparison"
        spinnerText="Comparing studies…"
        error={error}
        errorPrefix="Study comparison failed"
        onRun={() => job.run(() => runStudyComparison({ notebook_id: notebookId }))}
        onClear={job.clear}
      />

      {state === "done" &&
        result &&
        jobId &&
        (result.study_comparison ? (
          <>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.study_comparison}</ReactMarkdown>
            <TextExportButtons
              jobId={jobId}
              artifact="study-comparison"
              filenameBase="study_comparison"
              documentArtifact="study-comparison"
            />
          </>
        ) : (
          <p className="sr-info">No study comparison was generated for this run.</p>
        ))}
    </div>
  );
}

export default StudyComparisonPanel;
