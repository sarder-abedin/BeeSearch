import { runMindmap } from "../../../api/notebookAdvanced";
import { DotExportPanel, RunControls } from "./shared";
import { useAdvancedToolJob, useModelOverrides } from "./useAdvancedToolJob";

interface MindmapPanelProps {
  notebookId: string;
}

/** Mirrors ui/tabs/notebook.py::_tab_mindmap. */
function MindmapPanel({ notebookId }: MindmapPanelProps) {
  const job = useAdvancedToolJob();
  const overrides = useModelOverrides();
  const { state, jobId, result, error } = job;

  return (
    <div className="advanced-tools-tab__panel">
      <h3>Mind Map</h3>
      <p>Extracts key concepts and their relationships from your sources and renders them as a mind map.</p>

      <RunControls
        state={state}
        runLabel="Generate Mind Map"
        rerunLabel="Regenerate Mind Map"
        spinnerText="Extracting mind map…"
        error={error}
        errorPrefix="Mind map generation failed"
        onRun={() => job.run(() => runMindmap({ notebook_id: notebookId, ...overrides }))}
        onClear={job.clear}
      />

      {state === "done" &&
        result &&
        jobId &&
        (result.mindmap_dot ? (
          <DotExportPanel jobId={jobId} dot={result.mindmap_dot} artifact="mindmap" filenameBase="mindmap" previewAlt="Mind map" />
        ) : (
          <p className="sr-info">No mind map was generated for this run.</p>
        ))}
    </div>
  );
}

export default MindmapPanel;
