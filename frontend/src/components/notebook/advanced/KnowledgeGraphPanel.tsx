import { runKnowledgeGraph } from "../../../api/notebookAdvanced";
import { DotExportPanel, RunControls } from "./shared";
import { useAdvancedToolJob } from "./useAdvancedToolJob";

interface KnowledgeGraphPanelProps {
  notebookId: string;
}

/** Mirrors ui/tabs/notebook.py::_tab_knowledge_graph (the standalone Knowledge
 * Graph tab, distinct from the 7-agent pipeline's own knowledge graph step). */
function KnowledgeGraphPanel({ notebookId }: KnowledgeGraphPanelProps) {
  const job = useAdvancedToolJob();
  const { state, jobId, result, error } = job;

  return (
    <div className="advanced-tools-tab__panel">
      <h3>Knowledge Graph</h3>
      <p>Extracts entities and relationships from your sources and renders them as a knowledge graph.</p>

      <RunControls
        state={state}
        runLabel="Extract Knowledge Graph"
        rerunLabel="Regenerate Knowledge Graph"
        spinnerText="Extracting knowledge graph…"
        error={error}
        errorPrefix="Knowledge graph generation failed"
        onRun={() => job.run(() => runKnowledgeGraph({ notebook_id: notebookId }))}
        onClear={job.clear}
      />

      {state === "done" &&
        result &&
        jobId &&
        (result.knowledge_graph_dot ? (
          <DotExportPanel
            jobId={jobId}
            dot={result.knowledge_graph_dot}
            artifact="knowledge-graph"
            filenameBase="knowledge_graph"
            previewAlt="Knowledge graph"
          />
        ) : (
          <p className="sr-info">No knowledge graph was generated for this run.</p>
        ))}
    </div>
  );
}

export default KnowledgeGraphPanel;
