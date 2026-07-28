import { useState } from "react";
import type { SavedReview, SourceMeta } from "../../api/notebookTypes";
import AudioSummaryPanel from "./advanced/AudioSummaryPanel";
import CitationTimelinePanel from "./advanced/CitationTimelinePanel";
import CompareSourcesPanel from "./advanced/CompareSourcesPanel";
import CrossDocumentSummaryPanel from "./advanced/CrossDocumentSummaryPanel";
import FaqPanel from "./advanced/FaqPanel";
import KnowledgeGraphPanel from "./advanced/KnowledgeGraphPanel";
import LiteratureReviewPanel from "./advanced/LiteratureReviewPanel";
import MindmapPanel from "./advanced/MindmapPanel";
import ReviewerPanel from "./advanced/ReviewerPanel";
import StudyComparisonPanel from "./advanced/StudyComparisonPanel";
import "../sr/sr-common.css";
import "./AdvancedToolsTab.css";

interface AdvancedToolsTabProps {
  notebookId: string;
  sources: SourceMeta[];
  savedReviews?: Record<string, SavedReview>;
}

type AdvancedTool =
  | "cross-document-summary"
  | "faq"
  | "literature-review"
  | "mindmap"
  | "audio-summary"
  | "compare-sources"
  | "knowledge-graph"
  | "citation-timeline"
  | "study-comparison"
  | "reviewer";

/** Order mirrors ui/tabs/notebook.py's tab_summary..tab_study Streamlit tabs
 * (Tab 2 through Tab 10 of its 13-tab st.tabs(...) call). */
const ADVANCED_TOOLS: { key: AdvancedTool; label: string }[] = [
  { key: "cross-document-summary", label: "Summary" },
  { key: "faq", label: "FAQ" },
  { key: "literature-review", label: "Lit Review" },
  { key: "mindmap", label: "Mind Map" },
  { key: "audio-summary", label: "Audio" },
  { key: "compare-sources", label: "Compare" },
  { key: "knowledge-graph", label: "Graph" },
  { key: "citation-timeline", label: "Citation Timeline" },
  { key: "study-comparison", label: "Study Table" },
  { key: "reviewer", label: "Reviewer" },
];

/** Mirrors ExploreTab.tsx's radiogroup-of-tools pattern, adapted for Phase
 * C's 9 standalone notebook_advanced.py tools. Wrapping the radiogroup+panel
 * subtree in `key={notebookId}` forces a full remount on notebook switch, so
 * each of the 9 panels' independent useAdvancedToolJob state resets together
 * without each panel needing its own "did the notebook change" effect --
 * mirrors PipelineTab.tsx's own `key={jobId}` remount trick one level up. */
function AdvancedToolsTab({ notebookId, sources, savedReviews }: AdvancedToolsTabProps) {
  return (
    <div className="advanced-tools-tab" key={notebookId}>
      <AdvancedToolsToolSwitcher notebookId={notebookId} sources={sources} savedReviews={savedReviews} />
    </div>
  );
}

function AdvancedToolsToolSwitcher({ notebookId, sources, savedReviews }: AdvancedToolsTabProps) {
  const [choice, setChoice] = useState<AdvancedTool>("cross-document-summary");
  const sourceNames = sources.map((s) => s.filename);

  return (
    <>
      <p>Standalone analysis tools that run over this notebook&apos;s sources — pick one below.</p>

      <div className="advanced-tools-tab__radio" role="radiogroup" aria-label="Advanced tool">
        {ADVANCED_TOOLS.map((tool) => (
          <label
            key={tool.key}
            className={
              choice === tool.key
                ? "advanced-tools-tab__radio-option advanced-tools-tab__radio-option--selected"
                : "advanced-tools-tab__radio-option"
            }
          >
            <input
              type="radio"
              name="advanced-tool"
              value={tool.key}
              checked={choice === tool.key}
              onChange={() => setChoice(tool.key)}
            />
            {tool.label}
          </label>
        ))}
      </div>
      <hr />

      {choice === "cross-document-summary" && <CrossDocumentSummaryPanel notebookId={notebookId} />}
      {choice === "faq" && <FaqPanel notebookId={notebookId} sourceNames={sourceNames} />}
      {choice === "literature-review" && <LiteratureReviewPanel notebookId={notebookId} />}
      {choice === "mindmap" && <MindmapPanel notebookId={notebookId} />}
      {choice === "audio-summary" && <AudioSummaryPanel notebookId={notebookId} />}
      {choice === "compare-sources" && <CompareSourcesPanel notebookId={notebookId} sources={sources} />}
      {choice === "knowledge-graph" && <KnowledgeGraphPanel notebookId={notebookId} />}
      {choice === "citation-timeline" && <CitationTimelinePanel notebookId={notebookId} sourceNames={sourceNames} />}
      {choice === "study-comparison" && <StudyComparisonPanel notebookId={notebookId} />}
      {choice === "reviewer" && <ReviewerPanel notebookId={notebookId} sources={sources} savedReviews={savedReviews} />}
    </>
  );
}

export default AdvancedToolsTab;
