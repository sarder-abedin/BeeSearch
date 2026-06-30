import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { runLiteratureReview } from "../../../api/notebookAdvanced";
import type { ReferenceItem } from "../../../api/notebookAdvancedTypes";
import { formatPageLabel } from "./format";
import { RunControls, TextExportButtons } from "./shared";
import { useAdvancedToolJob, useModelOverrides } from "./useAdvancedToolJob";

interface LiteratureReviewPanelProps {
  notebookId: string;
}

interface ReferencesListProps {
  references: ReferenceItem[];
}

/** Mirrors NotebookCitations.tsx's "Sources (N)" item layout, adapted for
 * ReferenceItem (no `url` field, so always the "[n] doc_name · page" form). */
function ReferencesList({ references }: ReferencesListProps) {
  if (references.length === 0) return null;
  return (
    <div className="notebook-citations">
      <p>
        <strong>References ({references.length})</strong>
      </p>
      <div className="notebook-citations__body">
        {references.map((r, i) => (
          <div className="notebook-citations__item" key={r.n ?? i}>
            <p>
              <strong>
                [{r.n}] {r.doc_name}
              </strong>{" "}
              · {formatPageLabel(r.page)}
            </p>
            {r.snippet && <blockquote className="notebook-citations__snippet">{r.snippet}</blockquote>}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Mirrors ui/tabs/notebook.py::_tab_literature_review. The on-screen body
 * excludes references (shown separately, structured); the downloaded .md/
 * .docx/.pdf all use the server's composed body+references text via
 * TextExportButtons -- same split as the Streamlit tab's body vs. full_md. */
function LiteratureReviewPanel({ notebookId }: LiteratureReviewPanelProps) {
  const job = useAdvancedToolJob();
  const overrides = useModelOverrides();
  const { state, jobId, result, error } = job;

  return (
    <div className="advanced-tools-tab__panel">
      <h3>Literature Review</h3>
      <p>
        Generates a formal academic-style literature review with structured sections: introduction,
        background, methodology, key findings, critical analysis, and conclusion.
      </p>

      <RunControls
        state={state}
        runLabel="Generate Literature Review"
        rerunLabel="Regenerate Literature Review"
        spinnerText="Generating literature review…"
        error={error}
        errorPrefix="Literature review generation failed"
        onRun={() => job.run(() => runLiteratureReview({ notebook_id: notebookId, ...overrides }))}
        onClear={job.clear}
      />

      {state === "done" &&
        result &&
        jobId &&
        (result.review ? (
          <>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.review}</ReactMarkdown>
            <ReferencesList references={result.references} />
            <TextExportButtons
              jobId={jobId}
              artifact="review"
              filenameBase="literature_review"
              documentArtifact="review"
            />
          </>
        ) : (
          <p className="sr-info">No literature review was generated for this run.</p>
        ))}
    </div>
  );
}

export default LiteratureReviewPanel;
