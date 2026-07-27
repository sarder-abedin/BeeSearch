import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { runFaq } from "../../../api/notebookAdvanced";
import type { FaqItem } from "../../../api/notebookAdvancedTypes";
import { downloadText } from "../../../utils/download";
import CollapsibleCard from "../../sr/CollapsibleCard";
import { resolveValidSourceLabels } from "./format";
import { RunControls } from "./shared";
import { useAdvancedToolJob, useModelOverrides } from "./useAdvancedToolJob";

interface FaqPanelProps {
  notebookId: string;
  sourceNames: string[];
}

/** Mirrors ui/tabs/notebook.py::_tab_faq's "### question\nanswer" export
 * format exactly (no sources line in the exported markdown -- those are
 * display-only, via each item's caption). */
function composeFaqMarkdown(items: FaqItem[]): string {
  return items.map((it) => `### ${it.question}\n${it.answer}`).join("\n\n");
}

function FaqPanel({ notebookId, sourceNames }: FaqPanelProps) {
  const job = useAdvancedToolJob();
  const overrides = useModelOverrides();
  const { state, result, error } = job;
  const [nQuestions, setNQuestions] = useState(8);

  return (
    <div className="advanced-tools-tab__panel">
      <h3>FAQ</h3>
      <p>Auto-generates frequently asked questions with grounded answers drawn from your notebook sources.</p>

      <div className="sr-field">
        <label htmlFor="faq-n-questions">Number of questions</label>
        <input
          id="faq-n-questions"
          type="number"
          min={4}
          max={16}
          value={nQuestions}
          disabled={state === "running"}
          onChange={(e) => setNQuestions(Math.min(16, Math.max(4, Number(e.target.value) || 4)))}
        />
      </div>

      <RunControls
        state={state}
        runLabel="Generate FAQ"
        rerunLabel="Regenerate FAQ"
        spinnerText="Generating FAQ…"
        error={error}
        errorPrefix="FAQ generation failed"
        onRun={() => job.run(() => runFaq({ notebook_id: notebookId, n_questions: nQuestions, ...overrides }))}
        onClear={job.clear}
      />

      {state === "done" &&
        result &&
        (result.faqs.length === 0 ? (
          <p className="sr-info">No FAQ items were generated for this run.</p>
        ) : (
          <>
            {result.faqs.map((item, i) => {
              const labels = resolveValidSourceLabels(item.sources, sourceNames);
              return (
                <CollapsibleCard key={i} header={item.question}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.answer}</ReactMarkdown>
                  {labels.length > 0 && <p className="sr-caption">Sources: {labels.join(", ")}</p>}
                </CollapsibleCard>
              );
            })}
            <div className="sr-button-row">
              <button
                type="button"
                className="sr-button"
                onClick={() => downloadText(composeFaqMarkdown(result.faqs), "faq.md", "text/markdown")}
              >
                Download FAQ (.md)
              </button>
            </div>
          </>
        ))}
    </div>
  );
}

export default FaqPanel;
