import { useEffect, useState } from "react";
import { listTemplates } from "../../api/systematicReview";
import type { SRTemplate } from "../../api/systematicReviewTypes";
import "./TemplatePicker.css";

interface TemplatePickerProps {
  onApply: (template: SRTemplate) => void;
}

function TemplatePicker({ onApply }: TemplatePickerProps) {
  const [templates, setTemplates] = useState<SRTemplate[]>([]);
  const [choice, setChoice] = useState("_none");

  useEffect(() => {
    listTemplates()
      .then(setTemplates)
      .catch(() => setTemplates([]));
  }, []);

  const selected = templates.find((t) => t.key === choice) ?? null;

  return (
    <details className="template-picker">
      <summary>New to systematic reviews? Start from a guided template (optional)</summary>
      <div className="template-picker__body">
        <p className="template-picker__caption">
          Pick a starting point for a common review type — it pre-fills the research question
          and inclusion/exclusion criteria below, and you can edit anything afterwards. Prefer to
          write your own? Just skip this and start typing.
        </p>
        <label htmlFor="sr-template-select">Template</label>
        <select
          id="sr-template-select"
          value={choice}
          onChange={(e) => setChoice(e.target.value)}
        >
          <option value="_none">— Start from scratch —</option>
          {templates.map((t) => (
            <option key={t.key} value={t.key}>
              {t.label}
            </option>
          ))}
        </select>

        {selected && (
          <div className="template-picker__preview">
            <p className="template-picker__description">
              <em>{selected.description}</em>
            </p>
            <p>
              <strong>Suggested research question:</strong> {selected.research_question}
            </p>
            <div className="template-picker__criteria">
              <div>
                <strong>Inclusion criteria</strong>
                <ul>
                  {selected.inclusion.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <strong>Exclusion criteria</strong>
                <ul>
                  {selected.exclusion.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
            <p className="template-picker__note">{selected.note}</p>
            <button type="button" onClick={() => onApply(selected)}>
              Use this template
            </button>
          </div>
        )}
      </div>
    </details>
  );
}

export default TemplatePicker;
