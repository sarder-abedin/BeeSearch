import { forwardRef, useEffect, useImperativeHandle, useState } from "react";
import { checkGrammar } from "../../api/systematicReview";
import "./GrammarGate.css";

type GateMode = "as_typed" | "check_fix";

interface Cached {
  source: string;
  corrected: string;
  final: string | null;
}

export interface GrammarGateHandle {
  /** Mirrors `render_query_gate`'s return -- read only when the caller is about to submit. */
  resolve: () => { text: string; ready: boolean };
}

interface GrammarGateProps {
  rawText: string;
  contextHint: string;
  fieldId: string;
}

const GrammarGate = forwardRef<GrammarGateHandle, GrammarGateProps>(function GrammarGate(
  { rawText, contextHint, fieldId },
  ref,
) {
  const [mode, setMode] = useState<GateMode>("as_typed");
  const [cached, setCached] = useState<Cached | null>(null);
  const [checking, setChecking] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState("");

  useEffect(() => {
    if (mode !== "check_fix" || !rawText.trim()) return;
    if (cached && cached.source === rawText) return;

    let cancelled = false;
    setChecking(true);
    checkGrammar({ text: rawText, context_hint: contextHint })
      .then((result) => {
        if (cancelled) return;
        setCached({
          source: rawText,
          corrected: result.corrected,
          final: result.changed ? null : rawText,
        });
        setEditing(false);
      })
      .catch(() => {
        if (cancelled) return;
        // Fails safe like the backend's check_and_fix_grammar -- treat as unchanged.
        setCached({ source: rawText, corrected: rawText, final: rawText });
      })
      .finally(() => {
        if (!cancelled) setChecking(false);
      });

    return () => {
      cancelled = true;
    };
  }, [mode, rawText, contextHint, cached]);

  useImperativeHandle(
    ref,
    () => ({
      resolve: () => {
        if (mode === "as_typed" || !rawText.trim()) return { text: rawText, ready: true };
        if (cached && cached.source === rawText && cached.final !== null) {
          return { text: cached.final, ready: true };
        }
        return { text: rawText, ready: false };
      },
    }),
    [mode, rawText, cached],
  );

  const showSuggestion = mode === "check_fix" && rawText.trim() && cached?.source === rawText && cached.final === null;
  const resolvedCaption =
    mode === "check_fix" &&
    cached?.source === rawText &&
    cached.final !== null &&
    cached.final !== cached.source
      ? cached.final
      : null;

  return (
    <div className="grammar-gate">
      <div className="grammar-gate__toggle" role="radiogroup" aria-label="Grammar check">
        <label>
          <input
            type="radio"
            name={`gc-mode-${fieldId}`}
            checked={mode === "as_typed"}
            onChange={() => setMode("as_typed")}
          />
          Use as typed
        </label>
        <label>
          <input
            type="radio"
            name={`gc-mode-${fieldId}`}
            checked={mode === "check_fix"}
            onChange={() => setMode("check_fix")}
          />
          Check grammar before running
        </label>
      </div>

      {checking && <p className="grammar-gate__checking">Checking spelling and punctuation…</p>}

      {resolvedCaption && (
        <p className="grammar-gate__caption">
          Using corrected version: <em>{resolvedCaption.slice(0, 140)}{resolvedCaption.length > 140 ? "…" : ""}</em>
        </p>
      )}

      {showSuggestion && cached && (
        <div className="grammar-gate__suggestion">
          <p className="grammar-gate__suggestion-intro">
            <strong>Suggested grammar fixes</strong> — review below, then choose how to proceed.
          </p>
          <div className="grammar-gate__comparison">
            <div>
              <p className="grammar-gate__comparison-label">Your version</p>
              <blockquote>{cached.source}</blockquote>
            </div>
            <div>
              <p className="grammar-gate__comparison-label">Suggested correction</p>
              <blockquote>{cached.corrected}</blockquote>
            </div>
          </div>
          <div className="grammar-gate__actions">
            <button
              type="button"
              onClick={() => setCached({ ...cached, final: cached.corrected })}
            >
              Use suggested
            </button>
            <button type="button" onClick={() => setCached({ ...cached, final: cached.source })}>
              Keep my original
            </button>
            <button
              type="button"
              onClick={() => {
                setEditValue(cached.corrected);
                setEditing(true);
              }}
            >
              Edit it myself
            </button>
          </div>

          {editing && (
            <div className="grammar-gate__edit">
              <label htmlFor={`gc-edit-${fieldId}`}>Your edited version</label>
              <textarea
                id={`gc-edit-${fieldId}`}
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                rows={4}
              />
              <button
                type="button"
                onClick={() => {
                  setCached({ ...cached, final: editValue.trim() });
                  setEditing(false);
                }}
              >
                Use this edited version
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export default GrammarGate;
