import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ApiError } from "../../api/client";
import { getExplainHistory, pollExplainJob, runExplainTurn } from "../../api/notebookExplain";
import type {
  ExplainCitationItem,
  ExplainTurn as ExplainTurnData,
  ExplanationLevel,
  ExplanationStyle,
  SourceDecision,
} from "../../api/notebookExplainTypes";
import EvalResultPanel from "../EvalResultPanel";
import "../sr/sr-common.css";
import "./ExplainTab.css";

interface ExplainTabProps {
  notebookId: string;
  notebookName: string;
}

type TurnStatus = "idle" | "running" | "done" | "error";

/** Mirrors ui/tabs/notebook.py::_tab_explain's style_labels dict and radio options. */
const STYLE_OPTIONS: { key: ExplanationStyle; label: string }[] = [
  { key: "simple", label: "Simple Language" },
  { key: "analogy", label: "Extended Analogy" },
  { key: "walkthrough", label: "Step-by-Step" },
  { key: "debate", label: "For vs. Against" },
];

/** Mirrors the same function's level radio options (default index=1 -> "intermediate"). */
const LEVEL_OPTIONS: { key: ExplanationLevel; label: string }[] = [
  { key: "novice", label: "Novice" },
  { key: "intermediate", label: "Intermediate" },
  { key: "expert", label: "Expert" },
];

const STYLE_LABELS: Record<string, string> = Object.fromEntries(STYLE_OPTIONS.map((o) => [o.key, o.label]));

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.detail : (err as Error).message;
}

/** Mirrors NotebookCitations.tsx, adapted for ExplainCitationItem's string `n`
 * (document excerpts use an int n, online sources use a "Source N" string n --
 * see notebookExplainTypes.ts). */
function ExplainCitations({ citations }: { citations: ExplainCitationItem[] }) {
  const [expanded, setExpanded] = useState(false);
  if (citations.length === 0) return null;

  return (
    <div className="notebook-citations">
      <button
        type="button"
        className="notebook-citations__toggle"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span className="notebook-citations__chevron" aria-hidden="true">
          {expanded ? "▾" : "▸"}
        </span>
        Sources ({citations.length})
      </button>
      {expanded && (
        <div className="notebook-citations__body">
          {citations.map((c, i) => (
            <div className="notebook-citations__item" key={`${c.n}-${i}`}>
              {c.url ? (
                <p>
                  <strong>[{c.n}]</strong>{" "}
                  <a href={c.url} target="_blank" rel="noreferrer">
                    {(c.doc_name || c.url).slice(0, 60)}
                  </a>
                </p>
              ) : (
                <p>
                  <strong>
                    [{c.n}] {c.doc_name}
                  </strong>{" "}
                  · {c.page_label}
                </p>
              )}
              {c.snippet && <blockquote className="notebook-citations__snippet">{c.snippet}</blockquote>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Mirrors _tab_explain's st.info/st.warning/st.caption source-decision badge. */
function SourceDecisionBanner({ decision }: { decision: SourceDecision | null }) {
  if (!decision) return null;
  const { coverage_score: score, used_online, online_count, sources_searched, reason, search_attempted } = decision;

  if (used_online) {
    const labels: string[] = [];
    if (sources_searched.includes("academic")) labels.push("arXiv / Semantic Scholar");
    if (sources_searched.includes("web")) labels.push("web");
    const srcStr = labels.length > 0 ? labels.join(" + ") : "online";
    return (
      <p className="sr-info">
        Document coverage: {score}/10 — {reason}
        <br />
        The response below is split into sections: what your documents cover, why online search
        was needed, and what was found online ({online_count} source(s) from {srcStr}). Each
        online claim is cited with [Source N].
      </p>
    );
  }
  if (search_attempted) {
    return (
      <p className="sr-warning">
        Document coverage: {score}/10 — {reason}
        <br />
        Online search was attempted but found no additional results — this answer uses only your
        documents.
      </p>
    );
  }
  return <p className="sr-caption">Answered from your documents (coverage {score}/10)</p>;
}

function ExplainTab({ notebookId, notebookName }: ExplainTabProps) {
  const [history, setHistory] = useState<ExplainTurnData[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [loadedNotebookId, setLoadedNotebookId] = useState(notebookId);

  const [explanationStyle, setExplanationStyle] = useState<ExplanationStyle>("simple");
  const [explanationLevel, setExplanationLevel] = useState<ExplanationLevel>("intermediate");
  const [message, setMessage] = useState("");
  const [warning, setWarning] = useState<string | null>(null);
  const [status, setStatus] = useState<TurnStatus>("idle");
  const [stageLabel, setStageLabel] = useState("");
  const [lastSourceDecision, setLastSourceDecision] = useState<SourceDecision | null>(null);
  const [lastEvalResult, setLastEvalResult] = useState<Record<string, unknown> | null>(null);
  const [lastConceptHtml, setLastConceptHtml] = useState("");
  const [lastIsRepeat, setLastIsRepeat] = useState(false);
  const [lastUsedStyle, setLastUsedStyle] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  if (notebookId !== loadedNotebookId) {
    setLoadedNotebookId(notebookId);
    setHistory([]);
    setHistoryError(null);
    setStatus("idle");
    setStageLabel("");
    setWarning(null);
    setLastSourceDecision(null);
    setLastEvalResult(null);
    setLastConceptHtml("");
    setLastIsRepeat(false);
    setLastUsedStyle("");
  }

  useEffect(() => {
    abortRef.current?.abort();
    let cancelled = false;
    getExplainHistory(notebookId)
      .then((turns) => {
        if (!cancelled) {
          setHistory(turns);
          setHistoryError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setHistoryError(errorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [notebookId]);

  async function runTurn(text: string) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatus("running");
    setStageLabel("Starting…");
    setLastSourceDecision(null);
    setLastEvalResult(null);
    setLastConceptHtml("");
    setLastIsRepeat(false);
    setLastUsedStyle("");

    const userTurn: ExplainTurnData = {
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
      citations: null,
      suggested_questions: null,
      explanation_style: null,
    };
    setHistory((prev) => [...prev, userTurn]);

    try {
      const { job_id } = await runExplainTurn({
        notebook_id: notebookId,
        message: text,
        explanation_style: explanationStyle,
        explanation_level: explanationLevel,
      });

      const final = await pollExplainJob(
        job_id,
        (update) => {
          const info = update.stage_info ?? {};
          const label = typeof info.label === "string" ? info.label : null;
          if (label) setStageLabel(label);
        },
        controller.signal,
      );

      if (final.status === "done" && final.result) {
        const r = final.result;
        const assistantTurn: ExplainTurnData = {
          role: "assistant",
          content: r.assistant_response,
          timestamp: new Date().toISOString(),
          citations: r.citations,
          suggested_questions: r.suggested_questions,
          explanation_style: r.explanation_style,
        };
        setHistory((prev) => [...prev, assistantTurn]);
        setLastSourceDecision(r.source_decision);
        setLastEvalResult(r.eval_result);
        setLastConceptHtml(r.concept_visual_html);
        setLastIsRepeat(r.is_repeat_clarification);
        setLastUsedStyle(r.explanation_style);
        setStageLabel("Done.");
        setStatus("done");
      } else {
        setStageLabel(`Failed: ${final.error ?? "Unknown error."}`);
        setStatus("error");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setStageLabel(`Failed: ${errorMessage(err)}`);
      setStatus("error");
    }
  }

  function handleSend() {
    const trimmed = message.trim();
    if (!trimmed) {
      setWarning("Please enter a question.");
      return;
    }
    setMessage("");
    setWarning(null);
    void runTurn(trimmed);
  }

  function handleFollowup(q: string) {
    setWarning(null);
    void runTurn(q);
  }

  const lastAssistantTurn = [...history].reverse().find((t) => t.role === "assistant");

  return (
    <div className="explain-tab" key={notebookId}>
      <p>
        Ask questions about your notebook sources in plain language. Choose an explanation{" "}
        <strong>style</strong> — simple language, an extended analogy, a step-by-step walkthrough,
        or a structured debate — and an audience <strong>level</strong> — novice, intermediate, or
        expert — and the agent tailors its response to both.
      </p>

      <div className="explain-tab__radio" role="radiogroup" aria-label="Explanation style">
        {STYLE_OPTIONS.map((opt) => (
          <label
            key={opt.key}
            className={
              explanationStyle === opt.key
                ? "explain-tab__radio-option explain-tab__radio-option--selected"
                : "explain-tab__radio-option"
            }
          >
            <input
              type="radio"
              name="explain-style"
              value={opt.key}
              checked={explanationStyle === opt.key}
              onChange={() => setExplanationStyle(opt.key)}
            />
            {opt.label}
          </label>
        ))}
      </div>

      <div className="explain-tab__radio" role="radiogroup" aria-label="Explanation level">
        {LEVEL_OPTIONS.map((opt) => (
          <label
            key={opt.key}
            className={
              explanationLevel === opt.key
                ? "explain-tab__radio-option explain-tab__radio-option--selected"
                : "explain-tab__radio-option"
            }
          >
            <input
              type="radio"
              name="explain-level"
              value={opt.key}
              checked={explanationLevel === opt.key}
              onChange={() => setExplanationLevel(opt.key)}
            />
            {opt.label}
          </label>
        ))}
      </div>
      <hr />

      {historyError && <p className="sr-error">{historyError}</p>}

      {history.length === 0 ? (
        <p className="sr-caption">
          Type your first question below to start an explanation session grounded in this notebook.
        </p>
      ) : (
        <div className="explain-tab__transcript">
          {history.map((turn, i) => {
            const isLastAssistant = turn.role === "assistant" && turn === lastAssistantTurn;
            return (
              <div key={i} className={`explain-tab__turn explain-tab__turn--${turn.role}`}>
                <p className="explain-tab__turn-role">{turn.role === "user" ? "You" : "Assistant"}</p>
                {isLastAssistant && lastIsRepeat && (
                  <p className="sr-caption">
                    {lastUsedStyle && lastUsedStyle !== explanationStyle
                      ? `This looked like a repeat of an earlier question, so this answer uses ` +
                        `"${STYLE_LABELS[lastUsedStyle] ?? lastUsedStyle}" instead of your selected ` +
                        `style — explaining it differently, not just rewording it.`
                      : "This looked like a repeat of an earlier question — the explanation below " +
                        "takes a different angle than before."}
                  </p>
                )}
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
                {turn.role === "assistant" && (
                  <>
                    <ExplainCitations citations={turn.citations ?? []} />
                    {(turn.suggested_questions ?? []).length > 0 && (
                      <div className="explain-tab__followups">
                        {(turn.suggested_questions ?? []).map((q, qi) => (
                          <button
                            type="button"
                            key={qi}
                            className="explain-tab__followup-button"
                            onClick={() => handleFollowup(q)}
                            disabled={status === "running"}
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    )}
                    {isLastAssistant && lastConceptHtml && (
                      <div className="explain-tab__concept-visual">
                        <p className="explain-tab__concept-visual-label">Visualizing this concept:</p>
                        <iframe
                          title="Concept map"
                          srcDoc={lastConceptHtml}
                          sandbox="allow-scripts"
                          className="explain-tab__concept-visual-frame"
                        />
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          })}
        </div>
      )}

      <SourceDecisionBanner decision={lastSourceDecision} />
      <EvalResultPanel evalResult={lastEvalResult} />

      {warning && <p className="sr-warning">{warning}</p>}

      {status !== "idle" && (
        <div className={`explain-tab__status explain-tab__status--${status}`} role="status">
          {status === "running" && <span className="explain-tab__spinner" aria-hidden="true" />}
          <span>{stageLabel}</span>
        </div>
      )}

      <div className="explain-tab__composer">
        <label htmlFor="explain-message">Message</label>
        <textarea
          id="explain-message"
          rows={2}
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
            setWarning(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder={`Ask anything about ${notebookName}…`}
        />
        <button type="button" className="sr-button" onClick={handleSend} disabled={status === "running"}>
          Send
        </button>
      </div>
    </div>
  );
}

export default ExplainTab;
