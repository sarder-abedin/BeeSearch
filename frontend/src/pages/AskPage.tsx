import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ApiError } from "../api/client";
import { askResearchAssistant, pollAskJob } from "../api/researchAssistant";
import type { AskJobStatus, AskResult } from "../api/types";
import CitationCard from "../components/CitationCard";
import GrammarGate, { type GrammarGateHandle } from "../components/sr/GrammarGate";
import { useSettings } from "../context/SettingsContext";
import "./AskPage.css";

type RunStatus = "idle" | "running" | "done" | "error";

const VALIDATION_MESSAGE = "Please enter a research question.";
const GROUNDED_WARNING =
  "No published sources could be retrieved for this question — the answer below is from " +
  "general model knowledge and should be verified against primary literature.";
const INITIAL_LABEL = "Searching published literature…";

/** Byte-exact port of ui/tabs/research_assistant.py::_run_and_store's `_cb` stage→label map. */
function stageLabel(
  stage: string | null | undefined,
  info: Record<string, unknown>,
  includeWeb: boolean,
): string | null {
  switch (stage) {
    case "searching":
      return `Searching Google Scholar · arXiv · Semantic Scholar${includeWeb ? " · web" : ""}…`;
    case "reading":
      return `Reading ${Number(info.academic_count ?? 0)} paper(s) and ${Number(
        info.web_count ?? 0,
      )} web result(s)…`;
    case "answering":
      return info.grounded
        ? "Composing a grounded answer…"
        : "No sources found — answering from general knowledge…";
    case "done":
      return "Done.";
    default:
      return null;
  }
}

export default function AskPage() {
  const settings = useSettings();
  const [question, setQuestion] = useState("");
  const [includeWeb, setIncludeWeb] = useState(true);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [label, setLabel] = useState(INITIAL_LABEL);
  const [result, setResult] = useState<AskResult | null>(null);
  const [validationWarning, setValidationWarning] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const grammarRef = useRef<GrammarGateHandle | null>(null);

  async function runAsk(q: string, web: boolean) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatus("running");
    setLabel(INITIAL_LABEL);
    setResult(null);

    try {
      const { job_id } = await askResearchAssistant({
        question: q,
        include_web: web,
        include_crossref: settings.includeCrossref,
        model: settings.model,
        num_ctx: settings.numCtx,
        temperature_level: settings.temperatureLevel,
      });
      const final: AskJobStatus = await pollAskJob(
        job_id,
        (update) => {
          const next = stageLabel(update.stage, update.stage_info, web);
          if (next) setLabel(next);
        },
        controller.signal,
      );

      if (final.status === "done" && final.result) {
        setResult(final.result);
        setLabel("Done.");
        setStatus("done");
      } else {
        const message = final.error ?? "Unknown error.";
        setLabel(`Failed: ${message}`);
        setStatus("error");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const message = err instanceof ApiError ? err.detail : (err as Error).message;
      setLabel(`Failed: ${message}`);
      setStatus("error");
    }
  }

  function handleAsk() {
    const trimmed = question.trim();
    if (!trimmed) {
      setValidationWarning(VALIDATION_MESSAGE);
      return;
    }
    const resolved = grammarRef.current?.resolve() ?? { text: trimmed, ready: true };
    if (!resolved.ready) {
      setValidationWarning("Please resolve the grammar suggestion above, then click Ask again.");
      return;
    }
    setValidationWarning(null);
    void runAsk(resolved.text, includeWeb);
  }

  function handleFollowup(fq: string) {
    setQuestion(fq);
    setValidationWarning(null);
    void runAsk(fq, includeWeb);
  }

  const citations = result?.citations ?? [];
  const sources = result?.sources ?? [];
  const followups = result?.suggested_questions ?? [];

  return (
    <main className="ask-page">
      <h1>Mode 3 — AI Research Assistant</h1>
      <p>
        Ask a free-form research question and get an answer grounded in{" "}
        <strong>published literature</strong> with inline citations — no documents to
        upload, no PRISMA workflow. BeeSearch searches Google Scholar, arXiv, Semantic
        Scholar (and the web), reads what it finds, and cites its sources. Best for
        orienting questions; use <strong>Mode 1</strong> for an exhaustive systematic
        review.
      </p>
      <hr />

      <label className="ask-page__label" htmlFor="ra-question">
        Research question
      </label>
      <textarea
        id="ra-question"
        className="ask-page__textarea"
        rows={4}
        placeholder="e.g. Does intermittent fasting improve insulin sensitivity in adults?"
        value={question}
        onChange={(e) => {
          setQuestion(e.target.value);
          setValidationWarning(null);
        }}
      />
      <GrammarGate
        ref={grammarRef}
        rawText={question}
        contextHint="research question for an AI research assistant"
        fieldId="ra-question"
      />

      <div className="ask-page__controls">
        <label
          className="ask-page__checkbox"
          title="Supplement academic sources with general web results, cited the same way."
        >
          <input
            type="checkbox"
            checked={includeWeb}
            onChange={(e) => {
              setIncludeWeb(e.target.checked);
              setValidationWarning(null);
            }}
          />
          Also search the web (DuckDuckGo)
        </label>
        <button
          type="button"
          className="ask-page__ask-button"
          onClick={handleAsk}
          disabled={status === "running"}
        >
          Ask
        </button>
      </div>

      {validationWarning && <p className="ask-page__warning">{validationWarning}</p>}

      {status !== "idle" && (
        <div className={`ask-page__status ask-page__status--${status}`} role="status">
          {status === "running" && <span className="ask-page__spinner" aria-hidden="true" />}
          <span>{label}</span>
        </div>
      )}

      {result && (
        <>
          <hr />
          {!result.grounded && <p className="ask-page__warning">{GROUNDED_WARNING}</p>}

          <div className="ask-page__answer">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {result.answer || "*No answer generated.*"}
            </ReactMarkdown>
          </div>

          {citations.length > 0 && (
            <>
              <hr />
              <h2>Citations ({citations.length})</h2>
              {citations.map((c) => (
                <CitationCard key={c.n} citation={c} />
              ))}
            </>
          )}

          <p className="ask-page__caption">
            Searched {result.academic_count} paper(s) and {result.web_count} web
            result(s); {sources.length} used as context, {citations.length} cited in the
            answer.
          </p>

          {followups.length > 0 && (
            <>
              <hr />
              <p>
                <strong>Follow-up questions:</strong>
              </p>
              <div className="ask-page__followups">
                {followups.map((fq, i) => (
                  <button
                    type="button"
                    key={`${i}-${fq}`}
                    className="ask-page__followup-button"
                    onClick={() => handleFollowup(fq)}
                  >
                    {fq}
                  </button>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </main>
  );
}
