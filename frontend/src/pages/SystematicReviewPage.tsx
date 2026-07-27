import { useRef, useState } from "react";
import { ApiError } from "../api/client";
import { pollSRJob, runSystematicReview } from "../api/systematicReview";
import type { SRJobStatus, SRResult, SRTemplate } from "../api/systematicReviewTypes";
import CollapsibleCard from "../components/sr/CollapsibleCard";
import EvalResultPanel from "../components/EvalResultPanel";
import EvidenceTab from "../components/sr/EvidenceTab";
import ExploreTab from "../components/sr/ExploreTab";
import ExportTab from "../components/sr/ExportTab";
import GrammarGate, { type GrammarGateHandle } from "../components/sr/GrammarGate";
import PrismaFlowSummary from "../components/sr/PrismaFlowSummary";
import RagReflectionPanel from "../components/RagReflectionPanel";
import SynthesisTab from "../components/sr/SynthesisTab";
import TemplatePicker from "../components/sr/TemplatePicker";
import { useSettings } from "../context/SettingsContext";
import "../components/sr/sr-common.css";
import "./SystematicReviewPage.css";

type RunStatus = "idle" | "running" | "done" | "error";
type ResultTab = "synthesis" | "evidence" | "explore" | "export";

const RESULT_TABS: { key: ResultTab; label: string }[] = [
  { key: "synthesis", label: "Synthesis" },
  { key: "evidence", label: "Evidence" },
  { key: "explore", label: "Explore" },
  { key: "export", label: "Write-up & Export" },
];

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.detail : (err as Error).message;
}

function splitLines(text: string): string[] {
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
}

function SystematicReviewPage() {
  const settings = useSettings();
  const [question, setQuestion] = useState("");
  const [inclusion, setInclusion] = useState("");
  const [exclusion, setExclusion] = useState("");
  const [notice, setNotice] = useState<{ kind: "warning" | "info"; text: string } | null>(null);

  const [status, setStatus] = useState<RunStatus>("idle");
  const [progressPct, setProgressPct] = useState(0);
  const [progressLabel, setProgressLabel] = useState("Starting");
  const [progressDetail, setProgressDetail] = useState("");
  const [stepLog, setStepLog] = useState<string[]>([]);
  const [elapsedSeconds, setElapsedSeconds] = useState<number | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<SRResult | null>(null);
  const [activeTab, setActiveTab] = useState<ResultTab>("synthesis");

  const rqGateRef = useRef<GrammarGateHandle>(null);
  const incGateRef = useRef<GrammarGateHandle>(null);
  const excGateRef = useRef<GrammarGateHandle>(null);
  const abortRef = useRef<AbortController | null>(null);
  const startRef = useRef(0);

  function handleApplyTemplate(template: SRTemplate) {
    setQuestion(template.research_question);
    setInclusion(template.inclusion.join("\n"));
    setExclusion(template.exclusion.join("\n"));
    setNotice(null);
  }

  async function runReview(researchQuestion: string, inclusionCriteria: string[], exclusionCriteria: string[]) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatus("running");
    setProgressPct(0);
    setProgressLabel("Starting");
    setProgressDetail("");
    setStepLog([]);
    setElapsedSeconds(null);
    setRunError(null);
    setResult(null);
    setActiveTab("synthesis");
    startRef.current = performance.now();

    try {
      const { job_id } = await runSystematicReview({
        research_question: researchQuestion,
        inclusion_criteria: inclusionCriteria,
        exclusion_criteria: exclusionCriteria,
        model: settings.model,
        num_ctx: settings.numCtx,
        max_results: settings.maxResults,
        include_crossref: settings.includeCrossref,
      });
      setJobId(job_id);

      const final: SRJobStatus = await pollSRJob(
        job_id,
        (update) => {
          const info = update.stage_info ?? {};
          const pct = Number(info.progress_pct ?? 0);
          const label = String(info.label ?? update.stage ?? "");
          const detail = String(info.status_detail ?? "");
          setProgressPct(pct);
          if (label) setProgressLabel(label);
          setProgressDetail(detail);
          if (label) {
            setStepLog((prev) => [...prev, `${label} (${pct}%)${detail ? ` — ${detail}` : ""}`]);
          }
        },
        controller.signal,
      );

      if (final.status === "done" && final.result) {
        setProgressPct(100);
        setElapsedSeconds((performance.now() - startRef.current) / 1000);
        setResult(final.result);
        setStatus("done");
      } else {
        setRunError(final.error ?? "Unknown error.");
        setStatus("error");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setRunError(errorMessage(err));
      setStatus("error");
    }
  }

  function handleRun() {
    if (!question.trim()) {
      setNotice({ kind: "warning", text: "Please enter a research question." });
      return;
    }

    const resolvedQuestion = rqGateRef.current?.resolve() ?? { text: question, ready: true };
    const resolvedInclusion = incGateRef.current?.resolve() ?? { text: inclusion, ready: true };
    const resolvedExclusion = excGateRef.current?.resolve() ?? { text: exclusion, ready: true };

    if (!(resolvedQuestion.ready && resolvedInclusion.ready && resolvedExclusion.ready)) {
      setNotice({
        kind: "info",
        text: "Please resolve the grammar suggestion(s) above, then click Run Systematic Review again.",
      });
      return;
    }

    setNotice(null);
    void runReview(
      resolvedQuestion.text.trim(),
      splitLines(resolvedInclusion.text),
      splitLines(resolvedExclusion.text),
    );
  }

  return (
    <main className="sr-page">
      <h1>Mode 1 — Systematic Literature Review</h1>
      <p>
        Conduct a <strong>PRISMA-style systematic review</strong> powered by local LLM inference (Ollama).
        Describe your research question and criteria below — BeeSearch searches Google Scholar, arXiv,
        Semantic Scholar and CrossRef, screens papers, extracts evidence, and synthesises the findings into
        a full review you can explore, analyse further, and export.
      </p>
      <hr />

      <TemplatePicker onApply={handleApplyTemplate} />

      <div className="sr-field">
        <label htmlFor="sr-question">Research question</label>
        <textarea
          id="sr-question"
          rows={4}
          placeholder="e.g. What is the effect of sleep deprivation on working memory performance in university students?"
          value={question}
          onChange={(e) => {
            setQuestion(e.target.value);
            setNotice(null);
          }}
        />
      </div>
      <GrammarGate
        ref={rqGateRef}
        rawText={question}
        contextHint="systematic literature review research question"
        fieldId="sr-question"
      />
      <p className="sr-caption">
        These guide the screening step — be specific (study design, population, publication window,
        language, …) for sharper include/exclude decisions.
      </p>

      <div className="sr-two-col">
        <div>
          <div className="sr-field">
            <label htmlFor="sr-inclusion">Inclusion criteria (one per line)</label>
            <textarea
              id="sr-inclusion"
              rows={5}
              placeholder={"Peer-reviewed empirical studies\nHuman participants\nPublished 2010–2024\nEnglish language"}
              value={inclusion}
              onChange={(e) => {
                setInclusion(e.target.value);
                setNotice(null);
              }}
            />
          </div>
          <GrammarGate
            ref={incGateRef}
            rawText={inclusion}
            contextHint="inclusion criteria for a systematic review"
            fieldId="sr-inclusion"
          />
        </div>
        <div>
          <div className="sr-field">
            <label htmlFor="sr-exclusion">Exclusion criteria (one per line)</label>
            <textarea
              id="sr-exclusion"
              rows={5}
              placeholder={"Animal studies\nCase reports\nConference abstracts only\nNon-English publications"}
              value={exclusion}
              onChange={(e) => {
                setExclusion(e.target.value);
                setNotice(null);
              }}
            />
          </div>
          <GrammarGate
            ref={excGateRef}
            rawText={exclusion}
            contextHint="exclusion criteria for a systematic review"
            fieldId="sr-exclusion"
          />
        </div>
      </div>

      <div className="sr-button-row">
        <button
          type="button"
          className="sr-button sr-page__run-button"
          disabled={status === "running"}
          onClick={handleRun}
        >
          Run Systematic Review
        </button>
      </div>

      {notice?.kind === "warning" && <p className="sr-warning">{notice.text}</p>}
      {notice?.kind === "info" && <p className="sr-info">{notice.text}</p>}

      {status !== "idle" && (
        <>
          <hr />
          <h3>Running Systematic Review…</h3>
          <p className="sr-page__status-line">
            {status === "running" && <span className="sr-page__spinner" aria-hidden="true" />}
            <strong>{status === "done" ? "Done." : `${progressLabel}…`}</strong>{" "}
            <code>{status === "done" ? 100 : progressPct}%</code>
          </p>
          {status === "running" && progressDetail && <p className="sr-caption">{progressDetail}</p>}
          {status === "done" && elapsedSeconds !== null && (
            <p className="sr-caption">Finished in {elapsedSeconds.toFixed(1)}s.</p>
          )}
          {stepLog.length > 0 && (
            <CollapsibleCard header="Step log">
              <pre className="sr-page__step-log">{stepLog.join("\n")}</pre>
            </CollapsibleCard>
          )}
          {status === "error" && <p className="sr-error">Workflow error: {runError}</p>}
        </>
      )}

      {result && jobId && (
        // Keying on jobId remounts the whole results subtree per run, so a new
        // corpus never shows a previous run's cached citation network / trends / drift.
        <div key={jobId}>
          <hr />
          {result.errors.map((err, i) => (
            <p className="sr-warning" key={i}>
              {err}
            </p>
          ))}
          <EvalResultPanel evalResult={result.eval_result} />
          <RagReflectionPanel ragReflectionInfo={result.rag_reflection_info} />
          <hr />

          <p className="sr-caption">
            <strong>Reviewing:</strong> {result.research_question}
          </p>
          <h3>PRISMA Flow</h3>
          <PrismaFlowSummary flow={result.prisma_flow} />
          <p className="sr-caption">
            Queries: {result.search_queries.length} · Identified: {result.raw_papers.length} · Included:{" "}
            {result.included_papers.length} · Excluded: {result.excluded_papers.length}
          </p>

          <div className="sr-page__tabs" role="tablist" aria-label="Systematic review results">
            {RESULT_TABS.map((t) => (
              <button
                key={t.key}
                type="button"
                role="tab"
                aria-selected={activeTab === t.key}
                className={
                  activeTab === t.key ? "sr-page__tab-button sr-page__tab-button--active" : "sr-page__tab-button"
                }
                onClick={() => setActiveTab(t.key)}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div role="tabpanel">
            {activeTab === "synthesis" && <SynthesisTab result={result} />}
            {activeTab === "evidence" && <EvidenceTab result={result} />}
            {activeTab === "explore" && <ExploreTab jobId={jobId} result={result} />}
            {activeTab === "export" && <ExportTab jobId={jobId} result={result} />}
          </div>
        </div>
      )}
    </main>
  );
}

export default SystematicReviewPage;
