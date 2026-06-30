import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ApiError } from "../../api/client";
import { exportReportCitations, pollReportJob, runReport } from "../../api/notebookReport";
import type { ReportCitationFormat, ReportReference, ReportResult } from "../../api/notebookReportTypes";
import { useSettings } from "../../context/SettingsContext";
import CollapsibleCard from "../sr/CollapsibleCard";
import { downloadText } from "../../utils/download";
import "../sr/sr-common.css";
import "./ResearchReportTab.css";

interface ResearchReportTabProps {
  notebookId: string;
  sourceCount: number;
}

type RunStatus = "idle" | "running" | "done" | "error";
type AsyncState = "idle" | "running" | "error";
type ReportSubTab = "report" | "references";

const SUB_TABS: { key: ReportSubTab; label: string }[] = [
  { key: "report", label: "Report" },
  { key: "references", label: "References" },
];

const SOURCE_LABELS: Record<string, string> = {
  arxiv: "arXiv preprint",
  semantic_scholar: "Peer-reviewed",
  crossref: "CrossRef",
  web: "Web result",
};

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.detail : (err as Error).message;
}

function ResearchReportTab({ notebookId, sourceCount }: ResearchReportTabProps) {
  const settings = useSettings();
  const [goal, setGoal] = useState("");
  const [includeAcademic, setIncludeAcademic] = useState(true);
  const [includeWeb, setIncludeWeb] = useState(false);
  const [validationWarning, setValidationWarning] = useState<string | null>(null);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [progressPct, setProgressPct] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [runError, setRunError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<ReportResult | null>(null);
  const [activeSubTab, setActiveSubTab] = useState<ReportSubTab>("report");

  const [loadedNotebookId, setLoadedNotebookId] = useState(notebookId);
  if (notebookId !== loadedNotebookId) {
    setLoadedNotebookId(notebookId);
    setGoal("");
    setValidationWarning(null);
    setStatus("idle");
    setProgressPct(0);
    setProgressLabel("");
    setRunError(null);
    setJobId(null);
    setResult(null);
    setActiveSubTab("report");
  }

  async function handleRun() {
    if (!goal.trim()) {
      setValidationWarning("Please enter a research goal.");
      return;
    }
    setValidationWarning(null);

    const controller = new AbortController();

    setStatus("running");
    setProgressPct(0);
    setProgressLabel("Starting…");
    setRunError(null);
    setResult(null);
    setActiveSubTab("report");

    try {
      const { job_id } = await runReport({
        notebook_id: notebookId,
        goal: goal.trim(),
        include_academic: includeAcademic,
        include_web: includeWeb,
        model: settings.model,
        num_ctx: settings.numCtx,
        embed_model: settings.embedModel,
      });
      setJobId(job_id);

      const final = await pollReportJob(
        job_id,
        (update) => {
          const info = update.stage_info ?? {};
          const pct = Number(info.progress_pct ?? 0);
          const label = typeof info.label === "string" ? info.label : "";
          setProgressPct(pct);
          if (label) setProgressLabel(label);
        },
        controller.signal,
      );

      if (final.status === "done" && final.result) {
        setProgressPct(100);
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

  function handleClear() {
    setStatus("idle");
    setProgressPct(0);
    setProgressLabel("");
    setRunError(null);
    setJobId(null);
    setResult(null);
  }

  return (
    <div className="research-report-tab">
      <p>
        Generate a full research report grounded in your notebook sources and optionally augmented with
        peer-reviewed papers from arXiv and Semantic Scholar.
      </p>

      <div className="sr-field">
        <label htmlFor="report-goal">Research goal or question</label>
        <input
          id="report-goal"
          type="text"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="e.g. 'Summarise key findings on transformer attention mechanisms'"
          disabled={status === "running"}
        />
      </div>

      <label className="sr-explore-panel__checkbox">
        <input
          type="checkbox"
          checked={includeAcademic}
          disabled={status === "running"}
          onChange={(e) => setIncludeAcademic(e.target.checked)}
        />
        Search academic sources (arXiv + Semantic Scholar)
      </label>
      <label className="sr-explore-panel__checkbox">
        <input
          type="checkbox"
          checked={includeWeb}
          disabled={status === "running"}
          onChange={(e) => setIncludeWeb(e.target.checked)}
        />
        Include web search (DuckDuckGo)
      </label>

      {sourceCount === 0 && (
        <p className="sr-info">No sources in this notebook — will search academic literature only.</p>
      )}

      <div className="sr-button-row">
        <button type="button" className="sr-button" disabled={status === "running"} onClick={() => void handleRun()}>
          {result ? "Regenerate Report" : "Generate Research Report"}
        </button>
        {status !== "idle" && (
          <button type="button" onClick={handleClear} disabled={status === "running"}>
            Clear
          </button>
        )}
      </div>
      {validationWarning && <p className="sr-warning">{validationWarning}</p>}

      {status !== "idle" && (
        <p className="sr-page__status-line">
          {status === "running" && <span className="sr-page__spinner" aria-hidden="true" />}
          <strong>{status === "done" ? "Done." : progressLabel || "Working…"}</strong>{" "}
          <code>{status === "done" ? 100 : progressPct}%</code>
        </p>
      )}
      {status === "error" && <p className="sr-error">Research workflow failed: {runError}</p>}

      {result && jobId && (
        // Keying on jobId remounts the results subtree per run, so the References
        // panel never shows a previous run's cached export-button state.
        <div key={jobId}>
          <hr />
          {result.web_search_status === "empty" && (
            <p className="sr-warning">
              Web search was enabled but found no additional results — this report uses only academic/notebook
              sources.
            </p>
          )}
          {result.web_search_status === "error" && (
            <p className="sr-warning">
              Web search was enabled but failed — this report uses only academic/notebook sources.
            </p>
          )}

          {result.key_findings.length > 0 && (
            <>
              <h3>Key Findings</h3>
              <ol className="research-report-tab__findings">
                {result.key_findings.map((finding, i) => (
                  <li key={i}>{finding}</li>
                ))}
              </ol>
            </>
          )}

          <div className="research-report-tab__subtabs" role="tablist" aria-label="Research report results">
            {SUB_TABS.map((t) => (
              <button
                key={t.key}
                type="button"
                role="tab"
                aria-selected={activeSubTab === t.key}
                className={
                  activeSubTab === t.key
                    ? "research-report-tab__subtab-button research-report-tab__subtab-button--active"
                    : "research-report-tab__subtab-button"
                }
                onClick={() => setActiveSubTab(t.key)}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div role="tabpanel">
            {activeSubTab === "report" && <ReportPanel report={result.report} />}
            {activeSubTab === "references" && <ReferencesPanel references={result.references} jobId={jobId} />}
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Report: markdown render + client-side "Download Report (Markdown)" -- the
// report text is already in memory, so this needs no backend round-trip
// (mirrors ui/helpers.py::render_report).
// ─────────────────────────────────────────────────────────────────────────────

interface ReportPanelProps {
  report: string;
}

function ReportPanel({ report }: ReportPanelProps) {
  return (
    <div>
      <h3>Full Research Report</h3>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
      <div className="sr-button-row">
        <button
          type="button"
          className="sr-button"
          onClick={() => downloadText(report, "research_report.md", "text/markdown")}
        >
          Download Report (Markdown)
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// References: citation export buttons (server round-trip) + per-reference
// expandable cards (mirrors ui/helpers.py::render_references).
// ─────────────────────────────────────────────────────────────────────────────

interface ReferencesPanelProps {
  references: ReportReference[];
  jobId: string;
}

function ReferencesPanel({ references, jobId }: ReferencesPanelProps) {
  const [bibState, setBibState] = useState<AsyncState>("idle");
  const [risState, setRisState] = useState<AsyncState>("idle");
  const [bibError, setBibError] = useState<string | null>(null);
  const [risError, setRisError] = useState<string | null>(null);

  async function handleExport(fmt: ReportCitationFormat) {
    const setState = fmt === "bibtex" ? setBibState : setRisState;
    const setErr = fmt === "bibtex" ? setBibError : setRisError;
    setState("running");
    setErr(null);
    try {
      const text = await exportReportCitations(jobId, fmt);
      downloadText(text, `references.${fmt === "bibtex" ? "bib" : "ris"}`, "text/plain");
      setState("idle");
    } catch (err) {
      setErr(errorMessage(err));
      setState("error");
    }
  }

  if (references.length === 0) {
    return <p className="sr-info">No references found for this run.</p>;
  }

  return (
    <div>
      <h3>References ({references.length})</h3>
      <div className="sr-button-row">
        <button
          type="button"
          className="sr-button"
          disabled={bibState === "running"}
          onClick={() => void handleExport("bibtex")}
        >
          Export BibTeX (.bib)
        </button>
        <button
          type="button"
          className="sr-button"
          disabled={risState === "running"}
          onClick={() => void handleExport("ris")}
        >
          Export RIS (.ris)
        </button>
      </div>
      {bibState === "error" && <p className="sr-error">BibTeX export failed: {bibError}</p>}
      {risState === "error" && <p className="sr-error">RIS export failed: {risError}</p>}
      <hr />

      {references.map((ref) => (
        <CollapsibleCard key={ref.ref_num} header={`[${ref.ref_num}] ${ref.title.slice(0, 80)}`}>
          {ref.authors.length > 0 && (
            <p>
              <strong>Authors:</strong> {ref.authors.slice(0, 5).join("; ")}
            </p>
          )}
          <p>
            <strong>Journal/Venue:</strong> {ref.journal || "N/A"}
          </p>
          <p>
            <strong>Year:</strong> {ref.year || "N/A"}
          </p>
          {ref.doi && (
            <p>
              <strong>DOI:</strong>{" "}
              <a href={`https://doi.org/${ref.doi}`} target="_blank" rel="noreferrer">
                {ref.doi}
              </a>
            </p>
          )}
          {ref.url && (
            <p>
              <strong>URL:</strong>{" "}
              <a href={ref.url} target="_blank" rel="noreferrer">
                {ref.url.slice(0, 50)}
              </a>
            </p>
          )}
          {ref.abstract_snippet && (
            <p>
              <strong>Abstract:</strong> <em>{ref.abstract_snippet}</em>
            </p>
          )}
          <p className="sr-caption">
            Source: {SOURCE_LABELS[ref.source] ?? "Unknown"}
            {ref.citation_count !== null && ` · Citations: ${ref.citation_count}`}
          </p>
          <pre className="research-report-tab__apa">{ref.apa}</pre>
        </CollapsibleCard>
      ))}
    </div>
  );
}

export default ResearchReportTab;
