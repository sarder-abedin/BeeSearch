import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ApiError } from "../../api/client";
import { exportKnowledgeGraph, exportStudyGuide, pollPipelineJob, runPipeline } from "../../api/notebookPipeline";
import type {
  KnowledgeGraphFormat,
  PipelineChunk,
  PipelineResult,
  StudyGuideFormat,
} from "../../api/notebookPipelineTypes";
import { useSettings } from "../../context/SettingsContext";
import EvalResultPanel from "../EvalResultPanel";
import RagReflectionPanel from "../RagReflectionPanel";
import CollapsibleCard from "../sr/CollapsibleCard";
import { downloadBlob, downloadText } from "../../utils/download";
import "../sr/sr-common.css";
import "./PipelineTab.css";

interface PipelineTabProps {
  notebookId: string;
  sourceCount: number;
}

type RunStatus = "idle" | "running" | "done" | "error";
type AsyncState = "idle" | "running" | "error";

type PipelineSubTab =
  | "ingestion"
  | "summary"
  | "retrieval"
  | "citations"
  | "knowledge-graph"
  | "study-guide"
  | "podcast";

const SUB_TABS: { key: PipelineSubTab; label: string }[] = [
  { key: "ingestion", label: "Ingestion" },
  { key: "summary", label: "Summary" },
  { key: "retrieval", label: "Retrieval" },
  { key: "citations", label: "Citations" },
  { key: "knowledge-graph", label: "Knowledge Graph" },
  { key: "study-guide", label: "Study Guide" },
  { key: "podcast", label: "Podcast" },
];

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.detail : (err as Error).message;
}

/** Mirrors tools/text_parsing.py::format_page_label -- "n/a" for the -1
 * unknown/web-result sentinel (or any other non-positive value), else 1-based. */
function formatPageLabel(pageNum: number): string {
  if (typeof pageNum !== "number" || !Number.isInteger(pageNum) || pageNum < 0) return "n/a";
  return `p. ${pageNum + 1}`;
}

function PipelineTab({ notebookId, sourceCount }: PipelineTabProps) {
  const settings = useSettings();
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<RunStatus>("idle");
  const [progressPct, setProgressPct] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [runError, setRunError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [activeSubTab, setActiveSubTab] = useState<PipelineSubTab>("ingestion");

  const [loadedNotebookId, setLoadedNotebookId] = useState(notebookId);
  if (notebookId !== loadedNotebookId) {
    setLoadedNotebookId(notebookId);
    setStatus("idle");
    setProgressPct(0);
    setProgressLabel("");
    setRunError(null);
    setJobId(null);
    setResult(null);
    setActiveSubTab("ingestion");
  }

  async function handleRun() {
    const controller = new AbortController();

    setStatus("running");
    setProgressPct(0);
    setProgressLabel("Starting…");
    setRunError(null);
    setResult(null);
    setActiveSubTab("ingestion");

    try {
      const { job_id } = await runPipeline({
        notebook_id: notebookId,
        query: query.trim(),
        model: settings.model,
        num_ctx: settings.numCtx,
        embed_model: settings.embedModel,
        top_k: settings.hybridTopK,
        temperature_level: settings.temperatureLevel,
      });
      setJobId(job_id);

      const final = await pollPipelineJob(
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
    <div className="pipeline-tab">
      {sourceCount === 0 ? (
        <p className="sr-info">
          Add at least one source in the Sources panel before running the analysis pipeline.
        </p>
      ) : (
        <>
          <div className="sr-field">
            <label htmlFor="pipeline-query">Focus query (optional)</label>
            <input
              id="pipeline-query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. 'key findings on attention mechanisms'"
              disabled={status === "running"}
            />
          </div>

          <div className="sr-button-row">
            <button
              type="button"
              className="sr-button"
              disabled={status === "running"}
              onClick={() => void handleRun()}
            >
              {status === "idle" && !result ? "Run Full Pipeline" : "Run Pipeline Again"}
            </button>
            {status !== "idle" && (
              <button type="button" onClick={handleClear} disabled={status === "running"}>
                Clear
              </button>
            )}
          </div>

          {status !== "idle" && (
            <p className="sr-page__status-line">
              {status === "running" && <span className="sr-page__spinner" aria-hidden="true" />}
              <strong>{status === "done" ? "Done." : `${progressLabel || "Working…"}`}</strong>{" "}
              <code>{status === "done" ? 100 : progressPct}%</code>
            </p>
          )}
          {status === "error" && <p className="sr-error">Pipeline failed: {runError}</p>}

          {result && jobId && (
            // Keying on jobId remounts the results subtree per run, so the Knowledge
            // Graph / Study Guide panels never show a previous run's cached blob.
            <div key={jobId}>
              <hr />
              <EvalResultPanel evalResult={result.eval_result} />
              <RagReflectionPanel ragReflectionInfo={result.rag_reflection_info} />

              {result.errors.length > 0 && (
                <details className="sr-explore-panel__details">
                  <summary>{result.errors.length} warning(s)</summary>
                  <ul>
                    {result.errors.map((err, i) => (
                      <li key={i}>{err}</li>
                    ))}
                  </ul>
                </details>
              )}

              <div className="pipeline-tab__subtabs" role="tablist" aria-label="Pipeline results">
                {SUB_TABS.map((t) => (
                  <button
                    key={t.key}
                    type="button"
                    role="tab"
                    aria-selected={activeSubTab === t.key}
                    className={
                      activeSubTab === t.key
                        ? "pipeline-tab__subtab-button pipeline-tab__subtab-button--active"
                        : "pipeline-tab__subtab-button"
                    }
                    onClick={() => setActiveSubTab(t.key)}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              <div role="tabpanel">
                {activeSubTab === "ingestion" && (
                  <div>
                    <p>{result.ingestion_summary || "No ingestion summary available."}</p>
                    <p className="sr-caption">{result.doc_count} document(s) ingested.</p>
                  </div>
                )}

                {activeSubTab === "summary" && (
                  <div>
                    {result.cross_summary ? (
                      <>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.cross_summary}</ReactMarkdown>
                        <div className="sr-button-row">
                          <button
                            type="button"
                            className="sr-button"
                            onClick={() => downloadText(result.cross_summary, "summary.md", "text/markdown")}
                          >
                            Download .md
                          </button>
                        </div>
                      </>
                    ) : (
                      <p className="sr-info">No cross-document summary was generated for this run.</p>
                    )}

                    {Object.keys(result.per_doc_summaries).length > 0 && (
                      <>
                        <h4>Per-document summaries</h4>
                        {Object.entries(result.per_doc_summaries).map(([docName, summary]) => (
                          <CollapsibleCard key={docName} header={docName}>
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{summary}</ReactMarkdown>
                          </CollapsibleCard>
                        ))}
                      </>
                    )}
                  </div>
                )}

                {activeSubTab === "retrieval" && (
                  <div>
                    <p className="sr-caption">
                      Retrieval mode: <strong>{result.retrieval_mode}</strong> · {result.retrieved_chunks.length}{" "}
                      chunk(s) retrieved
                    </p>
                    {result.retrieved_chunks.length === 0 ? (
                      <p className="sr-info">No chunks were retrieved for this run.</p>
                    ) : (
                      result.retrieved_chunks.map((chunk: PipelineChunk, i: number) => (
                        <CollapsibleCard
                          key={chunk.chunk_id || i}
                          header={`[${i + 1}] ${chunk.doc_name} — ${formatPageLabel(chunk.page_num)}`}
                        >
                          <p>{chunk.text}</p>
                        </CollapsibleCard>
                      ))
                    )}
                  </div>
                )}

                {activeSubTab === "citations" && (
                  <div>
                    {result.citation_report ? (
                      <>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.citation_report}</ReactMarkdown>
                        <div className="sr-button-row">
                          <button
                            type="button"
                            className="sr-button"
                            onClick={() => downloadText(result.citation_report, "citations.md", "text/markdown")}
                          >
                            Download .md
                          </button>
                        </div>
                      </>
                    ) : (
                      <p className="sr-info">No citations were verified for this run.</p>
                    )}
                  </div>
                )}

                {activeSubTab === "knowledge-graph" && (
                  result.knowledge_graph_dot ? (
                    <KnowledgeGraphPanel jobId={jobId} dot={result.knowledge_graph_dot} />
                  ) : (
                    <p className="sr-info">No knowledge graph was generated for this run.</p>
                  )
                )}

                {activeSubTab === "study-guide" && (
                  result.study_guide ? (
                    <StudyGuidePanel jobId={jobId} studyGuide={result.study_guide} />
                  ) : (
                    <p className="sr-info">No study guide was generated for this run.</p>
                  )
                )}

                {activeSubTab === "podcast" && (
                  result.podcast_script ? (
                    <PodcastPanel script={result.podcast_script} />
                  ) : (
                    <p className="sr-info">No podcast script was generated for this run.</p>
                  )
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Knowledge Graph: fetches a rendered PNG preview (DOT has no native browser
// renderer) + DOT/PNG/SVG download buttons.
// ─────────────────────────────────────────────────────────────────────────────

interface KnowledgeGraphPanelProps {
  jobId: string;
  dot: string;
}

function KnowledgeGraphPanel({ jobId, dot }: KnowledgeGraphPanelProps) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [pngState, setPngState] = useState<AsyncState>("idle");
  const [svgState, setSvgState] = useState<AsyncState>("idle");
  const [pngError, setPngError] = useState<string | null>(null);
  const [svgError, setSvgError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    exportKnowledgeGraph(jobId, "png")
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setImageUrl(objectUrl);
      })
      .catch((err) => {
        if (!cancelled) setImageError(errorMessage(err));
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [jobId]);

  async function handleDownload(fmt: KnowledgeGraphFormat) {
    const setState = fmt === "png" ? setPngState : setSvgState;
    const setErr = fmt === "png" ? setPngError : setSvgError;
    setState("running");
    setErr(null);
    try {
      const blob = await exportKnowledgeGraph(jobId, fmt);
      downloadBlob(blob, `knowledge_graph.${fmt}`);
      setState("idle");
    } catch (err) {
      setErr(errorMessage(err));
      setState("error");
    }
  }

  return (
    <div className="pipeline-tab__kg">
      {imageUrl ? (
        <img src={imageUrl} alt="Knowledge graph" className="pipeline-tab__kg-image" />
      ) : imageError ? (
        <p className="sr-caption">Knowledge graph preview unavailable: {imageError}</p>
      ) : (
        <p className="sr-spinner-text">Rendering knowledge graph…</p>
      )}

      <div className="sr-button-row">
        <button type="button" className="sr-button" onClick={() => downloadText(dot, "knowledge_graph.dot", "text/vnd.graphviz")}>
          Download .dot
        </button>
        <button
          type="button"
          className="sr-button"
          disabled={pngState === "running"}
          onClick={() => void handleDownload("png")}
        >
          Download .png
        </button>
        <button
          type="button"
          className="sr-button"
          disabled={svgState === "running"}
          onClick={() => void handleDownload("svg")}
        >
          Download .svg
        </button>
      </div>
      {pngState === "error" && <p className="sr-error">PNG export failed: {pngError}</p>}
      {svgState === "error" && <p className="sr-error">SVG export failed: {svgError}</p>}

      <CollapsibleCard header="View DOT source">
        <pre className="pipeline-tab__dot-source">{dot}</pre>
      </CollapsibleCard>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Study Guide: markdown + Download .md (client-side) + DOCX/PDF (server export).
// ─────────────────────────────────────────────────────────────────────────────

interface StudyGuidePanelProps {
  jobId: string;
  studyGuide: string;
}

function StudyGuidePanel({ jobId, studyGuide }: StudyGuidePanelProps) {
  const [docxState, setDocxState] = useState<AsyncState>("idle");
  const [docxError, setDocxError] = useState<string | null>(null);
  const [pdfState, setPdfState] = useState<AsyncState>("idle");
  const [pdfError, setPdfError] = useState<string | null>(null);

  async function handleDownload(fmt: StudyGuideFormat) {
    const setState = fmt === "docx" ? setDocxState : setPdfState;
    const setErr = fmt === "docx" ? setDocxError : setPdfError;
    setState("running");
    setErr(null);
    try {
      const blob = await exportStudyGuide(jobId, fmt);
      downloadBlob(blob, `study_guide.${fmt}`);
      setState("idle");
    } catch (err) {
      setErr(errorMessage(err));
      setState("error");
    }
  }

  return (
    <div>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{studyGuide}</ReactMarkdown>
      <div className="sr-button-row">
        <button
          type="button"
          className="sr-button"
          onClick={() => downloadText(studyGuide, "study_guide.md", "text/markdown")}
        >
          Download .md
        </button>
        <button
          type="button"
          className="sr-button"
          disabled={docxState === "running"}
          onClick={() => void handleDownload("docx")}
        >
          Download .docx
        </button>
        <button
          type="button"
          className="sr-button"
          disabled={pdfState === "running"}
          onClick={() => void handleDownload("pdf")}
        >
          Download .pdf
        </button>
      </div>
      {docxState === "error" && <p className="sr-error">DOCX export failed: {docxError}</p>}
      {pdfState === "error" && <p className="sr-error">PDF export failed: {pdfError}</p>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Podcast: raw script (not markdown) + Download .txt + browser TTS playback.
// ─────────────────────────────────────────────────────────────────────────────

interface PodcastPanelProps {
  script: string;
}

function PodcastPanel({ script }: PodcastPanelProps) {
  const [speaking, setSpeaking] = useState(false);
  const ttsSupported = typeof window !== "undefined" && "speechSynthesis" in window;

  useEffect(() => {
    return () => {
      if (ttsSupported) window.speechSynthesis.cancel();
    };
  }, [ttsSupported]);

  function handleToggleSpeak() {
    if (!ttsSupported) return;
    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
      return;
    }
    const utterance = new SpeechSynthesisUtterance(script);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
    setSpeaking(true);
  }

  return (
    <div>
      <pre className="pipeline-tab__podcast-script">{script}</pre>
      <div className="sr-button-row">
        <button type="button" className="sr-button" onClick={() => downloadText(script, "podcast_script.txt")}>
          Download .txt
        </button>
        {ttsSupported && (
          <button type="button" className="sr-button" onClick={handleToggleSpeak}>
            {speaking ? "Stop reading" : "Read aloud"}
          </button>
        )}
      </div>
    </div>
  );
}

export default PipelineTab;
