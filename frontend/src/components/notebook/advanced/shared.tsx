import { useEffect, useState } from "react";
import { exportDocument, exportDot, exportText } from "../../../api/notebookAdvanced";
import type {
  DocumentArtifact,
  DocumentFormat,
  DotArtifact,
  DotFormat,
  TextArtifact,
} from "../../../api/notebookAdvancedTypes";
import { downloadBlob, downloadText } from "../../../utils/download";
import CollapsibleCard from "../../sr/CollapsibleCard";
import { type AsyncState, errorMessage } from "./format";
import type { AdvancedJobState } from "./useAdvancedToolJob";

interface RunControlsProps {
  state: AdvancedJobState;
  runLabel: string;
  rerunLabel: string;
  spinnerText: string;
  error: string | null;
  errorPrefix: string;
  onRun: () => void;
  onClear: () => void;
}

/** Generate/Regenerate + Clear button row, spinner text, and error line --
 * shared by all 9 panels, each of which only differs in label copy. Mirrors
 * ui/tabs/notebook.py::_gen_button's "Generate/Regenerate + Clear, always
 * paired" convention. */
export function RunControls({
  state,
  runLabel,
  rerunLabel,
  spinnerText,
  error,
  errorPrefix,
  onRun,
  onClear,
}: RunControlsProps) {
  return (
    <>
      <div className="sr-button-row">
        <button type="button" className="sr-button" disabled={state === "running"} onClick={onRun}>
          {state === "idle" ? runLabel : rerunLabel}
        </button>
        {state !== "idle" && (
          <button type="button" disabled={state === "running"} onClick={onClear}>
            Clear
          </button>
        )}
      </div>
      {state === "running" && <p className="sr-spinner-text">{spinnerText}</p>}
      {state === "error" && (
        <p className="sr-error">
          {errorPrefix}: {error}
        </p>
      )}
    </>
  );
}

interface TextExportButtonsProps {
  jobId: string;
  artifact: TextArtifact;
  filenameBase: string;
  documentArtifact?: DocumentArtifact;
}

/** Download .md (always via the server's /export/text/{artifact} -- the only
 * way to get the fully-composed text for "review", and a harmless round-trip
 * for the other 4 artifacts) plus optional .docx/.pdf (only the 3 artifacts
 * DocumentArtifact actually covers: summary/review/study-comparison). */
export function TextExportButtons({ jobId, artifact, filenameBase, documentArtifact }: TextExportButtonsProps) {
  const [mdState, setMdState] = useState<AsyncState>("idle");
  const [mdError, setMdError] = useState<string | null>(null);
  const [docxState, setDocxState] = useState<AsyncState>("idle");
  const [docxError, setDocxError] = useState<string | null>(null);
  const [pdfState, setPdfState] = useState<AsyncState>("idle");
  const [pdfError, setPdfError] = useState<string | null>(null);

  async function handleText() {
    setMdState("running");
    setMdError(null);
    try {
      const text = await exportText(jobId, artifact);
      downloadText(text, `${filenameBase}.md`, "text/markdown");
      setMdState("idle");
    } catch (err) {
      setMdError(errorMessage(err));
      setMdState("error");
    }
  }

  async function handleDocument(fmt: DocumentFormat) {
    if (!documentArtifact) return;
    const setState = fmt === "docx" ? setDocxState : setPdfState;
    const setErr = fmt === "docx" ? setDocxError : setPdfError;
    setState("running");
    setErr(null);
    try {
      const blob = await exportDocument(jobId, documentArtifact, fmt);
      downloadBlob(blob, `${filenameBase}.${fmt}`);
      setState("idle");
    } catch (err) {
      setErr(errorMessage(err));
      setState("error");
    }
  }

  return (
    <div>
      <div className="sr-button-row">
        <button type="button" className="sr-button" disabled={mdState === "running"} onClick={() => void handleText()}>
          Download .md
        </button>
        {documentArtifact && (
          <>
            <button
              type="button"
              className="sr-button"
              disabled={docxState === "running"}
              onClick={() => void handleDocument("docx")}
            >
              Download .docx
            </button>
            <button
              type="button"
              className="sr-button"
              disabled={pdfState === "running"}
              onClick={() => void handleDocument("pdf")}
            >
              Download .pdf
            </button>
          </>
        )}
      </div>
      {mdState === "error" && <p className="sr-error">Markdown export failed: {mdError}</p>}
      {docxState === "error" && <p className="sr-error">DOCX export failed: {docxError}</p>}
      {pdfState === "error" && <p className="sr-error">PDF export failed: {pdfError}</p>}
    </div>
  );
}

interface DotExportPanelProps {
  jobId: string;
  dot: string;
  artifact: DotArtifact;
  filenameBase: string;
  previewAlt: string;
}

/** PNG preview (fetched once, since DOT has no native browser renderer) plus
 * .dot/.png/.svg download buttons. Mirrors PipelineTab.tsx's KnowledgeGraphPanel,
 * parameterized over which of the 2 DotArtifact values this run is. */
export function DotExportPanel({ jobId, dot, artifact, filenameBase, previewAlt }: DotExportPanelProps) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [pngState, setPngState] = useState<AsyncState>("idle");
  const [svgState, setSvgState] = useState<AsyncState>("idle");
  const [pngError, setPngError] = useState<string | null>(null);
  const [svgError, setSvgError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    exportDot(jobId, artifact, "png")
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
  }, [jobId, artifact]);

  async function handleDownload(fmt: DotFormat) {
    const setState = fmt === "png" ? setPngState : setSvgState;
    const setErr = fmt === "png" ? setPngError : setSvgError;
    setState("running");
    setErr(null);
    try {
      const blob = await exportDot(jobId, artifact, fmt);
      downloadBlob(blob, `${filenameBase}.${fmt}`);
      setState("idle");
    } catch (err) {
      setErr(errorMessage(err));
      setState("error");
    }
  }

  return (
    <div className="advanced-tools-tab__dot">
      {imageUrl ? (
        <img src={imageUrl} alt={previewAlt} className="advanced-tools-tab__dot-image" />
      ) : imageError ? (
        <p className="sr-caption">Preview unavailable: {imageError}</p>
      ) : (
        <p className="sr-spinner-text">Rendering preview…</p>
      )}

      <div className="sr-button-row">
        <button
          type="button"
          className="sr-button"
          onClick={() => downloadText(dot, `${filenameBase}.dot`, "text/vnd.graphviz")}
        >
          Download .dot
        </button>
        <button type="button" className="sr-button" disabled={pngState === "running"} onClick={() => void handleDownload("png")}>
          Download .png
        </button>
        <button type="button" className="sr-button" disabled={svgState === "running"} onClick={() => void handleDownload("svg")}>
          Download .svg
        </button>
      </div>
      {pngState === "error" && <p className="sr-error">PNG export failed: {pngError}</p>}
      {svgState === "error" && <p className="sr-error">SVG export failed: {svgError}</p>}

      <CollapsibleCard header="View DOT source">
        <pre className="advanced-tools-tab__dot-source">{dot}</pre>
      </CollapsibleCard>
    </div>
  );
}
