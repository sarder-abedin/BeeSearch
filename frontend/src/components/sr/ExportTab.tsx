import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ApiError } from "../../api/client";
import {
  exportDocx,
  exportMarkdown,
  exportPdf,
  pollToolJob,
  triggerPlainLanguageSummary,
} from "../../api/systematicReview";
import type { PlainLanguageFormat, SRResult } from "../../api/systematicReviewTypes";
import { downloadBlob, downloadText } from "../../utils/download";
import "./sr-common.css";
import "./ExportTab.css";

interface ExportTabProps {
  jobId: string;
  result: SRResult;
}

type AsyncState = "idle" | "running" | "done" | "error";

const PLS_FORMATS: { key: PlainLanguageFormat; label: string }[] = [
  { key: "patient", label: "Patient / Public" },
  { key: "policy", label: "Policy Brief" },
  { key: "press", label: "Press Release" },
  { key: "all", label: "All Three" },
];

const PLS_SECTIONS: { key: "patient" | "policy" | "press"; title: string; filenamePrefix: string }[] = [
  { key: "patient", title: "Patient / Public Summary", filenamePrefix: "patient_summary" },
  { key: "policy", title: "Policy Brief", filenamePrefix: "policy_brief" },
  { key: "press", title: "Press Release", filenamePrefix: "press_release" },
];

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.detail : (err as Error).message;
}

function ExportTab({ jobId, result }: ExportTabProps) {
  const sessionId = result.session_id || "sr";

  const [mdState, setMdState] = useState<AsyncState>("idle");
  const [mdError, setMdError] = useState<string | null>(null);

  const [author, setAuthor] = useState("");
  const [institution, setInstitution] = useState("");

  const [docxState, setDocxState] = useState<AsyncState>("idle");
  const [docxError, setDocxError] = useState<string | null>(null);
  const docxBlobRef = useRef<Blob | null>(null);

  const [pdfState, setPdfState] = useState<AsyncState>("idle");
  const [pdfError, setPdfError] = useState<string | null>(null);
  const pdfBlobRef = useRef<Blob | null>(null);

  const [plsFormat, setPlsFormat] = useState<PlainLanguageFormat>("patient");
  const [plsState, setPlsState] = useState<AsyncState>("idle");
  const [plsError, setPlsError] = useState<string | null>(null);
  const [plsResult, setPlsResult] = useState<Record<string, string> | null>(null);

  async function handleDownloadMarkdown() {
    setMdState("running");
    setMdError(null);
    try {
      const md = await exportMarkdown(jobId);
      downloadText(md, `systematic_review_${sessionId}.md`, "text/markdown");
      setMdState("idle");
    } catch (err) {
      setMdError(errorMessage(err));
      setMdState("error");
    }
  }

  async function handleGenerateDocx() {
    setDocxState("running");
    setDocxError(null);
    try {
      docxBlobRef.current = await exportDocx(jobId, author, institution);
      setDocxState("done");
    } catch (err) {
      setDocxError(errorMessage(err));
      setDocxState("error");
    }
  }

  function handleDownloadDocx() {
    if (docxBlobRef.current) downloadBlob(docxBlobRef.current, `prisma_report_${sessionId}.docx`);
  }

  async function handleGeneratePdf() {
    setPdfState("running");
    setPdfError(null);
    try {
      pdfBlobRef.current = await exportPdf(jobId, author, institution);
      setPdfState("done");
    } catch (err) {
      setPdfError(errorMessage(err));
      setPdfState("error");
    }
  }

  function handleDownloadPdf() {
    if (pdfBlobRef.current) downloadBlob(pdfBlobRef.current, `prisma_report_${sessionId}.pdf`);
  }

  async function handleGenerateSummary() {
    setPlsState("running");
    setPlsError(null);
    try {
      const { job_id } = await triggerPlainLanguageSummary(jobId, { format: plsFormat });
      const final = await pollToolJob(job_id, () => {});
      if (final.status === "done" && final.result) {
        setPlsResult(final.result as Record<string, string>);
        setPlsState("done");
      } else {
        setPlsError(final.error ?? "Unknown error.");
        setPlsState("error");
      }
    } catch (err) {
      setPlsError(errorMessage(err));
      setPlsState("error");
    }
  }

  return (
    <div className="sr-export">
      <h3>Search Queries Used</h3>
      {result.search_queries.length === 0 ? (
        <p className="sr-info">No search queries recorded.</p>
      ) : (
        <ol>
          {result.search_queries.map((q, i) => (
            <li key={i}>{q}</li>
          ))}
        </ol>
      )}
      <hr />

      <h3>Export Systematic Review</h3>
      <div className="sr-button-row">
        <button
          type="button"
          className="sr-button"
          disabled={mdState === "running"}
          onClick={() => void handleDownloadMarkdown()}
        >
          Download as Markdown
        </button>
      </div>
      {mdState === "error" && <p className="sr-error">Markdown export failed: {mdError}</p>}
      <hr />

      <h3>PRISMA 2020 Manuscript Report</h3>
      <div className="sr-two-col">
        <div className="sr-field">
          <label htmlFor="export-author">Author name (optional)</label>
          <input id="export-author" type="text" value={author} onChange={(e) => setAuthor(e.target.value)} />
        </div>
        <div className="sr-field">
          <label htmlFor="export-institution">Institution (optional)</label>
          <input
            id="export-institution"
            type="text"
            value={institution}
            onChange={(e) => setInstitution(e.target.value)}
          />
        </div>
      </div>

      <div className="sr-two-col">
        <div>
          <div className="sr-button-row">
            <button
              type="button"
              className="sr-button"
              disabled={docxState === "running"}
              onClick={() => void handleGenerateDocx()}
            >
              Generate DOCX Report
            </button>
          </div>
          {docxState === "error" && <p className="sr-error">DOCX generation failed: {docxError}</p>}
          {docxState === "done" && (
            <>
              <p className="sr-success">DOCX ready.</p>
              <div className="sr-button-row">
                <button type="button" className="sr-button" onClick={handleDownloadDocx}>
                  Download DOCX
                </button>
              </div>
            </>
          )}
        </div>
        <div>
          <div className="sr-button-row">
            <button
              type="button"
              className="sr-button"
              disabled={pdfState === "running"}
              onClick={() => void handleGeneratePdf()}
            >
              Generate PDF Report
            </button>
          </div>
          {pdfState === "error" && <p className="sr-error">PDF generation failed: {pdfError}</p>}
          {pdfState === "done" && (
            <>
              <p className="sr-success">PDF ready.</p>
              <div className="sr-button-row">
                <button type="button" className="sr-button" onClick={handleDownloadPdf}>
                  Download PDF
                </button>
              </div>
            </>
          )}
        </div>
      </div>
      <hr />

      <h3>Plain-Language Summaries</h3>
      <p>Generate lay-audience summaries for different audiences.</p>

      <div className="sr-field">
        <label>Format</label>
        <div className="sr-button-row" role="radiogroup" aria-label="Plain-language summary format">
          {PLS_FORMATS.map((f) => (
            <label key={f.key}>
              <input
                type="radio"
                name="pls-format"
                checked={plsFormat === f.key}
                onChange={() => setPlsFormat(f.key)}
              />{" "}
              {f.label}
            </label>
          ))}
        </div>
      </div>

      <div className="sr-button-row">
        <button
          type="button"
          className="sr-button"
          disabled={plsState === "running"}
          onClick={() => void handleGenerateSummary()}
        >
          Generate Summary
        </button>
      </div>

      {plsState === "running" && <p className="sr-spinner-text">Generating plain-language summary…</p>}
      {plsState === "error" && <p className="sr-error">Summary generation failed: {plsError}</p>}

      {plsResult &&
        PLS_SECTIONS.filter((s) => plsResult[s.key]).map((s) => (
          <div className="sr-export__summary" key={s.key}>
            <h4>{s.title}</h4>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{plsResult[s.key]}</ReactMarkdown>
            <div className="sr-button-row">
              <button
                type="button"
                className="sr-button"
                onClick={() => downloadText(plsResult[s.key], `${s.filenamePrefix}_${sessionId}.txt`)}
              >
                Download (txt)
              </button>
            </div>
          </div>
        ))}
    </div>
  );
}

export default ExportTab;
