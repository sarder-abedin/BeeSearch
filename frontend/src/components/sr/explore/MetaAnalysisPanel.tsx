import { useEffect, useState } from "react";
import { ApiError } from "../../../api/client";
import { draftMetaAnalysis, poolMetaAnalysis, pollToolJob, seedMetaAnalysis } from "../../../api/systematicReview";
import type {
  MetaAnalysisPoolResponse,
  MetaAnalysisRow,
  PoolingModel,
  SRResult,
} from "../../../api/systematicReviewTypes";
import HtmlFrame from "./HtmlFrame";
import { arr, bool, num, obj, str } from "./parse";
import "../sr-common.css";

interface MetaAnalysisPanelProps {
  jobId: string;
  result: SRResult;
}

type AsyncState = "idle" | "loading" | "running" | "done" | "error";

function emptyRow(): MetaAnalysisRow {
  return { citation_key: "", label: "", effect: null, ci_low: null, ci_high: null, n: null };
}

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.detail : (err as Error).message;
}

function MetaAnalysisPanel({ jobId, result }: MetaAnalysisPanelProps) {
  const hasEvidence = result.evidence_table.length > 0;

  const [seedState, setSeedState] = useState<AsyncState>("loading");
  const [seedError, setSeedError] = useState<string | null>(null);
  const [measureLabels, setMeasureLabels] = useState<Record<string, string>>({});
  const [measure, setMeasure] = useState("OR");
  const [rows, setRows] = useState<MetaAnalysisRow[]>([]);

  const [draftState, setDraftState] = useState<AsyncState>("idle");
  const [draftError, setDraftError] = useState<string | null>(null);
  const [draftProgress, setDraftProgress] = useState<{ label: string; pct: number } | null>(null);
  const [draftFilledCount, setDraftFilledCount] = useState<number | null>(null);

  const [poolState, setPoolState] = useState<AsyncState>("idle");
  const [poolError, setPoolError] = useState<string | null>(null);
  const [poolResponse, setPoolResponse] = useState<MetaAnalysisPoolResponse | null>(null);
  const [modelChoice, setModelChoice] = useState<PoolingModel>("random");

  const requestKey = `${jobId}:${hasEvidence}`;
  const [loadedKey, setLoadedKey] = useState(requestKey);
  if (requestKey !== loadedKey) {
    setLoadedKey(requestKey);
    setSeedState("loading");
  }

  useEffect(() => {
    if (!hasEvidence) return;
    let cancelled = false;
    seedMetaAnalysis(jobId)
      .then((res) => {
        if (cancelled) return;
        setRows(res.rows);
        setMeasureLabels(res.measure_labels);
        const codes = Object.keys(res.measure_labels);
        if (codes.length > 0) setMeasure(codes[0]);
        setSeedState("done");
      })
      .catch((err) => {
        if (cancelled) return;
        setSeedError(errorMessage(err));
        setSeedState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, hasEvidence]);

  function updateRow(idx: number, field: "label" | "effect" | "ci_low" | "ci_high" | "n", value: string) {
    setRows((prev) =>
      prev.map((row, i) => {
        if (i !== idx) return row;
        if (field === "label") return { ...row, label: value };
        const parsed = value === "" ? null : Number(value);
        return { ...row, [field]: parsed };
      }),
    );
  }

  function addRow() {
    setRows((prev) => [...prev, emptyRow()]);
  }

  function removeRow(idx: number) {
    setRows((prev) => prev.filter((_, i) => i !== idx));
  }

  async function runDraft() {
    const beforeRows = rows;
    setDraftState("running");
    setDraftError(null);
    setDraftProgress(null);
    setDraftFilledCount(null);
    try {
      const { job_id } = await draftMetaAnalysis(jobId, { rows, measure });
      const final = await pollToolJob(job_id, (status) => {
        const info = obj(status.stage_info);
        setDraftProgress({ label: str(info.label, "Drafting effect sizes"), pct: num(info.progress_pct) });
      });
      if (final.status === "done" && final.result) {
        const draftedRows = arr<MetaAnalysisRow>(final.result.rows);
        const filled = draftedRows.filter((r, i) => {
          const before = beforeRows[i];
          return before && before.effect === null && r.effect !== null;
        }).length;
        setRows(draftedRows);
        setDraftFilledCount(filled);
        setPoolResponse(null);
        setDraftState("done");
      } else {
        setDraftError(final.error ?? "Unknown error.");
        setDraftState("error");
      }
    } catch (err) {
      setDraftError(errorMessage(err));
      setDraftState("error");
    }
  }

  async function runPool(modelOverride?: PoolingModel) {
    setPoolState("running");
    setPoolError(null);
    try {
      const res = await poolMetaAnalysis({ rows, measure, model: modelOverride ?? modelChoice });
      setPoolResponse(res);
      setPoolState("done");
    } catch (err) {
      setPoolError(errorMessage(err));
      setPoolState("error");
    }
  }

  function handleModelChoiceChange(next: PoolingModel) {
    setModelChoice(next);
    if (poolResponse) void runPool(next);
  }

  if (!hasEvidence) {
    return (
      <div className="sr-explore-panel">
        <h3>Statistical Meta-Analysis</h3>
        <p className="sr-info">Evidence table is empty — run the systematic review first.</p>
      </div>
    );
  }

  if (seedState === "loading") {
    return (
      <div className="sr-explore-panel">
        <h3>Statistical Meta-Analysis</h3>
        <p className="sr-spinner-text">Loading evidence rows…</p>
      </div>
    );
  }

  if (seedState === "error") {
    return (
      <div className="sr-explore-panel">
        <h3>Statistical Meta-Analysis</h3>
        <p className="sr-error">Failed to load meta-analysis rows: {seedError}</p>
      </div>
    );
  }

  const poolResult = obj(poolResponse?.result);
  const poolOk = poolResponse !== null && bool(poolResult.ok);

  return (
    <div className="sr-explore-panel">
      <h3>Statistical Meta-Analysis</h3>
      <p>
        Pool each study&apos;s reported effect size and 95% confidence interval into a single estimate
        with a forest plot — the standard step for combining evidence across studies. Enter the
        numbers from each paper&apos;s results (or draft them with the LLM below), then review and
        correct them before pooling — abstracts often round or omit statistics.
      </p>

      <div className="sr-field">
        <label htmlFor="meta-measure">Effect measure</label>
        <select id="meta-measure" value={measure} onChange={(e) => setMeasure(e.target.value)}>
          {Object.entries(measureLabels).map(([code, label]) => (
            <option key={code} value={code}>
              {label}
            </option>
          ))}
        </select>
      </div>

      <div className="sr-button-row">
        <button
          type="button"
          className="sr-button"
          disabled={draftState === "running"}
          onClick={() => void runDraft()}
        >
          Draft effect sizes from abstracts (LLM, best-effort)
        </button>
      </div>

      {draftState === "running" && (
        <p className="sr-spinner-text">
          Drafting effect sizes… {draftProgress ? `${draftProgress.pct}%` : ""}
        </p>
      )}
      {draftState === "error" && <p className="sr-error">Drafting effect sizes failed: {draftError}</p>}
      {draftState === "done" && draftFilledCount !== null && (
        <p className={draftFilledCount > 0 ? "sr-success" : "sr-error"}>
          {draftFilledCount > 0
            ? `Drafted ${draftFilledCount} of ${rows.length} effect sizes from abstracts. Review carefully before pooling — abstracts often round or omit statistics.`
            : "No usable effect sizes found in the abstracts. Enter them manually below."}
        </p>
      )}

      <p className="sr-caption">
        Edit any cell, or add/remove rows — leave a study blank to exclude it from pooling.
      </p>
      <table className="sr-table">
        <thead>
          <tr>
            <th>Study</th>
            <th>Effect</th>
            <th>95% CI low</th>
            <th>95% CI high</th>
            <th>N</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td>
                <input type="text" value={row.label} onChange={(e) => updateRow(i, "label", e.target.value)} />
              </td>
              <td>
                <input
                  type="number"
                  step="any"
                  value={row.effect ?? ""}
                  onChange={(e) => updateRow(i, "effect", e.target.value)}
                />
              </td>
              <td>
                <input
                  type="number"
                  step="any"
                  value={row.ci_low ?? ""}
                  onChange={(e) => updateRow(i, "ci_low", e.target.value)}
                />
              </td>
              <td>
                <input
                  type="number"
                  step="any"
                  value={row.ci_high ?? ""}
                  onChange={(e) => updateRow(i, "ci_high", e.target.value)}
                />
              </td>
              <td>
                <input
                  type="number"
                  step="any"
                  value={row.n ?? ""}
                  onChange={(e) => updateRow(i, "n", e.target.value)}
                />
              </td>
              <td>
                <button
                  type="button"
                  className="sr-table__remove"
                  aria-label="Remove row"
                  onClick={() => removeRow(i)}
                >
                  ✕
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="sr-button-row">
        <button type="button" className="sr-button" onClick={addRow}>
          Add row
        </button>
        <button
          type="button"
          className="sr-button"
          disabled={poolState === "running"}
          onClick={() => void runPool()}
        >
          Run Meta-Analysis
        </button>
      </div>

      {poolState === "error" && <p className="sr-error">Meta-analysis failed: {poolError}</p>}

      {poolResponse && !poolOk && (
        <p className="sr-info">{str(poolResult.reason, "Could not pool these studies.")}</p>
      )}

      {poolResponse && poolOk && (() => {
        const fe = obj(poolResult.fixed_effect);
        const rfx = obj(poolResult.random_effects);
        const het = obj(poolResult.heterogeneity);
        const measureLabel = str(poolResult.measure_label);
        const k = num(poolResult.k);

        return (
          <>
            <hr />
            <p>
              <strong>Pooled {measureLabel}</strong> — k = {k} studies
            </p>
            <div className="sr-metric-row">
              <div className="sr-metric">
                <span className="sr-metric__label">Fixed-effect</span>
                <span className="sr-metric__value">{num(fe.estimate).toFixed(2)}</span>
                <span className="sr-caption">
                  95% CI [{num(fe.ci_low).toFixed(2)}, {num(fe.ci_high).toFixed(2)}]
                </span>
              </div>
              <div className="sr-metric">
                <span className="sr-metric__label">Random-effects</span>
                <span className="sr-metric__value">{num(rfx.estimate).toFixed(2)}</span>
                <span className="sr-caption">
                  95% CI [{num(rfx.ci_low).toFixed(2)}, {num(rfx.ci_high).toFixed(2)}]
                </span>
              </div>
              <div className="sr-metric">
                <span className="sr-metric__label">I² (heterogeneity)</span>
                <span className="sr-metric__value">{num(het.i_squared).toFixed(0)}%</span>
              </div>
              <div className="sr-metric">
                <span className="sr-metric__label">τ² (between-study variance)</span>
                <span className="sr-metric__value">{num(het.tau_squared).toFixed(3)}</span>
              </div>
            </div>
            <p className="sr-caption">
              Cochran&apos;s Q = {num(het.q).toFixed(2)} (df = {num(het.df)}) · {str(het.interpretation)}
            </p>

            <div className="sr-field">
              <label>Pooling model shown in the forest plot</label>
              <div className="sr-button-row">
                <label>
                  <input
                    type="radio"
                    name="meta-model"
                    checked={modelChoice === "random"}
                    onChange={() => handleModelChoiceChange("random")}
                  />{" "}
                  Random-effects (recommended — allows for between-study variation)
                </label>
                <label>
                  <input
                    type="radio"
                    name="meta-model"
                    checked={modelChoice === "fixed"}
                    onChange={() => handleModelChoiceChange("fixed")}
                  />{" "}
                  Fixed-effect (assumes one true underlying effect)
                </label>
              </div>
            </div>

            <h4>Forest Plot</h4>
            {poolResponse.forest_html ? (
              <HtmlFrame
                html={poolResponse.forest_html}
                height={130 + 42 * (k + 1) + 20}
                title="Forest plot"
              />
            ) : (
              <p className="sr-info">Install plotly (pip install plotly) to see the forest plot.</p>
            )}
          </>
        );
      })()}
    </div>
  );
}

export default MetaAnalysisPanel;
