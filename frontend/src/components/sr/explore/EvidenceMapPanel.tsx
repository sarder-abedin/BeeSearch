import { useEffect, useState } from "react";
import type { SRResult } from "../../../api/systematicReviewTypes";
import { getEvidenceMap } from "../../../api/systematicReview";
import { ApiError } from "../../../api/client";
import HtmlFrame from "./HtmlFrame";
import { num, obj, nullableStr } from "./parse";
import "../sr-common.css";

interface EvidenceMapPanelProps {
  jobId: string;
  result: SRResult;
}

type LoadState = "loading" | "done" | "error";

function EvidenceMapPanel({ jobId, result }: EvidenceMapPanelProps) {
  const hasEvidence = result.evidence_table.length > 0;
  const [state, setState] = useState<LoadState>("loading");
  const [mapData, setMapData] = useState<Record<string, unknown> | null>(null);
  const [html, setHtml] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const requestKey = `${jobId}:${hasEvidence}`;
  const [loadedKey, setLoadedKey] = useState(requestKey);
  if (requestKey !== loadedKey) {
    setLoadedKey(requestKey);
    setState("loading");
  }

  useEffect(() => {
    if (!hasEvidence) return;
    let cancelled = false;
    getEvidenceMap(jobId)
      .then((res) => {
        if (cancelled) return;
        setMapData(res.map_data);
        setHtml(nullableStr(res.html));
        setState("done");
      })
      .catch((err) => {
        if (cancelled) return;
        const message = err instanceof ApiError ? err.detail : (err as Error).message;
        setError(message);
        setState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, hasEvidence]);

  const m = obj(mapData);
  const totalCells = num(m.total_cells);
  const totalStudies = num(m.total_studies);

  return (
    <div className="sr-explore-panel">
      <h3>Evidence Map</h3>
      <p>
        Bubble chart of evidence density across Population × Intervention dimensions. Bubble size =
        number of studies; colour = average quality. Renders instantly from this review&apos;s
        evidence table — no extra API calls.
      </p>

      {!hasEvidence ? (
        <p className="sr-info">Evidence table is empty — run the systematic review first.</p>
      ) : state === "loading" ? (
        <p className="sr-spinner-text">Building evidence map…</p>
      ) : state === "error" ? (
        <p className="sr-error">Evidence map failed: {error}</p>
      ) : totalStudies === 0 ? (
        <p className="sr-info">No evidence data to map.</p>
      ) : (
        <>
          <div className="sr-metric-row">
            <div className="sr-metric">
              <span className="sr-metric__label">Populated Cells</span>
              <span className="sr-metric__value">{totalCells}</span>
            </div>
            <div className="sr-metric">
              <span className="sr-metric__label">Total Studies Mapped</span>
              <span className="sr-metric__value">{totalStudies}</span>
            </div>
          </div>
          {html ? (
            <HtmlFrame html={html} height={480} title="Evidence map" />
          ) : (
            <p className="sr-info">Install plotly (pip install plotly) to see the evidence map chart.</p>
          )}
        </>
      )}
    </div>
  );
}

export default EvidenceMapPanel;
