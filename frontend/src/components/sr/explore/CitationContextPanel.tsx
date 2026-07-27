import { useState } from "react";
import type { SRResult } from "../../../api/systematicReviewTypes";
import { useExploreToolJob } from "./useExploreToolJob";
import { arr, nullableStr, obj, str } from "./parse";
import "../sr-common.css";

interface CitationContextPanelProps {
  jobId: string;
  result: SRResult;
}

interface CitationContext {
  sentence: string;
  matched_on: string;
}

function parseContext(raw: unknown): CitationContext {
  const c = obj(raw);
  return {
    sentence: str(c.sentence),
    matched_on: str(c.matched_on),
  };
}

function paperLabel(p: { citation_key: string; title: string; year: number | null }): string {
  const name = p.citation_key || p.title.slice(0, 40);
  return `${name} (${p.year ?? "n.d."})`;
}

function CitationContextPanel({ jobId, result }: CitationContextPanelProps) {
  const included = result.included_papers;
  const [citingIdx, setCitingIdx] = useState(0);
  const [citedIdx, setCitedIdx] = useState(0);
  const job = useExploreToolJob(jobId, "citation_context");

  return (
    <div className="sr-explore-panel">
      <h3>Citation Context</h3>
      <p>
        Find the exact sentence(s) where one included paper cites another, pulled from the{" "}
        <strong>citing paper&apos;s open-access full text</strong>. Best-effort: only works when an
        open-access PDF/HTML is available for the citing paper.
      </p>

      {included.length < 2 ? (
        <p className="sr-info">Need at least two included papers to look up a citation context.</p>
      ) : (
        <>
          <div className="sr-two-col">
            <div className="sr-field">
              <label htmlFor="cc-citing">Citing paper (A)</label>
              <select
                id="cc-citing"
                value={citingIdx}
                onChange={(e) => setCitingIdx(Number(e.target.value))}
              >
                {included.map((p, i) => (
                  <option key={`citing-${i}`} value={i}>
                    {paperLabel(p)}
                  </option>
                ))}
              </select>
            </div>
            <div className="sr-field">
              <label htmlFor="cc-cited">Cited paper (B)</label>
              <select
                id="cc-cited"
                value={citedIdx}
                onChange={(e) => setCitedIdx(Number(e.target.value))}
              >
                {included.map((p, i) => (
                  <option key={`cited-${i}`} value={i}>
                    {paperLabel(p)}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {citingIdx === citedIdx ? (
            <p className="sr-caption">Pick two different papers.</p>
          ) : (
            <>
              <div className="sr-button-row">
                <button
                  type="button"
                  className="sr-button"
                  disabled={job.state === "running"}
                  onClick={() => job.run({ citing_idx: citingIdx, cited_idx: citedIdx })}
                >
                  Find Citation Context
                </button>
              </div>

              {job.state === "running" && (
                <p className="sr-spinner-text">
                  Fetching the citing paper&apos;s full text and searching for the citation…
                </p>
              )}
              {job.state === "error" && (
                <p className="sr-error">Citation context lookup failed: {job.error}</p>
              )}

              {job.state === "done" && job.result && (() => {
                const status = nullableStr(job.result.status);
                const contexts = arr(job.result.contexts).map(parseContext);
                const sourceUrl = nullableStr(job.result.source_url);
                const reason = str(job.result.reason, "No citation context available.");

                if (status !== "ok") {
                  return <p className="sr-info">{reason}</p>;
                }

                return (
                  <>
                    <p className="sr-success">Found {contexts.length} citing sentence(s).</p>
                    {contexts.map((ctx, i) => (
                      <blockquote key={i}>
                        <p>{ctx.sentence}</p>
                        <p className="sr-caption">matched on {ctx.matched_on}</p>
                      </blockquote>
                    ))}
                    {sourceUrl && (
                      <p className="sr-caption">
                        Source:{" "}
                        <a href={sourceUrl} target="_blank" rel="noreferrer">
                          {sourceUrl}
                        </a>
                      </p>
                    )}
                  </>
                );
              })()}
            </>
          )}
        </>
      )}
    </div>
  );
}

export default CitationContextPanel;
