import type { SRResult } from "../../../api/systematicReviewTypes";
import { useExploreToolJob } from "./useExploreToolJob";
import { arr, num, obj, str, strArr } from "./parse";
import "../sr-common.css";

interface ConceptDriftPanelProps {
  jobId: string;
  result: SRResult;
}

interface Bucket {
  label: string;
  papers: number;
  top_terms: string[];
}

interface DriftTerm {
  term: string;
  growth: number;
  first_bucket: string;
  last_bucket: string;
}

function parseDriftTerm(raw: unknown): DriftTerm {
  const d = obj(raw);
  return {
    term: str(d.term),
    growth: num(d.growth),
    first_bucket: str(d.first_bucket),
    last_bucket: str(d.last_bucket),
  };
}

function ConceptDriftPanel({ jobId, result }: ConceptDriftPanelProps) {
  const job = useExploreToolJob(jobId, "concept_drift");
  const hasCorpus = result.raw_papers.length > 0;

  return (
    <div className="sr-explore-panel">
      <h3>Concept Drift Tracker</h3>
      <p>
        Detects vocabulary shifts across time periods in the included papers — which terms are
        rising, which are declining. Runs an LLM analysis pass over the corpus (~tens of seconds
        depending on corpus size).
      </p>

      {!hasCorpus ? (
        <p className="sr-info">No papers in corpus.</p>
      ) : (
        <>
          <div className="sr-button-row">
            <button
              type="button"
              className="sr-button"
              disabled={job.state === "running"}
              onClick={() => job.run()}
            >
              Detect Concept Drift
            </button>
          </div>

          {job.state === "running" && (
            <p className="sr-spinner-text">Analysing vocabulary evolution across time buckets…</p>
          )}
          {job.state === "error" && (
            <p className="sr-error">Concept drift analysis failed: {job.error}</p>
          )}

          {job.state === "done" && job.result && (() => {
            const bucketsObj = obj(job.result.buckets);
            const buckets: Bucket[] = Object.entries(bucketsObj)
              .slice(0, 6)
              .map(([label, meta]) => {
                const m = obj(meta);
                return { label, papers: num(m.papers), top_terms: strArr(m.top_terms) };
              });
            const rising = arr(job.result.rising_terms).map(parseDriftTerm).slice(0, 8);
            const declining = arr(job.result.declining_terms).map(parseDriftTerm).slice(0, 8);
            const llmAnalysis = str(job.result.llm_analysis);

            return (
              <>
                {buckets.length > 0 && (
                  <>
                    <p>
                      <strong>Vocabulary by time period:</strong>
                    </p>
                    {buckets.map((b) => (
                      <details key={b.label} className="sr-explore-panel__details">
                        <summary>
                          {b.label} — {b.papers} papers
                        </summary>
                        <p>{b.top_terms.join(", ")}</p>
                      </details>
                    ))}
                  </>
                )}

                <div className="sr-two-col">
                  <div>
                    <p>
                      <strong>Rising terms</strong> (becoming more prominent):
                    </p>
                    <ul>
                      {rising.map((r, i) => (
                        <li key={i}>
                          <strong>{r.term}</strong> (+{r.growth}) · {r.first_bucket} → {r.last_bucket}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <p>
                      <strong>Declining terms</strong> (becoming less prominent):
                    </p>
                    <ul>
                      {declining.map((d, i) => (
                        <li key={i}>
                          <strong>{d.term}</strong> ({d.growth}) · {d.first_bucket} → {d.last_bucket}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {llmAnalysis && (
                  <>
                    <hr />
                    <p>
                      <strong>LLM Analysis of Vocabulary Shifts:</strong>
                    </p>
                    <p>{llmAnalysis}</p>
                  </>
                )}
              </>
            );
          })()}
        </>
      )}
    </div>
  );
}

export default ConceptDriftPanel;
