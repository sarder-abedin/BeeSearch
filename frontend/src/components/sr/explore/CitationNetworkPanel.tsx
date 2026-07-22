import { useState } from "react";
import type { SRResult } from "../../../api/systematicReviewTypes";
import HtmlFrame from "./HtmlFrame";
import { useExploreToolJob } from "./useExploreToolJob";
import { arr, bool, nullableStr, num, obj, str } from "./parse";
import "../sr-common.css";

interface CitationNetworkPanelProps {
  jobId: string;
  result: SRResult;
}

interface NetworkStats {
  nodes: number;
  edges: number;
  isolated: number;
  mostCited: [string, number][];
  isolatedPapers: string[];
}

function parseStats(raw: unknown): NetworkStats {
  const s = obj(raw);
  return {
    nodes: num(s.nodes),
    edges: num(s.edges),
    isolated: num(s.isolated),
    mostCited: arr<[string, number]>(s.most_cited),
    isolatedPapers: arr<string>(s.isolated_papers),
  };
}

interface StanceCounts {
  Supporting: number;
  Contrasting: number;
  Mentioning: number;
  classified: number;
}

function parseStances(raw: unknown): StanceCounts {
  const s = obj(raw);
  return {
    Supporting: num(s.Supporting),
    Contrasting: num(s.Contrasting),
    Mentioning: num(s.Mentioning),
    classified: num(s.classified),
  };
}

interface GapCandidate {
  title: string;
  year: number | null;
  venue: string;
  url: string;
  cited_by_count: number;
}

function parseGap(raw: unknown): GapCandidate {
  const g = obj(raw);
  return {
    title: str(g.title, "Unknown title"),
    year: nullableStr(g.year) ? Number(g.year) : (typeof g.year === "number" ? g.year : null),
    venue: str(g.venue),
    url: str(g.url),
    cited_by_count: num(g.cited_by_count),
  };
}

function CitationNetworkPanel({ jobId, result }: CitationNetworkPanelProps) {
  const [classifyStances, setClassifyStances] = useState(false);
  const job = useExploreToolJob(jobId, "citation_network");

  const included = result.included_papers;

  return (
    <div className="sr-explore-panel">
      <h3>Citation Network</h3>
      <p>
        Ego network showing citation links <strong>between</strong> the included papers. Node
        colour = quality (Green High · Amber Medium · Red Low). Requires Semantic Scholar API
        calls — click below to fetch them (~30s for 20 papers).
      </p>

      {included.length === 0 ? (
        <p className="sr-info">No included papers to build a network from.</p>
      ) : result.citation_graph_html ? (
        <HtmlFrame html={result.citation_graph_html} height={520} title="Citation network" />
      ) : (
        <>
          <label className="sr-explore-panel__checkbox">
            <input
              type="checkbox"
              checked={classifyStances}
              onChange={(e) => setClassifyStances(e.target.checked)}
            />
            Classify citation stances (Smart Citations — LLM)
          </label>
          <div className="sr-button-row">
            <button
              type="button"
              className="sr-button"
              disabled={job.state === "running"}
              onClick={() => job.run({ classify_stances: classifyStances })}
            >
              Build Citation Network
            </button>
          </div>

          {job.state === "running" && (
            <p className="sr-spinner-text">Querying Semantic Scholar for citation links…</p>
          )}
          {job.state === "error" && <p className="sr-error">Citation network failed: {job.error}</p>}

          {job.state === "done" && job.result && (() => {
            const html = str(job.result.html);
            const stats = parseStats(job.result.stats);
            const stances = parseStances(job.result.stance_counts);
            const gaps = arr(job.result.gap_candidates).map(parseGap);

            let msg = `Network built: ${stats.nodes} nodes, ${stats.edges} citation edges, ${stats.isolated} isolated papers.`;
            if (stances.classified) {
              msg += ` Stances — ${stances.Supporting} supporting, ${stances.Contrasting} contrasting, ${stances.Mentioning} mentioning.`;
            }

            return (
              <>
                <p className="sr-success">{msg}</p>
                <HtmlFrame html={html} height={520} title="Citation network" />
                {bool(stances.classified > 0) && (
                  <p className="sr-caption">
                    Edge colours — 🟢 Supporting · 🔴 Contrasting · ⚪ Mentioning. ({stances.Supporting} /{" "}
                    {stances.Contrasting} / {stances.Mentioning})
                  </p>
                )}

                {stats.mostCited.length > 0 && (
                  <>
                    <p>
                      <strong>Most cited within corpus:</strong>
                    </p>
                    <ul>
                      {stats.mostCited.map(([node, deg]) => (
                        <li key={node}>
                          {node} — cited by {deg} included paper(s)
                        </li>
                      ))}
                    </ul>
                  </>
                )}

                {stats.isolatedPapers.length > 0 && (
                  <details className="sr-explore-panel__details">
                    <summary>
                      Isolated papers ({stats.isolatedPapers.length}) — no citation links to the rest of
                      the corpus
                    </summary>
                    <ul>
                      {stats.isolatedPapers.map((node) => (
                        <li key={node}>{node}</li>
                      ))}
                    </ul>
                  </details>
                )}

                {gaps.length > 0 && (
                  <>
                    <p title="Papers cited by 2+ of your included papers, but not themselves included.">
                      <strong>Frequently cited but not in your review — consider screening:</strong>
                    </p>
                    <ul>
                      {gaps.map((g, i) => {
                        let label = g.title;
                        if (g.year) label += ` (${g.year})`;
                        if (g.venue) label += ` — ${g.venue}`;
                        return (
                          <li key={`${g.title}-${i}`}>
                            {g.url ? <a href={g.url} target="_blank" rel="noreferrer">{label}</a> : label} —
                            cited by {g.cited_by_count} included paper(s)
                          </li>
                        );
                      })}
                    </ul>
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

export default CitationNetworkPanel;
