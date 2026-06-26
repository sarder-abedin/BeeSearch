import { useState } from "react";
import type { ExploreTool, SRResult } from "../../api/systematicReviewTypes";
import CitationContextPanel from "./explore/CitationContextPanel";
import CitationNetworkPanel from "./explore/CitationNetworkPanel";
import ConceptDriftPanel from "./explore/ConceptDriftPanel";
import EvidenceMapPanel from "./explore/EvidenceMapPanel";
import MetaAnalysisPanel from "./explore/MetaAnalysisPanel";
import PreprintStatusPanel from "./explore/PreprintStatusPanel";
import ReferenceCheckingPanel from "./explore/ReferenceCheckingPanel";
import ResearchTrendsPanel from "./explore/ResearchTrendsPanel";
import "./sr-common.css";
import "./ExploreTab.css";

interface ExploreTabProps {
  jobId: string;
  result: SRResult;
}

const EXPLORE_TOOLS: { key: ExploreTool; label: string }[] = [
  { key: "citation_network", label: "Citation Network" },
  { key: "citation_context", label: "Citation Context" },
  { key: "reference_checking", label: "Risk & Certainty" },
  { key: "preprint_status", label: "Preprint Status" },
  { key: "research_trends", label: "Research Trends" },
  { key: "evidence_map", label: "Evidence Map" },
  { key: "meta_analysis", label: "Meta-Analysis" },
  { key: "concept_drift", label: "Concept Drift" },
];

function ExploreTab({ jobId, result }: ExploreTabProps) {
  const [choice, setChoice] = useState<ExploreTool>("citation_network");

  return (
    <div className="sr-explore-tab">
      <p>
        Optional deep-dive tools that run on top of this review&apos;s corpus — pick one below.
        Each shows what it needs and roughly how long it takes before you run it.
      </p>

      <div className="sr-explore-tab__radio" role="radiogroup" aria-label="Explore tool">
        {EXPLORE_TOOLS.map((tool) => (
          <label
            key={tool.key}
            className={
              choice === tool.key
                ? "sr-explore-tab__radio-option sr-explore-tab__radio-option--selected"
                : "sr-explore-tab__radio-option"
            }
          >
            <input
              type="radio"
              name="explore-tool"
              value={tool.key}
              checked={choice === tool.key}
              onChange={() => setChoice(tool.key)}
            />
            {tool.label}
          </label>
        ))}
      </div>
      <hr />

      {choice === "citation_network" && <CitationNetworkPanel jobId={jobId} result={result} />}
      {choice === "citation_context" && <CitationContextPanel jobId={jobId} result={result} />}
      {choice === "reference_checking" && <ReferenceCheckingPanel jobId={jobId} result={result} />}
      {choice === "preprint_status" && <PreprintStatusPanel jobId={jobId} result={result} />}
      {choice === "research_trends" && <ResearchTrendsPanel jobId={jobId} result={result} />}
      {choice === "evidence_map" && <EvidenceMapPanel jobId={jobId} result={result} />}
      {choice === "meta_analysis" && <MetaAnalysisPanel jobId={jobId} result={result} />}
      {choice === "concept_drift" && <ConceptDriftPanel jobId={jobId} result={result} />}
    </div>
  );
}

export default ExploreTab;
