import { useState } from "react";
import DiscoveryNetworkPanel from "../components/paper-graph/DiscoveryNetworkPanel";
import SimilarityGraphPanel from "../components/paper-graph/SimilarityGraphPanel";
import "./PaperDiscoveryPage.css";

type FeatureTab = "similarity" | "discovery";

export default function PaperDiscoveryPage() {
  const [tab, setTab] = useState<FeatureTab>("similarity");

  return (
    <main className="pd-page">
      <h1>Mode 4 — Paper Discovery</h1>
      <p>
        Explore the academic neighborhood of any paper. <strong>Similarity Graph</strong>{" "}
        maps a single-origin paper's neighborhood via bibliographic coupling and
        co-citation (Semantic Scholar). <strong>Discovery Network</strong> lets you
        incrementally build a persistent collection by exploring references, citations,
        recommendations, and author networks.
      </p>
      <hr />

      <div className="pd-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "similarity"}
          className={`pd-tab${tab === "similarity" ? " pd-tab--active" : ""}`}
          onClick={() => setTab("similarity")}
        >
          Similarity Graph
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "discovery"}
          className={`pd-tab${tab === "discovery" ? " pd-tab--active" : ""}`}
          onClick={() => setTab("discovery")}
        >
          Discovery Network
        </button>
      </div>

      {tab === "similarity" ? <SimilarityGraphPanel /> : <DiscoveryNetworkPanel />}
    </main>
  );
}
