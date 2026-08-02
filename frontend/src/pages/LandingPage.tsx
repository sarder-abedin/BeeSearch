import beeSearchLogo from "../assets/logo.png";
import "./LandingPage.css";

export type ProjectId = "mode1" | "mode2" | "mode3" | "mode4";

interface Project {
  id: ProjectId;
  label: string;
  name: string;
  description: string;
  tags: string[];
  comingSoon?: boolean;
}

// Mirrors `ui/landing.py::_PROJECTS` -- keep name/description/tags in sync.
const PROJECTS: Project[] = [
  {
    id: "mode1",
    label: "01",
    name: "Systematic Literature Review",
    description:
      "Run a full PRISMA-compliant systematic review: search Google Scholar, arXiv, " +
      "Semantic Scholar and CrossRef; pre-rank abstracts with an LLM screener; screen " +
      "by inclusion/exclusion criteria; extract structured evidence; and synthesise " +
      "findings. Generate DOCX/PDF reports, plain-language summaries, citation networks, " +
      "preprint tracking, trend analysis, evidence maps, and concept drift detection.",
    tags: ["PRISMA 2020", "Google Scholar", "Abstract Screener", "Citation Network", "DOCX + PDF Export", "Trend Analysis"],
  },
  {
    id: "mode2",
    label: "02",
    name: "Research Notebook",
    description:
      "A NotebookLM-style workspace: upload PDFs, DOCX, TXT or web pages to build a " +
      "source notebook, then chat with grounded citations. Run a full 7-agent pipeline " +
      "for cross-document summary, citation verification, knowledge graph, study guide, " +
      "and podcast script. Advanced tools: FAQ, literature review, mind map, " +
      "citation timeline, source comparison, and study comparison table.",
    tags: ["Grounded Q&A", "7-Agent Pipeline", "Knowledge Graph", "Study Guide", "Hybrid RAG", "Source Citations"],
  },
  {
    id: "mode3",
    label: "03",
    name: "AI Research Assistant",
    description:
      "Ask a free-form research question and get an answer grounded in published " +
      "literature with inline citations — Elicit / Perplexity / Consensus style. No " +
      "documents to upload and no PRISMA workflow: BeeSearch searches Google Scholar, " +
      "arXiv, Semantic Scholar and the web, reads what it finds, and rebuilds an accurate " +
      "citation list from the sources it actually used.",
    tags: ["Free-form Q&A", "Literature-grounded", "Inline Citations", "Google Scholar", "arXiv", "Web Search"],
  },
  {
    id: "mode4",
    label: "04",
    name: "Paper Discovery",
    description:
      "Explore the academic neighborhood of any paper. Similarity Graph builds a " +
      "Connected Papers–style force-directed map using bibliographic coupling (Kessler, " +
      "1963) and co-citation (Small, 1973) scored via the Semantic Scholar API. " +
      "Discovery Network lets you incrementally grow a persistent collection by " +
      "exploring earlier work (references), later work (citations), similar papers " +
      "(recommendations), and author networks.",
    tags: ["Similarity Graph", "Discovery Network", "Bibliographic Coupling", "Co-citation", "Semantic Scholar", "Force Layout"],
  },
];

interface LandingPageProps {
  onSelect: (id: ProjectId) => void;
}

function LandingPage({ onSelect }: LandingPageProps) {
  return (
    <div className="landing-page">
      <img className="landing-page__logo" src={beeSearchLogo} alt="BeeSearch logo" />
      <p className="landing-page__tagline">
        Local AI tools for systematic literature review and source-grounded research notebooks —
        Ollama · LangGraph · Hybrid RAG · Google Scholar · arXiv · Semantic Scholar
      </p>

      <h2 className="landing-page__select-heading">Select a mode to get started</h2>
      <hr />

      <div className="landing-page__cards">
        {PROJECTS.map((project) => (
          <div className="mode-card" key={project.id}>
            <div className="mode-label">MODE {project.label}</div>
            <div className="mode-title">{project.name}</div>
            <p className="mode-desc">{project.description}</p>
            <div className="mode-tags">
              {project.tags.map((tag) => (
                <span className="mode-tag" key={tag}>
                  {tag}
                </span>
              ))}
            </div>
            <button
              type="button"
              className="mode-card__open-button"
              disabled={project.comingSoon}
              onClick={() => onSelect(project.id)}
            >
              {project.comingSoon ? "Coming soon" : `Open ${project.name}`}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default LandingPage;
