import { useEffect, useState } from "react";
import "./App.css";
import AskPage from "./pages/AskPage";
import LandingPage, { type ProjectId } from "./pages/LandingPage";
import SystematicReviewPage from "./pages/SystematicReviewPage";

const PROJECT_NAMES: Record<ProjectId, string> = {
  mode1: "Systematic Literature Review",
  mode2: "Research Notebook",
  mode3: "AI Research Assistant",
};

function readModeFromUrl(): ProjectId | null {
  const mode = new URLSearchParams(window.location.search).get("mode");
  return mode === "mode1" || mode === "mode2" || mode === "mode3" ? mode : null;
}

function App() {
  const [activeMode, setActiveMode] = useState<ProjectId | null>(() => readModeFromUrl());

  useEffect(() => {
    const url = new URL(window.location.href);
    if (activeMode) {
      url.searchParams.set("mode", activeMode);
    } else {
      url.searchParams.delete("mode");
    }
    window.history.replaceState(null, "", url);
  }, [activeMode]);

  if (!activeMode) {
    return <LandingPage onSelect={setActiveMode} />;
  }

  return (
    <div className="app">
      <div className="app__bar">
        <button type="button" className="app__back-button" onClick={() => setActiveMode(null)}>
          ← All Modes
        </button>
        <span className="app__bar-title">{PROJECT_NAMES[activeMode]}</span>
      </div>

      {activeMode === "mode1" && <SystematicReviewPage />}
      {activeMode === "mode3" && <AskPage />}
      {activeMode === "mode2" && (
        <p className="app__coming-soon">Research Notebook is coming soon.</p>
      )}
    </div>
  );
}

export default App;
