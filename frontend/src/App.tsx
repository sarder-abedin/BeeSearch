import { useEffect, useState } from "react";
import "./App.css";
import AskPage from "./pages/AskPage";
import LandingPage, { type ProjectId } from "./pages/LandingPage";
import NotebookPage from "./pages/NotebookPage";
import PaperDiscoveryPage from "./pages/PaperDiscoveryPage";
import SystematicReviewPage from "./pages/SystematicReviewPage";
import SettingsPanel from "./components/SettingsPanel";
import { SettingsProvider } from "./context/SettingsContext";

const PROJECT_NAMES: Record<ProjectId, string> = {
  mode1: "Systematic Literature Review",
  mode2: "Research Notebook",
  mode3: "AI Research Assistant",
  mode4: "Paper Discovery",
};

function readModeFromUrl(): ProjectId | null {
  const mode = new URLSearchParams(window.location.search).get("mode");
  return mode === "mode1" || mode === "mode2" || mode === "mode3" || mode === "mode4"
    ? mode
    : null;
}

function App() {
  const [activeMode, setActiveMode] = useState<ProjectId | null>(() => readModeFromUrl());
  const [settingsOpen, setSettingsOpen] = useState(false);

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
    return (
      <SettingsProvider>
        <LandingPage onSelect={setActiveMode} />
        <button
          type="button"
          className="app__settings-button app__settings-button--landing"
          aria-label="Open settings"
          onClick={() => setSettingsOpen(true)}
        >
          ⚙
        </button>
        {settingsOpen && <SettingsPanel onClose={() => setSettingsOpen(false)} />}
      </SettingsProvider>
    );
  }

  return (
    <SettingsProvider>
      <div className="app">
        <div className="app__bar">
          <button type="button" className="app__back-button" onClick={() => setActiveMode(null)}>
            ← All Modes
          </button>
          <span className="app__bar-title">{PROJECT_NAMES[activeMode]}</span>
          <button
            type="button"
            className="app__settings-button"
            aria-label="Open settings"
            onClick={() => setSettingsOpen(true)}
          >
            ⚙
          </button>
        </div>

        {activeMode === "mode1" && <SystematicReviewPage />}
        {activeMode === "mode3" && <AskPage />}
        {activeMode === "mode2" && <NotebookPage />}
        {activeMode === "mode4" && <PaperDiscoveryPage />}

        {settingsOpen && <SettingsPanel onClose={() => setSettingsOpen(false)} />}
      </div>
    </SettingsProvider>
  );
}

export default App;
