import { useEffect, useState } from "react";
import { exportAudioWav, exportText, runAudioSummary } from "../../../api/notebookAdvanced";
import { downloadBlob, downloadText } from "../../../utils/download";
import { type AsyncState, errorMessage } from "./format";
import { RunControls } from "./shared";
import { useAdvancedToolJob, useModelOverrides } from "./useAdvancedToolJob";

interface AudioSummaryPanelProps {
  notebookId: string;
}

/** Mirrors ui/tabs/notebook.py::_tab_audio. */
function AudioSummaryPanel({ notebookId }: AudioSummaryPanelProps) {
  const job = useAdvancedToolJob();
  const overrides = useModelOverrides();
  const { state, jobId, result, error } = job;

  return (
    <div className="advanced-tools-tab__panel">
      <h3>Audio Summary</h3>
      <p>
        Generates a spoken-word summary script (~300 words, ~2 min) and synthesizes it to a
        downloadable .wav audio file.
      </p>

      <RunControls
        state={state}
        runLabel="Generate Audio Script"
        rerunLabel="Regenerate Audio Script"
        spinnerText="Generating audio summary script…"
        error={error}
        errorPrefix="Audio script generation failed"
        onRun={() => job.run(() => runAudioSummary({ notebook_id: notebookId, ...overrides }))}
        onClear={job.clear}
      />

      {state === "done" &&
        result &&
        jobId &&
        (result.audio_script ? (
          <AudioScriptView jobId={jobId} script={result.audio_script} />
        ) : (
          <p className="sr-info">No audio script was generated for this run.</p>
        ))}
    </div>
  );
}

interface AudioScriptViewProps {
  jobId: string;
  script: string;
}

function AudioScriptView({ jobId, script }: AudioScriptViewProps) {
  const wordCount = script.split(/\s+/).filter(Boolean).length;
  const [textState, setTextState] = useState<AsyncState>("idle");
  const [textError, setTextError] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [wavState, setWavState] = useState<AsyncState>("idle");
  const [wavError, setWavError] = useState<string | null>(null);
  const [wavUrl, setWavUrl] = useState<string | null>(null);
  const [wavBlob, setWavBlob] = useState<Blob | null>(null);
  const ttsSupported = typeof window !== "undefined" && "speechSynthesis" in window;

  useEffect(() => {
    return () => {
      if (wavUrl) URL.revokeObjectURL(wavUrl);
    };
  }, [wavUrl]);

  useEffect(() => {
    return () => {
      if (ttsSupported) window.speechSynthesis.cancel();
    };
  }, [ttsSupported]);

  async function handleDownloadScript() {
    setTextState("running");
    setTextError(null);
    try {
      const text = await exportText(jobId, "audio-script");
      downloadText(text, "audio_summary_script.txt", "text/plain");
      setTextState("idle");
    } catch (err) {
      setTextError(errorMessage(err));
      setTextState("error");
    }
  }

  function handleToggleSpeak() {
    if (!ttsSupported) return;
    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
      return;
    }
    const utterance = new SpeechSynthesisUtterance(script);
    utterance.rate = 0.9;
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
    setSpeaking(true);
  }

  async function handleSynthesize() {
    setWavState("running");
    setWavError(null);
    try {
      const blob = await exportAudioWav(jobId);
      setWavUrl(URL.createObjectURL(blob));
      setWavBlob(blob);
      setWavState("idle");
    } catch (err) {
      setWavError(errorMessage(err));
      setWavState("error");
    }
  }

  return (
    <div>
      <p className="sr-caption">Word count: {wordCount}</p>
      <pre className="advanced-tools-tab__audio-script">{script}</pre>

      <div className="sr-button-row">
        <button
          type="button"
          className="sr-button"
          disabled={textState === "running"}
          onClick={() => void handleDownloadScript()}
        >
          Download script (.txt)
        </button>
        {ttsSupported && (
          <button type="button" className="sr-button" onClick={handleToggleSpeak}>
            {speaking ? "Stop reading" : "Play in browser"}
          </button>
        )}
        <button
          type="button"
          className="sr-button"
          disabled={wavState === "running"}
          onClick={() => void handleSynthesize()}
        >
          Synthesize .wav
        </button>
      </div>
      {textState === "error" && <p className="sr-error">Script export failed: {textError}</p>}
      {wavState === "error" && <p className="sr-error">Audio synthesis failed: {wavError}</p>}

      {wavUrl && wavBlob && (
        <div className="advanced-tools-tab__audio-player">
          <audio controls src={wavUrl} />
          <div className="sr-button-row">
            <button type="button" className="sr-button" onClick={() => downloadBlob(wavBlob, "audio_summary.wav")}>
              Download .wav
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default AudioSummaryPanel;
