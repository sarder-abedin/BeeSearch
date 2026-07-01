import { useState } from "react";
import { useSettings } from "../context/SettingsContext";
import "./SettingsPanel.css";

const GPU_LABELS: Record<string, string> = {
  apple_silicon: "Apple Silicon (Metal)",
  nvidia: "NVIDIA (CUDA)",
  cpu: "CPU only",
};

const CTX_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

interface SettingsPanelProps {
  onClose: () => void;
}

export default function SettingsPanel({ onClose }: SettingsPanelProps) {
  const s = useSettings();
  const [ramOverride, setRamOverride] = useState("");
  const [tightFitChoice, setTightFitChoice] = useState<"recommended" | "safe">("recommended");
  const [shutdownConfirm, setShutdownConfirm] = useState(false);
  const [shutdownDone, setShutdownDone] = useState(false);

  const status = s.status;
  const rec = status?.recommendation;
  const tier = status?.tier;
  const hw = status?.hardware;

  async function handleApplyRamOverride() {
    const val = Number(ramOverride);
    if (!val || val <= 0) return;
    await s.refresh(val);
  }

  async function handleConfirmShutdown() {
    await s.requestShutdown();
    setShutdownDone(true);
  }

  return (
    <div className="settings-panel__overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Settings">
        <div className="settings-panel__header">
          <h2>Settings</h2>
          <button type="button" className="settings-panel__close" onClick={onClose} aria-label="Close settings">
            ×
          </button>
        </div>

        <div className="settings-panel__body">
          {s.statusLoading && <p className="sr-spinner-text">Loading hardware status…</p>}
          {s.statusError && <p className="sr-error">{s.statusError}</p>}

          {hw && tier && rec && (
            <>
              {/* ── Hardware ────────────────────────────── */}
              <section className="settings-panel__section">
                <h3>Hardware</h3>
                <div className="sr-metric-row">
                  <div className="sr-metric">
                    <span className="sr-metric__label">RAM</span>
                    <span className="sr-metric__value">{hw.ram_gb.toFixed(0)} GB</span>
                  </div>
                  <div className="sr-metric">
                    <span className="sr-metric__label">Accelerator</span>
                    <span className="sr-metric__value">{GPU_LABELS[hw.gpu_type] ?? "Unknown"}</span>
                  </div>
                </div>
                <p className="sr-caption">CPU: {hw.cpu}</p>
                <p className="sr-caption">OS: {hw.os} ({hw.arch})</p>
                <p className="sr-caption">
                  <strong>Performance tier:</strong> {tier.label} — {tier.description}
                </p>

                {hw.in_docker && (
                  <div className="sr-info">
                    Running in <strong>Docker</strong> — detected {hw.ram_gb.toFixed(0)} GB (container
                    allocation, not host RAM). Enter your machine&apos;s actual RAM below for accurate
                    model recommendations.
                    <div className="sr-button-row">
                      <input
                        type="number"
                        min={1}
                        max={512}
                        step={8}
                        value={ramOverride}
                        onChange={(e) => setRamOverride(e.target.value)}
                        placeholder={hw.ram_gb.toFixed(0)}
                      />
                      <button type="button" className="sr-button" onClick={() => void handleApplyRamOverride()}>
                        Apply
                      </button>
                    </div>
                  </div>
                )}
              </section>

              {/* ── Model Recommendation ───────────────────── */}
              <section className="settings-panel__section">
                <h3>Model Recommendation</h3>
                {rec.can_run ? (
                  rec.tight_fit && rec.safe_alternative ? (
                    <>
                      <p className="sr-warning">{rec.reasoning}</p>
                      <label className="settings-panel__radio">
                        <input
                          type="radio"
                          name="tight-fit-choice"
                          checked={tightFitChoice === "recommended"}
                          onChange={() => setTightFitChoice("recommended")}
                        />
                        {rec.model} — higher capability, tight fit
                      </label>
                      <label className="settings-panel__radio">
                        <input
                          type="radio"
                          name="tight-fit-choice"
                          checked={tightFitChoice === "safe"}
                          onChange={() => setTightFitChoice("safe")}
                        />
                        {rec.safe_alternative.name} — {rec.safe_alternative.ram_gb} GB, more headroom
                      </label>
                      <div className="sr-button-row">
                        <button
                          type="button"
                          className="sr-button"
                          onClick={() =>
                            s.applyRecommended(
                              tightFitChoice === "recommended" ? rec.model ?? undefined : rec.safe_alternative?.name,
                              rec.num_ctx,
                            )
                          }
                        >
                          Apply Selection
                        </button>
                        <button type="button" className="sr-button" onClick={() => void s.refresh()}>
                          Refresh
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <p className="sr-success">
                        <strong>{rec.model}</strong> — {rec.reasoning}
                      </p>
                      <div className="sr-button-row">
                        <button type="button" className="sr-button" onClick={() => s.applyRecommended()}>
                          Apply Recommendation
                        </button>
                        <button type="button" className="sr-button" onClick={() => void s.refresh()}>
                          Refresh
                        </button>
                      </div>
                    </>
                  )
                ) : (
                  <>
                    <p className="sr-warning">No compatible models found. {rec.hardware_note}</p>
                    {rec.pull_command && <pre className="settings-panel__code">{rec.pull_command}</pre>}
                    <button type="button" className="sr-button" onClick={() => void s.refresh()}>
                      Refresh after pulling
                    </button>
                  </>
                )}
              </section>

              {/* ── Recommended Configuration ──────────────── */}
              <section className="settings-panel__section">
                <h3>Recommended Configuration</h3>
                <div className="sr-metric-row">
                  <div className="sr-metric">
                    <span className="sr-metric__label">Context (tokens)</span>
                    <span className="sr-metric__value">{tier.num_ctx.toLocaleString()}</span>
                  </div>
                  <div className="sr-metric">
                    <span className="sr-metric__label">Chunks per query</span>
                    <span className="sr-metric__value">{tier.hybrid_top_k}</span>
                  </div>
                  <div className="sr-metric">
                    <span className="sr-metric__label">Chunk size (chars)</span>
                    <span className="sr-metric__value">{tier.chunk_size}</span>
                  </div>
                  <div className="sr-metric">
                    <span className="sr-metric__label">Max papers</span>
                    <span className="sr-metric__value">{tier.max_results}</span>
                  </div>
                </div>
                {rec.can_run && rec.model && (
                  <div className="sr-button-row">
                    <button type="button" className="sr-button" onClick={s.applyAllRecommended}>
                      Apply All Recommended Settings
                    </button>
                  </div>
                )}
              </section>

              {/* ── LLM Model ───────────────────────────────── */}
              <section className="settings-panel__section">
                <h3>LLM Model</h3>
                {status.available_models.length > 0 ? (
                  <div className="sr-field">
                    <label htmlFor="settings-model">Active model</label>
                    <select
                      id="settings-model"
                      value={s.model ?? status.available_models[0]}
                      onChange={(e) => s.setModel(e.target.value)}
                    >
                      {status.available_models.map((m) => (
                        <option key={m} value={m}>
                          {m === rec.model ? `${m} (recommended)` : m}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <div className="sr-field">
                    <label htmlFor="settings-model-manual">Model name (manual entry)</label>
                    <input
                      id="settings-model-manual"
                      type="text"
                      value={s.model ?? rec.model ?? ""}
                      onChange={(e) => s.setModel(e.target.value)}
                      placeholder="llama3.2:3b"
                    />
                  </div>
                )}
              </section>

              {/* ── Response Tuning ─────────────────────────── */}
              <section className="settings-panel__section">
                <h3>Response Tuning</h3>
                <div className="sr-field">
                  <label htmlFor="settings-temperature">Temperature level</label>
                  <select
                    id="settings-temperature"
                    value={s.temperatureLevel}
                    onChange={(e) => s.setTemperatureLevel(e.target.value as typeof s.temperatureLevel)}
                  >
                    {status.temperature_levels.map((lvl) => (
                      <option key={lvl.key} value={lvl.key}>
                        {lvl.label}
                      </option>
                    ))}
                  </select>
                </div>
                <p className="sr-caption">
                  {status.temperature_levels.find((l) => l.key === s.temperatureLevel)?.description}
                </p>
                <p className="sr-caption">Applies to Research Notebook chat, summaries, and explanations.</p>
              </section>

              {/* ── Context Window ──────────────────────────── */}
              <section className="settings-panel__section">
                <h3>Context Window</h3>
                <div className="sr-field">
                  <label htmlFor="settings-ctx">Tokens</label>
                  <select id="settings-ctx" value={s.numCtx} onChange={(e) => s.setNumCtx(Number(e.target.value))}>
                    {CTX_OPTIONS.map((c) => (
                      <option key={c} value={c}>
                        {c.toLocaleString()}
                      </option>
                    ))}
                  </select>
                </div>
              </section>

              {/* ── Hybrid RAG ───────────────────────────────── */}
              <section className="settings-panel__section">
                <h3>Hybrid RAG</h3>
                <p className="sr-caption">FAISS dense + BM25 sparse search, fused with Reciprocal Rank Fusion.</p>
                <div className="sr-field">
                  <label htmlFor="settings-embed">Embedding model</label>
                  <select
                    id="settings-embed"
                    value={s.embedModel ?? status.embed_models[0]?.name ?? ""}
                    onChange={(e) => s.setEmbedModel(e.target.value)}
                  >
                    {status.embed_models.map((m) => (
                      <option key={m.name} value={m.name}>
                        {m.name} ({m.dim}d, {m.size_gb} GB){m.pulled ? "" : " — not pulled"}
                      </option>
                    ))}
                  </select>
                </div>
                {s.embedModel && status.embed_models.find((m) => m.name === s.embedModel)?.pulled === false && (
                  <p className="sr-warning">Run: ollama pull {s.embedModel}</p>
                )}
                <div className="sr-field">
                  <label htmlFor="settings-topk">Chunks per query: {s.hybridTopK}</label>
                  <input
                    id="settings-topk"
                    type="range"
                    min={3}
                    max={20}
                    value={s.hybridTopK}
                    onChange={(e) => s.setHybridTopK(Number(e.target.value))}
                  />
                </div>
              </section>

              {/* ── Search Settings ─────────────────────────── */}
              <section className="settings-panel__section">
                <h3>Search Settings</h3>
                <div className="sr-field">
                  <label htmlFor="settings-max-results">Max papers per query: {s.maxResults}</label>
                  <input
                    id="settings-max-results"
                    type="range"
                    min={3}
                    max={20}
                    value={s.maxResults}
                    onChange={(e) => s.setMaxResults(Number(e.target.value))}
                  />
                </div>
                <label className="settings-panel__radio">
                  <input
                    type="checkbox"
                    checked={s.includeCrossref}
                    onChange={(e) => s.setIncludeCrossref(e.target.checked)}
                  />
                  Include CrossRef search
                </label>
              </section>

              {/* ── Document Settings ───────────────────────── */}
              <section className="settings-panel__section">
                <h3>Document Settings</h3>
                <div className="sr-field">
                  <label htmlFor="settings-chunk-size">Chunk size (chars): {s.chunkSize}</label>
                  <input
                    id="settings-chunk-size"
                    type="range"
                    min={400}
                    max={1200}
                    step={100}
                    value={s.chunkSize}
                    onChange={(e) => s.setChunkSize(Number(e.target.value))}
                  />
                </div>
                <div className="sr-field">
                  <label htmlFor="settings-chunk-overlap">Chunk overlap (chars): {s.chunkOverlap}</label>
                  <input
                    id="settings-chunk-overlap"
                    type="range"
                    min={50}
                    max={300}
                    step={25}
                    value={s.chunkOverlap}
                    onChange={(e) => s.setChunkOverlap(Number(e.target.value))}
                  />
                </div>
                <label className="sr-toggle-label">
                  <input
                    type="checkbox"
                    checked={s.useDocling}
                    onChange={(e) => s.setUseDocling(e.target.checked)}
                  />
                  Advanced Parsing (Docling)
                </label>
                <label className="sr-toggle-label" style={{ opacity: s.useDocling ? 1 : 0.4 }}>
                  <input
                    type="checkbox"
                    checked={s.useOcr}
                    disabled={!s.useDocling}
                    onChange={(e) => s.setUseOcr(e.target.checked)}
                  />
                  Enable OCR (slower)
                </label>
                <div className="sr-field">
                  <label htmlFor="settings-large-doc-threshold">
                    Large-PDF page threshold: {s.largeDocPageThreshold}
                    <span className="settings-panel__help" title="PDFs above this page count use pdfplumber (streaming) instead of Docling ML models to avoid RAM spikes."> (?)</span>
                  </label>
                  <input
                    id="settings-large-doc-threshold"
                    type="range"
                    min={10}
                    max={150}
                    step={10}
                    value={s.largeDocPageThreshold}
                    onChange={(e) => s.setLargeDocPageThreshold(Number(e.target.value))}
                  />
                </div>
              </section>
            </>
          )}

          {/* ── Safe Shutdown ─────────────────────────────── */}
          <section className="settings-panel__section">
            <h3>Safe Shutdown</h3>
            {shutdownDone ? (
              <p className="sr-info">Shutdown requested — the server is stopping.</p>
            ) : !shutdownConfirm ? (
              <button type="button" className="sr-button settings-panel__danger" onClick={() => setShutdownConfirm(true)}>
                Shut Down Safely
              </button>
            ) : (
              <>
                <p className="sr-warning">
                  This will stop the server for every connected user. Sessions are saved to disk.
                </p>
                <div className="sr-button-row">
                  <button
                    type="button"
                    className="sr-button settings-panel__danger"
                    onClick={() => void handleConfirmShutdown()}
                    disabled={s.shuttingDown}
                  >
                    Confirm Shutdown
                  </button>
                  <button type="button" className="sr-button" onClick={() => setShutdownConfirm(false)}>
                    Cancel
                  </button>
                </div>
                {s.shutdownError && <p className="sr-error">{s.shutdownError}</p>}
              </>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
