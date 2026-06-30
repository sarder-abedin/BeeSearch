import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getSystemStatus, shutdownServer } from "../api/system";
import type { SystemStatusResponse, TemperatureLevel } from "../api/systemTypes";

const STORAGE_KEY = "beesearch.settings.v1";

export interface PersistedSettings {
  model: string | null;
  numCtx: number;
  temperatureLevel: TemperatureLevel;
  embedModel: string | null;
  hybridTopK: number;
  maxResults: number;
  includeCrossref: boolean;
  chunkSize: number;
  chunkOverlap: number;
}

const DEFAULT_SETTINGS: PersistedSettings = {
  model: null,
  numCtx: 8192,
  temperatureLevel: "focused",
  embedModel: null,
  hybridTopK: 8,
  maxResults: 6,
  includeCrossref: true,
  chunkSize: 800,
  chunkOverlap: 150,
};

function loadPersisted(): PersistedSettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<PersistedSettings>) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export interface SettingsContextValue extends PersistedSettings {
  setModel: (model: string | null) => void;
  setNumCtx: (numCtx: number) => void;
  setTemperatureLevel: (level: TemperatureLevel) => void;
  setEmbedModel: (model: string | null) => void;
  setHybridTopK: (topK: number) => void;
  setMaxResults: (max: number) => void;
  setIncludeCrossref: (include: boolean) => void;
  setChunkSize: (size: number) => void;
  setChunkOverlap: (overlap: number) => void;

  status: SystemStatusResponse | null;
  statusLoading: boolean;
  statusError: string | null;
  refresh: (ramOverrideGb?: number) => Promise<void>;

  applyRecommended: (modelName?: string, numCtxOverride?: number) => void;
  applyAllRecommended: () => void;

  shuttingDown: boolean;
  shutdownError: string | null;
  requestShutdown: () => Promise<void>;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<PersistedSettings>(() => loadPersisted());
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [shuttingDown, setShuttingDown] = useState(false);
  const [shutdownError, setShutdownError] = useState<string | null>(null);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  const refresh = useCallback(async (ramOverrideGb?: number) => {
    setStatusLoading(true);
    setStatusError(null);
    try {
      const next = await getSystemStatus(ramOverrideGb);
      setStatus(next);
      setSettings((prev) => ({
        ...prev,
        embedModel: prev.embedModel ?? next.embed_models.find((m) => m.pulled)?.name ?? next.embed_models[0]?.name ?? null,
      }));
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : String(err));
    } finally {
      setStatusLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => void refresh(), 0);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const applyRecommended = useCallback(
    (modelName?: string, numCtxOverride?: number) => {
      if (!status) return;
      const rec = status.recommendation;
      const chosenModel = modelName ?? rec.model ?? undefined;
      const chosenCtx = numCtxOverride ?? rec.num_ctx;
      if (!chosenModel) return;
      setSettings((prev) => ({ ...prev, model: chosenModel, numCtx: chosenCtx }));
    },
    [status],
  );

  const applyAllRecommended = useCallback(() => {
    if (!status) return;
    const { recommendation: rec, tier } = status;
    if (!rec.can_run || !rec.model) return;
    setSettings((prev) => ({
      ...prev,
      model: rec.model as string,
      numCtx: tier.num_ctx,
      hybridTopK: tier.hybrid_top_k,
      maxResults: tier.max_results,
      chunkSize: tier.chunk_size,
      chunkOverlap: tier.chunk_overlap,
    }));
  }, [status]);

  const requestShutdown = useCallback(async () => {
    setShuttingDown(true);
    setShutdownError(null);
    try {
      await shutdownServer();
    } catch (err) {
      setShutdownError(err instanceof Error ? err.message : String(err));
    } finally {
      setShuttingDown(false);
    }
  }, []);

  const value = useMemo<SettingsContextValue>(
    () => ({
      ...settings,
      setModel: (model) => setSettings((p) => ({ ...p, model })),
      setNumCtx: (numCtx) => setSettings((p) => ({ ...p, numCtx })),
      setTemperatureLevel: (temperatureLevel) => setSettings((p) => ({ ...p, temperatureLevel })),
      setEmbedModel: (embedModel) => setSettings((p) => ({ ...p, embedModel })),
      setHybridTopK: (hybridTopK) => setSettings((p) => ({ ...p, hybridTopK })),
      setMaxResults: (maxResults) => setSettings((p) => ({ ...p, maxResults })),
      setIncludeCrossref: (includeCrossref) => setSettings((p) => ({ ...p, includeCrossref })),
      setChunkSize: (chunkSize) => setSettings((p) => ({ ...p, chunkSize })),
      setChunkOverlap: (chunkOverlap) => setSettings((p) => ({ ...p, chunkOverlap })),
      status,
      statusLoading,
      statusError,
      refresh,
      applyRecommended,
      applyAllRecommended,
      shuttingDown,
      shutdownError,
      requestShutdown,
    }),
    [settings, status, statusLoading, statusError, refresh, applyRecommended, applyAllRecommended, shuttingDown, shutdownError, requestShutdown],
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components -- hook must live alongside its context
export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used within a SettingsProvider");
  return ctx;
}
