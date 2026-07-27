import { apiFetch } from "./client";
import type { ShutdownResult, SystemStatusResponse } from "./systemTypes";

const BASE = "/api/system";

export function getSystemStatus(ramOverrideGb?: number): Promise<SystemStatusResponse> {
  const query = ramOverrideGb ? `?ram_override_gb=${ramOverrideGb}` : "";
  return apiFetch<SystemStatusResponse>(`${BASE}/status${query}`);
}

export function shutdownServer(): Promise<ShutdownResult> {
  return apiFetch<ShutdownResult>(`${BASE}/shutdown`, { method: "POST" });
}
