import { ApiError } from "../../../api/client";

export type AsyncState = "idle" | "running" | "error";

export function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.detail : (err as Error).message;
}

/** Mirrors tools/text_parsing.py::format_page_label -- "n/a" for the -1
 * unknown sentinel (or null/any other non-positive value), else 1-based. */
export function formatPageLabel(pageNum: number | null): string {
  if (typeof pageNum !== "number" || !Number.isInteger(pageNum) || pageNum < 0) return "n/a";
  return `p. ${pageNum + 1}`;
}

/** Resolves a 1-based source index to its filename, truncated to `maxLen`
 * when given. Returns "—" for an out-of-range index -- mirrors Citation
 * Timeline's `src_names[src_n - 1][:20] if ... else "—"` in ui/tabs/notebook.py. */
export function resolveSourceLabel(n: number, sourceNames: string[], maxLen?: number): string {
  if (!Number.isInteger(n) || n < 1 || n > sourceNames.length) return "—";
  const name = sourceNames[n - 1];
  return maxLen ? name.slice(0, maxLen) : name;
}

/** Resolves a list of 1-based source indices, silently dropping any
 * out-of-range index -- mirrors FAQ's `[src_names[n-1] for n in cited if
 * isinstance(n, int) and 1 <= n <= len(src_names)]` (no "—" fallback; an
 * invalid index is omitted, not shown, since FAQ never renders a per-source
 * cell the way Citation Timeline's table does). */
export function resolveValidSourceLabels(indices: number[], sourceNames: string[]): string[] {
  return indices.filter((n) => Number.isInteger(n) && n >= 1 && n <= sourceNames.length).map((n) => sourceNames[n - 1]);
}
