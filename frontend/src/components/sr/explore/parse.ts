/** Small type-narrowing helpers for parsing the untyped `Record<string, unknown>`
 * results that come back from the generic Explore-tool job endpoint. */

export function str(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}

export function num(v: unknown, fallback = 0): number {
  return typeof v === "number" ? v : fallback;
}

export function bool(v: unknown, fallback = false): boolean {
  return typeof v === "boolean" ? v : fallback;
}

export function obj(v: unknown): Record<string, unknown> {
  return typeof v === "object" && v !== null ? (v as Record<string, unknown>) : {};
}

export function arr<T = unknown>(v: unknown): T[] {
  return Array.isArray(v) ? (v as T[]) : [];
}

export function strArr(v: unknown): string[] {
  return arr(v).filter((x): x is string => typeof x === "string");
}

export function nullableStr(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

export function nullableNum(v: unknown): number | null {
  return typeof v === "number" ? v : null;
}
