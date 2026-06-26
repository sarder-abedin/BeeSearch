export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

interface FastApiValidationError {
  msg?: string;
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    const detail = (body as { detail?: unknown } | null)?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const messages = (detail as FastApiValidationError[])
        .map((d) => d.msg)
        .filter((m): m is string => Boolean(m));
      if (messages.length) return messages.join("; ");
    }
    return res.statusText;
  } catch {
    return res.statusText;
  }
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new ApiError(res.status, detail);
  }

  return res;
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await request(path, init);
  return (await res.json()) as T;
}

/** For binary downloads (DOCX/PDF exports). */
export async function apiFetchBlob(path: string, init?: RequestInit): Promise<Blob> {
  const res = await request(path, init);
  return res.blob();
}

/** For plain-text endpoints (Markdown export). */
export async function apiFetchText(path: string, init?: RequestInit): Promise<string> {
  const res = await request(path, init);
  return res.text();
}
