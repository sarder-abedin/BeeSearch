import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiFetch, apiFetchBlob, apiFetchText } from "./client";

describe("apiFetch", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resolves with the parsed JSON body on a 200", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await apiFetch<{ ok: boolean }>("/api/health");

    expect(result).toEqual({ ok: true });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers["Content-Type"]).toBe("application/json");
  });

  it("throws ApiError with the string detail from a JSON error body", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Job not found." }), {
          status: 404,
          statusText: "Not Found",
        }),
      ),
    );

    await expect(apiFetch("/api/research-assistant/jobs/x")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      detail: "Job not found.",
    });
  });

  it("joins FastAPI validation-error messages from a detail array", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: [
              { msg: "Value error, Please enter a research question." },
              { msg: "field required" },
            ],
          }),
          { status: 422, statusText: "Unprocessable Entity" },
        ),
      ),
    );

    let error: unknown;
    try {
      await apiFetch("/api/research-assistant/ask");
    } catch (e) {
      error = e;
    }
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).detail).toBe(
      "Value error, Please enter a research question.; field required",
    );
  });

  it("falls back to statusText when the error body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("<not json>", { status: 500, statusText: "Internal Server Error" })),
    );

    await expect(apiFetch("/api/health")).rejects.toMatchObject({
      status: 500,
      detail: "Internal Server Error",
    });
  });
});

describe("apiFetchBlob", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resolves with a Blob on a 200", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("binary content", { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await apiFetchBlob("/api/systematic-review/jobs/x/export/docx");

    expect(typeof result.text).toBe("function");
    expect(await result.text()).toBe("binary content");
  });

  it("throws ApiError with the parsed detail on a non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Review job failed: boom" }), {
          status: 409,
          statusText: "Conflict",
        }),
      ),
    );

    await expect(apiFetchBlob("/api/systematic-review/jobs/x/export/docx")).rejects.toMatchObject({
      name: "ApiError",
      status: 409,
      detail: "Review job failed: boom",
    });
  });
});

describe("apiFetchText", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resolves with the raw text body on a 200", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("# Systematic Review Report", { status: 200 })),
    );

    const result = await apiFetchText("/api/systematic-review/jobs/x/export/markdown");

    expect(result).toBe("# Systematic Review Report");
  });

  it("throws ApiError on a non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Job not found." }), {
          status: 404,
          statusText: "Not Found",
        }),
      ),
    );

    await expect(apiFetchText("/api/systematic-review/jobs/x/export/markdown")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      detail: "Job not found.",
    });
  });
});
