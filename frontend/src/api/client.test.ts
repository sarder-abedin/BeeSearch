import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiFetch } from "./client";

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
