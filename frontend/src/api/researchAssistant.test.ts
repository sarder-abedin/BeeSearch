import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AskJobStatus } from "./types";

const apiFetchMock = vi.fn();
vi.mock("./client", () => ({
  apiFetch: (...args: unknown[]) => apiFetchMock(...args),
}));

const { askResearchAssistant, getAskJobStatus, pollAskJob } = await import("./researchAssistant");

function jobStatus(overrides: Partial<AskJobStatus> = {}): AskJobStatus {
  return {
    id: "job-1",
    status: "running",
    stage: null,
    stage_info: {},
    error: null,
    result: null,
    ...overrides,
  };
}

describe("askResearchAssistant / getAskJobStatus", () => {
  beforeEach(() => {
    apiFetchMock.mockReset();
  });

  it("POSTs the request body to /api/research-assistant/ask", async () => {
    apiFetchMock.mockResolvedValue({ job_id: "job-1" });
    const result = await askResearchAssistant({ question: "q", include_web: false });

    expect(result).toEqual({ job_id: "job-1" });
    expect(apiFetchMock).toHaveBeenCalledWith("/api/research-assistant/ask", {
      method: "POST",
      body: JSON.stringify({ question: "q", include_web: false }),
    });
  });

  it("GETs job status by id", async () => {
    const status = jobStatus();
    apiFetchMock.mockResolvedValue(status);

    const result = await getAskJobStatus("job-1");

    expect(result).toBe(status);
    expect(apiFetchMock).toHaveBeenCalledWith("/api/research-assistant/jobs/job-1");
  });
});

describe("pollAskJob", () => {
  beforeEach(() => {
    apiFetchMock.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("resolves immediately when the first poll is already terminal", async () => {
    const done = jobStatus({ status: "done", result: null });
    apiFetchMock.mockResolvedValue(done);
    const onUpdate = vi.fn();

    const result = await pollAskJob("job-1", onUpdate);

    expect(result).toBe(done);
    expect(onUpdate).toHaveBeenCalledWith(done);
    expect(apiFetchMock).toHaveBeenCalledTimes(1);
  });

  it("polls repeatedly until a terminal status, calling onUpdate every time", async () => {
    const running1 = jobStatus({ status: "running", stage: "searching" });
    const running2 = jobStatus({ status: "running", stage: "reading" });
    const done = jobStatus({ status: "done", stage: "done" });
    apiFetchMock
      .mockResolvedValueOnce(running1)
      .mockResolvedValueOnce(running2)
      .mockResolvedValueOnce(done);
    const onUpdate = vi.fn();

    const promise = pollAskJob("job-1", onUpdate);

    await vi.advanceTimersByTimeAsync(700);
    await vi.advanceTimersByTimeAsync(700);
    const result = await promise;

    expect(result).toBe(done);
    expect(onUpdate.mock.calls.map((c) => c[0])).toEqual([running1, running2, done]);
  });

  it("rejects with AbortError when the signal fires before a terminal status", async () => {
    apiFetchMock.mockResolvedValue(jobStatus({ status: "running" }));
    const controller = new AbortController();
    const onUpdate = vi.fn();

    const promise = pollAskJob("job-1", onUpdate, controller.signal);
    controller.abort();

    await expect(promise).rejects.toMatchObject({ name: "AbortError" });
  });

  it("propagates errors raised while fetching job status", async () => {
    apiFetchMock.mockRejectedValue(new Error("network down"));

    await expect(pollAskJob("job-1", vi.fn())).rejects.toThrow("network down");
  });
});
