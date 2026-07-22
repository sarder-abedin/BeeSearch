import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AskJobStatus, AskResult } from "../api/types";
import AskPage from "./AskPage";

const askResearchAssistantMock = vi.fn();
const pollAskJobMock = vi.fn();

vi.mock("../api/researchAssistant", () => ({
  askResearchAssistant: (...args: unknown[]) => askResearchAssistantMock(...args),
  pollAskJob: (...args: unknown[]) => pollAskJobMock(...args),
}));

vi.mock("../context/SettingsContext", () => ({
  useSettings: () => ({
    model: null,
    numCtx: 8192,
    temperatureLevel: "focused",
    embedModel: null,
    hybridTopK: 8,
    maxResults: 6,
    includeCrossref: true,
    chunkSize: 800,
    chunkOverlap: 150,
  }),
}));

const SOURCE = {
  n: 1,
  kind: "academic" as const,
  title: "Sleep & Memory",
  authors: ["A Smith"],
  year: 2020,
  url: "http://p1",
  snippet: "sleep helps memory",
  apa: "Smith, A. (2020). Sleep & Memory.",
  source: "arxiv",
};

function makeResult(overrides: Partial<AskResult> = {}): AskResult {
  return {
    question: "Does sleep help memory?",
    answer: "Sleep improves memory [1].",
    citations: [SOURCE],
    sources: [SOURCE],
    academic_count: 1,
    web_count: 0,
    suggested_questions: ["What about naps?"],
    grounded: true,
    ...overrides,
  };
}

/** Returns a controllable pollAskJob double for the NEXT call to pollAskJob(). */
function controllablePoll() {
  let onUpdateRef: ((s: AskJobStatus) => void) | undefined;
  let resolveRef: ((s: AskJobStatus) => void) | undefined;
  let rejectRef: ((e: unknown) => void) | undefined;
  pollAskJobMock.mockImplementationOnce((_jobId: string, onUpdate: (s: AskJobStatus) => void) => {
    onUpdateRef = onUpdate;
    return new Promise<AskJobStatus>((resolve, reject) => {
      resolveRef = resolve;
      rejectRef = reject;
    });
  });
  return {
    update: async (s: AskJobStatus) => {
      await act(async () => {
        onUpdateRef?.(s);
      });
    },
    settle: async (s: AskJobStatus) => {
      await act(async () => {
        resolveRef?.(s);
      });
    },
    fail: async (e: unknown) => {
      await act(async () => {
        rejectRef?.(e);
      });
    },
  };
}

describe("AskPage", () => {
  beforeEach(() => {
    askResearchAssistantMock.mockReset();
    pollAskJobMock.mockReset();
  });

  it("renders the header, intro, and default controls", () => {
    render(<AskPage />);
    expect(
      screen.getByRole("heading", { name: "Mode 3 — AI Research Assistant" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Research question")).toHaveValue("");
    expect(
      screen.getByRole("checkbox", { name: "Also search the web (DuckDuckGo)" }),
    ).toBeChecked();
    expect(screen.getByRole("button", { name: "Ask" })).toBeEnabled();
  });

  it("shows a validation warning for a blank question and does not call the API", async () => {
    const user = userEvent.setup();
    render(<AskPage />);

    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(screen.getByText("Please enter a research question.")).toBeInTheDocument();
    expect(askResearchAssistantMock).not.toHaveBeenCalled();
  });

  it("clears a stale validation warning as soon as the user types", async () => {
    const user = userEvent.setup();
    render(<AskPage />);

    await user.click(screen.getByRole("button", { name: "Ask" }));
    expect(screen.getByText("Please enter a research question.")).toBeInTheDocument();

    await user.type(screen.getByLabelText("Research question"), "a");
    expect(screen.queryByText("Please enter a research question.")).not.toBeInTheDocument();
  });

  it("runs the happy path: stage labels update live, then renders the grounded result", async () => {
    const user = userEvent.setup();
    askResearchAssistantMock.mockResolvedValue({ job_id: "job-1" });
    const poll = controllablePoll();

    render(<AskPage />);
    await user.type(screen.getByLabelText("Research question"), "Does sleep help memory?");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("Searching published literature…")).toBeInTheDocument();
    expect(askResearchAssistantMock).toHaveBeenCalledWith({
      question: "Does sleep help memory?",
      include_web: true,
      include_crossref: true,
      model: null,
      num_ctx: 8192,
      temperature_level: "focused",
    });

    await poll.update({
      id: "job-1",
      status: "running",
      stage: "searching",
      stage_info: {},
      error: null,
      result: null,
    });
    expect(
      await screen.findByText("Searching Google Scholar · arXiv · Semantic Scholar · web…"),
    ).toBeInTheDocument();

    await poll.update({
      id: "job-1",
      status: "running",
      stage: "reading",
      stage_info: { academic_count: 1, web_count: 0 },
      error: null,
      result: null,
    });
    expect(await screen.findByText("Reading 1 paper(s) and 0 web result(s)…")).toBeInTheDocument();

    await poll.update({
      id: "job-1",
      status: "running",
      stage: "answering",
      stage_info: { grounded: true },
      error: null,
      result: null,
    });
    expect(await screen.findByText("Composing a grounded answer…")).toBeInTheDocument();

    const result = makeResult();
    await poll.settle({
      id: "job-1",
      status: "done",
      stage: "done",
      stage_info: { citations: 1 },
      error: null,
      result,
    });

    expect(await screen.findByText("Sleep improves memory [1].")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Citations (1)" })).toBeInTheDocument();
    expect(screen.getByText("[1] 📄 paper — Sleep & Memory (2020)")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Searched 1 paper(s) and 0 web result(s); 1 used as context, 1 cited in the answer.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "What about naps?" })).toBeInTheDocument();
    expect(screen.queryByText(/No published sources could be retrieved/)).not.toBeInTheDocument();
  });

  it("omits the citations section and shows the general-knowledge warning when ungrounded", async () => {
    const user = userEvent.setup();
    askResearchAssistantMock.mockResolvedValue({ job_id: "job-2" });
    const poll = controllablePoll();
    render(<AskPage />);
    await user.type(screen.getByLabelText("Research question"), "Obscure question?");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    const result = makeResult({
      citations: [],
      sources: [],
      academic_count: 0,
      web_count: 0,
      suggested_questions: [],
      grounded: false,
      answer: "Generally, X.",
    });
    await poll.settle({
      id: "job-2",
      status: "done",
      stage: "done",
      stage_info: {},
      error: null,
      result,
    });

    expect(
      await screen.findByText(/No published sources could be retrieved for this question/),
    ).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /Citations/ })).not.toBeInTheDocument();
  });

  it("forwards include_web=false when the checkbox is unchecked", async () => {
    const user = userEvent.setup();
    askResearchAssistantMock.mockResolvedValue({ job_id: "job-3" });
    controllablePoll();
    render(<AskPage />);

    await user.click(screen.getByRole("checkbox", { name: "Also search the web (DuckDuckGo)" }));
    await user.type(screen.getByLabelText("Research question"), "q");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(askResearchAssistantMock).toHaveBeenCalledWith({
      question: "q",
      include_web: false,
      include_crossref: true,
      model: null,
      num_ctx: 8192,
      temperature_level: "focused",
    });
  });

  it("shows a job-error result as a Failed status line", async () => {
    const user = userEvent.setup();
    askResearchAssistantMock.mockResolvedValue({ job_id: "job-4" });
    const poll = controllablePoll();
    render(<AskPage />);
    await user.type(screen.getByLabelText("Research question"), "q");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    await poll.settle({
      id: "job-4",
      status: "error",
      stage: null,
      stage_info: {},
      error: "search backend exploded",
      result: null,
    });

    expect(await screen.findByText("Failed: search backend exploded")).toBeInTheDocument();
  });

  it("shows a Failed status line when askResearchAssistant itself rejects", async () => {
    const user = userEvent.setup();
    askResearchAssistantMock.mockRejectedValue(new Error("network down"));
    render(<AskPage />);
    await user.type(screen.getByLabelText("Research question"), "q");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("Failed: network down")).toBeInTheDocument();
  });

  it("disables the Ask button while a request is in flight", async () => {
    const user = userEvent.setup();
    askResearchAssistantMock.mockResolvedValue({ job_id: "job-5" });
    controllablePoll();
    render(<AskPage />);
    await user.type(screen.getByLabelText("Research question"), "q");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByRole("button", { name: "Ask" })).toBeDisabled();
  });

  it("clicking a follow-up question fills the box and immediately re-asks", async () => {
    const user = userEvent.setup();
    askResearchAssistantMock
      .mockResolvedValueOnce({ job_id: "job-6" })
      .mockResolvedValueOnce({ job_id: "job-7" });
    const poll = controllablePoll();
    render(<AskPage />);
    await user.type(screen.getByLabelText("Research question"), "Does sleep help memory?");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    const result = makeResult();
    await poll.settle({
      id: "job-6",
      status: "done",
      stage: "done",
      stage_info: {},
      error: null,
      result,
    });
    const followupButton = await screen.findByRole("button", { name: "What about naps?" });

    const poll2 = controllablePoll();
    await user.click(followupButton);

    expect(screen.getByLabelText("Research question")).toHaveValue("What about naps?");
    expect(askResearchAssistantMock).toHaveBeenLastCalledWith({
      question: "What about naps?",
      include_web: true,
      include_crossref: true,
      model: null,
      num_ctx: 8192,
      temperature_level: "focused",
    });

    const result2 = makeResult({
      question: "What about naps?",
      answer: "Napping helps too [1].",
      suggested_questions: [],
    });
    await poll2.settle({
      id: "job-7",
      status: "done",
      stage: "done",
      stage_info: {},
      error: null,
      result: result2,
    });
    expect(await screen.findByText("Napping helps too [1].")).toBeInTheDocument();
  });
});
