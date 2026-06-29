import { act, render, screen } from "@testing-library/react";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../api/client";
import type { ReportJobStatus, ReportReference, ReportResult } from "../../api/notebookReportTypes";
import ResearchReportTab from "./ResearchReportTab";

const runReportMock = vi.fn();
const pollReportJobMock = vi.fn();
const exportReportCitationsMock = vi.fn();

vi.mock("../../api/notebookReport", () => ({
  runReport: (...args: unknown[]) => runReportMock(...args),
  pollReportJob: (...args: unknown[]) => pollReportJobMock(...args),
  exportReportCitations: (...args: unknown[]) => exportReportCitationsMock(...args),
}));

const downloadTextMock = vi.fn();

vi.mock("../../utils/download", () => ({
  downloadText: (...args: unknown[]) => downloadTextMock(...args),
}));

function makeReference(overrides: Partial<ReportReference> = {}): ReportReference {
  return {
    ref_num: 1,
    title: "Attention Is All You Need",
    authors: ["Vaswani A", "Shazeer N"],
    journal: "NeurIPS",
    year: "2017",
    doi: "10.0000/abc",
    url: "https://arxiv.org/abs/1706.03762",
    abstract_snippet: "We propose a new architecture, the Transformer.",
    source: "arxiv",
    citation_count: 12,
    apa: "Vaswani A; Shazeer N (2017). Attention Is All You Need. NeurIPS.",
    ...overrides,
  };
}

function makeReportResult(overrides: Partial<ReportResult> = {}): ReportResult {
  return {
    notebook_id: "nb-1",
    goal: "What is self-attention?",
    mode: "hybrid",
    report: "## Executive Summary\n\nSelf-attention lets a model weigh every token [Paper 1].",
    key_findings: ["Self-attention scales quadratically with sequence length."],
    references: [makeReference()],
    web_search_status: "disabled",
    eval_result: {},
    errors: [],
    progress_pct: 100,
    ...overrides,
  };
}

/** Returns a controllable pollReportJob double for the NEXT call to pollReportJob(). */
function controllablePoll() {
  let onUpdateRef: ((s: ReportJobStatus) => void) | undefined;
  let resolveRef: ((s: ReportJobStatus) => void) | undefined;
  let rejectRef: ((e: unknown) => void) | undefined;
  pollReportJobMock.mockImplementationOnce((_jobId: string, onUpdate: (s: ReportJobStatus) => void) => {
    onUpdateRef = onUpdate;
    return new Promise<ReportJobStatus>((resolve, reject) => {
      resolveRef = resolve;
      rejectRef = reject;
    });
  });
  return {
    update: async (s: ReportJobStatus) => {
      await act(async () => {
        onUpdateRef?.(s);
      });
    },
    settle: async (s: ReportJobStatus) => {
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

/** Runs the report to a successful "done" state and returns the user-event instance. */
async function runToCompletion(result: ReportResult, sourceCount = 1): Promise<UserEvent> {
  const user = userEvent.setup();
  runReportMock.mockResolvedValue({ job_id: "job-1" });
  const poll = controllablePoll();
  render(<ResearchReportTab notebookId="nb-1" sourceCount={sourceCount} />);

  await user.type(screen.getByLabelText("Research goal or question"), "What is self-attention?");
  await user.click(screen.getByRole("button", { name: "Generate Research Report" }));
  await poll.settle({
    id: "job-1",
    status: "done",
    stage: "research_eval",
    stage_info: {},
    error: null,
    result,
  });
  await screen.findByText("Done.");
  return user;
}

describe("ResearchReportTab", () => {
  beforeEach(() => {
    runReportMock.mockReset();
    pollReportJobMock.mockReset();
    exportReportCitationsMock.mockReset();
    downloadTextMock.mockReset();
  });

  it("renders the goal input and defaults the academic toggle on, web toggle off", () => {
    render(<ResearchReportTab notebookId="nb-1" sourceCount={1} />);

    expect(screen.getByLabelText("Research goal or question")).toHaveValue("");
    expect(
      screen.getByRole("checkbox", { name: "Search academic sources (arXiv + Semantic Scholar)" }),
    ).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Include web search (DuckDuckGo)" })).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Generate Research Report" })).toBeInTheDocument();
  });

  it("shows the no-sources info banner only when the notebook has no sources", () => {
    const { rerender } = render(<ResearchReportTab notebookId="nb-1" sourceCount={0} />);
    expect(
      screen.getByText("No sources in this notebook — will search academic literature only."),
    ).toBeInTheDocument();

    rerender(<ResearchReportTab notebookId="nb-1" sourceCount={2} />);
    expect(
      screen.queryByText("No sources in this notebook — will search academic literature only."),
    ).not.toBeInTheDocument();
  });

  it("warns instead of calling the API when the goal is blank", async () => {
    const user = userEvent.setup();
    render(<ResearchReportTab notebookId="nb-1" sourceCount={1} />);

    await user.click(screen.getByRole("button", { name: "Generate Research Report" }));

    expect(await screen.findByText("Please enter a research goal.")).toBeInTheDocument();
    expect(runReportMock).not.toHaveBeenCalled();
  });

  it("sends the trimmed goal plus toggle values, shows live progress, then completes", async () => {
    const user = userEvent.setup();
    runReportMock.mockResolvedValue({ job_id: "job-1" });
    const poll = controllablePoll();
    render(<ResearchReportTab notebookId="nb-1" sourceCount={1} />);

    await user.type(screen.getByLabelText("Research goal or question"), "  What is self-attention?  ");
    await user.click(screen.getByRole("checkbox", { name: "Include web search (DuckDuckGo)" }));
    await user.click(screen.getByRole("button", { name: "Generate Research Report" }));

    expect(runReportMock).toHaveBeenCalledWith({
      notebook_id: "nb-1",
      goal: "What is self-attention?",
      include_academic: true,
      include_web: true,
    });
    expect(await screen.findByText("Starting…")).toBeInTheDocument();

    await poll.update({
      id: "job-1",
      status: "running",
      stage: "academic_search",
      stage_info: { label: "Searching arXiv + Semantic Scholar", progress_pct: 50 },
      error: null,
      result: null,
    });
    expect(await screen.findByText("Searching arXiv + Semantic Scholar")).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();

    await poll.settle({
      id: "job-1",
      status: "done",
      stage: "research_eval",
      stage_info: {},
      error: null,
      result: makeReportResult(),
    });

    expect(await screen.findByText("Done.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Regenerate Report" })).toBeInTheDocument();
  });

  it("shows an error message when the report job ends in error", async () => {
    const user = userEvent.setup();
    runReportMock.mockResolvedValue({ job_id: "job-err" });
    const poll = controllablePoll();
    render(<ResearchReportTab notebookId="nb-1" sourceCount={1} />);

    await user.type(screen.getByLabelText("Research goal or question"), "What is X?");
    await user.click(screen.getByRole("button", { name: "Generate Research Report" }));
    await poll.settle({
      id: "job-err",
      status: "error",
      stage: null,
      stage_info: {},
      error: "Ollama not reachable",
      result: null,
    });

    expect(await screen.findByText("Research workflow failed: Ollama not reachable")).toBeInTheDocument();
  });

  it("shows an error message when runReport itself rejects", async () => {
    const user = userEvent.setup();
    runReportMock.mockRejectedValue(new ApiError(404, "Notebook not found."));
    render(<ResearchReportTab notebookId="nb-1" sourceCount={1} />);

    await user.type(screen.getByLabelText("Research goal or question"), "What is X?");
    await user.click(screen.getByRole("button", { name: "Generate Research Report" }));

    expect(await screen.findByText("Research workflow failed: Notebook not found.")).toBeInTheDocument();
  });

  it("Clear resets the run state back to idle", async () => {
    const user = await runToCompletion(makeReportResult());

    expect(screen.getByRole("button", { name: "Regenerate Report" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear" }));

    expect(screen.getByRole("button", { name: "Generate Research Report" })).toBeInTheDocument();
    expect(screen.queryByRole("tablist", { name: "Research report results" })).not.toBeInTheDocument();
  });

  it.each([
    ["empty", "Web search was enabled but found no additional results — this report uses only academic/notebook sources."],
    ["error", "Web search was enabled but failed — this report uses only academic/notebook sources."],
  ])("shows a warning banner when web_search_status is %s", async (status, expectedText) => {
    await runToCompletion(makeReportResult({ web_search_status: status }));
    expect(await screen.findByText(expectedText)).toBeInTheDocument();
  });

  it("does not show a web-search warning banner when web search was disabled or succeeded", async () => {
    await runToCompletion(makeReportResult({ web_search_status: "ok" }));
    expect(screen.queryByText(/Web search was enabled/)).not.toBeInTheDocument();
  });

  it("renders Key Findings as a numbered list when present", async () => {
    await runToCompletion(
      makeReportResult({ key_findings: ["Finding one.", "Finding two."] }),
    );

    expect(screen.getByText("Key Findings")).toBeInTheDocument();
    expect(screen.getByText("Finding one.")).toBeInTheDocument();
    expect(screen.getByText("Finding two.")).toBeInTheDocument();
  });

  it("omits the Key Findings section when there are none", async () => {
    await runToCompletion(makeReportResult({ key_findings: [] }));
    expect(screen.queryByText("Key Findings")).not.toBeInTheDocument();
  });

  describe("Report sub-tab", () => {
    it("renders the report markdown by default and supports the Markdown download", async () => {
      const user = await runToCompletion(
        makeReportResult({ report: "## Executive Summary\n\nSelf-attention lets a model weigh tokens." }),
      );

      expect(screen.getByRole("tab", { name: "Report" })).toHaveAttribute("aria-selected", "true");
      expect(await screen.findByText("Executive Summary")).toBeInTheDocument();
      expect(screen.getByText("Self-attention lets a model weigh tokens.")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Download Report (Markdown)" }));
      expect(downloadTextMock).toHaveBeenCalledWith(
        "## Executive Summary\n\nSelf-attention lets a model weigh tokens.",
        "research_report.md",
        "text/markdown",
      );
    });
  });

  describe("References sub-tab", () => {
    it("lists references with citation export buttons and expandable detail cards", async () => {
      exportReportCitationsMock.mockResolvedValue("@article{vaswani2017,\n  title = {...}\n}\n");
      const user = await runToCompletion(
        makeReportResult({ references: [makeReference(), makeReference({ ref_num: 2, title: "Second Paper" })] }),
      );

      await user.click(screen.getByRole("tab", { name: "References" }));
      expect(await screen.findByText("References (2)")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Export BibTeX (.bib)" }));
      expect(exportReportCitationsMock).toHaveBeenCalledWith("job-1", "bibtex");
      expect(downloadTextMock).toHaveBeenCalledWith(
        "@article{vaswani2017,\n  title = {...}\n}\n",
        "references.bib",
        "text/plain",
      );

      exportReportCitationsMock.mockResolvedValue("TY  - JOUR\nER  - \n");
      await user.click(screen.getByRole("button", { name: "Export RIS (.ris)" }));
      expect(exportReportCitationsMock).toHaveBeenCalledWith("job-1", "ris");
      expect(downloadTextMock).toHaveBeenCalledWith("TY  - JOUR\nER  - \n", "references.ris", "text/plain");

      const card = screen.getByRole("button", { name: "[1] Attention Is All You Need" });
      await user.click(card);
      expect(await screen.findByText("Vaswani A; Shazeer N")).toBeInTheDocument();
      expect(screen.getByText("NeurIPS")).toBeInTheDocument();
      expect(screen.getByText("2017")).toBeInTheDocument();
      expect(screen.getByRole("link", { name: "10.0000/abc" })).toHaveAttribute(
        "href",
        "https://doi.org/10.0000/abc",
      );
      expect(screen.getByText("We propose a new architecture, the Transformer.")).toBeInTheDocument();
      expect(screen.getByText(/Source: arXiv preprint/)).toBeInTheDocument();
      expect(screen.getByText(/Citations: 12/)).toBeInTheDocument();
      expect(screen.getByText("Vaswani A; Shazeer N (2017). Attention Is All You Need. NeurIPS.")).toBeInTheDocument();
    });

    it("falls back to an info message when there are no references", async () => {
      await runToCompletion(makeReportResult({ references: [] }));

      await act(async () => {
        screen.getByRole("tab", { name: "References" }).click();
      });
      expect(await screen.findByText("No references found for this run.")).toBeInTheDocument();
    });

    it("omits optional fields and shows Unknown source when not provided", async () => {
      const user = await runToCompletion(
        makeReportResult({
          references: [
            makeReference({
              doi: "",
              url: "",
              abstract_snippet: "",
              source: "",
              citation_count: null,
              authors: [],
              journal: "",
              year: "",
            }),
          ],
        }),
      );

      await user.click(screen.getByRole("tab", { name: "References" }));
      await user.click(screen.getByRole("button", { name: "[1] Attention Is All You Need" }));

      expect(await screen.findAllByText("N/A", { exact: false })).toHaveLength(2);
      expect(screen.queryByText(/DOI:/)).not.toBeInTheDocument();
      expect(screen.queryByText(/URL:/)).not.toBeInTheDocument();
      expect(screen.queryByText(/Abstract:/)).not.toBeInTheDocument();
      expect(screen.getByText(/Source: Unknown/)).toBeInTheDocument();
      expect(screen.queryByText(/Citations:/)).not.toBeInTheDocument();
    });

    it("shows an error message when a citation export fails", async () => {
      exportReportCitationsMock.mockRejectedValue(new Error("export service down"));
      const user = await runToCompletion(makeReportResult());

      await user.click(screen.getByRole("tab", { name: "References" }));
      await user.click(screen.getByRole("button", { name: "Export BibTeX (.bib)" }));

      expect(await screen.findByText("BibTeX export failed: export service down")).toBeInTheDocument();
    });
  });

  it("resets all state when switching to a different notebook", async () => {
    await runToCompletion(makeReportResult());

    const { rerender } = render(<ResearchReportTab notebookId="nb-1" sourceCount={1} />);
    rerender(<ResearchReportTab notebookId="nb-2" sourceCount={1} />);

    expect(screen.getAllByRole("button", { name: "Generate Research Report" }).length).toBeGreaterThan(0);
  });
});
