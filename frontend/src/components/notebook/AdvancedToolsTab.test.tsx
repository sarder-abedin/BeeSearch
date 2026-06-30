import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../api/client";
import type { AdvancedJobStatus, AdvancedResult } from "../../api/notebookAdvancedTypes";
import type { SourceMeta } from "../../api/notebookTypes";
import AdvancedToolsTab from "./AdvancedToolsTab";

const runCrossDocumentSummaryMock = vi.fn();
const runFaqMock = vi.fn();
const runLiteratureReviewMock = vi.fn();
const runMindmapMock = vi.fn();
const runAudioSummaryMock = vi.fn();
const runCompareSourcesMock = vi.fn();
const runKnowledgeGraphMock = vi.fn();
const runCitationTimelineMock = vi.fn();
const runStudyComparisonMock = vi.fn();
const pollAdvancedJobMock = vi.fn();
const exportTextMock = vi.fn();
const exportDocumentMock = vi.fn();
const exportDotMock = vi.fn();
const exportAudioWavMock = vi.fn();

vi.mock("../../api/notebookAdvanced", () => ({
  runCrossDocumentSummary: (...args: unknown[]) => runCrossDocumentSummaryMock(...args),
  runFaq: (...args: unknown[]) => runFaqMock(...args),
  runLiteratureReview: (...args: unknown[]) => runLiteratureReviewMock(...args),
  runMindmap: (...args: unknown[]) => runMindmapMock(...args),
  runAudioSummary: (...args: unknown[]) => runAudioSummaryMock(...args),
  runCompareSources: (...args: unknown[]) => runCompareSourcesMock(...args),
  runKnowledgeGraph: (...args: unknown[]) => runKnowledgeGraphMock(...args),
  runCitationTimeline: (...args: unknown[]) => runCitationTimelineMock(...args),
  runStudyComparison: (...args: unknown[]) => runStudyComparisonMock(...args),
  pollAdvancedJob: (...args: unknown[]) => pollAdvancedJobMock(...args),
  exportText: (...args: unknown[]) => exportTextMock(...args),
  exportDocument: (...args: unknown[]) => exportDocumentMock(...args),
  exportDot: (...args: unknown[]) => exportDotMock(...args),
  exportAudioWav: (...args: unknown[]) => exportAudioWavMock(...args),
}));

const downloadBlobMock = vi.fn();
const downloadTextMock = vi.fn();

vi.mock("../../utils/download", () => ({
  downloadBlob: (...args: unknown[]) => downloadBlobMock(...args),
  downloadText: (...args: unknown[]) => downloadTextMock(...args),
}));

function makeSource(overrides: Partial<SourceMeta> = {}): SourceMeta {
  return {
    doc_id: "doc-1",
    filename: "paper1.pdf",
    file_type: "pdf",
    source_type: "upload",
    url: "",
    total_pages: 10,
    total_chunks: 20,
    content_md5: "abc123",
    added_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

const SOURCES: SourceMeta[] = [
  makeSource({ doc_id: "doc-1", filename: "paper1.pdf" }),
  makeSource({ doc_id: "doc-2", filename: "paper2.pdf" }),
];

function makeAdvancedResult(overrides: Partial<AdvancedResult> = {}): AdvancedResult {
  return {
    notebook_id: "nb-1",
    summary: "",
    faqs: [],
    review: "",
    references: [],
    mindmap_dot: "",
    audio_script: "",
    comparison: "",
    knowledge_graph_dot: "",
    timeline: [],
    study_comparison: "",
    ...overrides,
  };
}

/** Returns a controllable pollAdvancedJob double for the NEXT call. The hook's
 * onUpdate passed to pollAdvancedJob is a hardcoded no-op, so unlike
 * PipelineTab's controllablePoll there is no `update()` step -- only settle/fail. */
function controllablePoll() {
  let resolveRef: ((s: AdvancedJobStatus) => void) | undefined;
  let rejectRef: ((e: unknown) => void) | undefined;
  pollAdvancedJobMock.mockImplementationOnce(() => {
    return new Promise<AdvancedJobStatus>((resolve, reject) => {
      resolveRef = resolve;
      rejectRef = reject;
    });
  });
  return {
    settle: async (s: AdvancedJobStatus) => {
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

function resetAllMocks() {
  runCrossDocumentSummaryMock.mockReset();
  runFaqMock.mockReset();
  runLiteratureReviewMock.mockReset();
  runMindmapMock.mockReset();
  runAudioSummaryMock.mockReset();
  runCompareSourcesMock.mockReset();
  runKnowledgeGraphMock.mockReset();
  runCitationTimelineMock.mockReset();
  runStudyComparisonMock.mockReset();
  pollAdvancedJobMock.mockReset();
  exportTextMock.mockReset();
  exportDocumentMock.mockReset();
  exportDotMock.mockReset();
  exportAudioWavMock.mockReset();
  downloadBlobMock.mockReset();
  downloadTextMock.mockReset();
}

/** Selects a tool via its radio button (disambiguated from same-text headings by role). */
async function selectTool(user: UserEvent, label: string) {
  await user.click(screen.getByRole("radio", { name: label }));
}

/** Runs the currently-selected tool to a "done" state with the given result. */
async function runToCompletion(
  triggerMock: ReturnType<typeof vi.fn>,
  runLabel: string,
  result: AdvancedResult,
  jobId = "job-1",
): Promise<UserEvent> {
  const user = userEvent.setup();
  triggerMock.mockResolvedValue({ job_id: jobId });
  const poll = controllablePoll();

  await user.click(screen.getByRole("button", { name: runLabel }));
  await poll.settle({ id: jobId, status: "done", stage: "done", stage_info: {}, error: null, result });
  return user;
}

describe("AdvancedToolsTab", () => {
  beforeEach(() => {
    resetAllMocks();
    URL.createObjectURL = vi.fn(() => "blob:mock-url");
    URL.revokeObjectURL = vi.fn();
  });

  it("renders all 9 tools in the radiogroup with the Cross-Document Summary panel selected by default", () => {
    render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);

    const group = screen.getByRole("radiogroup", { name: "Advanced tool" });
    expect(group).toBeInTheDocument();
    for (const label of [
      "Summary",
      "FAQ",
      "Lit Review",
      "Mind Map",
      "Audio",
      "Compare",
      "Graph",
      "Citation Timeline",
      "Study Table",
    ]) {
      expect(screen.getByRole("radio", { name: label })).toBeInTheDocument();
    }
    expect(screen.getByRole("radio", { name: "Summary" })).toBeChecked();
    expect(screen.getByRole("heading", { name: "Cross-Document Summary" })).toBeInTheDocument();
  });

  it("switches the visible panel when a different tool radio is selected", async () => {
    const user = userEvent.setup();
    render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);

    await selectTool(user, "FAQ");
    expect(screen.getByRole("heading", { name: "FAQ" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Cross-Document Summary" })).not.toBeInTheDocument();

    await selectTool(user, "Mind Map");
    expect(screen.getByRole("heading", { name: "Mind Map" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "FAQ" })).not.toBeInTheDocument();
  });

  it("resets all per-tool state when notebookId changes", async () => {
    const user = userEvent.setup();
    const { rerender } = render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);

    await selectTool(user, "FAQ");
    expect(screen.getByRole("heading", { name: "FAQ" })).toBeInTheDocument();

    rerender(<AdvancedToolsTab notebookId="nb-2" sources={SOURCES} />);

    expect(screen.getByRole("radio", { name: "Summary" })).toBeChecked();
    expect(screen.getByRole("heading", { name: "Cross-Document Summary" })).toBeInTheDocument();
  });

  describe("Cross-Document Summary", () => {
    it("renders the generated summary with Markdown export", async () => {
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      const user = await runToCompletion(
        runCrossDocumentSummaryMock,
        "Generate Summary",
        makeAdvancedResult({ summary: "## Themes\n\nShared finding across papers." }),
      );

      expect(runCrossDocumentSummaryMock).toHaveBeenCalledWith({ notebook_id: "nb-1" });
      expect(await screen.findByText("Themes")).toBeInTheDocument();
      expect(screen.getByText("Shared finding across papers.")).toBeInTheDocument();

      exportTextMock.mockResolvedValue("## Themes\n\nShared finding across papers.");
      await user.click(screen.getByRole("button", { name: "Download .md" }));
      expect(exportTextMock).toHaveBeenCalledWith("job-1", "summary");
      expect(downloadTextMock).toHaveBeenCalledWith(
        "## Themes\n\nShared finding across papers.",
        "summary.md",
        "text/markdown",
      );

      exportDocumentMock.mockResolvedValue(new Blob(["fake-docx"]));
      await user.click(screen.getByRole("button", { name: "Download .docx" }));
      expect(exportDocumentMock).toHaveBeenCalledWith("job-1", "summary", "docx");
      expect(downloadBlobMock).toHaveBeenCalledWith(expect.any(Blob), "summary.docx");
    });

    it("falls back to an info message when no summary was generated", async () => {
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await runToCompletion(runCrossDocumentSummaryMock, "Generate Summary", makeAdvancedResult({ summary: "" }));

      expect(await screen.findByText("No summary was generated for this run.")).toBeInTheDocument();
    });

    it("shows an error message when the job ends in error", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      runCrossDocumentSummaryMock.mockResolvedValue({ job_id: "job-err" });
      const poll = controllablePoll();

      await user.click(screen.getByRole("button", { name: "Generate Summary" }));
      await poll.settle({ id: "job-err", status: "error", stage: null, stage_info: {}, error: "Ollama unreachable", result: null });

      expect(await screen.findByText("Summary generation failed: Ollama unreachable")).toBeInTheDocument();
    });

    it("shows an error message when the trigger request itself rejects", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      runCrossDocumentSummaryMock.mockRejectedValue(new ApiError(500, "Backend unavailable"));

      await user.click(screen.getByRole("button", { name: "Generate Summary" }));

      expect(await screen.findByText("Summary generation failed: Backend unavailable")).toBeInTheDocument();
    });

    it("Clear resets the run back to idle", async () => {
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      const user = await runToCompletion(
        runCrossDocumentSummaryMock,
        "Generate Summary",
        makeAdvancedResult({ summary: "Body." }),
      );
      await screen.findByText("Body.");

      await user.click(screen.getByRole("button", { name: "Clear" }));

      expect(screen.getByRole("button", { name: "Generate Summary" })).toBeInTheDocument();
      expect(screen.queryByText("Body.")).not.toBeInTheDocument();
    });
  });

  describe("FAQ", () => {
    async function openFaq(user: UserEvent) {
      await selectTool(user, "FAQ");
    }

    it("clamps the number-of-questions input between 4 and 16", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await openFaq(user);

      const input = screen.getByLabelText("Number of questions");
      expect(input).toHaveValue(8);

      fireEvent.change(input, { target: { value: "2" } });
      expect(input).toHaveValue(4);

      fireEvent.change(input, { target: { value: "99" } });
      expect(input).toHaveValue(16);
    });

    it("renders generated FAQ items as collapsible cards with resolved source labels, and exports Markdown", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await openFaq(user);

      runFaqMock.mockResolvedValue({ job_id: "job-1" });
      const poll = controllablePoll();
      await user.click(screen.getByRole("button", { name: "Generate FAQ" }));
      expect(runFaqMock).toHaveBeenCalledWith({ notebook_id: "nb-1", n_questions: 8 });

      await poll.settle({
        id: "job-1",
        status: "done",
        stage: "done",
        stage_info: {},
        error: null,
        result: makeAdvancedResult({
          faqs: [{ question: "What is X?", answer: "X is a thing.", sources: [1, 99] }],
        }),
      });

      const toggle = await screen.findByRole("button", { name: "What is X?" });
      expect(screen.queryByText("X is a thing.")).not.toBeInTheDocument();
      await user.click(toggle);
      expect(await screen.findByText("X is a thing.")).toBeInTheDocument();
      expect(screen.getByText("Sources: paper1.pdf")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Download FAQ (.md)" }));
      expect(downloadTextMock).toHaveBeenCalledWith("### What is X?\nX is a thing.", "faq.md", "text/markdown");
    });

    it("falls back to an info message when no FAQ items were generated", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await openFaq(user);
      await runToCompletion(runFaqMock, "Generate FAQ", makeAdvancedResult({ faqs: [] }));

      expect(await screen.findByText("No FAQ items were generated for this run.")).toBeInTheDocument();
    });
  });

  describe("Literature Review", () => {
    it("renders the review with a structured references list and supports .md/.docx/.pdf export", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Lit Review");

      await runToCompletion(
        runLiteratureReviewMock,
        "Generate Literature Review",
        makeAdvancedResult({
          review: "## Introduction\n\nBody text.",
          references: [{ n: 1, doc_name: "paper1.pdf", page: 2, snippet: "A snippet.", doc_id: "doc-1" }],
        }),
      );
      expect(runLiteratureReviewMock).toHaveBeenCalledWith({ notebook_id: "nb-1" });

      expect(await screen.findByText("Introduction")).toBeInTheDocument();
      expect(screen.getByText("References (1)")).toBeInTheDocument();
      expect(screen.getByText("[1] paper1.pdf")).toBeInTheDocument();
      expect(screen.getByText("p. 3", { exact: false })).toBeInTheDocument();
      expect(screen.getByText("A snippet.")).toBeInTheDocument();

      exportTextMock.mockResolvedValue("## Introduction\n\nBody text.");
      await user.click(screen.getByRole("button", { name: "Download .md" }));
      expect(exportTextMock).toHaveBeenCalledWith("job-1", "review");

      exportDocumentMock.mockResolvedValue(new Blob(["fake-pdf"]));
      await user.click(screen.getByRole("button", { name: "Download .pdf" }));
      expect(exportDocumentMock).toHaveBeenCalledWith("job-1", "review", "pdf");
    });

    it("falls back to an info message when no review was generated", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Lit Review");
      await runToCompletion(runLiteratureReviewMock, "Generate Literature Review", makeAdvancedResult({ review: "" }));

      expect(await screen.findByText("No literature review was generated for this run.")).toBeInTheDocument();
    });
  });

  describe("Mind Map", () => {
    it("fetches a PNG preview and supports .dot/.png/.svg downloads", async () => {
      const user = userEvent.setup();
      exportDotMock.mockResolvedValue(new Blob(["fake-png"], { type: "image/png" }));
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Mind Map");

      await runToCompletion(runMindmapMock, "Generate Mind Map", makeAdvancedResult({ mindmap_dot: "digraph { a -> b; }" }));
      expect(runMindmapMock).toHaveBeenCalledWith({ notebook_id: "nb-1" });
      expect(exportDotMock).toHaveBeenCalledWith("job-1", "mindmap", "png");
      expect(await screen.findByAltText("Mind map")).toHaveAttribute("src", "blob:mock-url");

      await user.click(screen.getByRole("button", { name: "Download .dot" }));
      expect(downloadTextMock).toHaveBeenCalledWith("digraph { a -> b; }", "mindmap.dot", "text/vnd.graphviz");

      await user.click(screen.getByRole("button", { name: "Download .svg" }));
      expect(exportDotMock).toHaveBeenCalledWith("job-1", "mindmap", "svg");
    });

    it("falls back to an info message when no mind map was generated", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Mind Map");
      await runToCompletion(runMindmapMock, "Generate Mind Map", makeAdvancedResult({ mindmap_dot: "" }));

      expect(await screen.findByText("No mind map was generated for this run.")).toBeInTheDocument();
      expect(exportDotMock).not.toHaveBeenCalled();
    });
  });

  describe("Audio Summary", () => {
    it("renders the script with word count, supports script download, and synthesizes a downloadable .wav", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Audio");

      await runToCompletion(
        runAudioSummaryMock,
        "Generate Audio Script",
        makeAdvancedResult({ audio_script: "This is the spoken summary script." }),
      );
      expect(runAudioSummaryMock).toHaveBeenCalledWith({ notebook_id: "nb-1" });
      expect(await screen.findByText("This is the spoken summary script.")).toBeInTheDocument();
      expect(screen.getByText("Word count: 6")).toBeInTheDocument();

      exportTextMock.mockResolvedValue("This is the spoken summary script.");
      await user.click(screen.getByRole("button", { name: "Download script (.txt)" }));
      expect(exportTextMock).toHaveBeenCalledWith("job-1", "audio-script");
      expect(downloadTextMock).toHaveBeenCalledWith(
        "This is the spoken summary script.",
        "audio_summary_script.txt",
        "text/plain",
      );

      exportAudioWavMock.mockResolvedValue(new Blob(["fake-wav"], { type: "audio/wav" }));
      await user.click(screen.getByRole("button", { name: "Synthesize .wav" }));
      expect(exportAudioWavMock).toHaveBeenCalledWith("job-1");
      const audioEl = await screen.findByRole("button", { name: "Download .wav" });
      expect(audioEl).toBeInTheDocument();

      await user.click(audioEl);
      expect(downloadBlobMock).toHaveBeenCalledWith(expect.any(Blob), "audio_summary.wav");
    });

    it("supports the Play in browser / Stop reading toggle via the Web Speech API", async () => {
      const speak = vi.fn();
      const cancel = vi.fn();
      vi.stubGlobal("speechSynthesis", { speak, cancel });
      vi.stubGlobal(
        "SpeechSynthesisUtterance",
        class {
          text: string;
          rate = 1;
          onend: (() => void) | null = null;
          onerror: (() => void) | null = null;
          constructor(text: string) {
            this.text = text;
          }
        },
      );

      const user = userEvent.setup();
      const { unmount } = render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      try {
        await selectTool(user, "Audio");
        await runToCompletion(runAudioSummaryMock, "Generate Audio Script", makeAdvancedResult({ audio_script: "Script body." }));
        await screen.findByText("Script body.");

        await user.click(screen.getByRole("button", { name: "Play in browser" }));
        expect(speak).toHaveBeenCalledTimes(1);
        expect(speak.mock.calls[0][0]).toMatchObject({ text: "Script body." });
        expect(await screen.findByRole("button", { name: "Stop reading" })).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "Stop reading" }));
        expect(cancel).toHaveBeenCalledTimes(1);
        expect(await screen.findByRole("button", { name: "Play in browser" })).toBeInTheDocument();
      } finally {
        unmount();
        vi.unstubAllGlobals();
      }
    });

    it("falls back to an info message when no audio script was generated", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Audio");
      await runToCompletion(runAudioSummaryMock, "Generate Audio Script", makeAdvancedResult({ audio_script: "" }));

      expect(await screen.findByText("No audio script was generated for this run.")).toBeInTheDocument();
    });
  });

  describe("Compare Sources", () => {
    it("shows a guard message instead of run controls when fewer than two sources exist", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={[makeSource()]} />);
      await selectTool(user, "Compare");

      expect(screen.getByText("Add at least two sources to use source comparison.")).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Compare Sources" })).not.toBeInTheDocument();
    });

    it("defaults to comparing the first two sources and renders the comparison with export", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Compare");

      expect(screen.getByLabelText("Source A")).toHaveValue("doc-1");
      expect(screen.getByLabelText("Source B")).toHaveValue("doc-2");

      await runToCompletion(
        runCompareSourcesMock,
        "Compare Sources",
        makeAdvancedResult({ comparison: "## Comparison\n\nBoth papers agree on X." }),
      );
      expect(runCompareSourcesMock).toHaveBeenCalledWith({ notebook_id: "nb-1", doc_id_a: "doc-1", doc_id_b: "doc-2" });
      expect(await screen.findByText("Comparison")).toBeInTheDocument();

      exportTextMock.mockResolvedValue("## Comparison\n\nBoth papers agree on X.");
      await user.click(screen.getByRole("button", { name: "Download .md" }));
      expect(exportTextMock).toHaveBeenCalledWith("job-1", "comparison");
    });

    it("sends the selected source pair when changed from the defaults", async () => {
      const user = userEvent.setup();
      const threeSources = [...SOURCES, makeSource({ doc_id: "doc-3", filename: "paper3.pdf" })];
      render(<AdvancedToolsTab notebookId="nb-1" sources={threeSources} />);
      await selectTool(user, "Compare");

      await user.selectOptions(screen.getByLabelText("Source B"), "doc-3");
      runCompareSourcesMock.mockResolvedValue({ job_id: "job-1" });
      const poll = controllablePoll();
      await user.click(screen.getByRole("button", { name: "Compare Sources" }));

      expect(runCompareSourcesMock).toHaveBeenCalledWith({ notebook_id: "nb-1", doc_id_a: "doc-1", doc_id_b: "doc-3" });
      await poll.settle({ id: "job-1", status: "done", stage: "done", stage_info: {}, error: null, result: makeAdvancedResult() });
    });

    it("falls back to an info message when no comparison was generated", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Compare");
      await runToCompletion(runCompareSourcesMock, "Compare Sources", makeAdvancedResult({ comparison: "" }));

      expect(await screen.findByText("No comparison was generated for this run.")).toBeInTheDocument();
    });
  });

  describe("Knowledge Graph", () => {
    it("fetches a PNG preview and supports .dot/.png/.svg downloads", async () => {
      const user = userEvent.setup();
      exportDotMock.mockResolvedValue(new Blob(["fake-png"], { type: "image/png" }));
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Graph");

      await runToCompletion(
        runKnowledgeGraphMock,
        "Extract Knowledge Graph",
        makeAdvancedResult({ knowledge_graph_dot: "digraph { a -> b; }" }),
      );
      expect(runKnowledgeGraphMock).toHaveBeenCalledWith({ notebook_id: "nb-1" });
      expect(exportDotMock).toHaveBeenCalledWith("job-1", "knowledge-graph", "png");
      expect(await screen.findByAltText("Knowledge graph")).toHaveAttribute("src", "blob:mock-url");

      await user.click(screen.getByRole("button", { name: "Download .png" }));
      expect(exportDotMock).toHaveBeenCalledWith("job-1", "knowledge-graph", "png");
    });

    it("shows an error caption when the PNG preview fetch fails", async () => {
      const user = userEvent.setup();
      exportDotMock.mockRejectedValue(new Error("render service down"));
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Graph");
      await runToCompletion(
        runKnowledgeGraphMock,
        "Extract Knowledge Graph",
        makeAdvancedResult({ knowledge_graph_dot: "digraph { a -> b; }" }),
      );

      expect(await screen.findByText("Preview unavailable: render service down")).toBeInTheDocument();
    });

    it("falls back to an info message when no graph was generated", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Graph");
      await runToCompletion(runKnowledgeGraphMock, "Extract Knowledge Graph", makeAdvancedResult({ knowledge_graph_dot: "" }));

      expect(await screen.findByText("No knowledge graph was generated for this run.")).toBeInTheDocument();
    });
  });

  describe("Citation Timeline", () => {
    it("toggles the enrich-with-abstracts checkbox into the trigger request", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Citation Timeline");

      const checkbox = screen.getByRole("checkbox", { name: "Enrich with abstracts (Semantic Scholar)" });
      expect(checkbox).not.toBeChecked();
      await user.click(checkbox);
      expect(checkbox).toBeChecked();

      runCitationTimelineMock.mockResolvedValue({ job_id: "job-1" });
      const poll = controllablePoll();
      await user.click(screen.getByRole("button", { name: "Extract Citation Timeline" }));
      expect(runCitationTimelineMock).toHaveBeenCalledWith({ notebook_id: "nb-1", enrich_with_abstracts: true });
      await poll.settle({ id: "job-1", status: "done", stage: "done", stage_info: {}, error: null, result: makeAdvancedResult() });
    });

    it("renders the composed markdown table and downloads the exact same string", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Citation Timeline");

      await runToCompletion(
        runCitationTimelineMock,
        "Extract Citation Timeline",
        makeAdvancedResult({
          timeline: [
            {
              year: "2020",
              title: "Attention | Is All You Need",
              authors: "Vaswani et al.",
              gist: "Introduces\nthe transformer.",
              source: 1,
              url: "https://example.com/paper",
            },
            {
              year: "n.d.",
              title: "Older Work",
              authors: "Smith",
              gist: "A foundational idea.",
              source: 99,
              url: "",
            },
          ],
        }),
      );

      expect(await screen.findByRole("link", { name: "Attention | Is All You Need" })).toHaveAttribute(
        "href",
        "https://example.com/paper",
      );
      expect(screen.getByText("Older Work")).toBeInTheDocument();

      const expectedMd = [
        "| Year | Title | Authors | Key Idea | Source |",
        "|------|-------|---------|----------|--------|",
        "| 2020 | [Attention \\| Is All You Need](https://example.com/paper) | Vaswani et al. | Introduces the transformer. | paper1.pdf |",
        "| n.d. | Older Work | Smith | A foundational idea. | — |",
      ].join("\n");

      await user.click(screen.getByRole("button", { name: "Download (.md)" }));
      expect(downloadTextMock).toHaveBeenCalledWith(expectedMd, "citation_timeline.md", "text/markdown");
    });

    it("falls back to an info message when no timeline was generated", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Citation Timeline");
      await runToCompletion(runCitationTimelineMock, "Extract Citation Timeline", makeAdvancedResult({ timeline: [] }));

      expect(await screen.findByText("No citation timeline was generated for this run.")).toBeInTheDocument();
    });
  });

  describe("Study Comparison", () => {
    it("renders the study comparison table with .md/.docx/.pdf export", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Study Table");

      await runToCompletion(
        runStudyComparisonMock,
        "Generate Study Comparison",
        makeAdvancedResult({ study_comparison: "| Study | N |\n| --- | --- |\n| A | 100 |" }),
      );
      expect(runStudyComparisonMock).toHaveBeenCalledWith({ notebook_id: "nb-1" });
      expect(await screen.findByText("Study")).toBeInTheDocument();

      exportDocumentMock.mockResolvedValue(new Blob(["fake-docx"]));
      await user.click(screen.getByRole("button", { name: "Download .docx" }));
      expect(exportDocumentMock).toHaveBeenCalledWith("job-1", "study-comparison", "docx");
      expect(downloadBlobMock).toHaveBeenCalledWith(expect.any(Blob), "study_comparison.docx");
    });

    it("falls back to an info message when no study comparison was generated", async () => {
      const user = userEvent.setup();
      render(<AdvancedToolsTab notebookId="nb-1" sources={SOURCES} />);
      await selectTool(user, "Study Table");
      await runToCompletion(runStudyComparisonMock, "Generate Study Comparison", makeAdvancedResult({ study_comparison: "" }));

      expect(await screen.findByText("No study comparison was generated for this run.")).toBeInTheDocument();
    });
  });
});
