import { act, render, screen } from "@testing-library/react";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../api/client";
import type { PipelineJobStatus, PipelineResult } from "../../api/notebookPipelineTypes";
import PipelineTab from "./PipelineTab";

const runPipelineMock = vi.fn();
const pollPipelineJobMock = vi.fn();
const exportStudyGuideMock = vi.fn();
const exportKnowledgeGraphMock = vi.fn();

vi.mock("../../api/notebookPipeline", () => ({
  runPipeline: (...args: unknown[]) => runPipelineMock(...args),
  pollPipelineJob: (...args: unknown[]) => pollPipelineJobMock(...args),
  exportStudyGuide: (...args: unknown[]) => exportStudyGuideMock(...args),
  exportKnowledgeGraph: (...args: unknown[]) => exportKnowledgeGraphMock(...args),
}));

const downloadBlobMock = vi.fn();
const downloadTextMock = vi.fn();

vi.mock("../../utils/download", () => ({
  downloadBlob: (...args: unknown[]) => downloadBlobMock(...args),
  downloadText: (...args: unknown[]) => downloadTextMock(...args),
}));

vi.mock("../../context/SettingsContext", () => ({
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

function makePipelineResult(overrides: Partial<PipelineResult> = {}): PipelineResult {
  return {
    notebook_id: "nb-1",
    doc_count: 1,
    ingestion_summary: "Loaded 1 source(s) with 5 chunk(s) from notebook 'My Notebook'. Sources: paper.pdf.",
    per_doc_summaries: {},
    cross_summary: "",
    retrieved_chunks: [],
    retrieval_mode: "hybrid",
    verified_citations: [],
    citation_report: "",
    knowledge_graph_dot: "",
    kg_data: {},
    study_guide: "",
    podcast_script: "",
    errors: [],
    completed_steps: [],
    eval_result: {},
    rag_reflection_info: {},
    progress_pct: 100,
    ...overrides,
  };
}

/** Returns a controllable pollPipelineJob double for the NEXT call to pollPipelineJob(). */
function controllablePoll() {
  let onUpdateRef: ((s: PipelineJobStatus) => void) | undefined;
  let resolveRef: ((s: PipelineJobStatus) => void) | undefined;
  let rejectRef: ((e: unknown) => void) | undefined;
  pollPipelineJobMock.mockImplementationOnce((_jobId: string, onUpdate: (s: PipelineJobStatus) => void) => {
    onUpdateRef = onUpdate;
    return new Promise<PipelineJobStatus>((resolve, reject) => {
      resolveRef = resolve;
      rejectRef = reject;
    });
  });
  return {
    update: async (s: PipelineJobStatus) => {
      await act(async () => {
        onUpdateRef?.(s);
      });
    },
    settle: async (s: PipelineJobStatus) => {
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

/** Runs the pipeline to a successful "done" state and returns the user-event instance. */
async function runToCompletion(result: PipelineResult): Promise<UserEvent> {
  const user = userEvent.setup();
  runPipelineMock.mockResolvedValue({ job_id: "job-1" });
  const poll = controllablePoll();
  render(<PipelineTab notebookId="nb-1" sourceCount={1} />);

  await user.click(screen.getByRole("button", { name: "Run Full Pipeline" }));
  await poll.settle({
    id: "job-1",
    status: "done",
    stage: "done",
    stage_info: {},
    error: null,
    result,
  });
  await screen.findByText("Done.");
  return user;
}

describe("PipelineTab", () => {
  beforeEach(() => {
    runPipelineMock.mockReset();
    pollPipelineJobMock.mockReset();
    exportStudyGuideMock.mockReset();
    exportKnowledgeGraphMock.mockReset();
    downloadBlobMock.mockReset();
    downloadTextMock.mockReset();
    exportKnowledgeGraphMock.mockResolvedValue(new Blob(["fake-png"], { type: "image/png" }));
    URL.createObjectURL = vi.fn(() => "blob:mock-url");
    URL.revokeObjectURL = vi.fn();
  });

  it("shows a guard message instead of the run controls when there are no sources", () => {
    render(<PipelineTab notebookId="nb-1" sourceCount={0} />);

    expect(
      screen.getByText("Add at least one source in the Sources panel before running the analysis pipeline."),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run Full Pipeline" })).not.toBeInTheDocument();
  });

  it("sends the trimmed focus query, shows live progress, then lands on the Ingestion sub-tab", async () => {
    const user = userEvent.setup();
    runPipelineMock.mockResolvedValue({ job_id: "job-1" });
    const poll = controllablePoll();
    render(<PipelineTab notebookId="nb-1" sourceCount={1} />);

    await user.type(screen.getByLabelText("Focus query (optional)"), "  attention mechanisms  ");
    await user.click(screen.getByRole("button", { name: "Run Full Pipeline" }));

    expect(runPipelineMock).toHaveBeenCalledWith({
      notebook_id: "nb-1",
      query: "attention mechanisms",
      model: null,
      num_ctx: 8192,
      embed_model: null,
      top_k: 8,
      temperature_level: "focused",
    });
    expect(await screen.findByText("Starting…")).toBeInTheDocument();

    await poll.update({
      id: "job-1",
      status: "running",
      stage: "summarize",
      stage_info: { label: "Summarizing documents…", progress_pct: 40 },
      error: null,
      result: null,
    });
    expect(await screen.findByText("Summarizing documents…")).toBeInTheDocument();
    expect(screen.getByText("40%")).toBeInTheDocument();

    await poll.settle({
      id: "job-1",
      status: "done",
      stage: "done",
      stage_info: {},
      error: null,
      result: makePipelineResult({ doc_count: 2, ingestion_summary: "Loaded 2 source(s)." }),
    });

    expect(await screen.findByText("Done.")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Ingestion" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Loaded 2 source(s).")).toBeInTheDocument();
    expect(screen.getByText("2 document(s) ingested.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Pipeline Again" })).toBeInTheDocument();
    expect(screen.queryByText(/warning\(s\)/)).not.toBeInTheDocument();
  });

  it("shows an error message when the pipeline job ends in error", async () => {
    const user = userEvent.setup();
    runPipelineMock.mockResolvedValue({ job_id: "job-err" });
    const poll = controllablePoll();
    render(<PipelineTab notebookId="nb-1" sourceCount={1} />);

    await user.click(screen.getByRole("button", { name: "Run Full Pipeline" }));
    await poll.settle({
      id: "job-err",
      status: "error",
      stage: null,
      stage_info: {},
      error: "Ollama not reachable",
      result: null,
    });

    expect(await screen.findByText("Pipeline failed: Ollama not reachable")).toBeInTheDocument();
  });

  it("shows an error message when runPipeline itself rejects", async () => {
    const user = userEvent.setup();
    runPipelineMock.mockRejectedValue(new ApiError(500, "Backend unavailable"));
    render(<PipelineTab notebookId="nb-1" sourceCount={1} />);

    await user.click(screen.getByRole("button", { name: "Run Full Pipeline" }));

    expect(await screen.findByText("Pipeline failed: Backend unavailable")).toBeInTheDocument();
  });

  it("Clear resets the run state back to idle", async () => {
    const user = await runToCompletion(makePipelineResult());

    expect(screen.getByRole("button", { name: "Run Pipeline Again" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear" }));

    expect(screen.getByRole("button", { name: "Run Full Pipeline" })).toBeInTheDocument();
    expect(screen.queryByRole("tablist", { name: "Pipeline results" })).not.toBeInTheDocument();
  });

  it("shows a warnings expander listing each error only when the run result has errors", async () => {
    await runToCompletion(
      makePipelineResult({ errors: ["Podcast generation failed", "Knowledge graph rendering skipped"] }),
    );

    expect(await screen.findByText("2 warning(s)")).toBeInTheDocument();
    expect(screen.getByText("Podcast generation failed")).toBeInTheDocument();
    expect(screen.getByText("Knowledge graph rendering skipped")).toBeInTheDocument();
  });

  describe("sub-tabs", () => {
    it("Summary: renders the cross-document summary with its download button, plus per-document summaries", async () => {
      const user = await runToCompletion(
        makePipelineResult({
          cross_summary: "## Key finding\n\nResult X was significant.",
          per_doc_summaries: { "paper.pdf": "Summary of paper.pdf." },
        }),
      );

      await user.click(screen.getByRole("tab", { name: "Summary" }));
      expect(await screen.findByText("Key finding")).toBeInTheDocument();
      expect(screen.getByText("Result X was significant.")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Download .md" }));
      expect(downloadTextMock).toHaveBeenCalledWith(
        "## Key finding\n\nResult X was significant.",
        "summary.md",
        "text/markdown",
      );

      expect(screen.getByText("Per-document summaries")).toBeInTheDocument();
      const cardToggle = screen.getByRole("button", { name: "paper.pdf" });
      await user.click(cardToggle);
      expect(await screen.findByText("Summary of paper.pdf.")).toBeInTheDocument();
    });

    it("Summary: falls back to an info message when no cross-document summary was generated", async () => {
      const user = await runToCompletion(makePipelineResult({ cross_summary: "", per_doc_summaries: {} }));

      await user.click(screen.getByRole("tab", { name: "Summary" }));
      expect(
        await screen.findByText("No cross-document summary was generated for this run."),
      ).toBeInTheDocument();
      expect(screen.queryByText("Per-document summaries")).not.toBeInTheDocument();
    });

    it("Retrieval: lists retrieved chunks with the retrieval mode and formatted page labels", async () => {
      const user = await runToCompletion(
        makePipelineResult({
          retrieval_mode: "hybrid",
          retrieved_chunks: [
            { chunk_id: "c1", doc_name: "a.pdf", page_num: 2, text: "Chunk text 1" },
            { chunk_id: "c2", doc_name: "b.pdf", page_num: -1, text: "Chunk text 2" },
          ],
        }),
      );

      await user.click(screen.getByRole("tab", { name: "Retrieval" }));
      expect(await screen.findByText("hybrid")).toBeInTheDocument();
      expect(screen.getByText(/2 chunk\(s\) retrieved/)).toBeInTheDocument();

      const first = screen.getByRole("button", { name: "[1] a.pdf — p. 3" });
      const second = screen.getByRole("button", { name: "[2] b.pdf — n/a" });
      await user.click(first);
      expect(await screen.findByText("Chunk text 1")).toBeInTheDocument();
      await user.click(second);
      expect(await screen.findByText("Chunk text 2")).toBeInTheDocument();
    });

    it("Retrieval: falls back to an info message when no chunks were retrieved", async () => {
      const user = await runToCompletion(makePipelineResult({ retrieved_chunks: [] }));

      await user.click(screen.getByRole("tab", { name: "Retrieval" }));
      expect(await screen.findByText("No chunks were retrieved for this run.")).toBeInTheDocument();
    });

    it("Citations: renders the citation report markdown with its download button", async () => {
      const user = await runToCompletion(
        makePipelineResult({ citation_report: "| Claim | Confidence |\n| --- | --- |\n| X | HIGH |" }),
      );

      await user.click(screen.getByRole("tab", { name: "Citations" }));
      expect(await screen.findByText("Claim")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Download .md" }));
      expect(downloadTextMock).toHaveBeenCalledWith(
        "| Claim | Confidence |\n| --- | --- |\n| X | HIGH |",
        "citations.md",
        "text/markdown",
      );
    });

    it("Citations: falls back to an info message when no citations were verified", async () => {
      const user = await runToCompletion(makePipelineResult({ citation_report: "" }));

      await user.click(screen.getByRole("tab", { name: "Citations" }));
      expect(await screen.findByText("No citations were verified for this run.")).toBeInTheDocument();
    });

    it("Knowledge Graph: fetches a PNG preview and supports dot/png/svg downloads", async () => {
      const user = await runToCompletion(makePipelineResult({ knowledge_graph_dot: "digraph { a -> b; }" }));

      await user.click(screen.getByRole("tab", { name: "Knowledge Graph" }));
      expect(exportKnowledgeGraphMock).toHaveBeenCalledWith("job-1", "png");
      expect(await screen.findByAltText("Knowledge graph")).toHaveAttribute("src", "blob:mock-url");

      await user.click(screen.getByRole("button", { name: "Download .dot" }));
      expect(downloadTextMock).toHaveBeenCalledWith(
        "digraph { a -> b; }",
        "knowledge_graph.dot",
        "text/vnd.graphviz",
      );

      await user.click(screen.getByRole("button", { name: "View DOT source" }));
      expect(await screen.findByText("digraph { a -> b; }")).toBeInTheDocument();
    });

    it("Knowledge Graph: download buttons request the matching export format", async () => {
      const user = await runToCompletion(makePipelineResult({ knowledge_graph_dot: "digraph { a -> b; }" }));

      await user.click(screen.getByRole("tab", { name: "Knowledge Graph" }));
      await screen.findByAltText("Knowledge graph");

      await user.click(screen.getByRole("button", { name: "Download .svg" }));
      expect(exportKnowledgeGraphMock).toHaveBeenCalledWith("job-1", "svg");

      await user.click(screen.getByRole("button", { name: "Download .png" }));
      expect(exportKnowledgeGraphMock).toHaveBeenCalledWith("job-1", "png");
    });

    it("Knowledge Graph: shows an error caption when the PNG preview fetch fails", async () => {
      exportKnowledgeGraphMock.mockReset();
      exportKnowledgeGraphMock.mockRejectedValue(new Error("render service down"));
      const user = await runToCompletion(makePipelineResult({ knowledge_graph_dot: "digraph { a -> b; }" }));

      await user.click(screen.getByRole("tab", { name: "Knowledge Graph" }));
      expect(
        await screen.findByText("Knowledge graph preview unavailable: render service down"),
      ).toBeInTheDocument();
    });

    it("Knowledge Graph: falls back to an info message when no graph was generated", async () => {
      const user = await runToCompletion(makePipelineResult({ knowledge_graph_dot: "" }));

      await user.click(screen.getByRole("tab", { name: "Knowledge Graph" }));
      expect(await screen.findByText("No knowledge graph was generated for this run.")).toBeInTheDocument();
      expect(exportKnowledgeGraphMock).not.toHaveBeenCalled();
    });

    it("Study Guide: renders the markdown and supports .md and .docx downloads", async () => {
      exportStudyGuideMock.mockResolvedValue(new Blob(["fake-docx"]));
      const user = await runToCompletion(makePipelineResult({ study_guide: "## Study Guide\n\nQ1. What is X?" }));

      await user.click(screen.getByRole("tab", { name: "Study Guide" }));
      expect(await screen.findByText("Q1. What is X?")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Download .md" }));
      expect(downloadTextMock).toHaveBeenCalledWith(
        "## Study Guide\n\nQ1. What is X?",
        "study_guide.md",
        "text/markdown",
      );

      await user.click(screen.getByRole("button", { name: "Download .docx" }));
      expect(exportStudyGuideMock).toHaveBeenCalledWith("job-1", "docx");
    });

    it("Study Guide: shows an error message when the DOCX export fails", async () => {
      exportStudyGuideMock.mockRejectedValue(new Error("export service down"));
      const user = await runToCompletion(makePipelineResult({ study_guide: "Body text." }));

      await user.click(screen.getByRole("tab", { name: "Study Guide" }));
      await user.click(screen.getByRole("button", { name: "Download .docx" }));

      expect(await screen.findByText("DOCX export failed: export service down")).toBeInTheDocument();
    });

    it("Study Guide: falls back to an info message when no study guide was generated", async () => {
      const user = await runToCompletion(makePipelineResult({ study_guide: "" }));

      await user.click(screen.getByRole("tab", { name: "Study Guide" }));
      expect(await screen.findByText("No study guide was generated for this run.")).toBeInTheDocument();
    });

    it("Podcast: renders the script, supports .txt download, and omits Read aloud when unsupported", async () => {
      const user = await runToCompletion(makePipelineResult({ podcast_script: "HOST: Welcome to the show." }));

      await user.click(screen.getByRole("tab", { name: "Podcast" }));
      expect(await screen.findByText("HOST: Welcome to the show.")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Download .txt" }));
      expect(downloadTextMock).toHaveBeenCalledWith("HOST: Welcome to the show.", "podcast_script.txt");

      expect(screen.queryByRole("button", { name: "Read aloud" })).not.toBeInTheDocument();
    });

    it("Podcast: falls back to an info message when no script was generated", async () => {
      const user = await runToCompletion(makePipelineResult({ podcast_script: "" }));

      await user.click(screen.getByRole("tab", { name: "Podcast" }));
      expect(await screen.findByText("No podcast script was generated for this run.")).toBeInTheDocument();
    });

    it("Podcast: supports the Read aloud / Stop reading toggle via the Web Speech API", async () => {
      const speak = vi.fn();
      const cancel = vi.fn();
      vi.stubGlobal("speechSynthesis", { speak, cancel });
      vi.stubGlobal(
        "SpeechSynthesisUtterance",
        class {
          text: string;
          onend: (() => void) | null = null;
          onerror: (() => void) | null = null;
          constructor(text: string) {
            this.text = text;
          }
        },
      );

      const user = userEvent.setup();
      runPipelineMock.mockResolvedValue({ job_id: "job-1" });
      const poll = controllablePoll();
      const { unmount } = render(<PipelineTab notebookId="nb-1" sourceCount={1} />);

      try {
        await user.click(screen.getByRole("button", { name: "Run Full Pipeline" }));
        await poll.settle({
          id: "job-1",
          status: "done",
          stage: "done",
          stage_info: {},
          error: null,
          result: makePipelineResult({ podcast_script: "HOST: Welcome to the show." }),
        });
        await screen.findByText("Done.");

        await user.click(screen.getByRole("tab", { name: "Podcast" }));
        await user.click(await screen.findByRole("button", { name: "Read aloud" }));

        expect(speak).toHaveBeenCalledTimes(1);
        expect(speak.mock.calls[0][0]).toMatchObject({ text: "HOST: Welcome to the show." });
        expect(await screen.findByRole("button", { name: "Stop reading" })).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "Stop reading" }));
        expect(cancel).toHaveBeenCalledTimes(1);
        expect(await screen.findByRole("button", { name: "Read aloud" })).toBeInTheDocument();
      } finally {
        unmount();
        vi.unstubAllGlobals();
      }
    });
  });
});
