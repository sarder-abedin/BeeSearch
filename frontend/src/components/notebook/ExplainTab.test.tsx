import { act, render, screen, within } from "@testing-library/react";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../api/client";
import type { ExplainJobStatus, ExplainResult, ExplainTurn } from "../../api/notebookExplainTypes";
import ExplainTab from "./ExplainTab";

const getExplainHistoryMock = vi.fn();
const runExplainTurnMock = vi.fn();
const pollExplainJobMock = vi.fn();

vi.mock("../../api/notebookExplain", () => ({
  getExplainHistory: (...args: unknown[]) => getExplainHistoryMock(...args),
  runExplainTurn: (...args: unknown[]) => runExplainTurnMock(...args),
  pollExplainJob: (...args: unknown[]) => pollExplainJobMock(...args),
}));

function makeExplainResult(overrides: Partial<ExplainResult> = {}): ExplainResult {
  return {
    notebook_id: "nb-1",
    user_message: "What is X?",
    assistant_response: "X is a thing.",
    explanation_style: "simple",
    citations: [],
    suggested_questions: [],
    is_repeat_clarification: false,
    repeated_question: "",
    new_concepts: [],
    concept_visual_html: "",
    source_decision: null,
    online_results: [],
    eval_result: {},
    errors: [],
    progress_pct: 100,
    ...overrides,
  };
}

/** Returns a controllable pollExplainJob double for the NEXT call to pollExplainJob(). */
function controllablePoll() {
  let onUpdateRef: ((s: ExplainJobStatus) => void) | undefined;
  let resolveRef: ((s: ExplainJobStatus) => void) | undefined;
  let rejectRef: ((e: unknown) => void) | undefined;
  pollExplainJobMock.mockImplementationOnce((_jobId: string, onUpdate: (s: ExplainJobStatus) => void) => {
    onUpdateRef = onUpdate;
    return new Promise<ExplainJobStatus>((resolve, reject) => {
      resolveRef = resolve;
      rejectRef = reject;
    });
  });
  return {
    update: async (s: ExplainJobStatus) => {
      await act(async () => {
        onUpdateRef?.(s);
      });
    },
    settle: async (s: ExplainJobStatus) => {
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

/** Types `text`, sends it, and resolves the turn to "done" with `result`. */
async function sendMessageAndComplete(
  user: UserEvent,
  text: string,
  result: ExplainResult,
  jobId = "job-1",
): Promise<void> {
  runExplainTurnMock.mockResolvedValueOnce({ job_id: jobId });
  const poll = controllablePoll();
  await user.type(screen.getByLabelText("Message"), text);
  await user.click(screen.getByRole("button", { name: "Send" }));
  await poll.settle({ id: jobId, status: "done", stage: "done", stage_info: {}, error: null, result });
  await screen.findByText("Done.");
}

describe("ExplainTab", () => {
  beforeEach(() => {
    getExplainHistoryMock.mockReset();
    runExplainTurnMock.mockReset();
    pollExplainJobMock.mockReset();
    getExplainHistoryMock.mockResolvedValue([]);
  });

  it("loads history on mount and shows the empty-state caption plus default style/level selection", async () => {
    render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);

    expect(getExplainHistoryMock).toHaveBeenCalledWith("nb-1");
    expect(await screen.findByText(/Type your first question/)).toBeInTheDocument();

    expect(screen.getByRole("radiogroup", { name: "Explanation style" })).toBeInTheDocument();
    expect(screen.getByRole("radiogroup", { name: "Explanation level" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Simple Language" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Intermediate" })).toBeChecked();
    for (const label of ["Extended Analogy", "Step-by-Step", "For vs. Against", "Novice", "Expert"]) {
      expect(screen.getByRole("radio", { name: label })).not.toBeChecked();
    }
  });

  it("renders previously-saved history, including the last assistant turn's citations and follow-ups", async () => {
    const user = userEvent.setup();
    const turns: ExplainTurn[] = [
      {
        role: "user",
        content: "What is X?",
        timestamp: "2026-01-01T00:00:00Z",
        citations: null,
        suggested_questions: null,
        explanation_style: null,
      },
      {
        role: "assistant",
        content: "X is a thing. [1]",
        timestamp: "2026-01-01T00:00:01Z",
        citations: [{ n: "1", doc_name: "paper.pdf", page: 2, page_label: "p. 2", snippet: "X is defined here.", url: "" }],
        suggested_questions: ["What about Y?"],
        explanation_style: "simple",
      },
    ];
    getExplainHistoryMock.mockResolvedValue(turns);

    render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);

    expect(await screen.findByText("X is a thing. [1]")).toBeInTheDocument();
    expect(screen.getByText("You")).toBeInTheDocument();
    expect(screen.getByText("Assistant")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "What about Y?" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Sources (1)" }));
    expect(screen.getByText("X is defined here.")).toBeInTheDocument();
  });

  it("shows an error message when loading history fails", async () => {
    getExplainHistoryMock.mockReset();
    getExplainHistoryMock.mockRejectedValue(new ApiError(500, "Backend unavailable"));

    render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);

    expect(await screen.findByText("Backend unavailable")).toBeInTheDocument();
  });

  it("shows a warning and does not call runExplainTurn when sending a blank message", async () => {
    const user = userEvent.setup();
    render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
    await screen.findByText(/Type your first question/);

    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Please enter a question.")).toBeInTheDocument();
    expect(runExplainTurnMock).not.toHaveBeenCalled();
  });

  it("sends a message with the selected style/level, shows live stage updates, then appends the assistant turn", async () => {
    const user = userEvent.setup();
    render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
    await screen.findByText(/Type your first question/);

    runExplainTurnMock.mockResolvedValue({ job_id: "job-1" });
    const poll = controllablePoll();

    await user.type(screen.getByLabelText("Message"), "What is X?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(runExplainTurnMock).toHaveBeenCalledWith({
      notebook_id: "nb-1",
      message: "What is X?",
      explanation_style: "simple",
      explanation_level: "intermediate",
    });
    expect(screen.getByLabelText("Message")).toHaveValue("");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(await screen.findByText("What is X?")).toBeInTheDocument();
    expect(screen.getByText("Starting…")).toBeInTheDocument();

    await poll.update({
      id: "job-1",
      status: "running",
      stage: "storyteller",
      stage_info: { label: "Crafting the explanation…" },
      error: null,
      result: null,
    });
    expect(await screen.findByText("Crafting the explanation…")).toBeInTheDocument();

    await poll.settle({
      id: "job-1",
      status: "done",
      stage: "done",
      stage_info: {},
      error: null,
      result: makeExplainResult({ assistant_response: "X is a thing.", suggested_questions: ["What about Y?"] }),
    });

    expect(await screen.findByText("Done.")).toBeInTheDocument();
    expect(screen.getByText("X is a thing.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "What about Y?" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).not.toBeDisabled();
  });

  it("sends on Enter but inserts a newline (no send) on Shift+Enter", async () => {
    const user = userEvent.setup();
    render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
    await screen.findByText(/Type your first question/);
    runExplainTurnMock.mockResolvedValue({ job_id: "job-1" });
    const poll = controllablePoll();

    const textarea = screen.getByLabelText("Message");
    await user.type(textarea, "Hello{Shift>}{Enter}{/Shift}");
    expect(runExplainTurnMock).not.toHaveBeenCalled();

    await user.type(textarea, "{Enter}");
    expect(runExplainTurnMock).toHaveBeenCalledTimes(1);
    expect(runExplainTurnMock).toHaveBeenCalledWith(
      expect.objectContaining({ message: "Hello" }),
    );

    await poll.settle({ id: "job-1", status: "done", stage: "done", stage_info: {}, error: null, result: makeExplainResult() });
  });

  it("sends the currently-selected style and level", async () => {
    const user = userEvent.setup();
    render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
    await screen.findByText(/Type your first question/);

    await user.click(screen.getByRole("radio", { name: "Extended Analogy" }));
    await user.click(screen.getByRole("radio", { name: "Expert" }));

    runExplainTurnMock.mockResolvedValue({ job_id: "job-1" });
    const poll = controllablePoll();
    await user.type(screen.getByLabelText("Message"), "Explain X");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(runExplainTurnMock).toHaveBeenCalledWith({
      notebook_id: "nb-1",
      message: "Explain X",
      explanation_style: "analogy",
      explanation_level: "expert",
    });

    await poll.settle({ id: "job-1", status: "done", stage: "done", stage_info: {}, error: null, result: makeExplainResult() });
  });

  it("disables a follow-up button while its turn is in flight, and sends its text verbatim", async () => {
    const user = userEvent.setup();
    render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
    await screen.findByText(/Type your first question/);

    await sendMessageAndComplete(user, "What is X?", makeExplainResult({ suggested_questions: ["What about Y?"] }));

    runExplainTurnMock.mockResolvedValueOnce({ job_id: "job-2" });
    const poll = controllablePoll();
    const followupButton = screen.getByRole("button", { name: "What about Y?" });
    await user.click(followupButton);

    expect(runExplainTurnMock).toHaveBeenLastCalledWith({
      notebook_id: "nb-1",
      message: "What about Y?",
      explanation_style: "simple",
      explanation_level: "intermediate",
    });
    expect(followupButton).toBeDisabled();

    await poll.settle({ id: "job-2", status: "done", stage: "done", stage_info: {}, error: null, result: makeExplainResult() });
  });

  it("shows a failure status when the job ends in error", async () => {
    const user = userEvent.setup();
    render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
    await screen.findByText(/Type your first question/);

    runExplainTurnMock.mockResolvedValue({ job_id: "job-err" });
    const poll = controllablePoll();
    await user.type(screen.getByLabelText("Message"), "What is X?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await poll.settle({
      id: "job-err",
      status: "error",
      stage: null,
      stage_info: {},
      error: "Ollama not reachable",
      result: null,
    });

    expect(await screen.findByText("Failed: Ollama not reachable")).toBeInTheDocument();
  });

  it("shows a failure status when runExplainTurn itself rejects", async () => {
    const user = userEvent.setup();
    render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
    await screen.findByText(/Type your first question/);

    runExplainTurnMock.mockRejectedValue(new ApiError(500, "Backend unavailable"));
    await user.type(screen.getByLabelText("Message"), "What is X?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Failed: Backend unavailable")).toBeInTheDocument();
  });

  describe("citations", () => {
    it("renders document and online citations with the right link/snippet shapes, and toggles collapse", async () => {
      const user = userEvent.setup();
      const { container } = render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
      await screen.findByText(/Type your first question/);

      await sendMessageAndComplete(
        user,
        "What is X?",
        makeExplainResult({
          citations: [
            { n: "1", doc_name: "paper.pdf", page: 2, page_label: "p. 2", snippet: "X is defined here.", url: "" },
            { n: "Source 2", doc_name: "A Survey of X", page: null, page_label: "n/a", snippet: "", url: "https://example.com/x" },
          ],
        }),
      );

      await user.click(screen.getByRole("button", { name: "Sources (2)" }));

      const items = container.querySelectorAll(".notebook-citations__item");
      expect(items).toHaveLength(2);
      expect(items[0].textContent).toContain("[1] paper.pdf");
      expect(items[0].textContent).toContain("p. 2");
      expect(screen.getByText("X is defined here.")).toBeInTheDocument();

      const link = within(items[1] as HTMLElement).getByRole("link", { name: "A Survey of X" });
      expect(link).toHaveAttribute("href", "https://example.com/x");

      await user.click(screen.getByRole("button", { name: "Sources (2)" }));
      expect(screen.queryByText("X is defined here.")).not.toBeInTheDocument();
    });
  });

  describe("source decision banner", () => {
    it("shows an info banner with the searched-source labels when online search was used", async () => {
      const user = userEvent.setup();
      const { container } = render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
      await screen.findByText(/Type your first question/);

      await sendMessageAndComplete(
        user,
        "What is X?",
        makeExplainResult({
          source_decision: {
            coverage_score: 4,
            used_docs: true,
            used_online: true,
            search_attempted: true,
            reason: "Docs only partially cover this.",
            sources_searched: ["academic", "web"],
            online_count: 3,
          },
        }),
      );

      const banner = container.querySelector("p.sr-info");
      expect(banner).not.toBeNull();
      expect(banner?.textContent).toContain("Document coverage: 4/10");
      expect(banner?.textContent).toContain("Docs only partially cover this.");
      expect(banner?.textContent).toContain("3 source(s) from arXiv / Semantic Scholar + web");
    });

    it("shows a warning banner when online search found nothing additional", async () => {
      const user = userEvent.setup();
      const { container } = render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
      await screen.findByText(/Type your first question/);

      await sendMessageAndComplete(
        user,
        "What is X?",
        makeExplainResult({
          source_decision: {
            coverage_score: 5,
            used_docs: true,
            used_online: false,
            search_attempted: true,
            reason: "Borderline coverage.",
            sources_searched: [],
            online_count: 0,
          },
        }),
      );

      const banner = container.querySelector("p.sr-warning");
      expect(banner).not.toBeNull();
      expect(banner?.textContent).toContain("Document coverage: 5/10");
      expect(banner?.textContent).toContain("Online search was attempted but found no additional results");
    });

    it("shows a plain caption when the answer came from documents alone", async () => {
      const user = userEvent.setup();
      render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
      await screen.findByText(/Type your first question/);

      await sendMessageAndComplete(
        user,
        "What is X?",
        makeExplainResult({
          source_decision: {
            coverage_score: 9,
            used_docs: true,
            used_online: false,
            search_attempted: false,
            reason: "Well covered.",
            sources_searched: [],
            online_count: 0,
          },
        }),
      );

      expect(await screen.findByText("Answered from your documents (coverage 9/10)")).toBeInTheDocument();
    });
  });

  describe("repeat-clarification caption", () => {
    it("mentions the actually-used style when it differs from the currently-selected style", async () => {
      const user = userEvent.setup();
      render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
      await screen.findByText(/Type your first question/);

      await sendMessageAndComplete(
        user,
        "I don't understand, explain again",
        makeExplainResult({ is_repeat_clarification: true, explanation_style: "analogy" }),
      );

      expect(
        await screen.findByText(
          'This looked like a repeat of an earlier question, so this answer uses "Extended Analogy" instead of your selected style — explaining it differently, not just rewording it.',
        ),
      ).toBeInTheDocument();
    });

    it("falls back to a generic caption when the used style matches the currently-selected style", async () => {
      const user = userEvent.setup();
      render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
      await screen.findByText(/Type your first question/);

      await sendMessageAndComplete(
        user,
        "I don't understand, explain again",
        makeExplainResult({ is_repeat_clarification: true, explanation_style: "simple" }),
      );

      expect(
        await screen.findByText(
          "This looked like a repeat of an earlier question — the explanation below takes a different angle than before.",
        ),
      ).toBeInTheDocument();
    });
  });

  describe("concept visualization", () => {
    it("renders a sandboxed iframe with the concept map HTML only for the latest assistant turn", async () => {
      const user = userEvent.setup();
      render(<ExplainTab notebookId="nb-1" notebookName="My Notebook" />);
      await screen.findByText(/Type your first question/);

      await sendMessageAndComplete(
        user,
        "I don't understand",
        makeExplainResult({ concept_visual_html: "<html><body>concept map</body></html>" }),
      );

      const iframe = screen.getByTitle("Concept map") as HTMLIFrameElement;
      expect(iframe).toHaveAttribute("sandbox", "allow-scripts");
      expect(iframe.srcdoc).toBe("<html><body>concept map</body></html>");

      await sendMessageAndComplete(user, "Got it, thanks", makeExplainResult({ concept_visual_html: "" }), "job-2");

      expect(screen.queryByTitle("Concept map")).not.toBeInTheDocument();
    });
  });

  it("reloads history and resets per-notebook UI state when notebookId changes", async () => {
    const user = userEvent.setup();
    const { rerender } = render(<ExplainTab notebookId="nb-1" notebookName="Notebook One" />);
    await screen.findByText(/Type your first question/);

    await sendMessageAndComplete(user, "What is X?", makeExplainResult());
    expect(screen.getByText("Done.")).toBeInTheDocument();

    getExplainHistoryMock.mockResolvedValueOnce([]);
    rerender(<ExplainTab notebookId="nb-2" notebookName="Notebook Two" />);

    expect(getExplainHistoryMock).toHaveBeenCalledWith("nb-2");
    expect(await screen.findByText(/Type your first question/)).toBeInTheDocument();
    expect(screen.queryByText("Done.")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("Ask anything about Notebook Two…")).toBeInTheDocument();
  });
});
