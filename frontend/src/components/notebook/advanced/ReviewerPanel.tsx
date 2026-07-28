import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { runPaperReview, runReviewerChat } from "../../../api/notebookAdvanced";
import type { ExternalReference, ReviewChatItem } from "../../../api/notebookAdvancedTypes";
import type { SavedReview, SourceMeta } from "../../../api/notebookTypes";
import { RunControls } from "./shared";
import { useAdvancedToolJob, useModelOverrides } from "./useAdvancedToolJob";

interface ReviewerPanelProps {
  notebookId: string;
  sources: SourceMeta[];
  savedReviews?: Record<string, SavedReview>;
}

function ReviewerPanel({ notebookId, sources, savedReviews }: ReviewerPanelProps) {
  const job = useAdvancedToolJob();
  const overrides = useModelOverrides();
  const { state, result, error } = job;

  const [docId, setDocId] = useState(sources[0]?.doc_id ?? "");
  const [chatHistory, setChatHistory] = useState<ReviewChatItem[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const chatRef = useRef<HTMLDivElement>(null);

  // Fresh job result takes precedence; fall back to persisted review.
  const freshReview = state === "done" ? (result?.paper_review ?? "") : "";
  const freshRefs = state === "done" ? (result?.paper_review_refs ?? []) : [];
  const saved = savedReviews?.[docId];
  const reviewText = freshReview || (state === "idle" ? (saved?.review_text ?? "") : "");
  const extRefs: ExternalReference[] = freshRefs.length > 0
    ? freshRefs
    : (state === "idle" ? ((saved?.external_refs ?? []) as ExternalReference[]) : []);
  const isShowingReview = reviewText.length > 0;
  const isFromMemory = !freshReview && !!saved?.review_text;

  // Reset chat when a new review is generated.
  useEffect(() => {
    if (state === "done") {
      setChatHistory([]);
      setChatError(null);
    }
  }, [state]);

  // Reset chat when doc changes so old conversation doesn't bleed through.
  useEffect(() => {
    setChatHistory([]);
    setChatError(null);
  }, [docId]);

  // Scroll chat to bottom after each new message.
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [chatHistory]);

  if (sources.length === 0) {
    return (
      <div className="advanced-tools-tab__panel">
        <h3>Reviewer</h3>
        <p className="sr-info">Add at least one source to use the Reviewer.</p>
      </div>
    );
  }

  async function handleSendChat() {
    if (!chatInput.trim() || !reviewText) return;
    const userMessage = chatInput.trim();
    setChatInput("");
    setChatLoading(true);
    setChatError(null);

    const userItem: ReviewChatItem = { role: "user", content: userMessage };
    const nextHistory = [...chatHistory, userItem];
    setChatHistory(nextHistory);

    try {
      const { job_id } = await runReviewerChat({
        notebook_id: notebookId,
        doc_id: docId,
        review_text: reviewText,
        chat_history: chatHistory,
        user_message: userMessage,
        external_refs: extRefs,
        ...overrides,
      });

      const { pollAdvancedJob } = await import("../../../api/notebookAdvanced");
      const final = await pollAdvancedJob(job_id, () => {});
      if (final.status === "done" && final.result?.reviewer_chat_response) {
        const assistantItem: ReviewChatItem = {
          role: "assistant",
          content: final.result.reviewer_chat_response,
        };
        setChatHistory([...nextHistory, assistantItem]);
      } else {
        setChatError(final.error ?? "No response received.");
        setChatHistory(chatHistory);
      }
    } catch (err) {
      setChatError((err as Error).message);
      setChatHistory(chatHistory);
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <div className="advanced-tools-tab__panel">
      <h3>Reviewer</h3>
      <p>
        Select a paper and generate a critical IEEE-style peer review grounded in the uploaded
        document <em>and</em> backed by evidence from arXiv and Semantic Scholar — external papers
        are cited inline as [E1], [E2], … in the review text.
      </p>
      <p className="sr-caption">
        After the review is generated, use the <strong>Critique Validation</strong> dialogue below
        to act as the reviewer yourself. Propose critique points and the assistant will check
        whether each is accurate, evidence-based, and justified — confirming what holds up and
        pushing back on anything unsupported or overstated.
      </p>

      <div className="sr-field">
        <label htmlFor="reviewer-source">Paper to review</label>
        <select
          id="reviewer-source"
          value={docId}
          disabled={state === "running"}
          onChange={(e) => {
            setDocId(e.target.value);
            job.clear();
          }}
        >
          {sources.map((s) => (
            <option key={s.doc_id} value={s.doc_id}>
              {s.filename}
            </option>
          ))}
        </select>
      </div>

      <RunControls
        state={state}
        runLabel={isFromMemory ? "Regenerate Review" : "Generate Review"}
        rerunLabel="Regenerate Review"
        spinnerText="Reviewing paper — this may take a minute…"
        error={error}
        errorPrefix="Review generation failed"
        onRun={() =>
          job.run(() =>
            runPaperReview({ notebook_id: notebookId, doc_id: docId, ...overrides }),
          )
        }
        onClear={job.clear}
      />

      {isShowingReview && (
        <>
          {isFromMemory && saved && (
            <p className="sr-caption reviewer-panel__saved-notice">
              Loaded from memory — generated {new Date(saved.generated_at).toLocaleString()}.
              Click "Regenerate Review" to refresh.
            </p>
          )}
          <div className="reviewer-panel__review-header">
            <button
              type="button"
              className="sr-button reviewer-panel__download-btn"
              title="Download review as Markdown"
              onClick={() => {
                const docName = sources.find((s) => s.doc_id === docId)?.filename ?? "review";
                const filename = docName.replace(/\.[^.]+$/, "_review.md");
                const blob = new Blob([reviewText], { type: "text/markdown" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }}
            >
              ⬇ Download (.md)
            </button>
          </div>
          {reviewText ? (
            <div className="reviewer-panel__review">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{reviewText}</ReactMarkdown>
            </div>
          ) : (
            <p className="sr-info">No review was generated for this run.</p>
          )}

          {extRefs.length > 0 && (
            <div className="reviewer-panel__refs">
              <h4>External References</h4>
              <p className="sr-caption">
                Papers retrieved from arXiv and Semantic Scholar before the review was written.
                Each is cited inline in the review text as [E1], [E2], …
              </p>
              <ul className="reviewer-panel__ref-list">
                {extRefs.map((ref, i) => (
                  <ExternalRefCard key={i} ref_={ref} />
                ))}
              </ul>
            </div>
          )}

          <div className="reviewer-panel__chat">
            <h4>Critique Validation</h4>
            <p className="sr-caption">
              You are the reviewer. Propose your critique points about this paper and the
              assistant will validate each one — confirming what is accurate and well-grounded,
              challenging what is unsupported, and helping you sharpen weak points into precise,
              evidence-based reviewer comments.
            </p>

            {chatHistory.length > 0 && (
              <div className="reviewer-panel__chat-history" ref={chatRef}>
                {chatHistory.map((msg, i) => (
                  <div
                    key={i}
                    className={
                      msg.role === "user"
                        ? "reviewer-panel__chat-msg reviewer-panel__chat-msg--user"
                        : "reviewer-panel__chat-msg reviewer-panel__chat-msg--assistant"
                    }
                  >
                    <span className="reviewer-panel__chat-role">
                      {msg.role === "user" ? "You (Reviewer)" : "Validator"}
                    </span>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                ))}
                {chatLoading && (
                  <div className="reviewer-panel__chat-msg reviewer-panel__chat-msg--assistant">
                    <span className="reviewer-panel__chat-role">Validator</span>
                    <p className="sr-spinner-text">Validating…</p>
                  </div>
                )}
              </div>
            )}

            {chatError && <p className="sr-error">Chat failed: {chatError}</p>}

            <div className="reviewer-panel__chat-input-row">
              <textarea
                className="reviewer-panel__chat-input"
                placeholder="Propose a critique point — the validator will check whether it's accurate and evidence-based…"
                rows={3}
                value={chatInput}
                disabled={chatLoading}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void handleSendChat();
                  }
                }}
              />
              <button
                type="button"
                className="sr-button"
                disabled={chatLoading || !chatInput.trim()}
                onClick={() => void handleSendChat()}
              >
                Send
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function ExternalRefCard({ ref_ }: { ref_: ExternalReference }) {
  const authors =
    ref_.authors.length > 0
      ? ref_.authors.slice(0, 3).join(", ") + (ref_.authors.length > 3 ? " et al." : "")
      : "Unknown authors";
  const year = ref_.year ? ` (${ref_.year})` : "";

  return (
    <li className="reviewer-panel__ref-card">
      <div className="reviewer-panel__ref-title">
        {ref_.ref_num && (
          <span className="reviewer-panel__ref-label">[{ref_.ref_num}]</span>
        )}
        {ref_.url ? (
          <a href={ref_.url} target="_blank" rel="noopener noreferrer">
            {ref_.title || "Untitled"}
          </a>
        ) : (
          <span>{ref_.title || "Untitled"}</span>
        )}
      </div>
      <div className="reviewer-panel__ref-meta">
        {authors}
        {year} &middot; <span className="reviewer-panel__ref-source">{ref_.source}</span>
      </div>
      {ref_.abstract_snippet && (
        <div className="reviewer-panel__ref-abstract">{ref_.abstract_snippet}</div>
      )}
    </li>
  );
}

export default ReviewerPanel;
