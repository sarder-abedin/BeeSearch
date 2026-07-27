import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { runPaperReview, runReviewerChat } from "../../../api/notebookAdvanced";
import type { ExternalReference, ReviewChatItem } from "../../../api/notebookAdvancedTypes";
import type { SourceMeta } from "../../../api/notebookTypes";
import { RunControls } from "./shared";
import { useAdvancedToolJob, useModelOverrides } from "./useAdvancedToolJob";

interface ReviewerPanelProps {
  notebookId: string;
  sources: SourceMeta[];
}

function ReviewerPanel({ notebookId, sources }: ReviewerPanelProps) {
  const job = useAdvancedToolJob();
  const overrides = useModelOverrides();
  const { state, result, error } = job;

  const [docId, setDocId] = useState(sources[0]?.doc_id ?? "");
  const [chatHistory, setChatHistory] = useState<ReviewChatItem[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const chatRef = useRef<HTMLDivElement>(null);

  const reviewText = result?.paper_review ?? "";
  const extRefs = result?.paper_review_refs ?? [];

  // Reset chat when a new review is generated.
  useEffect(() => {
    if (state === "done") {
      setChatHistory([]);
      setChatError(null);
    }
  }, [state]);

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
        document. Supporting literature from arXiv and Semantic Scholar is sourced automatically.
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
        runLabel="Generate Review"
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

      {state === "done" && result && (
        <>
          {reviewText ? (
            <div className="reviewer-panel__review">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{reviewText}</ReactMarkdown>
            </div>
          ) : (
            <p className="sr-info">No review was generated for this run.</p>
          )}

          {extRefs.length > 0 && (
            <div className="reviewer-panel__refs">
              <h4>Supporting Literature</h4>
              <p className="sr-caption">
                Papers found via arXiv and Semantic Scholar based on critique points in this review.
              </p>
              <ul className="reviewer-panel__ref-list">
                {extRefs.map((ref, i) => (
                  <ExternalRefCard key={i} ref_={ref} />
                ))}
              </ul>
            </div>
          )}

          <div className="reviewer-panel__chat">
            <h4>Follow-up Discussion</h4>
            <p className="sr-caption">
              Ask questions about specific critique points or request suggestions to address
              weaknesses.
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
                      {msg.role === "user" ? "You" : "Reviewer"}
                    </span>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                ))}
                {chatLoading && (
                  <div className="reviewer-panel__chat-msg reviewer-panel__chat-msg--assistant">
                    <span className="reviewer-panel__chat-role">Reviewer</span>
                    <p className="sr-spinner-text">Responding…</p>
                  </div>
                )}
              </div>
            )}

            {chatError && <p className="sr-error">Chat failed: {chatError}</p>}

            <div className="reviewer-panel__chat-input-row">
              <textarea
                className="reviewer-panel__chat-input"
                placeholder="Ask about a critique point or request suggestions…"
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
