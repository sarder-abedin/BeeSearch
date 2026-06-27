import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  createNotebook,
  deleteNotebook,
  getNotebook,
  listNotebooks,
  pollChatJob,
  removeSource,
  renameNotebook,
  sendChatMessage,
  uploadSource,
} from "../api/notebook";
import { ApiError } from "../api/client";
import type { ConversationTurn, NotebookDetail, NotebookSummary } from "../api/notebookTypes";
import EvalResultPanel from "../components/EvalResultPanel";
import NotebookCitations from "../components/NotebookCitations";
import RagReflectionPanel from "../components/RagReflectionPanel";
import "../components/sr/sr-common.css";
import "./NotebookPage.css";

type ChatStatus = "idle" | "running" | "done" | "error";

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.detail : (err as Error).message;
}

export default function NotebookPage() {
  const [notebooks, setNotebooks] = useState<NotebookSummary[]>([]);
  const [notebooksError, setNotebooksError] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<NotebookDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [sidebarNotice, setSidebarNotice] = useState<{ kind: "warning" | "info"; text: string } | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [autoWebSearch, setAutoWebSearch] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [message, setMessage] = useState("");
  const [chatStatus, setChatStatus] = useState<ChatStatus>("idle");
  const [chatLabel, setChatLabel] = useState("");
  const [chatWarning, setChatWarning] = useState<string | null>(null);
  const [lastEvalResult, setLastEvalResult] = useState<Record<string, unknown> | null>(null);
  const [lastRagReflection, setLastRagReflection] = useState<Record<string, unknown> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [loadedNotebookId, setLoadedNotebookId] = useState<string | null>(null);
  const [syncedNotebookId, setSyncedNotebookId] = useState<string | null>(null);

  if (activeId !== loadedNotebookId) {
    setLoadedNotebookId(activeId);
    setChatStatus("idle");
    setChatLabel("");
    setChatWarning(null);
    setLastEvalResult(null);
    setLastRagReflection(null);
    if (!activeId) {
      setDetail(null);
      setDetailError(null);
    }
  }

  if ((detail?.notebook_id ?? null) !== syncedNotebookId) {
    setSyncedNotebookId(detail?.notebook_id ?? null);
    setRenameValue(detail?.name ?? "");
  }

  useEffect(() => {
    let cancelled = false;
    listNotebooks()
      .then((list) => {
        if (!cancelled) setNotebooks(list);
      })
      .catch((err) => {
        if (!cancelled) setNotebooksError(errorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    abortRef.current?.abort();
    if (!activeId) return;

    let cancelled = false;
    getNotebook(activeId)
      .then((d) => {
        if (!cancelled) {
          setDetail(d);
          setDetailError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setDetailError(errorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  async function refreshNotebooks() {
    try {
      setNotebooks(await listNotebooks());
      setNotebooksError(null);
    } catch (err) {
      setNotebooksError(errorMessage(err));
    }
  }

  async function refreshDetail(id: string) {
    try {
      setDetail(await getNotebook(id));
      setDetailError(null);
    } catch (err) {
      setDetailError(errorMessage(err));
    }
  }

  async function handleCreate() {
    setCreating(true);
    setSidebarNotice(null);
    try {
      const nb = await createNotebook(newName.trim());
      setNewName("");
      setActiveId(nb.notebook_id);
      void refreshNotebooks();
    } catch (err) {
      setSidebarNotice({ kind: "warning", text: errorMessage(err) });
    } finally {
      setCreating(false);
    }
  }

  async function handleUpload(files: FileList | null) {
    if (!activeId || !files || files.length === 0) return;
    setUploading(true);
    setSidebarNotice(null);
    const duplicates: string[] = [];
    try {
      for (const file of Array.from(files)) {
        const result = await uploadSource(activeId, file);
        if (result.duplicate) duplicates.push(file.name);
      }
      if (duplicates.length > 0) {
        setSidebarNotice({ kind: "info", text: `Already in this notebook (skipped): ${duplicates.join(", ")}` });
      }
      await refreshDetail(activeId);
      void refreshNotebooks();
    } catch (err) {
      setSidebarNotice({ kind: "warning", text: errorMessage(err) });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleRemoveSource(docId: string) {
    if (!activeId) return;
    setSidebarNotice(null);
    try {
      await removeSource(activeId, docId);
      await refreshDetail(activeId);
      void refreshNotebooks();
    } catch (err) {
      setSidebarNotice({ kind: "warning", text: errorMessage(err) });
    }
  }

  async function handleRename() {
    if (!activeId) return;
    const trimmed = renameValue.trim();
    if (!trimmed) return;
    setSidebarNotice(null);
    try {
      await renameNotebook(activeId, trimmed);
      await refreshDetail(activeId);
      void refreshNotebooks();
    } catch (err) {
      setSidebarNotice({ kind: "warning", text: errorMessage(err) });
    }
  }

  async function handleDelete() {
    if (!activeId) return;
    setSidebarNotice(null);
    try {
      await deleteNotebook(activeId);
      setActiveId(null);
      void refreshNotebooks();
    } catch (err) {
      setSidebarNotice({ kind: "warning", text: errorMessage(err) });
    }
  }

  async function runChat(text: string) {
    if (!activeId || !detail) return;

    if (detail.source_count === 0 && !autoWebSearch) {
      setChatWarning(
        "Add at least one source before asking questions, or enable Auto web search below to let the agent search the web (DuckDuckGo).",
      );
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setChatStatus("running");
    setChatLabel("Searching your notebook…");
    setLastEvalResult(null);
    setLastRagReflection(null);

    const userTurn: ConversationTurn = {
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
      citations: null,
      suggested_questions: null,
    };
    setDetail((prev) => (prev ? { ...prev, conversation: [...prev.conversation, userTurn] } : prev));

    try {
      const { job_id } = await sendChatMessage({
        notebook_id: activeId,
        message: text,
        include_web_search: autoWebSearch,
      });

      const final = await pollChatJob(
        job_id,
        (update) => {
          const info = update.stage_info ?? {};
          const label = typeof info.label === "string" ? info.label : null;
          if (label) setChatLabel(label);
        },
        controller.signal,
      );

      if (final.status === "done" && final.result) {
        const r = final.result;
        const assistantTurn: ConversationTurn = {
          role: "assistant",
          content: r.assistant_response,
          timestamp: new Date().toISOString(),
          citations: r.citations,
          suggested_questions: r.suggested_questions,
        };
        setDetail((prev) => (prev ? { ...prev, conversation: [...prev.conversation, assistantTurn] } : prev));
        setLastEvalResult(r.eval_result);
        setLastRagReflection(r.rag_reflection_info);
        setChatLabel("Done.");
        setChatStatus("done");
      } else {
        setChatLabel(`Failed: ${final.error ?? "Unknown error."}`);
        setChatStatus("error");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setChatLabel(`Failed: ${errorMessage(err)}`);
      setChatStatus("error");
    }
  }

  function handleSend() {
    const trimmed = message.trim();
    if (!trimmed) {
      setChatWarning("Please enter a question.");
      return;
    }
    setMessage("");
    setChatWarning(null);
    void runChat(trimmed);
  }

  function handleFollowup(q: string) {
    setChatWarning(null);
    void runChat(q);
  }

  return (
    <main className="notebook-page">
      <h1>Mode 2 — Research Notebook</h1>
      <p>
        Upload PDFs, DOCX, or text files and ask grounded questions — BeeSearch retrieves the
        most relevant passages from your own documents (or the web, if enabled) and cites its
        sources inline.
      </p>
      <hr />

      <div className="notebook-page__layout">
        <aside className="notebook-page__sidebar">
          <h2>Notebook</h2>
          {notebooksError && <p className="sr-error">{notebooksError}</p>}

          <div className="sr-field">
            <label htmlFor="nb-select">Select a notebook</label>
            <select id="nb-select" value={activeId ?? ""} onChange={(e) => setActiveId(e.target.value || null)}>
              <option value="">+ New notebook</option>
              {notebooks.map((nb) => (
                <option key={nb.notebook_id} value={nb.notebook_id}>
                  {nb.name.slice(0, 30)} ({nb.source_count} src, {nb.turn_count} turns)
                </option>
              ))}
            </select>
          </div>

          {!activeId && (
            <div className="sr-field">
              <label htmlFor="nb-new-name">Notebook name</label>
              <input
                id="nb-new-name"
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Untitled Notebook"
              />
              <button type="button" className="sr-button" onClick={() => void handleCreate()} disabled={creating}>
                Create notebook
              </button>
            </div>
          )}

          {sidebarNotice?.kind === "warning" && <p className="sr-warning">{sidebarNotice.text}</p>}
          {sidebarNotice?.kind === "info" && <p className="sr-info">{sidebarNotice.text}</p>}

          {activeId && detail && (
            <div className="notebook-page__sources">
              <h3>Sources</h3>
              <label htmlFor="nb-file-input">Upload sources</label>
              <input
                id="nb-file-input"
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.doc,.txt,.md,.rst"
                onChange={(e) => void handleUpload(e.target.files)}
                disabled={uploading}
              />
              {uploading && <p className="sr-spinner-text">Uploading…</p>}

              {detail.sources.length === 0 ? (
                <p className="sr-caption">
                  No sources yet — upload a PDF, DOCX, TXT, or Markdown file to get started.
                </p>
              ) : (
                <ul className="notebook-page__source-list">
                  {detail.sources.map((s) => (
                    <li key={s.doc_id}>
                      <span>{s.filename}</span>{" "}
                      <span className="sr-caption">({s.total_chunks} chunks)</span>
                      <button type="button" onClick={() => void handleRemoveSource(s.doc_id)}>
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <label className="notebook-page__checkbox">
                <input
                  type="checkbox"
                  checked={autoWebSearch}
                  onChange={(e) => setAutoWebSearch(e.target.checked)}
                />
                Auto web search
              </label>

              <details className="sr-explore-panel__details">
                <summary>Rename / Delete</summary>
                <div className="sr-field">
                  <label htmlFor="nb-rename">Notebook name</label>
                  <input
                    id="nb-rename"
                    type="text"
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                  />
                  <button type="button" className="sr-button" onClick={() => void handleRename()}>
                    Save name
                  </button>
                </div>
                <button
                  type="button"
                  className="notebook-page__delete-button"
                  onClick={() => void handleDelete()}
                >
                  Delete notebook
                </button>
              </details>
            </div>
          )}
        </aside>

        <section className="notebook-page__main">
          {!activeId || !detail ? (
            <p className="sr-caption">Create or select a notebook on the left to begin.</p>
          ) : (
            <>
              {detailError && <p className="sr-error">{detailError}</p>}
              <h2>{detail.name}</h2>
              {detail.sources.length > 0 && (
                <p className="sr-caption">
                  Grounded in: {detail.sources.slice(0, 6).map((s) => s.filename.slice(0, 24)).join(", ")}
                  {detail.sources.length > 6 ? " …" : ""}
                </p>
              )}
              <hr />

              <div className="notebook-page__transcript">
                {detail.conversation.map((turn, i) => (
                  <div key={i} className={`notebook-page__turn notebook-page__turn--${turn.role}`}>
                    <p className="notebook-page__turn-role">{turn.role === "user" ? "You" : "Assistant"}</p>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
                    {turn.role === "assistant" && (
                      <>
                        <NotebookCitations citations={turn.citations ?? []} />
                        {(turn.suggested_questions ?? []).length > 0 && (
                          <div className="notebook-page__followups">
                            {(turn.suggested_questions ?? []).map((q, qi) => (
                              <button
                                type="button"
                                key={qi}
                                className="notebook-page__followup-button"
                                onClick={() => handleFollowup(q)}
                                disabled={chatStatus === "running"}
                              >
                                {q}
                              </button>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                ))}
              </div>

              <EvalResultPanel evalResult={lastEvalResult} />
              <RagReflectionPanel ragReflectionInfo={lastRagReflection} />

              {chatWarning && <p className="sr-warning">{chatWarning}</p>}

              {chatStatus !== "idle" && (
                <div className={`notebook-page__status notebook-page__status--${chatStatus}`} role="status">
                  {chatStatus === "running" && <span className="notebook-page__spinner" aria-hidden="true" />}
                  <span>{chatLabel}</span>
                </div>
              )}

              <div className="notebook-page__composer">
                <label htmlFor="nb-message">Message</label>
                <textarea
                  id="nb-message"
                  rows={2}
                  value={message}
                  onChange={(e) => {
                    setMessage(e.target.value);
                    setChatWarning(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Ask a question about your sources…"
                />
                <button
                  type="button"
                  className="sr-button"
                  onClick={handleSend}
                  disabled={chatStatus === "running"}
                >
                  Send
                </button>
              </div>
            </>
          )}
        </section>
      </div>
    </main>
  );
}
