# Frequently Asked Questions

> For installation instructions, see the [README](../README.md).
> For a full project description and architecture, see the [Overview](overview.md).

---

**Q: How do I run this with Docker?**
A: Run `docker compose up --build` from the project root. This starts two services: Ollama (LLM) and the Streamlit app (port 8501). A one-shot `model-init` container pulls the default model on first start. Open http://localhost:8501 in your browser. See the **Option A — Docker** section in the README for full details.

---

**Q: Does web search require a separate service or API key?**
A: No. Web search uses [DuckDuckGo](https://github.com/deedy5/ddgs) directly via the `ddgs` library — it runs in-process with no separate service, no API key, and no Docker container. If a DuckDuckGo query fails, web results are silently skipped and academic search (arXiv, Semantic Scholar, CrossRef) continues normally.

---

**Q: Does this send my documents to any server?**
A: No. Documents are processed entirely locally — parsed with Docling (or pdfplumber when Docling is disabled), embedded with a local Ollama model, stored in a local ChromaDB instance, and analysed by a local Ollama LLM. The only network calls are to arXiv, Semantic Scholar, and CrossRef — these receive only your search queries, not your documents.

---

**Q: Does Docling require a separate download or installation?**
A: Docling is included in requirements.txt and is enabled by default. On first use, it downloads its ML models (approximately 500 MB) to models/docling/ — this happens automatically. Subsequent runs use the cached models. To use the lightweight pdfplumber-only parser instead, toggle "Advanced Parsing (Docling)" off in the sidebar or pass --no-docling on the CLI.

---

**Q: Do I need to pull a separate embedding model for Hybrid RAG?**
A: Yes. Pull `nomic-embed-text` (274 MB) before the first document analysis run: `ollama pull nomic-embed-text`. If you skip this, the system automatically falls back to BM25 keyword search for all document modes and shows a warning banner in the UI — retrieval still works, but without dense vector ranking. You can use a different embedding model with `--embed-model mxbai-embed-large` (or via the sidebar dropdown).

---

**Q: Will the same document be re-embedded every time I upload it?**
A: No. Embeddings are cached in ChromaDB on disk (`outputs/chroma_db/`). On subsequent runs with the same document, the cached embeddings are used and Ollama is not called again for that content. Use `python main.py --clear-store` to free disk space.

---

**Q: Which Ollama model should I use?**
A: Run `python main.py --check-system` (CLI) or open the app (Streamlit) — the hardware detector will read your RAM and GPU type and recommend the best model that fits. As a rule of thumb: `llama3.2:3b` for machines with less than 8 GB RAM, `llama3.1:8b` for most laptops and desktops, `mistral-nemo:12b` when you need a 128k context window for long documents. On Apple Silicon, Ollama uses Metal GPU automatically so you can typically run one size larger than the RAM numbers suggest.

---

**Q: How do I handle very long documents?**
A: Increase the "Context window (tokens)" slider in the sidebar and use a model that supports a larger context (e.g. `mistral-nemo:12b` supports 128k tokens). Set `NUM_CTX=131072` in your `.env` to make 128k the default.

---

**Q: How many papers does ProposalGPT search?**
A: ProposalGPT (Mode 2) runs a 9-agent LangGraph pipeline. The `literature_review_agent` generates 5 targeted sub-queries and fetches up to 5 papers per query per source (arXiv + Semantic Scholar), giving typically 20–40 unique papers. The assembly node then selects only those papers that were actually cited in the text — usually 8–15 references.

---

**Q: Are the citations in the proposal real?**
A: Yes. The LLM is explicitly instructed not to invent citations. Every `[citation_key]` in the proposal corresponds to a paper that was retrieved from arXiv or Semantic Scholar during the search step. The reference list contains the real title, authors, year, journal, DOI, and URL for each cited paper.

---

**Q: Can I continue revising a proposal after closing the browser?**
A: Yes — this is the purpose of the long-term memory. Every ProposalGPT session is saved to the `proposal_sessions` table in `outputs/memory/sessions.db`. Open the app later, go to Mode 2 (ProposalGPT) → "Revise Existing", select your session, and issue a new instruction. On the CLI, use `--list-proposals` to see saved sessions and `--revise SESSION_ID` to resume one. The revision history is preserved indefinitely.

---

**Q: Can I restore a previous research run without re-running the workflow?**
A: Yes. Completed Literature Search (Mode 1) runs are automatically saved to the `research_sessions` table in `outputs/memory/sessions.db`. The sidebar shows your 5 most recent sessions — click any entry to restore the full report without re-running the workflow.

---

**Q: Does the Research Notebook use a vector database for document search?**
A: Yes — for dense retrieval it uses FAISS (in-memory) plus ChromaDB (for persistent embedding cache). Chunk text itself is stored in the `notebook_chunks` table in `outputs/memory/sessions.db`, separate from the notebook metadata in the `notebooks` table. This split means `list_notebooks()` never loads chunk text, keeping the sidebar fast. The notebook's 7-agent pipeline works from this structured session state with no additional setup required.

---

**Q: How does the Research Notebook remember what topics have been covered?**
A: After each Q&A turn, a micro LLM call extracts the names of concepts that were explained. These are stored in the `concepts_covered` list in the notebook's `meta_json` column in `sessions.db`. On the next turn, this list is injected into the system prompt so the LLM avoids re-explaining concepts from scratch. The full chat history and all source metadata are also persisted in the same database, enabling the notebook to pick up exactly where it left off across browser restarts.

---

**Q: What makes Wisdom Mode different from just asking the chatbot a question?**
A: Three things: (1) the agent first asks Socratic clarifying questions to understand your specific situation before searching, so it doesn't give generic advice; (2) every insight is grounded in real academic papers from arXiv and Semantic Scholar, with per-claim confidence ratings and a devil's advocate caveat so you know how solid the evidence is; (3) sessions persist across browser restarts and past sessions silently enrich future ones through topic-tag overlap — the agent builds a personalised knowledge graph over time.

---

**Q: What is the Socratic Clarification step?**
A: Before running certain modes, the assistant generates 2–3 focused questions tailored to your specific goal. A fast LLM call (`temperature=0.2`, `num_predict=512`) creates questions based on both your goal text and a mode-specific context. If the LLM call fails, hardcoded per-mode fallback questions are used automatically. Answers are injected into every prose-generating node's system prompt as a `USER CLARIFICATIONS` block.

---

**Q: Which modes use Socratic Clarification?**
A: Primarily Modes 1 and 3. Mode 1 (Literature Search) shows a collapsible "Clarify Requirements" form before the Run button. Mode 3 (Wisdom) uses a full Socratic clarification phase as the first step of its pipeline before any knowledge search occurs. Mode 2 (ProposalGPT) incorporates clarification within its `funding_call_analyzer` agent at the start of the 9-agent pipeline. Mode 4 (Systematic Review) uses explicit inclusion/exclusion criteria text fields instead of open Socratic questions. Mode 5 (Research Notebook) uses the ongoing session context and chat history rather than an upfront clarification form — follow-up questions are handled naturally within the Q&A interface.

---

**Q: How do I skip the clarification questions?**
A: In the UI, simply don't click "Clarify Requirements" — the form is optional and unanswered questions are silently skipped. In the CLI, pass `--no-clarify` to bypass the entire step:
```bash
python main.py --mode search --goal "..." --no-clarify
```

---

**Q: Does the Socratic Clarification step require an extra LLM call?**
A: Yes — one small call per run (or new session for modes that use session-based clarification). It uses `num_predict=512` so it completes in a few seconds. If the call fails for any reason, the system falls back to hardcoded questions immediately without user-visible errors. Runs where all questions are skipped incur zero overhead in the main workflow nodes.

---

**Q: How does cross-session memory work in Wisdom Mode?**
A: When generating new wisdom, the agent searches your past wisdom sessions for topic-tag overlap. If it finds related sessions (e.g. you previously explored "stress and sleep" and now ask about "concentration"), those sessions' key insights are silently injected into the synthesis prompt as background context. The agent doesn't name them explicitly — the influence is passive, making the new wisdom naturally richer without feeling like a history lecture.

---

**Q: What is the quality score that appears after every run?**
A: Each mode runs a **self-evaluation node** after its primary workflow completes. A micro LLM call (`temperature=0.1`, `num_predict=300`) scores the output on mode-specific dimensions (all 1–5):

| Mode | Dimensions |
|------|-----------|
| Mode 1 — Literature Search | Goal alignment · Evidence quality · Clarity |
| Mode 2 — ProposalGPT | Goal alignment · Objectives quality · Methodology soundness |
| Mode 3 — Wisdom | Evidence grounding · Confidence calibration · Actionability |
| Mode 4 — Systematic Review | Comprehensiveness · Evidence quality · Synthesis quality · Methodological rigour · Clinical utility |
| Mode 5 — Research Notebook (Q&A) | Answer grounding · Citation accuracy · Relevance |
| Mode 5 — Research Notebook (Pipeline) | Summary quality · Citation coverage · Study guide quality |
| Mode 6 — Grammar Proofreading | Polish quality · Context fit · Error coverage · Fluency |

In the Streamlit UI the score appears as a collapsed expander with a colour-coded badge and per-dimension metrics. In the CLI it renders as a Rich colour-coded table. The eval is entirely non-blocking — if the LLM call fails for any reason the primary output is unaffected and the score is simply omitted.

---

**Q: What is Self-Reflective RAG and how does it improve results?**
A: Self-Reflective RAG adds a relevance-grading step after every retrieval call. A single batched LLM call (`temperature=0.0`) reads all retrieved items at once and returns a `{"grades": [true/false, ...]}` array. Items graded `false` are filtered out before they reach the main LLM context window — so the LLM only sees evidence that is actually relevant to the query. For document-based modes (1 and 5), if fewer than 3 chunks pass grading the system automatically reformulates the query and runs a second retrieval cycle. For paper-based modes (2, 3, 4), a single-pass filter is applied. Mode 6 (Grammar Proofreading) has no external retrieval, so SR-RAG is a no-op for that mode (`rag_reflection_info = {}`). Any grading failure (LLM timeout, parse error, Ollama unreachable) silently falls back to returning all retrieved items unchanged — the pipeline always continues.

---

**Q: What is the difference between Hybrid RAG and Self-Reflective RAG?**
A: They solve different problems and are active in different modes.

**Hybrid RAG** (`tools/hybrid_store.py`) is a retrieval engine — it combines dense vector search (FAISS + Ollama embeddings) and sparse keyword search (BM25), fused with Reciprocal Rank Fusion. It answers the question *"how do we find relevant chunks from uploaded documents?"* It is only active in **Modes 1 and 5** (when documents are uploaded and `nomic-embed-text` is pulled). If no documents are uploaded, or the embedding model is missing, it falls back to BM25 keyword search.

**Self-Reflective RAG** (`agents/self_reflective_rag.py`) is a post-retrieval filter — it grades whatever was retrieved (chunks or papers) and removes irrelevant items before they enter the LLM context. It answers the question *"of the things we retrieved, which ones actually help answer the query?"* It is active in **the five retrieval modes (1–5)**: `grade_chunks()` for Modes 1 and 5, and `grade_papers()` for Modes 2, 3, and 4 (which retrieve academic papers via API, not HybridStore). Mode 6 (Grammar Proofreading) has no external retrieval and is excluded.

In practice, Modes 1 and 5 with uploaded documents use both — Hybrid RAG retrieves, then Self-Reflective RAG filters. Modes 2–4 use only Self-Reflective RAG (on API-sourced papers). Mode 6 uses neither.

---

**Q: Does Self-Reflective RAG slow down the pipeline?**
A: There is a small added latency — one extra LLM call per retrieval site (typically 1–3 seconds with a local Ollama model). The grading LLM uses a small context window (`num_ctx=4096`) and short output (`num_predict=100`) to stay fast. If a second retrieval cycle fires the overhead is approximately doubled for that query. The fallback (all grades True) is free.

Grading results are shown automatically after every run without any extra steps. **In the Streamlit UI** a collapsed "Self-Reflective RAG" expander appears directly below the quality score expander — it shows retrieved count, relevant count, pass rate, number of cycles, and any rewritten query. **In the CLI** a cyan Rich table is printed immediately after the quality score table with the same metrics. The raw metadata is also available in `state["rag_reflection_info"]` — it records `cycles`, `total_retrieved`, `total_relevant`, `rewritten_queries`, and `grading_skipped` for chunk-based modes, and `papers_retrieved` / `papers_after_grading` for paper-based modes.

---

**Q: Can I refine the output after it is generated?**
A: Yes — every mode has a built-in feedback refinement loop. After the pipeline completes, provide a plain-English instruction and the system will apply it to the output in a single LLM call (`temperature=0.4`). Up to 3 refinement rounds are allowed per session; each round is capped and the previous outputs are preserved in a revision history. In the Streamlit UI a collapsible "Refine this output" expander appears below the results. In the CLI, after displaying results the tool prompts `Feedback> ` — press Enter to skip.

---

**Q: What kinds of feedback work best for refinement?**
A: Specific, actionable instructions. Examples that work well: "Add more detail to the methodology section", "Simplify the language for a non-specialist audience", "Make the conclusions section stronger with more citations", "Expand the discussion of limitations". Vague feedback ("make it better") produces modest changes. The agent preserves all inline citations (`[1]`, `[Smith et al., 2023]`) and Markdown formatting unless the feedback explicitly requests otherwise.

---

**Q: Does feedback refinement affect the original saved report?**
A: No — refinements are stored separately. The original pipeline output is always saved at its standard path (e.g. `outputs/report_<session_id>.md`). If you refine via the CLI, the refined version is saved as `report_<session_id>_refined.md` alongside it. In the Streamlit UI the refined output is available as a separate download without overwriting the original. The revision history (all intermediate versions + feedback instructions) is held in browser session state and is not persisted to disk.

---

**Q: Is there a test suite?**
A: Yes. Run `python -m pytest tests/ -v` from the project root (install pytest first: `pip install pytest`). The suite has **496 tests** covering: citation export (BibTeX/RIS format, key collision handling), memory CRUD, Socratic clarification fallback questions, all state factory functions, graph-level integration tests across all six modes, self-evaluation nodes, wisdom routing regex, node-level streaming, explicit proposal memory saver, word-level cross-session tag overlap, and Grammar Proofreading nodes/graph/memory (36 tests). All run fully offline — no Ollama, no network.

---

**Q: How is the BibTeX citation key generated?**
A: The key is `<first_token_of_first_author_lastname><year>`. For example, `"Vaswani A"` → first token is `"Vaswani"` → key is `vaswani2017`. If two papers share the same key, a suffix is appended: `smith2023`, `smith2023a`, `smith2023b`, etc. The key is generated in `_make_bibtex_key()` in `tools/citation_tools.py` using `first_author.split()[0]` — the **first** space-separated token (the last name in `"Lastname Initials"` format).

---

**Q: Why does importing `tools.citation_tools` not require tenacity, faiss, or chromadb?**
A: `tools/__init__.py` uses Python's `__getattr__` hook for lazy loading — no submodule is imported until the name is actually accessed. When you `from tools.citation_tools import refs_to_bibtex` directly, you bypass the `tools` package entirely and only load the pure-stdlib citation module. If you access, say, `tools.OllamaEmbedder`, only then does the embeddings submodule (which pulls in faiss and chromadb) get imported. This design keeps the citation tests fast and dependency-free.

---

**Q: Why do the wisdom and proposal node files use `_get_memory()` instead of a module-level singleton?**
A: Creating the memory object at module level (`_memory = ResearchMemory()`) would create `outputs/memory/` as a side effect of importing the module — including during tests. The lazy getter pattern (`_memory: T | None = None`, populated on first call) delays creation until the object is actually needed, and lets tests inject an isolated memory object via `monkeypatch.setattr` before running the graph. All node files use this pattern.

---

**Q: How are the graphs integration-tested without a running Ollama?**
A: Each integration test file patches `ChatOllama` (and singleton searchers like `_academic`, `_web`) with `unittest.mock.patch` / `monkeypatch.setattr`. Real memory objects backed by `tmp_path` are injected so persistence is tested. Tests cover all five modes end-to-end: Literature Search (arXiv + Semantic Scholar search paths), ProposalGPT (new-proposal and revision paths through the 9-agent pipeline), Wisdom (clarification path, forced-proceed knowledge-search path, and follow-up path), Systematic Review (full PRISMA pipeline), and Research Notebook (ingestion, Q&A, and study guide generation).

**Q: What happens when the recommended model uses >= 85% of available RAM?**
A: The hardware detector flags this as a "tight fit" and surfaces two choices instead of one. In the Streamlit sidebar, a warning banner and a radio button appear — one option for the higher-capability tight-fit model, another for a smaller safe-headroom alternative. In the CLI, both options are listed and the user types `1` or `2` to choose. The chosen model is applied to the session; no guessing occurs.

---

**Q: How is the Streamlit app structured internally?**
A: `app.py` shows a **landing page** (`ui/landing.py`) with **six mode cards**. Clicking "Launch" sets `st.session_state["active_project"]` and the app lazy-imports only that mode's `projects/modeN_*.py` module. This keeps startup fast — unused modes are never imported. The sidebar (`ui/sidebar.py`) shows hardware detection, model settings, and style profiles regardless of which mode is active. Each mode's tab logic lives in `ui/tabs/<name>.py`. The main workflow runs in a background thread so the Streamlit UI stays responsive during long research jobs. Adding a new mode means creating `ui/tabs/<name>.py`, a `projects/modeN_<name>.py` with a `run(settings)` function, and a card in `ui/landing.py`.

---

**Q: Can I export citations to Zotero or Mendeley?**
A: Yes. Click "Export BibTeX (.bib)" or "Export RIS (.ris)" on any References tab. Both formats are universally supported by reference managers. The `.bib` file can also be used directly with LaTeX (`\bibliography{references}`).

---

**Q: Does this work on Apple Silicon Macs (M1/M2/M3/M4)?**
A: Yes, and it's one of the best platforms for this project. Ollama automatically uses Metal GPU and the Neural Engine on Apple Silicon — no configuration needed. The hardware detector identifies your chip and adjusts the RAM budget to use unified memory efficiently. An M2 Pro with 16 GB, for example, can comfortably run `llama3.1:8b`. Run `python main.py --check-system` to see the recommendation for your specific chip.

---

**Q: Does this work on machines without a GPU?**
A: Yes — Ollama runs on CPU-only machines. Performance will be slower (expect 5–30 tokens/second for 8B models on a modern CPU vs. 50–100 tokens/second on a mid-range GPU). For CPU use, the hardware detector will automatically recommend a smaller model like `llama3.2:3b` that fits comfortably in your available RAM.

---

**Q: What if the PDF or DOCX export fails?**
A: The Markdown version is always saved to `outputs/proposal_<id>.md` first, so you never lose the content. DOCX requires `python-docx` and PDF requires `reportlab` — both are listed in `requirements.txt`. Run `pip install reportlab python-docx` if they are missing.

---

**Q: What is a Writing Style Profile and how does it work?**
A: A Writing Style Profile captures your personal writing style from 2–5 of your own documents (past papers, reports, grant proposals) and applies it to AI-generated prose. The system runs two LLM calls: one to extract four style dimensions (tone/formality, structure/format, vocabulary/complexity, citation style), and one to synthesise a compact instruction block (~280 words). That block is then appended to the system prompt of every prose-generating node in the affected modes — no model fine-tuning, no extra dependencies, no GPU required.

---

**Q: Does enabling a Style Profile slow down research runs?**
A: Negligibly. The profile's instruction block (~280 words) is simply appended to existing system prompts, adding roughly 300 tokens per LLM call. The analysis itself (two LLM calls) runs only once when you create the profile. After that, using the profile adds no extra LLM calls — only a slightly longer prompt.

---

**Q: Can I have multiple Style Profiles for different writing contexts?**
A: Yes. Create as many named profiles as you need (e.g. "Academic Papers", "Grant Proposals", "Industry Reports"). Each is stored in the `style_profiles` table in `outputs/memory/sessions.db`. Select the active profile from the sidebar dropdown in the UI, or pass `--style-profile "Name"` on the CLI. Only one profile is active at a time.

---

**Q: Which modes are affected by the Writing Style Profile?**
A: Mode 1 (Literature Search) applies the style block to its report-generation nodes, so your exported literature reviews and summaries match your writing voice. Mode 2 (ProposalGPT) is **not** affected — it has its own structured prompt architecture across its 9 agents (`funding_call_analyzer`, `research_planner`, `literature_review_agent`, `proposal_writer`, `impact_agent`, `budget_agent`, `compliance_agent`, `reviewer_agent`, `improvement_agent`) which overrides any global style injection. Mode 3 (Wisdom) and Mode 4 (Systematic Review) use fixed analytical prose styles by design. Mode 5 (Research Notebook) uses conversational tone for Q&A and structured templates for study guides and summaries.

---

**Q: Can I target a specific funding agency for my proposal?**
A: Yes. ProposalGPT (Mode 2) has built-in agency templates: Horizon Europe, Vinnova, SSF, and VR Swedish Research Council. Each template adjusts the proposal's tone, word-count targets, and required section notes for that funder. Pass `--funding-agency "Horizon Europe"` (or any of the supported agency names) on the CLI, or expand the **Funding Agency & Scope** panel in the Mode 2 tab to select an agency and optionally set budget range, project duration, consortium size, and TRL level. The `budget_agent` also auto-detects indirect cost rates from the agency: Horizon Europe/ERC/MSCA → 25% indirect, Swedish agencies (Vinnova/SSF/VR) → 20%, Generic → 25%. Omit the flag for a general-purpose proposal.

---

**Q: Can I export my Research Notebook session?**
A: Yes. When a Research Notebook (Mode 5) session is active, download buttons are available in the session panel for the full conversation transcript, the generated study guide, and the source summaries — all in Markdown or Word (.docx) format. The underlying session state (sources, chat history, knowledge graph, concepts covered) is persisted in `outputs/memory/sessions.db` and can be resumed at any time.

---

**Q: What happens when I re-upload a document with updated content?**
A: The system computes an MD5 hash of the uploaded content and compares it to the hash stored in ChromaDB for that filename. If they differ, the stale embeddings for that document are automatically removed from the cache before re-embedding. This means you always get retrieval results that reflect the current version of the file, without needing to manually run `--clear-store`.

---

**Q: Is there a word limit for Grammar Proofreading (Mode 6)?**
A: There is no hard word limit. The practical limit is determined by the model's context window (set via the "Context window (tokens)" slider in the sidebar, or `--num-ctx` on the CLI). If the estimated character count of your input text exceeds approximately 60% of the configured context window, Mode 6 adds an informational warning to the output but does **not** truncate your text — the model's own context window governs what fits. For very long documents, use a model with a large context window such as `mistral-nemo:12b` (128k tokens) and set `--num-ctx 131072`.

---

**Q: Which file types does Grammar Proofreading accept?**
A: With Docling enabled by default, Mode 6 accepts PDF, DOCX, TXT, MD, PPTX, XLSX, HTML, and image formats via the file uploader in the UI, or via `--files` on the CLI. If Docling is disabled (via `--no-docling` or the sidebar toggle), only PDF, DOCX, TXT, and MD are supported, using pdfplumber and python-docx. You can also paste text directly into the text area (UI) or pass it via `--goal` (CLI).

---

**Q: Does Grammar Proofreading rewrite my entire text?**
A: Yes — the polished text is a **complete rewrite** optimised for clarity, fluency, and correctness for the chosen writing context. This goes beyond mechanical error correction: the agent restructures sentences, removes redundancy, and adjusts register. The original text is always preserved in the session state and shown in the Issues tab as the baseline for comparison. Download buttons for the polished output (Markdown and TXT) are always visible; for file-upload inputs they appear prominently at the top of the Polished Text tab.

---

**Q: Can I give feedback on the polished text?**
A: Yes. In the Streamlit UI, a **Refine this output** feedback box appears below the output tabs — revision rounds are unlimited, so you can iterate as many times as needed. Type your instruction (e.g. "Make the tone less formal" or "Tighten the second paragraph") and click Refine — the agent produces a revised version that incorporates your feedback while maintaining the chosen writing context. In the CLI, an interactive `Feedback>` prompt appears after the initial output; up to 3 interactive rounds are supported. Press Enter to skip.

---

**Q: Can I download the proofread output?**
A: Yes. In the Streamlit UI, two download buttons always appear in the **Polished Text** tab: "Download as Markdown (.md)" and "Download as TXT (.txt)". For file-upload inputs, these buttons are displayed at the top of the tab for easy access. In the CLI, the polished text is automatically saved to `outputs/grammar_<session_id>.md` after the initial run.

---

**Q: Does the Systematic Review search Google Scholar, and does it need an API key?**
A: Yes — Google Scholar is searched **first** (as the primary source) using the `scholarly` library, which scrapes Google Scholar directly with no API key and no cost. Because it scrapes HTML, it can occasionally be rate-limited; when that happens the screener silently skips Google Scholar and continues with arXiv, Semantic Scholar, and CrossRef. Install the library with `pip install scholarly` — it is listed in `requirements.txt`. The `--systematic-review` CLI flag automatically uses all four sources.

---

**Q: What is the Abstract Screener in Mode 4?**
A: The abstract screener is an LLM-powered pre-ranking step that runs immediately after literature search, before the formal inclusion/exclusion screening node. It scores each candidate paper 0–100 against your research question and returns a verdict (include / uncertain / exclude) with a short rationale. Papers are sorted by score so the formal screener sees the most relevant ones first. If the screener LLM call fails for any reason, the pipeline continues unaffected with an empty `screener_scores` list.

---

**Q: How does the Citation Network work in Mode 4?**
A: After synthesis, you can build an ego-only citation network — edges are drawn only between the papers that were included in your review (no 1-hop expansion to external papers). The tool queries the Semantic Scholar API to find which included papers cite each other, then renders an interactive Pyvis HTML graph where node colour reflects quality rating (green = High, amber = Medium, red = Low). Network statistics (most-cited, most-citing, isolated nodes) are shown alongside the graph. In the UI, click **Build Citation Network** in the Discovery tab. In the CLI, add `--sr-citation-network` to the run command.

---

**Q: What does the Preprint Tracker do?**
A: For each paper in the review, the preprint tracker cross-references CrossRef to determine its publication status: **journal** (published in a peer-reviewed journal with a DOI), **published** (non-arXiv paper confirmed via CrossRef), **preprint** (arXiv paper not yet found in CrossRef), or **retracted** (CrossRef `update-policy` or `is-retracted-by` relation present). Non-arXiv papers with a DOI are classified immediately without an API call. A summary count by status is shown in the UI Discovery tab or printed as a Rich table in the CLI (`--sr-preprints`).

---

**Q: Can I get a PRISMA-compliant Word or PDF report from Mode 4?**
A: Yes. After the review pipeline completes, use the **Export & Reports** tab in the UI (or `--sr-docx` / `--sr-pdf` on the CLI) to generate a ready-to-submit document. The DOCX is built with python-docx and the PDF with reportlab — no LibreOffice required. Both follow the PRISMA 2020 section structure: Title page, Abstract, Introduction, Methods (Search Strategy, Eligibility Criteria, Data Extraction, Quality Assessment), Results (Study Selection with PRISMA flow table, Study Characteristics, Evidence Table, Narrative Synthesis), Discussion (Summary, Research Gaps, Limitations, Conclusions), and References. Pass `--sr-author` and `--sr-institution` to personalise the title page.

---

**Q: What are the plain-language summary formats in Mode 4?**
A: Three formats are available, each generated by a targeted LLM prompt from the completed review state:
- **Patient summary** — approximately 350 words at an 8th-grade reading level; four paragraphs covering what was studied, what was found, what it means for patients, and what questions remain.
- **Policy brief** — Markdown with six fixed headers: Executive Summary, Background, Key Findings, Evidence Quality, Policy Recommendations, and Research Gaps.
- **Press release** — inverted-pyramid structure with a news-style headline, dateline, opening paragraph stating the main finding, supporting context, a fictional expert quote, and a boilerplate close.

In the UI select the format from the radio buttons in the **Export & Reports** tab. On the CLI use `--sr-plain-language patient`, `--sr-plain-language policy`, `--sr-plain-language press`, or `--sr-plain-language all`.

---

**Q: How does Trend Analysis work in Mode 4?**
A: The trend analyzer queries the CrossRef facet API (`facet=published:*`) to retrieve field-wide year-by-year publication counts for the review's search queries — not just the papers retrieved locally. When CrossRef returns fewer than 30 records for a query, Semantic Scholar is queried as a supplement. Trend direction is classified by comparing the 3-year recent average against the 3-year earlier average: ratio > 1.2 = growing, < 0.8 = declining, else stable. Results are displayed as a Plotly line chart in the **Trends & Analysis** UI tab. On the CLI, add `--sr-trends`.

---

**Q: What is the Evidence Map in Mode 4?**
A: The evidence map is a Plotly bubble chart that visualises where in the Population × Intervention space your included papers cluster. Each bubble represents a unique (Population, Intervention) pair, sized by paper count and coloured by mean quality rating. If PICO fields are not present in the evidence table, the tool falls back to using `study_design` as a proxy for both axes. A matplotlib PNG is generated as fallback when Plotly is unavailable. The evidence map appears in the **Trends & Analysis** tab of the UI and in the CLI output when `--sr-trends` is passed.

---

**Q: What is Concept Drift detection in Mode 4?**
A: Concept drift analysis groups included papers into 5-year buckets and computes TF-IDF keyword rankings for each bucket using only Python's stdlib (no scikit-learn). Keywords that rise 3 or more rank positions across buckets are classified as **rising**; those that fall 3 or more as **declining**; others as **stable**. An optional LLM call produces a prose narrative describing the detected shifts. This helps you understand how the field's vocabulary and focus have evolved over time. Enable it in the **Trends & Analysis** tab or with `--sr-concept-drift` on the CLI.
