# Agentic Research Assistant — Project Overview

> Detailed description of the project, its architecture, and all six modes.
> For installation and running instructions, see the [README](../README.md).
> For CLI usage and examples, see the [Tutorial](tutorial.md).
> For common questions, see the [FAQ](FAQ.md).

---

## What is this?

An **agentic AI workflow** that acts as a full research companion — it can:

1. **Search peer-reviewed literature** (arXiv, Semantic Scholar, CrossRef) with automatic citation formatting and synthesise a structured report
2. **Generate publication-quality grant proposals** with a 9-agent LangGraph pipeline — funding call analysis, win strategy, literature review, 20 proposal sections, budget, compliance checking, virtual reviewer simulation, and iterative improvement
3. **Transform knowledge into wisdom** — describe any life or scientific scenario and get validated wisdom: a rigorous scientific explanation, a plain-language version, actionable steps, and claim-level confidence scores with a devil's advocate caveat — all backed by real academic sources
4. **Run a simplified systematic review** — automate a PRISMA-style literature review: generate queries from your research question, screen papers against inclusion/exclusion criteria, extract structured evidence, and synthesise themes, research gaps, and a conclusion
5. **Ask questions about your own documents** in a NotebookLM-style workspace — upload PDFs, papers, or web pages and get answers grounded strictly in those sources, with inline citations to document and page
6. **Proofread and polish English text** — paste or upload any document and get a professionally rewritten version suited to your writing context (Academic, Professional Email, Formal, or Informal), with per-error explanations, style tips, and iterative feedback refinement

**Additional capabilities across all modes:**
- Upload multiple documents simultaneously for cross-document comparative analysis
- Export any reference list to **BibTeX** (`.bib`) or **RIS** (`.ris`) for Zotero, Mendeley, and JabRef
- Save and restore research sessions — completed analyses persist and reload from the sidebar
- **Automatic hardware detection** recommends the best model and context window for your machine (Apple Silicon Metal, NVIDIA CUDA, or CPU-only) — the sidebar only shows models you've actually downloaded
- **Self-Reflective RAG** — a post-retrieval relevance filter active in all 5 retrieval modes: a single batched LLM call grades every retrieved item (document chunks or academic papers) and removes irrelevant evidence before it reaches the main LLM context window; grading results are shown in the UI and CLI output. Mode 6 (Grammar Proofreading) does not retrieve external content, so `rag_reflection_info` is always `{}` and the SR-RAG display is a no-op

Everything runs **100% locally** — no OpenAI, no cloud, no API fees.

---

## What is Hybrid RAG?

**Hybrid RAG** combines two complementary retrieval signals — dense semantic search and sparse keyword search — then fuses their results to find the most relevant document chunks for each query.

```
Document -> chunks -> Ollama embed -> FAISS (dense)  --+
                   -> BM25 index  (sparse) -----------+--> RRF fusion -> top-k chunks -> LLM
```

### The three components

| Component | Role | Why it matters |
|-----------|------|----------------|
| **FAISS** (dense) | Semantic vector search using Ollama embeddings | Finds conceptually related passages even without exact word matches |
| **BM25** (sparse) | Keyword frequency search (rank-bm25) | Catches exact terms, acronyms, and technical jargon that dense search misses |
| **RRF** (fusion) | Reciprocal Rank Fusion merges the two ranked lists | Score-normalisation-free; empirically the best-performing fusion strategy |

### How it compares

| | Pure Vector RAG | Vectorless | **Hybrid RAG (this project)** |
|--|-----------------|------------|-------------------------------|
| **Relevance** | Good (semantic only) | Full doc (no selection) | Best (semantic + keyword) |
| **Exact-term match** | Weak | Yes (full text) | Yes (BM25) |
| **Context efficiency** | Top-k chunks | Entire excerpts | Top-k, precisely ranked |
| **Persistent cache** | Yes | No | Yes (ChromaDB) |
| **Setup** | Embedding model | Nothing extra | Embedding model |

### Persistence and caching

Embeddings are stored in **ChromaDB** on disk (`outputs/chroma_db/`). When you upload the same document a second time, embeddings are retrieved from the cache — Ollama is not called again. This makes repeat analyses and sessions fast.

**FAISS** is rebuilt in-memory each session for ultra-fast retrieval within a run.

### Retrieval flow per query

```
Search query (from Node 2)
  +- embed query -> FAISS -> top-2k dense results
  +- tokenise   -> BM25  -> top-2k sparse results
                          |
              RRF(k=60): score = sum 1/(60 + rank_i)
                          |
              Top-k unique chunks (default k=8)
                          |
              Self-Reflective Grading
                Single LLM call: {"grades": [true, false, ...]}
                Irrelevant items removed; < 3 pass -> rewrite + cycle 2
                          |
              Relevant chunks -> LLM context + display in UI
```

All queries from Node 2 are searched, results are deduplicated by `chunk_id`, and the total context is capped at ~50% of the model's context window. After retrieval, `grade_chunks()` (Self-Reflective RAG) filters out irrelevant chunks; see the dedicated **Self-Reflective RAG** section below for full details.

### Fallback mode

If the Ollama embedding model is not pulled, the system automatically falls back to **BM25 keyword search** — document retrieval still works using keyword matching, just without dense vector ranking. A warning appears in the UI.

```bash
# Enable full hybrid RAG — pull the embedding model first
ollama pull nomic-embed-text
```

---

## Self-Reflective RAG

**Self-Reflective RAG** (Corrective/Adaptive variant) adds a **post-retrieval relevance filter** to every mode. After each retrieval call — whether from HybridStore (document chunks) or an academic API (papers) — a single batched LLM call grades all retrieved items for relevance before they reach the main LLM context window. Irrelevant items are filtered out; only graded-relevant evidence enters the prompt.

This is distinct from Hybrid RAG: Hybrid RAG determines *how* items are retrieved; Self-Reflective RAG determines *which* retrieved items are actually relevant to the query.

### Coverage across retrieval modes

| Mode | What is graded | Grading function | Cycles |
|------|----------------|-----------------|--------|
| Mode 1 — Literature Search | Document chunks (HybridStore) | `grade_chunks()` | Up to 2 (with query rewrite) |
| Mode 2 — ProposalGPT | Academic papers (arXiv + SS) | `grade_papers()` | One-pass |
| Mode 3 — Wisdom | Academic papers (arXiv + SS) | `grade_papers()` | One-pass |
| Mode 4 — Systematic Review | Academic papers (pre-screening) | `grade_papers()` | One-pass |
| Mode 5 — Research Notebook | Document chunks (HybridStore) | `grade_chunks()` | Up to 2 (with query rewrite) |
| Mode 6 — Grammar Proofreading | *(no retrieval)* | *(not applicable)* | — |

### Two grading paths

**Chunk-based modes (1 and 5)** use `self_reflective_retrieve()`, which orchestrates up to 2 retrieval cycles:

```
retrieve chunks (Hybrid RAG -- FAISS + BM25 + RRF)
    |
    v
grade_chunks()  <-  single LLM call, temperature=0.0
  Returns {"grades": [true, false, ...]}
    |
    +- >= 3 relevant -> pass to LLM context
    |
    +- < 3 relevant?
         |
         v
       rewrite_query()  <-  temperature=0.3, reformulates the query
         |
         v
       cycle 2 retrieval -> grade new chunks only
       merge + deduplicate by chunk_id
         |
         v
       pass merged relevant chunks to LLM context (capped at top_k)
```

**Paper-based modes (2, 3, 4)** use `grade_papers()` directly as a one-pass filter on the academic API results — no cycle logic because API search is already global:

```
academic API search (arXiv + Semantic Scholar)
    |
    v
grade_papers()  <-  single LLM call, temperature=0.0
  Input: title + first 300 chars of abstract per paper
  Returns {"grades": [true, false, ...]}
    |
    v
filtered papers -> LLM context  (fallback: original list if all filtered out)
```

### Grading mechanics

- **LLM call:** `temperature=0.0`, `num_predict=100`, `num_ctx=min(num_ctx, 4096)` — fast and deterministic
- **Prompt:** numbered list of items (chunks truncated to 400 chars; papers as title + 300-char abstract)
- **Response:** `{"grades": [true, false, true, ...]}` — one bool per item, parsed with `re.search(r"\{.*\}", response, re.DOTALL)`
- **Grading skipped detection:** if all items grade `true` on a batch of more than one item, `grading_skipped=True` is set and original items are returned (silent LLM failure guard)
- **Minimum threshold (chunk modes):** if fewer than 3 items pass, cycle 2 fires

### Fallback guarantees

The pipeline **never blocks** due to grading:

| Failure condition | Behaviour |
|------------------|-----------|
| LLM call raises any exception | All items treated as relevant |
| JSON parse error | All items treated as relevant |
| Wrong array length returned | All items treated as relevant |
| All items grade `true` on a multi-item list | `grading_skipped=True`; original items returned |
| Cycle 2 yields no new relevant items | Cycle-1 results returned |
| Empty input list | Returns empty list immediately (no LLM call) |

### Where results appear

Grading metadata is surfaced automatically in both interfaces after every run.

**Streamlit UI** — a collapsed "Self-Reflective RAG" expander appears directly below the quality score expander:
```
Self-Reflective RAG -- 6/8 items passed grading (75%)
  [ Retrieved: 8 ]  [ Relevant: 6 ]  [ Pass Rate: 75% ]
  Cycles: 1
```
If cycle 2 fired, the rewritten query is shown inside the expander. If grading was skipped, a warning note appears.

**CLI** — a cyan Rich table is printed immediately after the quality score table:
```
+------------------ Self-Reflective RAG ------------------+
| Metric          | Value                                  |
| Retrieved       | 8                                      |
| Passed grading  | 6                                      |
| Pass rate       | 75%                                    |
| Cycles          | 1                                      |
+--------------------------------------------------------+
```

### State field

Grading metadata is stored in `state["rag_reflection_info"]`:

- **Chunk-based modes (1, 5):** `List[Dict]` — one entry per query, each with `query`, `cycles`, `total_retrieved`, `total_relevant`, `rewritten_queries`, `grading_skipped`
- **Paper-based modes (2, 3, 4):** `Dict` — `papers_retrieved`, `papers_after_grading`

The field defaults to `[]` / `{}` and is never absent from the state — both UI helpers and CLI printers are no-ops when it is empty (e.g. BM25 fallback path).

---

## Architecture Overview

### Literature Search (Mode 1)

```
                     +-------------------------------------+
                     |         USER INPUT                   |
                     |  Goal + Documents (1 or more)        |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |      LangGraph State Machine         |
                     |                                      |
                     |  [1] Document Ingestion              |
                     |       +- Docling (default) /         |
                     |          pdfplumber (--no-docling)   |
                     |       +- Validate + log char count   |
                     |                                      |
                     |  [2] Query Generation (Ollama LLM)   |
                     |       +- Decomposes goal -> queries  |
                     |                                      |
                     |  [3] Academic Search                 |
                     |       +- arXiv API                   |
                     |       +- Semantic Scholar API        |
                     |       +- CrossRef API (optional)     |
                     |                                      |
                     |  [4] Web Search (optional)           |
                     |       +- DuckDuckGo (ddgs, in-proc)  |
                     |                                      |
                     |  [5] Document Analysis               |
                     |       +- Hybrid RAG retrieval        |
                     |          FAISS (dense) + BM25 (sparse|
                     |          -> RRF -> top-k chunks       |
                     |       +- SR-RAG: grade_chunks()       |
                     |          Batch LLM relevance filter   |
                     |          (cycle 2 + rewrite if < 3   |
                     |          pass) -> relevant chunks->LLM|
                     |       +- Per-doc analysis (multi)    |
                     |       +- Cross-doc synthesis (multi) |
                     |                                      |
                     |  [6] Reference Compilation           |
                     |       +- Dedup + APA formatting      |
                     |                                      |
                     |  [7] Report Generation               |
                     |       +- Markdown report + citations |
                     |       +- Per-Document Breakdown      |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |         OUTPUT                       |
                     |  - Executive Summary                  |
                     |  - Key Findings (bullet points)      |
                     |  - Per-Document Breakdown (multi)    |
                     |  - Detailed Analysis                  |
                     |  - Mapped References (APA)           |
                     |  - BibTeX / RIS export               |
                     |  - Downloadable Markdown Report      |
                     |  - Session saved to sidebar          |
                     |  - Optional: PRISMA Systematic       |
                     |    Review (checkbox before Run)      |
                     +--------------------------------------+
```

### Progress visibility

Every pipeline node returns a `status_detail` string that is displayed in both the UI and the CLI directly below the progress percentage. This gives a real-time description of what the pipeline has just completed — for example, "14 chunks indexed · FAISS + BM25 + RRF" after the document ingestion node, or "23 papers via arXiv · Semantic Scholar · CrossRef" after the academic search node. The CLI prints each `status_detail` line alongside the percentage bar, and the Streamlit UI updates the status text live as each node finishes.

### ProposalGPT (Mode 2)

```
                     +-------------------------------------+
                     |         USER INPUT                   |
                     |  Funding call (file/URL/paste)       |
                     |  Researcher ideas + CVs/pubs (opt.) |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |   ProposalGPT LangGraph (9 agents)   |
                     |                                      |
                     |  [1] Funding Call Analyzer           |
                     |       +- Extracts objectives, criteria|
                     |       +- Budget constraints, keywords |
                     |       +- Compliance checklist         |
                     |                                      |
                     |  [2] Research Planner                |
                     |       +- Win strategy, SWOT analysis  |
                     |       +- Reviewer perspective         |
                     |       +- Risk analysis                |
                     |                                      |
                     |  [3] Literature Review Agent         |
                     |       +- arXiv + Semantic Scholar     |
                     |       +- SR-RAG: grade_papers()       |
                     |          Filters irrelevant papers     |
                     |       +- State of art, research gaps  |
                     |                                      |
                     |  [4] Proposal Writer                 |
                     |       +- 14 core sections (exec      |
                     |          summary -> data management) |
                     |       +- Work packages, deliverables  |
                     |          & milestones (JSON tables)  |
                     |                                      |
                     |  [5] Impact Agent                    |
                     |       +- Impact, dissemination,      |
                     |          exploitation, ethics,       |
                     |          sustainability              |
                     |                                      |
                     |  [6] Budget Agent                    |
                     |       +- Personnel/equipment/travel  |
                     |       +- Indirect costs (25%/20%)    |
                     |       +- Auto-scaling if over budget  |
                     |                                      |
                     |  [7] Compliance Checker              |
                     |       +- Missing sections check      |
                     |       +- Keyword coverage scoring    |
                     |       +- 0-100 compliance score      |
                     |                                      |
                     |  [8] Reviewer Simulation             |
                     |       +- 5 virtual reviewers         |
                     |          (scientific 30%, impact 25%,|
                     |          innovation 20%, impl 15%,   |
                     |          agency 10%)                 |
                     |       +- Weighted 1-5 overall score  |
                     |                                      |
                     |  [9] Improvement Agent               |
                     |       +- Identifies weak sections    |
                     |       +- Rewrites top 2 sections     |
                     |       +- Improvement plan            |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |   Session Memory (SQLite)            |
                     |   outputs/memory/sessions.db         |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |         OUTPUT (8 tabs)              |
                     |  Funding Call Analysis               |
                     |  Win Strategy & SWOT                 |
                     |  Literature Review                   |
                     |  Draft Proposal (17 sections)        |
                     |  Reviewer Simulation (1-5)           |
                     |  Compliance Report                   |
                     |  Export (MD, DOCX, PDF, CSV)         |
                     |  Improvement Plan                    |
                     +--------------------------------------+
```

### Wisdom Mode (Mode 3)

```
                     +-------------------------------------+
                     |         USER INPUT                   |
                     |  Scenario / Question + optional doc  |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |   Wisdom LangGraph (7 nodes)         |
                     |                                      |
                     |  [1] Context Loader                  |
                     |       +- Load history + doc context  |
                     |       +- Find related sessions       |
                     |          (passive topic-tag overlap) |
                     |                                      |
                     |  [2] Clarification (temp=0.6)        |
                     |       +- Socratic Q&A (max 3 Qs)    |
                     |       +- Decides when to proceed     |
                     |                                      |
                     |  [3] Knowledge Search                |
                     |       +- arXiv + Semantic Scholar    |
                     |       +- DuckDuckGo (optional)       |
                     |       +- SR-RAG: grade_papers()       |
                     |          Filters irrelevant papers     |
                     |                                      |
                     |  [4] Wisdom Synthesis (temp=0.5)     |
                     |       +- Scientific explanation      |
                     |       +- Simple explanation (ELI)   |
                     |       +- Actionable takeaways        |
                     |       +- Topic tags extracted        |
                     |                                      |
                     |  [5] Wisdom Validator (temp=0.2)     |
                     |       +- Per-claim confidence score  |
                     |       +- Consensus / Emerging label  |
                     |       +- Devil's advocate caveat     |
                     |                                      |
                     |  [6] Wisdom Follow-up (post-gen Q&A) |
                     |  [7] Memory Saver                    |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |   Long-Term Memory (SQLite)          |
                     |   outputs/memory/sessions.db         |
                     |   Stores: conversation, wisdom,      |
                     |           papers, topic_tags         |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |         OUTPUT (4 tabs)              |
                     |  Scientific explanation + cites      |
                     |  Plain-language version              |
                     |  Evidence-based action steps         |
                     |  Confidence scores + devil's adv.    |
                     |  "Run PRISMA Review on this          |
                     |    topic" button in left panel       |
                     +--------------------------------------+
```

### Systematic Review (Mode 4)

Mode 4 is available as a **standalone tab** and is also accessible from **every other mode** as an optional add-on.

```
                     +-------------------------------------+
                     |         USER INPUT                   |
                     |  Research question                   |
                     |  Inclusion / Exclusion criteria      |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |   Systematic Review LangGraph        |
                     |   (6 nodes, linear pipeline)         |
                     |                                      |
                     |  [1] Query Generation                |
                     |       +- 4-6 varied search queries   |
                     |                                      |
                     |  [2] Literature Search               |
                     |       +- arXiv + Semantic Scholar    |
                     |       +- Dedup + sort by citations   |
                     |       +- SR-RAG: grade_papers()       |
                     |          Pre-screening filter         |
                     |          reduces Screening workload   |
                     |                                      |
                     |  [3] Screening                       |
                     |       +- Per-paper INCLUDE/EXCLUDE   |
                     |       +- Records exclusion reason    |
                     |                                      |
                     |  [4] Evidence Extraction             |
                     |       +- Study design, sample size   |
                     |       +- Key finding, quality rating |
                     |                                      |
                     |  [5] Synthesis                       |
                     |       +- PRISMA flow counts          |
                     |       +- Narrative synthesis         |
                     |       +- Key themes + research gaps  |
                     |       +- Conclusion + limitations    |
                     |                                      |
                     |  [6] Self-Evaluation                 |
                     |       +- 5-dimension quality score   |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |         OUTPUT                       |
                     |  - PRISMA flow metrics               |
                     |  - Evidence table (per paper)        |
                     |  - Narrative synthesis + themes      |
                     |  - Research gaps + conclusion        |
                     |  - Downloadable Markdown report      |
                     +--------------------------------------+
```

**Integration into all modes:**

| Mode | How to trigger SR |
|------|------------------|
| Mode 1 — Literature Search | Tick **"Also run Systematic Review"** checkbox before Run |
| Mode 2 — ProposalGPT | Tick **"Also run Systematic Review on this proposal topic"** checkbox before Generate |
| Mode 3 — Wisdom Mode | Click **"Run PRISMA Review on this topic"** button in the left panel |
| Mode 4 (standalone) | Use the dedicated **Mode 4: Systematic Review** tab directly |
| Mode 5 — Research Notebook | Available from within the notebook via dedicated pipeline run |

**CLI (with any research mode):**
```bash
python main.py --mode search --goal "Effect of sleep deprivation on working memory" \
  --with-systematic-review
```

### Grammar Proofreading (Mode 6)

```
                     +-------------------------------------+
                     |         USER INPUT                   |
                     |  Free-text paste OR file upload      |
                     |  (PDF / DOCX / TXT / MD)             |
                     |  Writing context: Academic /         |
                     |  Professional Email / Formal /       |
                     |  Informal                            |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |   Grammar LangGraph (5 nodes)        |
                     |                                      |
                     |  [1] Text Loader                     |
                     |       +- Strip whitespace            |
                     |       +- Count words / sentences     |
                     |       +- Warn if near context limit  |
                     |                                      |
                     |  [2] Grammar Analysis (temp=0.1)     |
                     |       +- JSON array of errors        |
                     |          {type, original, suggestion,|
                     |           explanation, severity}     |
                     |                                      |
                     |  [3] Polish (temp=0.2) PRIMARY       |
                     |       +- Context-specific rewrite    |
                     |          via _STYLE_PROMPTS dict     |
                     |       +- Fully fluent polished text  |
                     |       +- Change summary (Markdown)   |
                     |       +- Feedback revision path      |
                     |          (refinement_round > 0)      |
                     |                                      |
                     |  [4] Style Advisor (temp=0.3)        |
                     |       +- Clarity / conciseness tips  |
                     |       +- Skipped if style not in     |
                     |          focus_areas                 |
                     |                                      |
                     |  [5] Grammar Eval (temp=0.1)         |
                     |       +- polish_quality              |
                     |       +- context_fit                 |
                     |       +- error_coverage              |
                     |       +- fluency                     |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |   Session Memory (SQLite)            |
                     |   outputs/memory/sessions.db         |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |         OUTPUT (4 tabs)              |
                     |  Polished Text (primary output)      |
                     |     +- Download MD / TXT             |
                     |  Issues Found (error list)           |
                     |  Style Tips                          |
                     |  Summary (word count, error stats)   |
                     |  Feedback -> revision rounds         |
                     +--------------------------------------+
```

---

## Open-Source Stack

| Component | Tool | Why |
|-----------|------|-----|
| **Local LLM** | [Ollama](https://ollama.ai) + 14 supported models | Free, private, no API key |
| **Agent Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) | State-machine agents |
| **Document RAG** | Hybrid RAG: FAISS (dense) + BM25 (sparse) + ChromaDB + RRF | Best of both retrieval strategies; falls back to BM25 keyword search if embeddings unavailable |
| **Self-Reflective RAG** | Corrective/Adaptive post-retrieval filter (`agents/self_reflective_rag.py`) | Batch LLM relevance grading after every retrieval call — active in the 5 retrieval modes (1–5); Mode 6 has no retrieval so SR-RAG is a no-op |
| **Document Parsing (Advanced)** | [Docling](https://github.com/DS4SD/docling) (default) | Layout-aware PDF parsing, table extraction as Markdown, PPTX/XLSX/HTML/image support; models cached in models/docling/ |
| **PDF Processing** | [pdfplumber](https://github.com/jsvine/pdfplumber) (fallback, used with --no-docling) | Text + tables from PDFs when Docling is not used |
| **Academic Search** | [arXiv API](https://arxiv.org/help/api) | Free preprint access |
| **Academic Search** | [Semantic Scholar](https://api.semanticscholar.org) | Free, peer-reviewed papers |
| **DOI Resolution** | [CrossRef API](https://api.crossref.org) | Free DOI -> metadata |
| **Web Search** | [ddgs](https://github.com/deedy5/ddgs) (DuckDuckGo) | No API key — runs in-process |
| **Session Memory** | [SQLite](https://www.sqlite.org/) via `tools/session_db.py` | Single `sessions.db` for all modes, WAL mode, 10–50x faster than JSON |
| **MCP Server** | [FastMCP](https://github.com/jlowin/fastmcp) (`mcp_servers/research_tools_server.py`) | Exposes 6 research tools via MCP protocol for Claude Desktop / Claude Code |
| **Citation Export** | stdlib `re` (built-in) | BibTeX + RIS, no extra deps |
| **DOCX Export** | [python-docx](https://python-docx.readthedocs.io/) | Word document generation |
| **PDF Export** | [ReportLab](https://www.reportlab.com/opensource/) | PDF generation, pure Python |
| **Graph Export** | [graphviz](https://pypi.org/project/graphviz/) + system `graphviz` | Render DOT to PNG / SVG |
| **TTS / Audio** | [pyttsx3](https://pypi.org/project/pyttsx3/) + system `espeak-ng` | Offline text-to-speech -> WAV |
| **UI** | [Streamlit](https://streamlit.io) | Python web app |
| **CLI** | [Rich](https://github.com/Textualize/rich) | Beautiful terminal output |

---

## Hardware Detection & Auto-Configuration

The assistant detects your hardware at startup and recommends the best Ollama model and context window for your specific machine — no manual tuning required.

### What gets detected

| Property | How it's detected |
|----------|------------------|
| **CPU name** | `sysctl machdep.cpu.brand_string` (Mac) / `platform.processor()` (others) |
| **RAM** | `psutil` -> `sysctl hw.memsize` -> `/proc/meminfo` (fallback chain) |
| **Apple Silicon** | `platform.system() == "Darwin"` + `platform.machine() == "arm64"` |
| **NVIDIA GPU** | `nvidia-smi --query-gpu=name` |

### How the recommendation works

Available RAM is budgeted conservatively:

| Hardware | Usable RAM budget |
|----------|------------------|
| Apple Silicon | 75% of total (unified memory shared with GPU) |
| NVIDIA GPU | 80% of total |
| CPU-only | Total - 4 GB (reserved for OS) |

The best **already-pulled** model that fits within the budget is recommended. Models are ranked by quality:

| Model | RAM needed | Context | Why |
|-------|-----------|---------|-----|
| `phi4:14b` | 14 GB | 16k | Highest reasoning quality |
| `mistral-nemo:12b` | 12 GB | 128k | Best context window |
| `gemma2:9b` | 9 GB | 32k | Strong general reasoning |
| `llama3.1:8b` | 8 GB | 32k | Best all-rounder |
| `qwen2.5:7b` | 7 GB | 32k | Efficient, multilingual |
| `llama3.2:3b` | 3 GB | 32k | Fastest, lowest RAM |

### Apple Silicon (M-series Macs)

Ollama automatically uses **Metal GPU** and the Neural Engine on Apple Silicon — no configuration needed. The hardware detector recognises this and adjusts the RAM budget accordingly (unified memory is shared between CPU and GPU, so the full chip RAM is available for model weights).

### Tight-fit warning

When the best available model uses >= 85% of usable RAM, the system surfaces both the high-capability model (tight fit) and the next-best pulled model (safe headroom) as a choice rather than silently picking one:

```
Warning: Tight memory fit detected. Choose which model to apply:
  o llama3.1:8b -- higher capability, tight fit
  o qwen2.5:7b  -- 7 GB, more headroom (Efficient, good multilingual support)
  [Apply Selection]  [Refresh]
```

The CLI equivalent prompts `[1/2]` and updates `args.model` for the session.

### Streamlit UI — sidebar hardware panel

When you open the app, the sidebar shows:

```
Your Hardware
  RAM: 16.0 GB    Accelerator: Apple Silicon (Metal)
  CPU: Apple M2 Pro
  OS: Darwin (arm64)

Recommendation
  llama3.1:8b -- Reliable all-rounder.
  Fits comfortably in 12 GB usable memory.

  [Apply Recommendation]  [Refresh]
```

- **Apply Recommendation** (or **Apply Selection** when a tight-fit choice is shown) pre-fills both the model dropdown and the context-window slider with the optimal values in one click
- **Refresh** re-queries Ollama for newly pulled models without reloading the page (cache expires every 30 seconds automatically)
- The model dropdown shows **only models currently pulled in Ollama** — no risk of selecting a model that isn't downloaded
- The recommended model is marked with a star in the list

If no models are pulled yet, the sidebar shows the exact pull command for the best model that fits your RAM:

```
Warning: No compatible models found.
Pull the recommended model:
  ollama pull llama3.1:8b
```

### CLI — system check

```bash
# Print hardware profile + available models + recommendation, then exit
python main.py --check-system
```

Example output:

```
-------------- System Check --------------
+--------------+----------------------------------+
| CPU          | Apple M2 Pro                     |
| RAM          | 16.0 GB                          |
| Accelerator  | Apple Silicon (Metal GPU)        |
| OS           | Darwin (arm64)                   |
+--------------+----------------------------------+

Pulled Ollama Models (2)
+--------------------+------------------+
| Model              | Status           |
+--------------------+------------------+
| llama3.1:8b        | recommended      |
| llama3.2:3b        |                  |
+--------------------+------------------+

Recommendation
  Recommended model:   llama3.1:8b
  Recommended num_ctx: 32,768

  Meta Llama 3.1 8B -- Reliable all-rounder.
  Fits comfortably in 12 GB usable memory.
  Apple Silicon (Metal GPU) -- Ollama uses Metal
  and the Neural Engine automatically.
```

The hardware check also runs automatically at the start of every research or proposal workflow so you always know what configuration is active.

---

## Six Modes — Explained

### Mode 1 — Academic Literature Search

**When to use:** You want a literature review, need citations for a paper, or are exploring a new research area.

**How it works:**
1. The LLM decomposes your topic into 4–6 focused search queries
2. Each query is sent to arXiv, Semantic Scholar, and optionally CrossRef
3. Results are deduplicated and ranked (peer-reviewed > preprint, by citation count)
4. If documents are uploaded, Hybrid RAG (FAISS + BM25 + RRF) retrieves the top-k chunks per query; **Self-Reflective RAG** then grades each chunk for relevance and removes irrelevant ones before the LLM sees them
5. The LLM selects the most relevant papers and formats APA citations
6. A structured report is generated with in-text citation mapping

**Citation format (APA 7th edition):**
```
Smith, J.; Jones, A. (2023). Attention is all you need. *NeurIPS*. https://doi.org/10.xxxx
```

---

### Mode 2 — ProposalGPT

**When to use:** You need to write a competitive grant proposal — for Horizon Europe, NSF, Vinnova, or any other funding agency — starting from the official call document.

**What it produces:**
- Complete call analysis: objectives, evaluation criteria, budget constraints, compliance checklist
- Win strategy: hidden priorities, SWOT analysis, reviewer perspective, risk assessment
- Academic literature review grounded in arXiv + Semantic Scholar
- A full 17-section proposal including work packages, deliverables, milestones, and a data management plan
- Structured budget with personnel/equipment/travel breakdown, indirect costs, and auto-scaling
- Compliance score (0–100) with per-section status and keyword coverage analysis
- Virtual reviewer simulation from 5 perspectives with a weighted 1–5 score
- AI-rewritten improved versions of the weakest sections
- Export to Markdown, Word (.docx), PDF, and budget CSV

#### The 9-Agent Pipeline

| Agent | Node | What it does |
|-------|------|-------------|
| 1 | `funding_call_analyzer` | Extracts structured fields from the call text; generates funding summary, evaluation matrix, compliance checklist |
| 2 | `research_planner` | Generates hidden priorities, win strategy, SWOT analysis, reviewer perspective, risk register |
| 3 | `literature_review_agent` | Searches arXiv + Semantic Scholar; applies SR-RAG `grade_papers()` to filter irrelevant papers; generates literature review, state of art, research gaps |
| 4 | `proposal_writer` | Writes 14 core sections (executive summary -> data management); JSON-parses work packages, deliverables, milestones |
| 5 | `impact_agent` | Writes impact, dissemination, exploitation, ethics, sustainability sections |
| 6 | `budget_agent` | Generates itemised personnel/equipment/travel budget; calculates indirect costs; scales to max budget if needed |
| 7 | `compliance_agent` | Checks mandatory sections, keyword coverage, page estimate; calculates 0–100 score |
| 8 | `reviewer_agent` | Simulates 5 virtual reviewers (scientific 30%, impact 25%, innovation 20%, implementation 15%, agency 10%); weighted overall score |
| 9 | `improvement_agent` | Identifies weak sections from reviewer feedback; rewrites top 2; generates improvement plan |

#### Budget Format Detection

The budget agent auto-detects the funding agency format:

| Format | Indirect rate | Agencies |
|--------|--------------|----------|
| `horizon_europe` | 25% on personnel | Horizon Europe, ERC, MSCA |
| `swedish` | 20% on personnel | Vinnova, VR, Formas, SSF |
| `generic` | 25% on personnel | NSF, DARPA, NIH, and others |

If the generated budget exceeds `max_budget × 1.05`, all line items are proportionally scaled down to fit within `max_budget × 0.98`.

#### Reviewer Simulation

Five virtual reviewers independently score the proposal on a 1–5 scale and provide strengths, weaknesses, and suggestions. The weighted overall score is:

```
overall = (scientific × 0.30) + (impact × 0.25) + (innovation × 0.20)
        + (implementation × 0.15) + (agency × 0.10)
```

#### Session Persistence

Every session is saved to `outputs/memory/proposal_gpt_<session_id>.json`. The CLI can export any saved session to all output formats without re-running the pipeline.

#### Export formats

| Format | Content |
|--------|---------|
| `.md` | Full Markdown proposal (all 17 sections + tables) |
| `.docx` | Word document (python-docx) |
| `.pdf` | PDF (ReportLab) |
| `.csv` | Itemised budget with totals |

---

### Mode 3 — Wisdom Mode

**When to use:** You face a complex life or scientific scenario and want rigorously grounded, personalised wisdom — not generic advice.

**What it does:**
- Asks Socratic clarifying questions (up to 3 rounds) to understand your specific context before searching
- Searches real academic papers (arXiv + Semantic Scholar) to ground every insight
- Delivers wisdom in two registers: **scientific depth** and **plain-language simplicity**
- Provides **per-claim confidence scores** (High / Medium / Low), consensus labels (Scientific consensus / Emerging evidence / Debated / Minority view), and a devil's advocate caveat
- Proposes concrete **actionable steps** derived from the literature
- Persists all sessions to disk; answers follow-up questions within the same session
- Silently enriches new sessions using topic-tag overlap with past ones — no explicit citations of history

#### The Wisdom Generation Pipeline

```
User message
    |
context_loader  -- loads history + prior wisdom + related sessions
    |
clarification   -- asks one Socratic question at a time (max 3 rounds)
    |  (when ready)
knowledge_search -- LLM generates academic queries -> arXiv + Semantic Scholar
    |
wisdom_synthesis -- LLM synthesises deep understanding + simple explanation + action steps
    |
wisdom_validator -- LLM self-critiques: per-claim confidence, consensus label, devil's advocate
    |
memory_saver    -- persists turn + wisdom output to JSON
```

#### Long-Term Wisdom Memory

Every session is saved to `outputs/memory/wisdom_<session_id>.json`:

- Scenario and topic you described
- Full conversation (user + assistant turns) with metadata flags (`is_question`, `has_wisdom`)
- Validated wisdom output (deep understanding, simple explanation, actionable steps, validation report)
- Topic tags extracted by the synthesis LLM — used for passive cross-session enrichment

#### Cross-Session Passive Enrichment

When synthesising new wisdom, the agent searches all past wisdom sessions for topic-tag overlap. Matching sessions' key insights are silently injected as background context into the synthesis prompt. The agent never names past sessions — the influence is passive, producing naturally richer wisdom without feeling like a history lecture.

```
New session: "decision fatigue at work"
  -> topic_tags: ["decision fatigue", "cognitive load", "executive function"]
  -> matched past sessions:
      +-- "stress and memory" (overlap: cognitive load)
      +-- "willpower and habits" (overlap: executive function)
  -> their wisdom snippets injected silently into synthesis prompt
```

#### Related Sessions Panel

When viewing an active session in the UI, related sessions (those sharing topic-tag word overlap with the current one) appear as clickable buttons under **"Related sessions:"** in the left panel. Click any button to load and browse that session's wisdom output.

---

### Mode 4 — Simplified PRISMA Systematic Review

**When to use:** You need a structured literature review that documents its own selection process — for a thesis, evidence synthesis, grant application, or any context where PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) reporting is expected or helpful.

**What it does:**
Given a PICO-style research question and optional inclusion/exclusion criteria, the agent automatically runs the full screening and synthesis pipeline and returns a structured review with PRISMA flow counts, an evidence table, and a narrative synthesis. Post-synthesis tools generate PRISMA-compliant reports, plain-language summaries, citation networks, preprint tracking, trend analysis, evidence maps, and concept drift detection — all on demand from the same result.

**The pipeline (6 nodes):**

```
query_generation -> literature_search -> screening -> evidence_extraction -> synthesis -> sr_eval
```

| Node | What happens |
|------|-------------|
| `query_generation` | LLM decomposes the research question into targeted search queries |
| `literature_search` | Queries **Google Scholar · arXiv · Semantic Scholar · CrossRef**; runs LLM abstract screener (0–100 relevance score) to pre-rank papers; deduplicates and sorts by screener score; applies SR-RAG `grade_papers()` as a second filter before `screening_node` evaluates them |
| `screening` | Each paper is assessed against inclusion/exclusion criteria; excluded papers are logged with a reason |
| `evidence_extraction` | For included papers: study design, quality rating (High/Medium/Low), and key finding are extracted |
| `synthesis` | LLM synthesises the evidence into narrative text, key themes, research gaps, and a conclusion |
| `sr_eval` | Five-dimension quality self-evaluation |

**Post-synthesis on-demand tools:**

| Tool | File | What it produces |
|------|------|-----------------|
| Abstract Screener | `tools/abstract_screener.py` | 0–100 relevance score per paper with include/uncertain/exclude verdict and rationale |
| Citation Network | `tools/citation_network.py` | Interactive Pyvis HTML graph of citation links between included papers (ego-only, via Semantic Scholar) |
| Preprint Tracker | `tools/preprint_tracker.py` | Status for each paper: journal / published / preprint / retracted (CrossRef lookup) |
| PRISMA Report | `tools/prisma_report.py` | Ready-to-submit DOCX and PDF with PRISMA 2020 section scaffold (Methods → Results → Discussion) |
| Plain-Language Summaries | `tools/plain_language.py` | Patient summary (8th-grade), policy brief (Markdown), press release (inverted pyramid) |
| Trend Analyzer | `tools/trend_analyzer.py` | Year-by-year publication counts via CrossRef facet API + Semantic Scholar; growing/declining/stable classification |
| Evidence Map | `tools/evidence_map.py` | Plotly bubble chart of Population × Intervention coverage; matplotlib PNG fallback |
| Concept Drift | `tools/concept_drift.py` | TF-IDF keyword shift across 5-year buckets; rising/declining/stable terms; optional LLM narrative |

**New state fields (added to `SystematicReviewState`):**
`screener_scores`, `preprint_tracking`, `citation_graph_html`, `trend_data`, `evidence_map_data`, `concept_drift_data`

**What PRISMA means:** PRISMA is a reporting standard for systematic reviews that requires documenting how many records were identified, screened, and included or excluded at each stage. Mode 4 produces this flow automatically from the pipeline's internal counts — it does not replace a full Cochrane-style review, but gives you a traceable, reproducible evidence summary from free academic databases.

**UI tabs:** Synthesis | Evidence Table | Discovery | Trends & Analysis | Export & Reports

**Quality dimensions scored:** `search_comprehensiveness`, `screening_rigor`, `evidence_quality`, `synthesis_depth`, `gap_identification`.

---

### Mode 5 — Research Notebook (NotebookLM-style grounded Q&A)

**When to use:** You have a set of your own sources — papers, reports, notes, web pages — and you want to ask questions about them and get answers that are grounded **only** in those sources, with citations to the exact document and page. This is the closest mode to Google's NotebookLM.

**What it does:**
You create one or more **notebooks**, add sources to each (upload PDF/DOCX/TXT/Markdown files, or paste a web page URL), and then chat with the notebook. Every answer is generated strictly from retrieved passages of your sources; if the sources don't contain the answer, the assistant says so rather than guessing.

**The pipeline (3 nodes, single-turn per question):**

```
retrieve -> answer -> save
```

| Node | What happens |
|------|-------------|
| `retrieve` | Loads the notebook, (re)builds the Hybrid RAG index from the notebook's stored chunks, and runs hybrid retrieval (FAISS dense + BM25 sparse, fused with RRF) for the question. SR-RAG `grade_chunks()` then filters irrelevant chunks (up to 2 cycles with query rewriting if fewer than 3 pass). Degrades to BM25 keyword fallback if the embedding model isn't pulled. |
| `answer` | The LLM receives the numbered source excerpts and answers using only them, citing each claim inline as `[n]`. It also proposes 2–3 follow-up questions. |
| `save` | Persists the question and the cited answer to the notebook's JSON so the conversation survives browser restarts. |

**Grounding contract:** the answer node is instructed to use only the provided excerpts, cite every claim with `[n]`, and explicitly decline when the sources don't cover the question. Each `[n]` resolves to a citation showing the source filename and page number.

**Persistence:** each notebook is one JSON file (`outputs/memory/notebook_<id>.json`) holding its sources, chunks, and full conversation. Because chunk IDs are stable content hashes, reopening a notebook rebuilds its in-memory FAISS/BM25 indexes from the **ChromaDB embedding cache** — no re-embedding.

**Reused infrastructure:** `DocumentProcessor` (chunking), `HybridStore` (FAISS + BM25 + ChromaDB + RRF), `OllamaEmbedder` (local embeddings), and `fetch_call_text` (web-page fetch + HTML cleaning). No new dependencies or services.

**7-Agent LangGraph Pipeline:** Mode 5 includes a full multi-agent pipeline where seven LangGraph nodes communicate through shared state (`NotebookPipelineState`). Run it via the **Pipeline** tab in the UI or `--notebook-pipeline` in the CLI — one button produces all seven outputs in sequence.

| Agent | Node | What it does |
|-------|------|-------------|
| 1 | `ingest` | Loads sources and chunks from `NotebookMemory` into pipeline state |
| 2 | `summarize` | Per-document summaries + cross-document synthesis (Overview · Common Themes · Contradictions · Takeaways) |
| 3 | `retrieve` | Hybrid RAG (FAISS + BM25 + RRF) on a focus query; then SR-RAG `grade_chunks()` filters irrelevant chunks before they reach the LLM; BM25 keyword fallback if embeddings unavailable |
| 4 | `verify_citations` | Identifies 5–8 claims in the summary; verifies each against source material (HIGH/MEDIUM/LOW confidence) |
| 5 | `build_kg` | Extracts entity–relationship graph -> Graphviz DOT (reuses `_knowledge_graph_to_dot()`) |
| 6 | `generate_study_guide` | Key Concepts · Glossary · Review Q&A · Quick Summary -> Markdown + DOCX + PDF |
| 7 | `generate_podcast` | Two-speaker dialogue (HOST: Alex, EXPERT: Dr. Jordan) -> TXT + browser TTS |

State flows linearly: `START -> ingest -> summarize -> retrieve -> verify_citations -> build_kg -> generate_study_guide -> generate_podcast -> END`

**Advanced analysis features (Phase 2):** beyond grounded Q&A and the full pipeline, Mode 5 provides nine individual one-click analysis tools available via the tabbed UI and CLI commands:

| Feature | UI tab | CLI flag | Output |
|---------|--------|----------|--------|
| Cross-document summary | Summary | `--notebook-summary <id>` | Markdown: common themes, contradictions, key takeaways; DOCX + PDF download |
| FAQ generation | FAQ | `--notebook-faq <id>` | 4–16 grounded Q&A pairs with source attribution |
| Literature review | Lit Review | `--notebook-review <id>` | Formal academic review (Introduction -> Conclusion); DOCX + PDF download |
| Audio summary script | Audio | `--notebook-audio <id>` | 300-word spoken-word script; browser TTS playback; `.wav` synthesis via pyttsx3 (CLI: saves `notebook_<id>_audio.wav`) |
| Mind map | Mind Map | `--notebook-mindmap <id>` | Concept tree as Graphviz DOT, rendered in browser; export as `.dot`, `.png`, `.svg` |
| Source comparison | Compare | `--notebook-compare <id> --compare-docs A B` | Side-by-side Markdown with table |
| Knowledge graph | Graph | `--notebook-graph <id>` | Entity–relationship graph as Graphviz DOT; export as `.dot`, `.png`, `.svg` |
| Timeline extraction | Timeline | `--notebook-timeline <id>` | Chronological events table (year, event, significance, source index) |
| Study comparison table | Study Table | `--notebook-study-table <id>` | Markdown comparison table across research type, sample, method, findings, limitations; DOCX + PDF download |

All advanced features load data from the notebook's stored chunks (`NotebookMemory`) — no additional indexing is required. The LLM is called once per feature.

**Export formats summary:**

| Content type | Available exports |
|---|---|
| Markdown text (summary, lit review, study table) | `.md` download, `.docx` (Word), `.pdf` |
| Graphviz graphs (mind map, knowledge graph) | `.dot` (source), `.png` (raster), `.svg` (vector) |
| Audio script | `.txt` download, browser TTS, `.wav` (pyttsx3) |

Graphviz PNG/SVG export requires the system `graphviz` package (`apt install graphviz` or included in the Docker image). WAV synthesis requires `pyttsx3` + `espeak-ng` (`apt install espeak-ng`, also in Docker).

---

### Mode 6 — English Grammar Proofreading

**When to use:** You need to polish a piece of writing — a journal paper draft, a professional email, a formal letter, or a blog post — and want both corrected and fluently rewritten output tailored to the appropriate register.

**What it does:**
- Accepts free-text paste or file upload (PDF, DOCX, TXT, MD)
- Applies a **writing context** that shapes the entire rewrite — not just spell-check:

| Context | Writing rules applied |
|---------|----------------------|
| `academic` | Third-person, no contractions, technical precision, hedging, suitable for peer review |
| `professional_email` | Clear subject -> context -> action -> close structure; professional but warm; active voice |
| `formal` | Strict formal register, elevated vocabulary, no contractions; for legal/official/policy documents |
| `informal` | Preserves author voice, allows contractions; focuses on clarity and natural flow |

- **Primary output** — a fully polished, fluent, context-appropriate rewrite (not just a list of errors)
- Per-error annotations: type (grammar / spelling / punctuation), original text, suggestion, explanation, severity
- Style suggestions: clarity, conciseness, tone, vocabulary, structure
- Change summary — Markdown list of what was changed and why
- **Feedback revision loop** — critique the polished output and the agent revises it; unlimited rounds in the Streamlit UI, up to 3 rounds via the CLI
- **Downloadable output** — Markdown (.md) and plain text (.txt) download buttons always shown; for file uploads the download section is displayed prominently at the top of the Polished Text tab

#### The 5-Node Pipeline

```
text_loader -> grammar_analysis -> polish -> style_advisor -> grammar_eval -> END
```

| Node | Temperature | What it does |
|------|-------------|-------------|
| `text_loader` | — | Strips whitespace, counts words/sentences, warns if text is near the context window limit |
| `grammar_analysis` | 0.1 | Detects errors as a JSON array — type, original, suggestion, explanation, severity |
| `polish` | 0.2 | **Primary output** — holistic rewrite for clarity, fluency, and context; uses `_STYLE_PROMPTS[style_level]` as the system prompt; on revision rounds, injects previous polished text + feedback |
| `style_advisor` | 0.3 | Clarity/conciseness/tone tips; skipped if "style" and "clarity" are not in `focus_areas` |
| `grammar_eval` | 0.1 | Self-evaluation on 4 dimensions: `polish_quality`, `context_fit`, `error_coverage`, `fluency` |

#### No External Retrieval

Mode 6 does not query any academic API or document store. `rag_reflection_info` is always `{}` — the Self-Reflective RAG display is a no-op. All processing happens locally via the configured Ollama LLM.

#### Session Persistence

Every session is saved to `outputs/memory/grammar_<session_id>.json`. Polished text is also written to `outputs/grammar_<session_id>.md` via the CLI.

#### CLI Examples

```bash
# Basic proofreading (professional email context by default)
python main.py --grammar-check --goal "She dont like cats."

# Academic context
python main.py --grammar-check --files paper_draft.pdf --style-level academic

# Formal document, focus only on grammar and punctuation
python main.py --grammar-check --files contract.docx --style-level formal \
  --focus grammar punctuation

# Informal blog post, all focus areas
python main.py --grammar-check --goal "So I been thinking about this for ages..."

# List saved grammar sessions
python main.py --list-grammar
```

---

## Writing Style Profiles

The assistant can learn your writing style from your own documents and apply it consistently across every mode that generates prose (Modes 1–4).

### How it works

1. **Upload 2–5 of your own documents** (past papers, reports, proposals) in the Style Profiles tab
2. An LLM analyses the combined text across four dimensions: tone & formality, structure & format, vocabulary & complexity, citation & evidence style
3. The analysis is stored as a named profile in `outputs/memory/style_<id>.json`
4. Select the profile in the sidebar — a compact ~280-word instruction block is injected into the system prompt of every writing node

### What gets captured

| Dimension | Examples |
|-----------|---------|
| **Tone & formality** | Register (formal/semi-formal), person (first/third), hedging words, assertiveness level |
| **Structure & format** | Paragraph length, sentence complexity, transition phrases, bullet vs. prose preference |
| **Vocabulary & complexity** | Technical density, preferred terms, words to avoid, readability level |
| **Citation & evidence style** | Citation density, placement (end-of-sentence vs. inline), evidence hierarchy, confidence language |

### Where style is injected

Style injection only applies to **prose-generating** nodes — not to search, retrieval, or citation-compilation nodes:

| Mode | Injected nodes |
|------|---------------|
| Mode 1 — Literature Search | Executive summary, report synthesis |
| Mode 2 — ProposalGPT | Excellence, methodology, impact, all prose-generating sections |
| Mode 3 — Wisdom | Wisdom synthesis |

### No fine-tuning required

Style profiles work entirely via prompt engineering. No GPU, no model retraining, no extra packages. The profile is loaded from disk and injected at inference time — zero overhead for runs without a profile.

### Multiple named profiles

Create as many profiles as needed:
- "Academic Writing" — formal, heavily cited, third-person
- "Grant Proposals" — persuasive, outcome-focused, first-person plural
- "Technical Reports" — direct, numbered lists, minimal hedging

Switch between them by selecting a different profile in the sidebar dropdown.

---

## Socratic Clarification

Before any workflow runs, the assistant asks 2–3 focused questions tailored to the user's specific goal and mode. This replaces guesswork with intent — the LLM never has to infer audience, scope, or depth because the user has already stated them.

### How it works

A fast LLM call (`temperature=0.2`, `num_predict=512`) generates questions using both the user's goal text and a mode-specific context string. If the LLM call fails, hardcoded per-mode fallback questions are used instead.

Questions come in two formats:

| Type | UI control | CLI input |
|------|-----------|-----------|
| `select` | Radio buttons (3–4 options) | Numbered choice (Enter to skip) |
| `text` | Single-line text field | Free-text prompt (Enter to skip) |

### Per-mode question focus

| Mode | What is clarified |
|------|------------------|
| 1 — Literature Search | Date range, discipline, intended use (background / gap analysis / survey) |
| 2 — ProposalGPT | Funding agency, consortium size, whether preliminary results exist |
| 3 — Wisdom Mode | Personal vs professional context, what has already been tried, urgency |
| 4 — Systematic Review | PICO criteria, date range, evidence quality threshold |
| 5 — Research Notebook | Focus question, source scope, analysis depth |

### UI flow

```
[Enter goal] -> [Clarify Requirements] -> [form renders] -> [answer 2-3 questions] -> [Run]
```

Click **"Clarify Requirements"** to expand the form. Unanswered questions are silently skipped. Click **"Reset questions"** to regenerate new questions for a changed goal. Answers are stored per-session and survive page refreshes.

### CLI flow

```
$ python main.py --mode search --goal "transformer architectures for NLP"

  Clarifying question 1 of 2
  How recent should the literature be?
    1. Last 2 years
    2. Last 5 years
    3. Last 10 years
    4. No restriction
  Your choice [1-4, Enter to skip]: 2

  Clarifying question 2 of 2
  What will you use these results for? (free text, Enter to skip):
  > Background for my PhD literature review
```

Skip the entire step with `--no-clarify`:

```bash
python main.py --mode search --goal "..." --no-clarify
```

### Injection into prose nodes

Answers are stored in `clarifications: Dict[str, str]` on every state TypedDict. A `_clarification_context(state)` helper in each node file returns `""` when the dict is empty (zero overhead for unanswered runs) or a formatted block:

```
USER CLARIFICATIONS (follow these when generating text):
- audience: Academic researchers
- purpose: Background review
```

This block is appended to the system prompt of every prose-generating node.

---

## Citation Export (BibTeX / RIS)

Every References panel across all modes includes two export buttons:

| Format | Extension | Compatible with |
|--------|-----------|-----------------|
| **BibTeX** | `.bib` | Zotero, JabRef, Mendeley, LaTeX (`\bibliography{}`) |
| **RIS** | `.ris` | Zotero, Mendeley, EndNote, RefWorks, Papers |

**BibTeX key format:** `<lastname><year>` — the **first token** of the first author's name string (i.e. the last name in `"Lastname Initials"` format), lowercased. Collision handling appends `a`, `b`, `c`... when two papers share the same key.

```bibtex
@article{vaswani2017,
  author    = {Vaswani A and Shazeer N and Parmar N},
  title     = {Attention Is All You Need},
  year      = {2017},
  journal   = {NeurIPS},
  doi       = {10.48550/arXiv.1706.03762}
}
```

---

## Research Session Persistence (Mode 1)

Completed literature search runs are automatically saved to `outputs/memory/research_<session_id>.json`. The sidebar shows the 5 most recent sessions — click any entry to instantly restore the full report, key findings, and reference list without re-running the workflow.

```
Recent Research Sessions (sidebar)
  +-- Survey of transformer architectures... (2024-03-06)
  +-- Antibiotic resistance prediction...   (2024-03-05)
  +-- Quantum error correction overview...  (2024-03-04)
```

---

## Multi-Document Cross-Analysis

Upload two or more files in any mode that accepts documents (Modes 1, 3, 4). When multiple documents are present, the analysis pipeline runs additional LLM calls:

1. **Per-document analysis** — a focused 200–300 word analysis for each uploaded file, identifying its specific contribution to the research goal
2. **Cross-document synthesis** — a paragraph identifying common themes across all documents, any contradictions or tensions, how the documents complement each other, and collective gaps

These appear as a dedicated "Per-Document Analysis" tab in the results and as a "Per-Document Breakdown" section in the downloadable Markdown report. Single-document workflows are unaffected (zero overhead).

---

## Self-Evaluation Framework

Every mode automatically scores its own output after completing the primary workflow. A dedicated **eval node** appended to each LangGraph makes one micro LLM call (`temperature=0.1`, `num_predict=300`) and returns a JSON object with mode-specific quality dimensions plus an overall score (1–5) and a one-sentence summary.

### Quality dimensions per mode

| Mode | Dimensions scored |
|------|-------------------|
| **Mode 1 — Literature Search** | Goal alignment · Evidence quality · Clarity |
| **Mode 2 — ProposalGPT** | Compliance score · Reviewer score · Section coverage |
| **Mode 3 — Wisdom** | Evidence grounding · Confidence calibration · Actionability |
| **Mode 4 — Systematic Review** | Search comprehensiveness · Screening rigor · Evidence quality · Synthesis depth · Gap identification |
| **Mode 5 — Research Notebook** | Grounding · Citation coverage · Response completeness |
| **Mode 6 — Grammar Proofreading** | Polish quality · Context fit · Error coverage · Fluency |

All modes also receive an **Overall (1–5)** score and a **one-sentence summary**.

### Design principles

- **Non-blocking** — any LLM failure is caught silently; the primary output is always delivered regardless
- **Cheap** — capped at 300 predicted tokens and a context window of at most 8,192 tokens; adds only a few seconds to the run
- **Skips gracefully** — Wisdom eval is skipped on clarification-only turns where no wisdom has been generated yet

### Where scores appear

**Streamlit UI** — a collapsed expander appears directly below the primary output for all four modes:
```
Quality Score 4/5 -- Well-grounded in evidence with clear actionable steps...
  [ Evidence Grounding 4/5 ]  [ Confidence Calibration 4/5 ]  [ Actionability 5/5 ]
```
Score badge colours: 4–5 (strong) · 3 (acceptable) · 1–2 (weak)

**CLI** — a Rich colour-coded table is printed immediately after the primary output, with per-dimension rows and an overall row.

---

## Project Structure

The app is organised as a **main project with six sub-projects** — one per mode. Opening the Streamlit app shows a landing page; clicking a mode card loads only that mode's code (lazy imports), keeping startup fast regardless of how many modes exist.

```
agentic-research-assistant/
|
+-- README.md                      <- Installation and running instructions
+-- docs/
|   +-- overview.md                <- You are here -- full project description
|   +-- FAQ.md                     <- Frequently asked questions
|   +-- architecture.md            <- Detailed system architecture diagrams
+-- requirements.txt               <- All dependencies (open-source)
+-- .env.example                   <- Configuration template
+-- .gitignore
+-- Dockerfile                     <- App container (Python + Streamlit)
+-- docker-compose.yml             <- App + Ollama (CPU)
+-- docker-compose.gpu.yml         <- GPU override (NVIDIA)
+-- .dockerignore
+-- main.py                        <- CLI entry point (--project flag selects sub-project)
+-- app.py                         <- Streamlit entry point -- landing page + dispatcher (~80 lines)
|
+-- projects/                      <- Sub-project modules (one per mode)
|   +-- __init__.py                <- PROJECT_REGISTRY metadata dict (6 modes)
|   +-- mode1_literature_search.py <- Mode 1: Literature Search -- run(settings)
|   +-- mode2_proposal.py          <- Mode 2: ProposalGPT -- run(settings)
|   +-- mode3_wisdom.py            <- Mode 3: Wisdom Mode -- run(settings)
|   +-- mode4_systematic_review.py <- Mode 4: Systematic Review -- run(settings)
|   +-- mode5_notebook.py          <- Mode 5: Research Notebook -- run(settings)
|   +-- mode6_grammar.py           <- Mode 6: Grammar Proofreading -- run(settings)
|
+-- ui/                            <- Streamlit UI modules
|   +-- landing.py                 <- Landing page -- 6 mode cards, Launch buttons
|   +-- helpers.py                 <- Shared helpers (uploads, clarification form, citations, report)
|   +-- sidebar.py                 <- Hardware, model, RAG, and style settings (shared across all modes)
|   +-- tabs/
|       +-- proposal_gpt.py        <- Mode 2 tab content (9-agent ProposalGPT UI)
|       +-- wisdom.py              <- Mode 3 tab content + render_wisdom_output()
|       +-- systematic_review.py   <- Mode 4 tab content
|       +-- notebook.py            <- Mode 5 tab content (Research Notebook)
|       +-- grammar_proofreading.py <- Mode 6 tab content (Grammar Proofreading)
|       +-- style_profiles.py      <- Style Profiles (accessible from landing page)
|
+-- tests/                         <- pytest test suite (496 tests, fully offline)
|   +-- test_citation_tools.py         <- BibTeX/RIS format, key collision
|   +-- test_story_memory.py           <- StorytellerMemory CRUD
|   +-- test_research_memory.py        <- ResearchMemory CRUD
|   +-- test_clarifier.py              <- Fallback questions, schema validation
|   +-- test_state_factories.py        <- All 4 state factory functions
|   +-- test_session_db.py             <- SQLite backend: all 9 tables, CRUD, WAL, pack/unpack (14 tests)
|   +-- test_integration_story.py      <- Story graph (6): 3-node, JSON parsing, memory
|   +-- test_integration_research.py   <- Research graph (4): search mode, mocked LLM+searcher
|   +-- test_integration_proposal.py   <- Proposal graph (4): new + revision paths
|   +-- test_integration_wisdom.py     <- Wisdom graph (5): clarify, force-proceed, followup
|   +-- test_notebook_memory.py        <- NotebookMemory CRUD (20 tests)
|   +-- test_notebook_nodes.py         <- retrieve/answer/save nodes + graph smoke (18 tests)
|   +-- test_notebook_advanced.py      <- Phase-2 advanced features (66 tests)
|   +-- test_notebook_pipeline.py      <- 7-agent pipeline (34 tests)
|   +-- test_proposal_gpt.py           <- ProposalGPT 9-agent pipeline (45 tests)
|   +-- test_feedback_agent.py         <- Feedback refinement (18 tests)
|   +-- test_self_reflective_rag.py    <- Self-Reflective RAG (38 tests)
|   +-- test_grammar_nodes.py          <- Grammar Proofreading nodes + graph (36 tests)
|   +-- test_web_loader.py             <- URL ingestion helper (6 tests)
|
+-- config/
|   +-- __init__.py
|   +-- settings.py                <- Pydantic-validated config from .env
|   +-- hardware.py                <- Hardware detection + model recommendation (CPU/GPU/Apple Silicon)
|
+-- tools/
|   +-- __init__.py
|   +-- document_tools.py          <- PDF/DOCX/TXT extraction + chunking
|   +-- search_tools.py            <- Google Scholar (scholarly), arXiv, Semantic Scholar, CrossRef, DuckDuckGo (ddgs)
|   +-- abstract_screener.py       <- LLM 0-100 relevance scorer for SR pre-screening
|   +-- citation_network.py        <- Ego-only citation graph: networkx DiGraph + Pyvis HTML (Semantic Scholar)
|   +-- preprint_tracker.py        <- CrossRef status lookup: journal/published/preprint/retracted
|   +-- prisma_report.py           <- PRISMA 2020 DOCX (python-docx) + PDF (reportlab) report generator
|   +-- plain_language.py          <- Patient summary, policy brief, press release generators
|   +-- trend_analyzer.py          <- CrossRef facet API + Semantic Scholar year-count trends
|   +-- evidence_map.py            <- Plotly Population x Intervention bubble chart + PNG fallback
|   +-- concept_drift.py           <- TF-IDF keyword shift across 5-year buckets (pure stdlib)
|   +-- session_db.py              <- SQLite backend: _tx(), init_db(), pack/unpack (all modes share sessions.db)
|   +-- export_tools.py            <- DOCX + PDF export (python-docx + ReportLab)
|   +-- citation_tools.py          <- BibTeX + RIS export (pure stdlib)
|   +-- proposal_tools.py          <- ProposalGPT assembly: assemble_full_proposal_md(), build_proposal_docx(), build_budget_csv()
|   +-- embeddings.py              <- OllamaEmbedder -- batched /api/embed calls
|   +-- hybrid_store.py            <- HybridStore: FAISS + ChromaDB + BM25 + RRF
|   +-- style_profiler.py          <- Writing style analyser (LLM -> structured profile)
|   +-- cli_recommender.py         <- Smart recommendation engine for CLI modes
|
+-- mcp_servers/
|   +-- research_tools_server.py   <- FastMCP server: search_arxiv, search_semantic_scholar,
|                                     search_crossref, web_search, query_notebook, ingest_document
|
+-- .mcp.json                      <- MCP server registration for Claude Desktop / Claude Code
|
+-- agents/
|   +-- __init__.py
|   +-- state.py                   <- ResearchState TypedDict (Mode 1)
|   +-- nodes.py                   <- Research nodes + Hybrid RAG
|   +-- graph.py                   <- Research LangGraph StateGraph
|   +-- memory.py                  <- ProposalMemory + ResearchMemory (accepts memory_dir)
|   +-- proposal_gpt_state.py      <- ProposalGPTState TypedDict + create_proposal_gpt_state()
|   +-- proposal_gpt_nodes.py      <- 9 ProposalGPT agent nodes
|   +-- proposal_gpt_graph.py      <- build_proposal_gpt_pipeline() + run_proposal_gpt()
|   +-- wisdom_memory.py           <- Persistent store for Wisdom Mode sessions
|   +-- wisdom_state.py            <- WisdomState TypedDict (Mode 3)
|   +-- wisdom_nodes.py            <- 7 nodes: clarification->search->synthesis->validate
|   +-- wisdom_graph.py            <- build_wisdom_graph() + run_wisdom_turn()
|   +-- systematic_review_state.py <- SystematicReviewState TypedDict (Mode 4)
|   +-- systematic_review_nodes.py <- 6 PRISMA nodes
|   +-- systematic_review_graph.py <- SR LangGraph StateGraph
|   +-- style_memory.py            <- CRUD for Writing Style Profiles (style_*.json)
|   +-- notebook_state.py          <- NotebookState TypedDict (Mode 5)
|   +-- notebook_memory.py         <- NotebookMemory -- SQLite persistence (notebooks + notebook_chunks tables)
|   +-- notebook_nodes.py          <- 3 nodes: retrieve -> answer -> save
|   +-- notebook_graph.py          <- build_notebook_graph() + run_notebook_turn()
|   +-- notebook_advanced.py       <- Phase-2 features: summary, FAQ, lit-review, mind-map, audio, compare, KG, timeline, study-table
|   +-- notebook_pipeline_state.py <- NotebookPipelineState TypedDict + create_pipeline_state()
|   +-- notebook_pipeline_nodes.py <- 7 pipeline nodes (ingestion->summarize->retrieve->verify->KG->study->podcast)
|   +-- notebook_pipeline_graph.py <- build_notebook_pipeline() + run_notebook_pipeline()
|   +-- feedback_agent.py          <- refine_with_feedback() -- up to 3 feedback rounds, all modes
|   +-- self_reflective_rag.py     <- grade_chunks(), grade_papers(), self_reflective_retrieve()
|   +-- grammar_state.py           <- GrammarState TypedDict + create_grammar_state() (Mode 6)
|   +-- grammar_nodes.py           <- 4 nodes: text_loader, grammar_analysis, polish, style_advisor
|   +-- grammar_graph.py           <- build_grammar_graph() + run_grammar_check() (Mode 6)
|   +-- grammar_memory.py          <- GrammarMemory -- SQLite persistence for grammar sessions
|
+-- outputs/                       <- All generated files saved here
    +-- .gitkeep
    +-- memory/
    |   +-- sessions.db            <- Single SQLite database for ALL modes
    |                                (grammar_sessions, wisdom_sessions, wisdom_tags,
    |                                 story_sessions, style_profiles, proposal_sessions,
    |                                 research_sessions, notebooks, notebook_chunks)
    +-- chroma_db/                 <- ChromaDB embedding cache (FAISS source vectors)
    +-- report_<id>.md             <- Research reports
    +-- proposal_<id>.md           <- ProposalGPT full proposals
    +-- grammar_<id>.md            <- Grammar proofreading polished output (CLI)
    +-- proposal_<id>.docx         <- Word exports
    +-- proposal_<id>.pdf          <- PDF exports
    +-- budget_<id>.csv            <- Budget exports (Mode 2)
```

---

## Understanding the Code — Tutorial Walkthrough

### 1. State-based Agent Design

Every workflow shares a TypedDict state. Each node reads fields from state, does its work, and returns a partial update — LangGraph merges the updates automatically.

**Research state (Mode 1):**
```python
# agents/state.py
class ResearchState(TypedDict, total=False):
    goal: str
    uploaded_docs: List[ProcessedDocument]
    search_queries: List[str]
    academic_papers: List[Paper]
    doc_context: List[Dict]          # retrieved chunks (Hybrid RAG: FAISS + BM25 + RRF)
    references: List[Dict]
    report: str
    num_ctx: int                     # LLM context window (tokens)
```

**ProposalGPT state (Mode 2):**
```python
# agents/proposal_gpt_state.py
class ProposalGPTState(TypedDict, total=False):
    # INPUT
    funding_call_text: str
    user_ideas: str
    funding_agency: str
    budget_format: str               # "horizon_europe"|"swedish"|"generic"
    session_id: str
    model_name: str
    # AGENT 1 -- Call Analysis
    call_title: str
    call_objectives: List[str]
    keywords: List[str]
    budget_constraints: Dict         # max_budget, currency, duration_months
    mandatory_sections: List[str]
    # AGENT 2 -- Strategy
    win_strategy: str
    swot_analysis: str
    # AGENT 3 -- Literature
    literature_review: str
    state_of_art: str
    research_gaps: List[str]
    # AGENT 4 -- Proposal sections
    executive_summary: str
    excellence: str
    methodology: str
    work_packages: List[Dict]
    deliverables: List[Dict]
    milestones: List[Dict]
    # AGENT 6 -- Budget
    budget_personnel: List[Dict]
    budget_equipment: List[Dict]
    budget_travel: List[Dict]
    budget_total: float
    budget_indirect_rate: float
    # AGENT 7 -- Compliance
    compliance_score: int            # 0-100
    missing_sections: List[str]
    # AGENT 8 -- Reviewer
    reviewer_scores: Dict[str, Dict]
    overall_score: float             # 1.0-5.0 weighted average
    # AGENT 9 -- Improvement
    weak_sections: List[str]
    improved_sections: Dict[str, str]
    improvement_plan: str
    # PIPELINE META
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    progress_pct: int
```

### 2. Hybrid RAG — Document Analysis (Modes 1 and 3)

Documents are chunked and indexed by both a dense (FAISS) and a sparse (BM25) retriever. Reciprocal Rank Fusion merges the two ranked lists:

```python
# tools/hybrid_store.py -- HybridStore.search()
dense_results  = self._faiss_search(query_embedding, top_k=top_k * 2)   # cosine similarity
sparse_results = self._bm25_search(query_tokens, top_k=top_k * 2)       # BM25Okapi

# Reciprocal Rank Fusion (k=60)
scores: Dict[str, float] = {}
for rank, doc_id in enumerate(dense_results):
    scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank + 1)
for rank, doc_id in enumerate(sparse_results):
    scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank + 1)

top_chunks = sorted(scores, key=scores.get, reverse=True)[:top_k]
# top_chunks pass to Self-Reflective RAG before reaching the LLM
```

```python
# agents/self_reflective_rag.py -- grade_chunks() (runs after RRF)
grades = grade_chunks(top_chunks, query, model_name, num_ctx)
# grades = [True, False, True, ...]  -- one bool per chunk, single LLM call
relevant_chunks = [c for c, g in zip(top_chunks, grades) if g]
# relevant_chunks are injected into the LLM's human message
# If fewer than 3 pass and max_cycles > 1, query is rewritten and cycle 2 fires
```

If the Ollama embedding model is unavailable, the system falls back to BM25 keyword search and shows a warning banner. The context window size is controlled by the **"Context window (tokens)"** slider in the sidebar.

### 3. Academic Citation Flow

```python
# Search -> normalise -> deduplicate -> rank -> format APA
papers = searcher.search("transformer protein folding")

apa = paper.to_apa()
# "LeCun, Y.; Bengio, Y. (2023). Deep learning. *Nature*. https://doi.org/10.xxxx"
```

### 4. Conditional Graph Routing (Research workflow)

```python
# agents/graph.py
def _route_after_academic_search(state: ResearchState) -> str:
    if state.get("include_web_search", False):
        return "web_search"
    return "document_analysis"

graph.add_conditional_edges(
    "academic_search",
    _route_after_academic_search,
    {"web_search": "web_search", "document_analysis": "document_analysis"},
)
```

### 5. ProposalGPT — Linear 9-Node Pipeline

```python
# agents/proposal_gpt_graph.py
graph = StateGraph(ProposalGPTState)
graph.add_node("funding_call_analyzer", funding_call_analyzer_node)
graph.add_node("research_planner", research_planner_node)
# ... 7 more nodes
graph.add_edge(START, "funding_call_analyzer")
graph.add_edge("funding_call_analyzer", "research_planner")
# ... linear chain through all 9 nodes
graph.add_edge("improvement_agent", END)

# One call runs the full pipeline; stream_callback fires after each node
final_state = run_proposal_gpt(
    create_proposal_gpt_state(funding_call_text=call_text, ...),
    stream_callback=lambda node, state: print(f"Done: {node}"),
)
```

### 6. Long-Term Memory

```python
# Research memory -- outputs/memory/research_<id>.json
research_mem = ResearchMemory()
research_mem.save_session(session_id, goal, report, references,
                          key_findings, document_names, mode, model)
recent = research_mem.list_sessions(limit=5)

# Story memory -- outputs/memory/story_<id>.json
story_mem = StorytellerMemory()
sid = story_mem.new_session(topic="Transformers", document_context="...")
story_mem.add_turn(sid, "user", "What is self-attention?")
story_mem.add_turn(sid, "assistant", "Think of it like a spotlight...",
                   suggested_questions=["What is Q,K,V?"])
story_mem.add_concepts(sid, ["self-attention", "query-key-value"])
```

### 7. Wisdom Mode — Socratic Search and Validation Graph

Wisdom Mode uses a multi-phase LangGraph with conditional routing to move the conversation from clarification -> academic search -> synthesis -> validation:

```
START -> context_loader -> [route]
    +-- "wisdom_followup"  -> wisdom_followup -> memory_saver -> END
    +-- "clarification"    -> clarification -> [route]
            +-- "knowledge_search" -> wisdom_synthesis -> wisdom_validator -> memory_saver -> END
            +-- "memory_saver"     -> memory_saver -> END   (still asking Qs)
```

```python
# One call per user message
final_state = run_wisdom_turn(create_wisdom_state(
    user_message="How does chronic stress affect memory consolidation?",
    session_id="w9f2b4d1",
    topic="stress and memory",
    model_name="llama3.1:8b",
    num_ctx=32768,
))
# phase is "clarifying", "ready_to_generate", or "done"
response = final_state["assistant_response"]
```

**Wisdom memory — SQLite (`outputs/memory/sessions.db`, `wisdom_sessions` + `wisdom_tags` tables):**
```python
wisdom_mem = WisdomMemory()
sid = wisdom_mem.new_session(topic="stress and memory", scenario="I can't retain...")
wisdom_mem.add_turn(sid, "user", "I struggle to remember after deadlines")
wisdom_mem.add_turn(sid, "assistant", "Can you clarify...", metadata={"is_question": True})
wisdom_mem.save_wisdom(sid, deep_understanding="...", simple_explanation="...",
                       actionable_takeaways="...", validation={...}, topic_tags=[...])
related = wisdom_mem.find_related_sessions(["stress", "memory"], current_session_id=sid)
```

The `find_related_sessions()` method uses an indexed SQL JOIN on the `wisdom_tags` table (individual tag words stored as rows) — no embedding required, and faster than the previous full-file scan.

### 8. Systematic Review — PRISMA Pipeline (Mode 4)

Mode 4 runs a **simplified PRISMA systematic review** as a linear 6-node LangGraph:

```
START -> query_generation -> literature_search -> screening ->
        evidence_extraction -> synthesis -> sr_eval -> END
```

| Node | What it does |
|------|-------------|
| `query_generation` | LLM generates 4–6 varied academic search queries (broad + narrow) from the research question |
| `literature_search` | Fans out across all queries via `AcademicSearcher`; deduplicates by normalised title slug; sorts by citation count; applies SR-RAG `grade_papers()` as a pre-screening filter — reduces the number of papers `screening` must evaluate per-paper |
| `screening` | LLM-per-paper INCLUDE/EXCLUDE decisions against PICO criteria; records exclusion reasons |
| `evidence_extraction` | Extracts study design, sample size, key finding, quality (High/Medium/Low), relevance score for up to 20 included papers |
| `synthesis` | Builds PRISMA flow metrics; generates narrative synthesis with inline citations, key themes, research gaps, limitations, and conclusion |
| `sr_eval` | Five-dimension quality self-evaluation |

**PRISMA flow output:**
```json
{"identified": 48, "screened": 48, "eligibility": 32, "included": 18, "excluded": 30}
```

**CLI:**
```bash
python main.py --systematic-review \
  --goal "Effect of sleep deprivation on working memory in university students" \
  --inclusion "Peer-reviewed empirical studies" "Human participants" "Published 2010-2024" \
  --exclusion "Animal studies" "Review papers only"
```

### 10. BibTeX / RIS Export

```python
# tools/citation_tools.py -- pure stdlib, no new dependencies
from tools.citation_tools import refs_to_bibtex, refs_to_ris

bib = refs_to_bibtex(references)    # -> .bib file string
ris = refs_to_ris(references)        # -> .ris file string
```

BibTeX key generation uses the **first token** of the first author's name (the last name in `"Lastname Initials"` format): `"Vaswani A"` -> `vaswani2017`. Collision handling appends `a`, `b`, `c`... for same-author same-year papers.

### 10. Lazy Tool Imports

`tools/__init__.py` uses Python's module `__getattr__` hook to load submodules on first access, not at import time:

```python
# tools/__init__.py -- nothing is imported until you use the name
def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_path, attr = _EXPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value  # cache: subsequent access is O(1)
        return value
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
```

This means `from tools.citation_tools import refs_to_bibtex` (pure stdlib) does **not** trigger the import of `tenacity`, `faiss`, `chromadb`, or any other heavy dependency. Tests that only need citation tools run without the full dependency stack installed.

### 11. Lazy Memory Singletons in Node Files

`wisdom_nodes.py` and `notebook_nodes.py` expose a `_get_memory()` getter instead of creating the memory object at module level:

```python
# Pattern used in all three node files
_memory: T | None = None

def _get_memory() -> T:
    global _memory
    if _memory is None:
        _memory = T()          # T = WisdomMemory / NotebookMemory
    return _memory
```

This prevents `outputs/memory/` from being created on import (which breaks tests using `tmp_path`), and allows tests to inject a custom instance:

```python
# In tests -- inject an isolated memory object before the graph runs
monkeypatch.setattr(wisdom_nodes_module, "_memory", WisdomMemory(tmp_path))
monkeypatch.setattr(notebook_nodes_module, "_memory", NotebookMemory(tmp_path))
```

### 12. DOCX and PDF Export

```python
# tools/export_tools.py

# Word document
docx_bytes = build_docx(proposal_markdown, references, metadata={"title": "..."})
Path("proposal.docx").write_bytes(docx_bytes)

# PDF
pdf_bytes = build_pdf(proposal_markdown, references, metadata={"title": "..."})
Path("proposal.pdf").write_bytes(pdf_bytes)
```

Both functions accept the raw Markdown string. They parse it into sections, apply styles, render inline formatting, and produce a professional document layout without any external tools (no LibreOffice, no LaTeX, no Pandoc).

---

## Infrastructure Notes

**Rate limit handling (Semantic Scholar):** 429 responses from Semantic Scholar trigger tenacity exponential backoff (up to 30 s wait, 4 attempts) instead of a fixed 5 s sleep, so the search step recovers automatically under load without failing the workflow.

**Embedding cache invalidation:** If you re-upload a document with the same filename but changed content, the system computes an MD5 hash of the new content and compares it against the stored hash. On a mismatch, stale embeddings are removed from ChromaDB before re-embedding, so your retrieval results always reflect the current file.

**Background thread for ProposalGPT:** The 9-agent pipeline runs in a background thread with progress polled every 3 seconds via `st.rerun()`, keeping the Streamlit UI responsive during the full proposal generation run (typically 5–15 minutes depending on model speed).

---

## Contributing

Pull requests are welcome. For major changes, open an issue first.

**Adding a new academic database:**
1. Add a new searcher class to `tools/search_tools.py`
2. Import and call it in `AcademicSearcher.search()`
3. Add config variables to `.env.example` and `config/settings.py`

**Adding a new ProposalGPT agent:**
1. Add the new state fields to `ProposalGPTState` in `agents/proposal_gpt_state.py`
2. Write a new node function in `agents/proposal_gpt_nodes.py`
3. Register the node and edge in `agents/proposal_gpt_graph.py`
4. Add output rendering to the relevant tab in `ui/tabs/proposal_gpt.py`

**Adding a new mode (sub-project):**
1. Create `ui/tabs/<name>.py` with a `tab_<name>(settings)` function
2. Create `projects/modeN_<name>.py` with a `run(settings)` function that lazy-imports the tab
3. Register it in `projects/__init__.py` (`PROJECT_REGISTRY`), `app.py` (`_PROJECT_MODULES`), and `ui/landing.py` (`_PROJECTS` list)
4. Add any shared rendering helpers to `ui/helpers.py`

---

## License

MIT — see [LICENSE](../LICENSE).
