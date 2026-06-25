# Frequently Asked Questions

> For installation instructions, see the [README](../README.md).
> For a full project description and architecture, see the [Overview](overview.md).

---

**Q: How do I run this with Docker?**
A: Run `docker compose up --build` from the project root. This starts two services: Ollama (LLM) and the Streamlit app (port 8501). A one-shot `model-init` container pulls the default model on first start. Open http://localhost:8501 in your browser.

---

**Q: Does this send my documents to any server?**
A: No. Documents are processed entirely locally — parsed with Docling (or pdfplumber when Docling is disabled), embedded with a local Ollama model, stored in a local ChromaDB instance, and analysed by a local Ollama LLM. The only network calls are to Google Scholar, arXiv, Semantic Scholar, CrossRef, and — only when web search is enabled or automatically triggered (see "Can I use BeeSearch entirely offline?" below) — DuckDuckGo. All of these receive only your search queries, never your document text.

---

**Q: Which Ollama model should I use?**
A: Run `python main.py --check-system` or open the Streamlit sidebar — the hardware detector reads your RAM and GPU type and recommends the best model. Rule of thumb: `llama3.2:3b` for < 8 GB RAM, `llama3.1:8b` for most laptops, `mistral-nemo:12b` for long documents (128k context). On Apple Silicon, Ollama uses Metal automatically.

---

**Q: Do I need to pull a separate embedding model for the Research Notebook?**
A: Yes. Pull `nomic-embed-text` (274 MB) before your first notebook session: `ollama pull nomic-embed-text`. If you skip this, the system automatically falls back to BM25 keyword search and shows a warning — retrieval still works, just without dense vector ranking. You can specify a different embedding model with `--embed-model mxbai-embed-large`.

---

**Q: Will the same document be re-embedded every time I upload it?**
A: No. Embeddings are cached in ChromaDB on disk (`outputs/chroma_db/`). On subsequent runs with the same document, the cached embeddings are used and Ollama is not called again. If you modify a document, the cache is invalidated automatically via MD5 hash comparison — no manual clearing required. Run `python -c "import shutil; shutil.rmtree('outputs/chroma_db', ignore_errors=True)"` to free disk space.

---

**Q: Does Docling require a separate download?**
A: Docling is included in `requirements.txt` and enabled by default. On first use, it downloads its ML models (~500 MB) automatically to `models/docling/`. Subsequent runs use the cached models. To use the lightweight pdfplumber-only parser instead, toggle "Advanced Parsing (Docling)" off in the sidebar or pass `--no-docling` on the CLI.

---

**Q: Uploading a large PDF crashes or freezes my machine — what can I do?**
A: Docling loads ~500 MB of ML models into RAM, which can exhaust memory on resource-constrained machines. BeeSearch automatically switches to the lightweight pdfplumber parser for PDFs that exceed the `LARGE_DOC_PAGE_THRESHOLD` (default: 50 pages). pdfplumber streams pages one at a time and uses a fraction of the RAM — Docling is never loaded for those files. To lower the threshold (e.g. on an 8 GB machine), set `LARGE_DOC_PAGE_THRESHOLD=20` in your `.env`, pass `--large-doc-threshold 20` on the CLI, or toggle "Advanced Parsing (Docling)" off in the sidebar to disable Docling entirely.

---

**Q: How do I handle very long documents in the Notebook?**
A: Two levers: (1) **Parser RAM** — large PDFs auto-switch to pdfplumber (page-by-page streaming) to stay within available RAM; lower `LARGE_DOC_PAGE_THRESHOLD` in `.env` if needed. (2) **LLM context** — increase the "Context window (tokens)" slider in the sidebar and use a model with a larger context (e.g. `mistral-nemo:12b` supports 128k tokens). Set `NUM_CTX=131072` in your `.env` to make 128k the default. Pass `--num-ctx 32768` on the CLI for a one-off increase.

---

**Q: Does Google Scholar search require an API key?**
A: No. The Systematic Literature Review uses the `scholarly` Python library, which queries Google Scholar without an API key. No account or subscription is required. For very high-volume use, Google may temporarily throttle requests — in that case, the review continues with results from arXiv, Semantic Scholar, and CrossRef.

---

**Q: What does the Abstract Screener do?**
A: Before formal inclusion/exclusion screening, the Abstract Screener sends each paper's title and abstract to the LLM and asks it to assign a relevance score from 0 to 100 against your research question and criteria. Papers scoring ≥ 60 are marked *include*, 40–59 *uncertain*, and < 40 *exclude*. Scores are visible in the Explore tab (pick *Abstract Screener*). This pre-ranks papers and reduces the load on the formal screening step.

---

**Q: What is the Citation Network and how is it built?**
A: The Citation Network is an ego-only graph that shows which of your included papers cite each other. After synthesis, you can trigger it from the Explore tab. It queries the Semantic Scholar API for the references of each included paper, then draws edges between pairs where one included paper cites another. It uses networkx for the graph and Pyvis for an interactive HTML visualisation. The graph itself stays ego-only — no external papers are added as nodes — but two extra panels help with screening: an **Isolated papers** list names any included papers with no citation links to the rest of your corpus, and a **"Frequently cited but not in your review"** list surfaces papers cited by 2+ of your included papers but not themselves included, as candidates for a second screening pass.

---

**Q: What are Smart Citations (Supporting / Contrasting / Mentioning)?**
A: When you tick *"Classify citation stances"* before building the Citation Network, BeeSearch runs one LLM call per citation edge that compares the two papers' abstracts and labels how the citing paper engages with the cited one: **Supporting** (agrees with / builds on it, green edge), **Contrasting** (disputes / contradicts it, red edge), or **Mentioning** (neutral reference, gray edge — the safe default). This is scite.ai-style citation classification. Because only abstracts are available (not full text), it is best-effort and pinned to a deterministic temperature; any edge that can't be classified stays neutral.

---

**Q: What is the Citation Context tool?**
A: Citation Context (in the Explore tab) surfaces the exact sentence(s) where one included paper cites another — the scite.ai "snippet" view. Pick a citing paper and a cited paper; BeeSearch fetches the citing paper's **open-access full text** (PDF via pdfplumber, or HTML), then searches it for sentences mentioning the cited work by author surname or a distinctive title token. It is deliberately descoped and fails safe: when no open-access full text is available, the fetch fails, or the cited work isn't found, it reports "unavailable"/"not found" rather than erroring.

---

**Q: What does the Preprint Tracker report?**
A: For each included paper, the Preprint Tracker queries CrossRef by title and returns one of four statuses: *journal* (published in a peer-reviewed journal), *published* (formerly a preprint, now journal-published), *preprint* (still on arXiv or a preprint server), or *retracted* (CrossRef retraction notice found). This lets you flag any included papers that should be treated with extra caution.

---

**Q: What are the Risk of Bias, GRADE, and Contradiction results?**
A: Every systematic review now runs a **quality assessment** step between evidence extraction and synthesis (you'll also find it in Explore → *Risk & Certainty*, and `--sr-quality` on the CLI):
- **Risk of Bias** — each paper is rated with the appropriate Cochrane tool: **RoB 2** for randomised trials, **ROBINS-I** for observational studies, across that tool's standard bias domains, with an overall Low / Some concerns / High judgement.
- **GRADE** — the overall body of evidence gets a certainty rating (**High / Moderate / Low / Very low**) across the five GRADE domains (risk of bias, inconsistency, indirectness, imprecision, publication bias), starting High for RCT evidence and Low otherwise.
- **Contradictions** — the LLM scans the evidence for conflicting findings across papers and reports each as a contested claim with a **0–100 consensus score** and the papers on each side.

These results feed the narrative synthesis (so it reflects certainty and disagreement instead of guessing) and are included in the Markdown/report exports. Each assessment fails safe: if one can't be completed it degrades to an empty result rather than blocking the review.

---

**Q: What is the AI Research Assistant (Mode 3)?**
A: Mode 3 answers a **free-form research question** from published literature in general — no documents to upload and no PRISMA workflow. It searches Google Scholar, arXiv, Semantic Scholar (and optionally the web), numbers every retrieved source, grounds an LLM answer in them, then rebuilds the citation list **in code** from whichever `[n]` markers the answer actually used (never the LLM's self-written references). If no sources can be retrieved it answers from general knowledge with an explicit caveat and no citations. Use it from the Mode 3 tab or `python main.py --ask "your question"` (add `--no-web` for academic-only).

---

**Q: How does the Trend Analyzer work?**
A: The Trend Analyzer queries the CrossRef facet API with your research query to get field-wide publication counts per year. If CrossRef returns fewer than 30 records, it supplements from Semantic Scholar. It classifies the overall trend as *growing*, *stable*, *declining*, or *insufficient data* based on the slope over the most recent 5 years. Results appear in the Explore tab (pick *Research Trends*) and can be exported as a table.

---

**Q: What does the Evidence Map show?**
A: The Evidence Map is a Plotly bubble chart that groups your included papers by Population and Intervention (extracted from the evidence table). Bubble size reflects the number of studies in that cell; bubble colour reflects average quality (green = High, amber = Medium, red = Low). This gives a quick visual summary of where the evidence is concentrated and where gaps exist. A matplotlib PNG is generated as a fallback if Plotly is unavailable.

---

**Q: What is Concept Drift detection?**
A: Concept Drift groups all retrieved papers into 5-year buckets, extracts the top TF-IDF keywords from each bucket using only the Python standard library (no scikit-learn), and compares keyword rankings between the earliest and most recent bucket. Keywords that rise by 3 or more rank positions are classified *rising*; those that fall by 3 or more are *declining*; others are *stable*. Optionally, the LLM writes a short narrative describing the conceptual shift.

---

**Q: What formats does the PRISMA Report support?**
A: The report is generated in both DOCX (via python-docx) and PDF (via reportlab) with identical content: title page with author/institution, abstract, PRISMA 2020-structured sections (Introduction, Methods, Results, Discussion, Conclusion), evidence table, and references. No LibreOffice or Microsoft Word installation is required.

---

**Q: What are the three plain-language summary formats?**
A: After synthesis, you can generate up to three plain-language summaries:
- **Public** — written at an 8th-grade reading level (~350 words) with no jargon, for general audience or the general public
- **Policy** — a one-page Markdown brief with a policy context, key evidence points, and recommendations for policy-makers
- **Press** — an inverted-pyramid press release with a headline, a lede, body paragraphs, and a closing quote, for journalists

Pass `--sr-plain-language all` to generate all three in one run.

---

**Q: What does the 7-agent Notebook pipeline produce?**
A: The pipeline runs 7 agents in sequence: (1) ingest documents into the hybrid store, (2) summarise each document and synthesise cross-document themes, (3) retrieve the most relevant chunks for your focus query, (4) verify 5–8 key claims against source material with confidence ratings, (5) build a knowledge graph (DOT + PNG + SVG), (6) generate a structured study guide with key concepts, glossary, Q&A pairs, summary — exported as Markdown, DOCX, and PDF, (7) generate a two-speaker podcast script (HOST: Alex, EXPERT: Dr. Jordan). Run with `python main.py --notebook-pipeline <id>`.

---

**Q: How is a Research Notebook saved and restored?**
A: Notebooks are stored in `outputs/memory/sessions.db` (SQLite). The `notebooks` table stores metadata, the source list, and the full conversation history. Chunks are stored in a separate `notebook_chunks` table. Embeddings are cached in ChromaDB. Reopening a notebook with `python main.py --notebook --notebook-id <id>` restores the full conversation and all sources without re-embedding. Run `python main.py --list-notebooks` to see all saved notebooks.

---

**Q: What does Self-Reflective RAG do?**
A: After every retrieval call, a single batched LLM call (`temperature=0.0`) grades all retrieved items for relevance to your query. Irrelevant items are discarded before reaching the main LLM. In the Research Notebook, if fewer than 3 chunks pass grading, the query is automatically rewritten and retrieval is retried (up to 2 cycles). In the Systematic Review, papers are graded once (one-pass). Any grading failure silently passes all items — the main pipeline is never blocked.

---

**Q: How trustworthy are the citations in Chat, Literature Review, and Explain?**
A: All three are grounded the same way under the hood: every retrievable excerpt (a chunk, not a whole document) gets its own number and a real page reference baked in *before* the LLM ever sees it, and the prompt instructs it to cite only those numbers. After the LLM responds, BeeSearch discards any References list it wrote and rebuilds one in code from whichever numbers were actually used in the body text — the model's self-written list is never trusted. A citation number the model invents with no matching excerpt (e.g. a hallucinated `[99]`) is silently dropped instead of shown with made-up source details.

---

**Q: What happens if I ask the Explain tab the same question twice, or say I don't understand?**
A: The `repetition_tracker` node detects this automatically — no toggle needed. It compares your message to recent prior questions (word-overlap similarity) and checks for confusion phrases ("I don't understand", "still lost", "explain it differently", and similar). If either matches, two things happen: the next explanation is generated in a genuinely different style (e.g. switching from a plain explanation to an analogy, a step-by-step walkthrough, or a for/against debate, rotating away from whichever style the previous answer used) instead of just rewording the same text, and an interactive concept map (a central concept with its directly related ideas, rendered with Pyvis) appears alongside the new explanation as a second, visual way into the idea. Detection requires at least one prior answer in the session — your very first message can't be "a repeat." Either step failing (LLM error, missing `pyvis`) silently falls back to a normal explanation with no concept map — it never blocks the response.

---

**Q: How does BeeSearch rank web search results?**
A: `WebSearcher` (DuckDuckGo) fetches more results than requested, deduplicates them by URL and normalised title, then re-ranks: recognised research domains (arxiv.org, PubMed, IEEE Xplore, Nature, ScienceDirect, `.edu`/`.gov`, and similar) move to the front, a short list of low-signal sites (Pinterest, Quora) moves to the back, and everything else keeps DuckDuckGo's own relative order. Nothing is ever dropped for being non-academic — only reordered — so a relevant blog or vendor page is never hidden, just deprioritised behind primary sources.

---

**Q: Can I use BeeSearch entirely offline?**
A: Almost. The LLM and embedding model run fully offline via Ollama. Online calls happen in two places: (1) the Systematic Literature Review always queries Google Scholar, arXiv, Semantic Scholar, and CrossRef to find papers; (2) in the Research Notebook, the Chat and Research Report tabs only search the web (DuckDuckGo) when you turn on their "Auto web search" / "Include web search" toggle (off by default), while the Explain tab automatically runs a web + academic search whenever your uploaded sources don't sufficiently cover a question — this one has no off switch. Leave both Notebook toggles off to keep Chat and Research Report fully offline.
