# BeeSearch — Tutorial

## Installation

```bash
# 1. Install Ollama
# macOS/Linux: https://ollama.ai
# Windows: https://ollama.ai/download

# 2. Pull a model (choose one)
ollama pull llama3.1:8b          # recommended for most machines
ollama pull llama3.2:3b          # for < 8 GB RAM
ollama pull mistral-nemo:12b     # for 128k context window

# 3. Pull the embedding model (for Research Notebook Hybrid RAG)
ollama pull nomic-embed-text

# 4. Clone and install
git clone https://github.com/sarder-abedin/BeeSearch.git
cd BeeSearch
pip install -r requirements.txt

# 5. Check your hardware
python main.py --check-system
```

---

## Running the app

### Streamlit UI

```bash
streamlit run app.py
# Open http://localhost:8501
```

### Docker

```bash
# Recommended — React + FastAPI web app only (lighter)
./scripts/start-web.sh
# Open http://localhost:8000

# Full stack — React + Streamlit + CLI
docker compose up --build
# React app at http://localhost:8000; Streamlit also at http://localhost:8501
```

---

## Mode 1 — Systematic Literature Review

### UI walkthrough

1. Open the app and click **Open Systematic Literature Review**
2. Enter your PICO-style research question in the text area
3. Optionally add inclusion and exclusion criteria
4. Click **Run Systematic Review**
5. Watch the progress bar: query generation → literature search (Google Scholar + arXiv + Semantic Scholar + CrossRef) → screening → evidence extraction (incl. PICO fields) → quality assessment (risk of bias, GRADE certainty, contradictions) → synthesis → quality evaluation
6. Explore the 4 result tabs:
   - **Synthesis** — narrative synthesis, key themes, research gaps, conclusion, PRISMA flow counts, a GRADE certainty / contradiction callout
   - **Evidence** — structured table of included papers with PICO fields and quality ratings
   - **Explore** — pick one deep-dive tool at a time: Citation Network (with Smart Citations stance classification), Citation Context, Risk & Certainty (RoB/GRADE/contradictions), Abstract Screener, Preprint Tracker, Research Trends, Evidence Map, Meta-Analysis, Concept Drift
   - **Write-up & Export** — Markdown download, DOCX/PDF generation, plain-language summaries

### CLI walkthrough

```bash
# Basic review
python main.py --systematic-review \
  --goal "What is the effect of sleep deprivation on working memory?" \
  --inclusion "Peer-reviewed empirical studies" "Human participants" \
  --exclusion "Animal studies" "Review papers only"

# With DOCX and PDF reports
python main.py --systematic-review \
  --goal "Efficacy of CBT for treatment-resistant depression" \
  --sr-docx --sr-pdf \
  --sr-author "Dr. Jane Smith" --sr-institution "University of Oxford"

# With plain-language summaries (patient / policy / press / all)
python main.py --systematic-review \
  --goal "Effect of mindfulness on anxiety in adults" \
  --sr-plain-language all

# With trend analysis
python main.py --systematic-review \
  --goal "Machine learning in drug discovery" \
  --sr-trends

# With preprint tracking
python main.py --systematic-review \
  --goal "COVID-19 vaccine efficacy" \
  --sr-preprints

# With concept drift detection
python main.py --systematic-review \
  --goal "Antibiotic resistance mechanisms" \
  --sr-concept-drift

# With risk-of-bias / GRADE / contradiction results printed to console
python main.py --systematic-review \
  --goal "Efficacy of statins for cardiovascular risk reduction" \
  --sr-quality

# Full combined run
python main.py --systematic-review \
  --goal "Efficacy of CBT for treatment-resistant depression" \
  --inclusion "RCTs" "Adult patients (≥18)" \
  --exclusion "Children and adolescents" "Open-label studies" \
  --sr-docx --sr-pdf \
  --sr-author "Dr. Smith" --sr-institution "MIT" \
  --sr-plain-language all \
  --sr-quality \
  --sr-trends --sr-preprints --sr-concept-drift
```

### Progress output example

```
📋 PRISMA Systematic Review
Research question: Effect of sleep deprivation on working memory
Inclusion: Peer-reviewed empirical studies, Human participants
Exclusion: Animal studies, Review papers only
Model: llama3.1:8b

  Generating search queries  ─────────── 10%  0:00:03
  Searching Google Scholar · arXiv · Semantic Scholar · CrossRef ── 30%  0:00:18
  Screening papers ───────────────────── 50%  0:00:35
  Extracting evidence ─────────────────  70%  0:00:52
  Assessing risk of bias, GRADE certainty, contradictions ─ 80%  0:01:05
  Synthesising findings ───────────────  90%  0:01:15
  Evaluating review quality ───────────  100% 0:01:22

✓ Complete in 82.4s

PRISMA Flow
┌───────────────────┬───────┐
│ Stage             │ Count │
│ Identified        │ 87    │
│ After Dedup       │ 61    │
│ Screened          │ 61    │
│ Included          │ 14    │
│ Excluded          │ 47    │
└───────────────────┴───────┘

✓ Report saved: outputs/systematic_review_<id>.md
✓ DOCX saved: outputs/prisma_report_<id>.docx
✓ PDF saved: outputs/prisma_report_<id>.pdf
✓ Patient summary saved: outputs/summary_patient_<id>.txt
✓ Policy summary saved: outputs/summary_policy_<id>.txt
✓ Press summary saved: outputs/summary_press_<id>.txt
```

### Output files

| File | Contents |
|------|---------|
| `outputs/systematic_review_<id>.md` | Full SR report in Markdown |
| `outputs/prisma_report_<id>.docx` | PRISMA 2020 Word document |
| `outputs/prisma_report_<id>.pdf` | PRISMA 2020 PDF |
| `outputs/summary_patient_<id>.txt` | Patient plain-language summary (~350 words, 8th-grade) |
| `outputs/summary_policy_<id>.txt` | Policy brief (Markdown with 6 headers) |
| `outputs/summary_press_<id>.txt` | Press release (inverted pyramid) |

---

## Mode 2 — Research Notebook

### Table and figure support

BeeSearch understands both tables and figures inside your uploaded PDFs.

**Tables** work automatically — Docling extracts each table as full Markdown (column headers, rows, alignment) and presents it to the LLM in that format so it can compare values and cite them correctly with a `[TABLE]` label.

**Figures** require an optional Ollama vision model. To enable:

```bash
# 1. Pull a vision model
ollama pull llava:7b          # good balance of quality and RAM (needs ~4 GB)
# or:
ollama pull llama3.2-vision:11b   # best quality, needs ~16 GB RAM
# or:
ollama pull minicpm-v:8b      # strong on diagrams, 5 GB

# 2. Set it in .env
VISION_MODEL=llava:7b

# 3. Or pass it on the CLI
python main.py --notebook --notebook-name "My Research" \
  --files paper.pdf --vision-model llava:7b
```

BeeSearch calls the vision model once per figure at upload time to generate a caption, which is indexed alongside text and retrieved normally. In Chat, captions appear with a `[FIGURE]` label. Leave `VISION_MODEL` blank (the default) to skip figure extraction with no overhead.

### UI walkthrough

1. Click **Open Research Notebook** on the landing page
2. Create a new notebook or open an existing one
3. Add sources: upload PDFs/DOCX/TXT files or paste a web URL
4. Ask questions in the chat — answers are grounded in your sources with inline citations `[1]`, `[2]`…
5. Use the tab buttons for advanced tools:
   - **Summary** — cross-document synthesis
   - **FAQ** — auto-generated question/answer pairs
   - **Lit Review** — formal academic review
   - **Audio** — spoken-word script + WAV synthesis
   - **Mind Map** — concept tree as graph
   - **Graph** — entity–relationship knowledge graph
   - **Compare** — side-by-side source comparison
   - **Citation Timeline** — cited works by year, with one-line gists (optional Semantic Scholar abstract enrichment)
   - **Study Table** — structured research comparison
   - **Pipeline** — run all 7 agents in sequence

### CLI walkthrough

```bash
# Start a new notebook
python main.py --notebook --notebook-name "Sleep Research"

# Open an existing notebook
python main.py --list-notebooks
python main.py --notebook --notebook-id <id>

# Add sources when opening
python main.py --notebook --notebook-id <id> \
  --files paper1.pdf paper2.pdf notes.txt

# Document parsing options
python main.py --notebook --files paper.pdf              # default: Docling (layout-aware)
python main.py --notebook --files scanned.pdf --ocr     # Docling + OCR (scanned PDFs)
python main.py --notebook --files paper.pdf --no-docling  # always use pdfplumber
python main.py --notebook --files big.pdf --large-doc-threshold 20  # custom RAM threshold

# Advanced one-shot analysis
python main.py --notebook-summary <id>
python main.py --notebook-faq <id>
python main.py --notebook-review <id>
python main.py --notebook-audio <id>
python main.py --notebook-mindmap <id>
python main.py --notebook-graph <id>
python main.py --notebook-compare <id> --compare-docs paper1.pdf paper2.pdf
python main.py --notebook-timeline <id>                  # citation timeline
python main.py --notebook-timeline <id> --enrich-abstracts  # + Semantic Scholar abstracts
python main.py --notebook-study-table <id>

# 7-agent pipeline (runs all agents in sequence)
python main.py --notebook-pipeline <id>
python main.py --notebook-pipeline <id> --pipeline-query "What are the main findings on sleep?"
```

### Interactive slash commands

Once inside `--notebook` mode:

| Command | What it does |
|---------|-------------|
| `/add <file>` | Add a local document to the notebook |
| `/url <url>` | Fetch and add a web page |
| `/sources` | List all sources and chunk counts |
| `/summary` | Generate cross-document summary |
| `/faq` | Generate FAQ from sources |
| `/review` | Generate literature review |
| `/audio` | Generate audio script and synthesise WAV |
| `/mindmap` | Extract mind map (DOT + PNG + SVG) |
| `/graph` | Extract knowledge graph (DOT + PNG + SVG) |
| `/compare` | Compare two sources interactively |
| `/timeline` | Extract citation timeline (cited works by year) |
| `/study-table` | Generate study comparison table |
| `/quit` | Exit (session is saved automatically) |

Numbers `1`, `2`, `3` select suggested follow-up questions from the previous answer.

---

## Mode 3 — AI Research Assistant

A single-screen, stateless mode for a quick, literature-grounded answer — no upload, no PRISMA criteria, no saved session.

### UI walkthrough

1. Click **Open AI Research Assistant** on the landing page
2. Type a free-form research question
3. Optionally toggle off "include web results" to use academic sources only
4. Click **Ask** — the assistant searches Google Scholar, arXiv, and Semantic Scholar (plus the web unless disabled), then answers with inline `[n]` citations
5. Review the source list and pick a suggested follow-up question, or ask a new one

If no sources are found, the answer is clearly marked as ungrounded general knowledge rather than presented as if it were literature-backed.

### CLI walkthrough

```bash
# Ask a question (academic sources + web)
python main.py --ask "What is the evidence for intermittent fasting and insulin sensitivity?"

# Academic sources only, skip the web search
python main.py --ask "Efficacy of mindfulness-based stress reduction in adults" --no-web
```

```
╭─────────────────────── Ask ───────────────────────╮
│ AI Research Assistant                              │
│                                                     │
│ What is the evidence for intermittent fasting and  │
│ insulin sensitivity?                                │
│                                                     │
│ Model: llama3.1:8b                                 │
╰─────────────────────────────────────────────────────╯
⠋ Searching Google Scholar · arXiv · Semantic Scholar · web…

╭─────────────────────── Answer ────────────────────╮
│ Several randomized trials report improved insulin  │
│ sensitivity with intermittent fasting [1][2]...     │
╰─────────────────────────────────────────────────────╯
        Citations (3)
┌───┬──────────┬──────────────────────────┬──────┐
│ # │ Source   │ Title                    │ Year │
├───┼──────────┼──────────────────────────┼──────┤
│ 1 │ semantic │ Intermittent fasting...  │ 2021 │
│ 2 │ arxiv    │ Metabolic effects of...  │ 2020 │
│ 3 │ web      │ NIH: Fasting and...      │ 2023 │
└───┴──────────┴──────────────────────────┴──────┘
Searched 8 paper(s), 4 web result(s); 3 cited.

Follow-up questions:
  • Does the effect differ between time-restricted eating and alternate-day fasting?
  • What sample sizes did these trials use?
```

---

## Tips and common patterns

### Sanity-check a question with the AI Research Assistant before a full SR

A Systematic Review run takes minutes and produces a formal PRISMA pipeline. If you just want to know whether a question is even worth that investment, ask it in Mode 3 first:

```bash
python main.py --ask "Is there existing evidence on X?"
# If the answer looks promising, commit to a full systematic review:
python main.py --systematic-review --goal "..." --inclusion "..." --exclusion "..."
```

### Combine SR with Notebook

Run an SR first, save the included papers, then create a notebook from those papers for deep Q&A:

```bash
# Step 1: Run the SR
python main.py --systematic-review --goal "..." --sr-docx

# Step 2: Create a notebook from the downloaded papers
python main.py --notebook --notebook-name "SR Deep Dive"
# Inside the notebook: /add paper1.pdf paper2.pdf ...
```

### Use --check-system before a long run

```bash
python main.py --check-system
```

This prints hardware specs, pulled models, and the recommended model for your machine.

### Increase context for long documents

```bash
python main.py --notebook --notebook-id <id> --num-ctx 32768
# Or in .env: NUM_CTX=32768
```

### Handle large PDFs on low-RAM machines

PDFs larger than `LARGE_DOC_PAGE_THRESHOLD` pages (default: 50) automatically switch from Docling to pdfplumber to avoid loading ~500 MB of ML models into RAM. Lower the threshold if you are on a machine with < 8 GB RAM:

```bash
# One-off: process a 100-page PDF with a lower threshold
python main.py --notebook --files big_paper.pdf --large-doc-threshold 20

# Permanent: set in .env
LARGE_DOC_PAGE_THRESHOLD=20

# Disable Docling entirely (always uses pdfplumber)
python main.py --notebook --files paper.pdf --no-docling
```

### Save disk space

```bash
# Clear ChromaDB embedding cache
python -c "import shutil; shutil.rmtree('outputs/chroma_db', ignore_errors=True)"
```
