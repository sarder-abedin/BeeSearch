# ResearchBuddy — Tutorial

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
git clone https://github.com/sarder-abedin/ResearchBuddy.git
cd ResearchBuddy
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
docker compose up --build
# Open http://localhost:8501
```

---

## Mode 1 — Systematic Literature Review

### UI walkthrough

1. Open the app and click **Open Systematic Literature Review**
2. Enter your PICO-style research question in the text area
3. Optionally add inclusion and exclusion criteria
4. Click **Run Systematic Review**
5. Watch the progress bar: query generation → literature search (Google Scholar + arXiv + Semantic Scholar + CrossRef) → screening → evidence extraction → synthesis → quality evaluation
6. Explore the 5 result tabs:
   - **Synthesis** — narrative synthesis, key themes, research gaps, conclusion, PRISMA flow counts
   - **Evidence Table** — structured table of included papers with quality ratings
   - **Discovery** — abstract screener scores, citation network builder, preprint tracker
   - **Trends & Analysis** — publication trend chart, evidence map, concept drift
   - **Export & Reports** — Markdown download, DOCX/PDF generation, plain-language summaries

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

# Full combined run
python main.py --systematic-review \
  --goal "Efficacy of CBT for treatment-resistant depression" \
  --inclusion "RCTs" "Adult patients (≥18)" \
  --exclusion "Children and adolescents" "Open-label studies" \
  --sr-docx --sr-pdf \
  --sr-author "Dr. Smith" --sr-institution "MIT" \
  --sr-plain-language all \
  --sr-trends --sr-preprints --sr-concept-drift
```

### Progress output example

```
📋 PRISMA Systematic Review
Research question: Effect of sleep deprivation on working memory
Inclusion: Peer-reviewed empirical studies, Human participants
Exclusion: Animal studies, Review papers only
Model: llama3.1:8b

  Generating search queries  ─────────── 17%  0:00:03
  Searching Google Scholar · arXiv · Semantic Scholar · CrossRef ── 33%  0:00:18
  Screening papers ───────────────────── 50%  0:00:35
  Extracting evidence ─────────────────  67%  0:00:52
  Synthesising findings ───────────────  83%  0:01:15
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
   - **Timeline** — chronological events
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

# Advanced one-shot analysis
python main.py --notebook-summary <id>
python main.py --notebook-faq <id>
python main.py --notebook-review <id>
python main.py --notebook-audio <id>
python main.py --notebook-mindmap <id>
python main.py --notebook-graph <id>
python main.py --notebook-compare <id> --compare-docs paper1.pdf paper2.pdf
python main.py --notebook-timeline <id>
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
| `/timeline` | Extract chronological timeline |
| `/study-table` | Generate study comparison table |
| `/quit` | Exit (session is saved automatically) |

Numbers `1`, `2`, `3` select suggested follow-up questions from the previous answer.

---

## Tips and common patterns

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

### Save disk space

```bash
# Clear ChromaDB embedding cache
python -c "import shutil; shutil.rmtree('outputs/chroma_db', ignore_errors=True)"
```
