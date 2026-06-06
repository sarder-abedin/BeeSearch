# ResearchBuddy — Tutorial

## Quick start

```bash
# 1. Clone and start (Docker)
git clone https://github.com/sarder-abedin/researchbuddy
cd researchbuddy
docker-compose up
# Open http://localhost:8501

# OR: without Docker
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## Mode 1 — Systematic Review

### What it does
Automates a PRISMA-style systematic review: searches arXiv, Semantic Scholar, and CrossRef;
screens papers against your criteria; extracts structured evidence; runs Risk of Bias,
GRADE, and contradiction analysis; and synthesises findings.

### Step-by-step

**Step 1 — Enter your research question**

Write a PICO-style question:
> *What is the effect of mindfulness meditation on anxiety in adults with generalised anxiety disorder?*

**Step 2 — Set inclusion / exclusion criteria** *(optional but recommended)*

```
Inclusion (one per line):
  Peer-reviewed empirical studies
  Human adults (≥18 years)
  Published 2015–2024
  English language

Exclusion:
  Animal studies
  Case reports or single-subject designs
  Conference abstracts only
```

**Step 3 — Click "Run Systematic Review"**

The pipeline runs through 6 nodes. A progress bar and step log keep you informed.

**Step 4 — Explore the result tabs**

| Tab | What you'll find |
|-----|------------------|
| **Synthesis** | Narrative synthesis, key themes, research gaps, conclusion |
| **Evidence Table** | Per-paper: design, sample size, quality, key finding |
| **PRISMA Diagram** | Interactive Mermaid flowchart + Graphviz version |
| **Risk of Bias** | RoB 2 (RCTs) or ROBINS-I (observational) per paper |
| **GRADE** | Overall certainty level with domain downgrading rationale |
| **Contradictions** | Conflicting findings with consensus scores |
| **Search Queries** | The 4–6 queries used across databases |
| **Advanced → Sensitivity** | Re-run with quality filters or modified criteria |
| **Advanced → Monitor** | Save state; check for new papers on return visits |
| **Advanced → Pre-registration** | OSF-style template + PRISMA 2020 checklist |
| **Export** | Full Markdown report with evidence table and RoB summary |

**Tips**
- The more specific your inclusion/exclusion criteria, the tighter the screening.
- After running, click **Save search state** in the Monitor tab to enable incremental monitoring.
- Click **Import SR synthesis into this notebook** in the Notebook's source panel to
  send the synthesis into a notebook session for follow-up Q&A.

---

## Mode 1 — CLI (Systematic Review)

```bash
# Basic run
python cli.py sr "Effect of caffeine on working memory in adults"

# With criteria and export
python cli.py sr "Effect of caffeine on working memory" \
  --include "human participants,RCT or cohort" \
  --exclude "animal studies" \
  --output review.md \
  --json-output review.json

# Use a specific model
python cli.py sr "Effect of caffeine on working memory" --model llama3.2:3b
```

Output includes: PRISMA flow, GRADE grading, RoB summary, contradictions, narrative synthesis, research gaps, and quality scores.

---

## Mode 2 — Research Notebook

### What it does
A grounded Q&A notebook: upload sources, ask questions, get citations. Every answer is
backed by exact document and page references. Persistent across sessions.

### Step-by-step

**Step 1 — Create or select a notebook**

Click **+ New notebook**, give it a name, and click **Create notebook**.

**Step 2 — Add sources**

You can add:
- **Files**: PDF, DOCX, TXT, MD, HTML, EPUB (drag-and-drop in the file uploader)
- **Web pages**: paste a URL in *Add a web page*
- **BibTeX**: upload a `.bib` file from Zotero/Mendeley via *Import BibTeX / Zotero*
- **SR synthesis**: after running a Systematic Review, use *Import from Systematic Review*
  to send the SR directly into this notebook

**Step 3 — Ask questions in the Chat tab**

> *What methodology did Smith et al. use?*  
> *Summarise the key findings on anxiety reduction.*  
> *What are the limitations discussed in these papers?*

Each answer cites the exact source and page.

**Step 4 — Use the advanced tabs**

| Tab | What you'll find |
|-----|------------------|
| **Summary** | Cross-document synthesis |
| **FAQ** | Auto-generated Q&A from all sources |
| **Lit Review** | Formal literature review (introduction, methods, findings, analysis) |
| **Mind Map** | Concept relationships as a Graphviz mind map |
| **Audio** | Spoken summary script + .wav synthesis |
| **Compare** | Side-by-side comparison of two sources |
| **Graph** | Knowledge graph (entities + relationships) |
| **Timeline** | Chronological events extracted from sources |
| **Study Table** | Structured comparison: design, scope, findings, limitations |
| **Pipeline** | 7-agent pipeline: ingestion → summary → retrieval → citation check → KG → study guide → podcast |
| **Extraction** | PICO extraction table (Author, Design, Population, Intervention, Outcome) with CSV export |
| **Gaps** | Research gap map across 5 dimensions with priority ranking |
| **Hypotheses** | Testable PICO hypotheses generated from identified gaps |

---

## Mode 2 — CLI (Research Notebook)

```bash
# List all notebooks
python cli.py nb --list

# Create a notebook and ask a question
python cli.py nb --new "Sleep Study" --question "What does the literature say about sleep?"

# Ask a follow-up (use the notebook ID from --list output)
python cli.py nb --notebook-id abc123 --question "What are the main limitations?"

# Interactive REPL (no question arg = REPL mode)
python cli.py nb --notebook-id abc123

# With auto web search
python cli.py nb --notebook-id abc123 --question "Latest RCTs on mindfulness" --web
```

### Import BibTeX from CLI

```bash
# Import a Zotero export into a notebook
python cli.py bib abc123 ~/zotero_export.bib
```

### Research Gap Mapping from CLI

```bash
python cli.py gap abc123 "sleep and cognitive performance" -o gaps.md
```

### Hypothesis Generation from CLI

```bash
python cli.py hyp abc123 "sleep and cognitive performance" \
  --gaps "no longitudinal RCTs,understudied in older adults" \
  -n 5 -o hypotheses.md
```

---

## Settings panel

Available in the sidebar:

| Setting | Description | Default |
|---------|-------------|--------|
| **Model** | Ollama model for LLM tasks | `llama3.1:8b` |
| **Embedding model** | For vector retrieval | `nomic-embed-text` |
| **Context window** | Token budget per call | 32768 |
| **Hybrid top-k** | Retrieved chunks per query | 10 |
| **Chunk size / overlap** | Document chunking parameters | 512 / 64 |

---

## FAQ

**Q: Does it require an internet connection?**  
A: Yes, to search arXiv, Semantic Scholar, and CrossRef. The LLM runs locally via Ollama — no API keys needed.

**Q: How do I add a new Ollama model?**  
A: `docker exec -it researchbuddy-ollama ollama pull <model>` or `ollama pull <model>` if running locally.

**Q: The systematic review returns very few papers. Why?**  
A: Your criteria may be too narrow, or the topic may not have coverage in the three databases searched. Relax the exclusion criteria or try a broader formulation.

**Q: Can I export results?**  
A: Yes. Every tab with generated content has a download button. The SR Export tab provides a full Markdown report. Extraction tables export as CSV and Markdown.

**Q: How is this different from the full Agentic Research Assistant?**  
A: ResearchBuddy focuses exclusively on the two research-intensive modes: Systematic Review and Research Notebook. It adds features not in the ARA (RoB, GRADE, contradiction detection, sensitivity analysis, literature monitor, BibTeX import, CLI).
