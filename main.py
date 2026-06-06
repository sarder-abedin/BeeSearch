"""
main.py — Command-Line Interface
──────────────────────────────────
Terminal entry point for the Agentic Research Assistant.

Literature Search (Mode 1)
──────────────────────────
# Search academic sources on a topic
python main.py --goal "Transformer models for protein folding"

ProposalGPT — AI-assisted proposal writing (Mode 2)
────────────────────────────────────────────────────
# Generate a full proposal from a funding call document (9-agent pipeline)
python main.py --propose --call-file funding_call.pdf \
  --ideas "Our team has 10 years experience in federated learning..." \
  --funding-agency "Horizon Europe"

# Quick proposal with just ideas (no call file)
python main.py --propose \
  --goal "ML framework for antibiotic resistance prediction" \
  --ideas "Novel graph neural network approach" \
  --funding-agency "Generic"

# Run only specific agents (skip reviewer simulation for speed)
python main.py --propose --call-file call.pdf --no-reviewer \
  --ideas "Quantum ML approach" --funding-agency "NSF"

# List saved ProposalGPT sessions
python main.py --list-proposals

# Export a saved proposal session to DOCX/PDF/Markdown/CSV
python main.py --export-proposal <session_id> --output ./my_proposal

Wisdom Mode — interactive chat (Mode 3)
────────────────────────────────────────
# Start a new wisdom session
python main.py --wisdom --topic "chronic stress and memory"

# Provide a scenario upfront
python main.py --wisdom --topic "decision fatigue" \
  --scenario "I struggle to make good choices late in the day at work."

# Continue an existing wisdom session
python main.py --wisdom-session <session_id>

# List all saved wisdom sessions
python main.py --list-wisdom

Systematic Review (Mode 4)
──────────────────────────
# Run a simplified PRISMA systematic review
python main.py --systematic-review \
  --goal "What is the effect of sleep deprivation on working memory?" \
  --inclusion "Peer-reviewed empirical studies" "Human participants" \
  --exclusion "Animal studies" "Review papers only"

# With funding agency template (affects section lengths and tone)
python main.py --propose \
  --goal "ML framework for antibiotic resistance prediction" \
  --funding-agency "Horizon Europe" \
  --duration 36 --budget "€1M – €3M" --trl "TRL 3–5" --consortium "3 universities + 1 SME"

Research Notebook (Mode 5)
──────────────────────────
# Start a new notebook (interactive Q&A with slash commands)
python main.py --notebook --notebook-name "Antibiotic Resistance"

# Continue an existing notebook
python main.py --notebook --notebook-id <notebook_id>

# Add files while opening a notebook
python main.py --notebook --notebook-id <id> --files paper.pdf notes.txt

# List all saved notebooks
python main.py --list-notebooks

# Advanced analysis (one-shot, no interactive loop)
python main.py --notebook-summary <notebook_id>          # cross-doc summary
python main.py --notebook-faq <notebook_id>              # generate FAQ
python main.py --notebook-review <notebook_id>           # literature review
python main.py --notebook-audio <notebook_id>            # audio script
python main.py --notebook-mindmap <notebook_id>          # mind map (saves .dot/.png/.svg, prints DOT)
python main.py --notebook-graph <notebook_id>            # knowledge graph (saves .dot/.png/.svg, prints DOT)
python main.py --notebook-compare <notebook_id> \
  --compare-docs "paper_0.pdf" "paper_1.pdf"             # source comparison
python main.py --notebook-timeline <notebook_id>         # timeline extraction
python main.py --notebook-study-table <notebook_id>      # cross-source study table
python main.py --notebook-pipeline <notebook_id>         # full 7-agent pipeline
  [--pipeline-query "focus query"]                       # optional focus query

Utilities
─────────
python main.py --list-docs          # list indexed documents
python main.py --clear-store        # wipe the vector store
python main.py --check-system       # hardware + model recommendation
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table
from tools.cli_recommender import (
    recommend_mode,
    recommend_post_research,
    recommend_proposal_pre,
    recommend_proposal_post,
    recommend_story_turn,
    recommend_wisdom_turn,
    print_recommendations,
)

console = Console()

# ── All recognised long-form flags (used for typo suggestions) ─────────────
_KNOWN_FLAGS = [
    "--goal", "--files", "--mode", "--model", "--web", "--output",
    "--list-docs", "--clear-store", "--propose", "--instructions",
    "--revise", "--revision", "--list-proposals", "--export-proposal",
    "--num-ctx", "--embed-model", "--top-k", "--check-system", "--verbose",
    "--story", "--story-session", "--list-stories", "--topic", "--style",
    "--wisdom", "--wisdom-session", "--scenario", "--list-wisdom",
    "--style-profile", "--create-style-profile", "--style-name", "--list-style-profiles",
    "--no-clarify",
    "--systematic-review", "--sr", "--inclusion", "--exclusion",
    "--with-systematic-review", "--with-sr",
    "--funding-agency", "--budget", "--duration", "--trl", "--consortium",
    "--project",
    "--shutdown",
    "--notebook", "--notebook-id", "--notebook-name", "--list-notebooks",
    "--notebook-summary", "--notebook-faq", "--notebook-review",
    "--notebook-audio", "--notebook-mindmap", "--notebook-graph",
    "--notebook-compare", "--compare-docs",
    "--notebook-timeline", "--notebook-study-table",
    "--notebook-pipeline", "--pipeline-query",
    "--call-file", "--ideas", "--institution", "--no-reviewer",
    "-g", "-f", "-m", "-o", "-i", "-v",
]

# Flags that indicate a mode choice — used for mode redirect hints
_MODE_ALIASES: dict[str, str] = {
    "story": "--story",
    "storytelling": "--story",
    "partner": "--story",
    "chat": "--story",
    "wisdom": "--wisdom",
    "wise": "--wisdom",
    "insight": "--wisdom",
    "notebook": "--notebook",
    "notebooklm": "--notebook",
    "rag": "--notebook",
}


def _configure_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class _SmartParser(argparse.ArgumentParser):
    """ArgumentParser that suggests corrections for unrecognised flags."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        console.print(f"\n[red]Error:[/red] {message}")

        # --- Suggest similar flags for typos ---
        match = re.search(r"(?:unrecognized argument|invalid choice)[s]?:?\s*'?(--?[\w-]+)", message)
        if match:
            bad = match.group(1)
            suggestions = difflib.get_close_matches(bad, _KNOWN_FLAGS, n=3, cutoff=0.5)
            if suggestions:
                console.print("\n[yellow]Did you mean:[/yellow]")
                for s in suggestions:
                    console.print(f"  [bold cyan]{s}[/bold cyan]")

        # --- Mode redirect hints ---
        mode_match = re.search(r"invalid choice:\s*'([\w-]+)'", message)
        if mode_match:
            bad_mode = mode_match.group(1).lower()
            redirect = _MODE_ALIASES.get(bad_mode)
            if redirect:
                console.print(
                    f"\n[yellow]Tip:[/yellow] [bold]{bad_mode}[/bold] is not a --mode value. "
                    f"Use [bold cyan]{redirect}[/bold cyan] instead.\n"
                    f"  Example: [dim]python main.py {redirect} --topic \"your topic\"[/dim]"
                )

        console.print(
            "\nRun [bold]python main.py --help[/bold] for full usage."
        )
        sys.exit(2)


def _parse_args():
    parser = _SmartParser(
        prog="research-agent",
        description="Agentic Research Assistant — 100% open-source AI for scientific research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Research modes (1–3) ──────────────────────────────────
    parser.add_argument(
        "--goal", "-g",
        type=str,
        help="Research goal or question (required for analysis modes)",
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        type=Path,
        help="One or more document files to analyse (PDF, DOCX, TXT)",
        default=[],
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["search"],
        default="search",
        help="search=academic literature search (default: search). "
             "For document + local source analysis, use --notebook.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model to use (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Supplement with Google web search (via FastAPI Google Search Service)",
    )
    parser.add_argument(
        "--with-systematic-review", "--with-sr",
        action="store_true",
        dest="with_systematic_review",
        help="After the main research workflow, also run a PRISMA systematic review on the same goal",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to save the Markdown report (default: ./outputs/report_<id>.md)",
    )
    parser.add_argument(
        "--list-docs",
        action="store_true",
        help="List all documents currently indexed in the vector store",
    )
    parser.add_argument(
        "--clear-store",
        action="store_true",
        help="Clear the vector store (removes all indexed documents)",
    )

    # ── ProposalGPT (Mode 2) ──────────────────────────────────
    parser.add_argument(
        "--propose",
        action="store_true",
        help="Run the ProposalGPT 9-agent pipeline to generate a full research proposal.",
    )
    parser.add_argument(
        "--call-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Funding call document (PDF, DOCX, TXT) to analyse.",
    )
    parser.add_argument(
        "--ideas",
        type=str,
        default="",
        help="Researcher ideas, project positioning, and background context.",
    )
    parser.add_argument(
        "--institution",
        type=str,
        default="",
        help="Institution/department description for the proposal.",
    )
    parser.add_argument(
        "--no-reviewer",
        action="store_true",
        dest="no_reviewer",
        help="Skip the reviewer simulation and improvement agents (faster).",
    )
    parser.add_argument(
        "--instructions", "-i",
        type=str,
        default="",
        help="Extra instructions for the proposal (tone, length, focus areas).",
    )
    parser.add_argument(
        "--list-proposals",
        action="store_true",
        help="List all saved ProposalGPT sessions.",
    )
    parser.add_argument(
        "--export-proposal",
        type=str,
        metavar="SESSION_ID",
        help="Export a saved ProposalGPT session to DOCX, PDF, Markdown, and CSV budget.",
    )

    # ── Research Partner — Mode 5 ─────────────────────────────
    parser.add_argument(
        "--story",
        action="store_true",
        help="Start an interactive Research Partner (storytelling) session",
    )
    parser.add_argument(
        "--story-session",
        type=str,
        metavar="SESSION_ID",
        help="Continue an existing Research Partner session by ID",
    )
    parser.add_argument(
        "--list-stories",
        action="store_true",
        help="List all saved Research Partner sessions",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Topic for a new Research Partner or Wisdom Mode session",
    )
    parser.add_argument(
        "--style",
        choices=["simple", "analogy", "walkthrough", "debate"],
        default="simple",
        help="Explanation style for Research Partner: simple | analogy | walkthrough | debate",
    )

    # ── Wisdom Mode — Mode 6 ──────────────────────────────────
    parser.add_argument(
        "--wisdom",
        action="store_true",
        help="Start an interactive Wisdom Mode session",
    )
    parser.add_argument(
        "--wisdom-session",
        type=str,
        metavar="SESSION_ID",
        help="Continue an existing Wisdom Mode session by ID",
    )
    parser.add_argument(
        "--list-wisdom",
        action="store_true",
        help="List all saved Wisdom Mode sessions",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="",
        help="Scenario description for Wisdom Mode (your specific situation)",
    )

    # ── Grammar Proofreading (Mode 6) ────────────────────────
    parser.add_argument(
        "--grammar-check",
        action="store_true",
        help="Run Grammar Proofreading mode (Mode 6) — proofread and polish English text",
    )
    parser.add_argument(
        "--grammar-session",
        type=str,
        metavar="SESSION_ID",
        help="Load an existing grammar proofreading session by ID",
    )
    parser.add_argument(
        "--list-grammar",
        action="store_true",
        help="List all saved grammar proofreading sessions",
    )
    parser.add_argument(
        "--style-level",
        type=str,
        default="professional_email",
        choices=["academic", "professional_email", "formal", "informal"],
        help="Writing context for Grammar Proofreading (default: professional_email)",
    )
    parser.add_argument(
        "--focus",
        nargs="*",
        choices=["grammar", "punctuation", "spelling", "style", "clarity"],
        default=[],
        help="Focus areas for proofreading (default: all areas)",
    )

    # ── Systematic Review (Mode 7) ────────────────────────────
    parser.add_argument(
        "--systematic-review", "--sr",
        action="store_true",
        help="Run a PRISMA systematic review (requires --goal). Searches Google Scholar, arXiv, "
             "Semantic Scholar, and CrossRef.",
    )
    parser.add_argument(
        "--inclusion",
        nargs="+",
        default=[],
        help="Inclusion criteria for systematic review (one string per criterion)",
    )
    parser.add_argument(
        "--exclusion",
        nargs="+",
        default=[],
        help="Exclusion criteria for systematic review (one string per criterion)",
    )
    parser.add_argument(
        "--sr-docx",
        action="store_true",
        help="Generate a PRISMA 2020-compliant DOCX manuscript after the SR run "
             "(saved to outputs/prisma_report_<id>.docx)",
    )
    parser.add_argument(
        "--sr-pdf",
        action="store_true",
        help="Generate a PRISMA 2020-compliant PDF manuscript after the SR run "
             "(saved to outputs/prisma_report_<id>.pdf)",
    )
    parser.add_argument(
        "--sr-plain-language",
        choices=["patient", "policy", "press", "all"],
        default="",
        metavar="FORMAT",
        help="Generate a plain-language summary after the SR run. "
             "Formats: patient (8th-grade), policy (policy brief), press (press release), all (all three). "
             "Saved to outputs/summary_<format>_<id>.txt",
    )
    parser.add_argument(
        "--sr-trends",
        action="store_true",
        help="Fetch field-wide publication-volume trend data from CrossRef after the SR run "
             "and print a year-by-year table",
    )
    parser.add_argument(
        "--sr-preprints",
        action="store_true",
        help="Check each included paper against CrossRef to flag unverified preprints and retractions",
    )
    parser.add_argument(
        "--sr-concept-drift",
        action="store_true",
        help="Analyse vocabulary evolution across time buckets in the retrieved corpus "
             "and print rising / declining terms",
    )
    parser.add_argument(
        "--sr-author",
        type=str,
        default="",
        metavar="NAME",
        help="Author name for the DOCX/PDF title page (used with --sr-docx / --sr-pdf)",
    )
    parser.add_argument(
        "--sr-institution",
        type=str,
        default="",
        metavar="NAME",
        help="Institution name for the DOCX/PDF title page (used with --sr-docx / --sr-pdf)",
    )

    # ── Research Notebook (Mode 8) ───────────────────────────
    parser.add_argument(
        "--notebook",
        action="store_true",
        help="Open the Research Notebook in interactive mode (Mode 8). "
             "Chat with your sources using grounded, cited answers.",
    )
    parser.add_argument(
        "--notebook-id",
        type=str,
        default="",
        metavar="ID",
        help="Notebook ID to open (use --list-notebooks to find IDs). "
             "If omitted with --notebook, you will be prompted to choose or create one.",
    )
    parser.add_argument(
        "--notebook-name",
        type=str,
        default="",
        metavar="NAME",
        help="Name for a new notebook (used with --notebook when no --notebook-id is given).",
    )
    parser.add_argument(
        "--list-notebooks",
        action="store_true",
        help="List all saved Research Notebooks.",
    )
    parser.add_argument(
        "--notebook-summary",
        type=str,
        default="",
        metavar="NOTEBOOK_ID",
        help="Generate a cross-document summary for the given notebook and print/save it.",
    )
    parser.add_argument(
        "--notebook-faq",
        type=str,
        default="",
        metavar="NOTEBOOK_ID",
        help="Generate an FAQ for the given notebook.",
    )
    parser.add_argument(
        "--notebook-review",
        type=str,
        default="",
        metavar="NOTEBOOK_ID",
        help="Generate a formal literature review for the given notebook.",
    )
    parser.add_argument(
        "--notebook-audio",
        type=str,
        default="",
        metavar="NOTEBOOK_ID",
        help="Generate a spoken-word audio summary script for the given notebook.",
    )
    parser.add_argument(
        "--notebook-mindmap",
        type=str,
        default="",
        metavar="NOTEBOOK_ID",
        help="Extract a mind map from the given notebook (saves .dot/.png/.svg; prints DOT to stdout).",
    )
    parser.add_argument(
        "--notebook-graph",
        type=str,
        default="",
        metavar="NOTEBOOK_ID",
        help="Extract a knowledge graph from the given notebook (saves .dot/.png/.svg; prints DOT to stdout).",
    )
    parser.add_argument(
        "--notebook-compare",
        type=str,
        default="",
        metavar="NOTEBOOK_ID",
        help="Compare two sources in the given notebook (requires --compare-docs).",
    )
    parser.add_argument(
        "--compare-docs",
        nargs=2,
        metavar=("SOURCE_A", "SOURCE_B"),
        default=[],
        help="Two source filenames to compare (used with --notebook-compare).",
    )
    parser.add_argument(
        "--notebook-timeline",
        type=str,
        default="",
        metavar="NOTEBOOK_ID",
        help="Extract a chronological timeline from the given notebook.",
    )
    parser.add_argument(
        "--notebook-study-table",
        type=str,
        default="",
        metavar="NOTEBOOK_ID",
        help="Generate a structured study comparison table for the given notebook.",
    )
    parser.add_argument(
        "--notebook-pipeline",
        type=str,
        default="",
        metavar="NOTEBOOK_ID",
        help="Run the full 7-agent pipeline for the given notebook "
             "(ingestion → summarization → retrieval → citation verification → "
             "knowledge graph → study guide → podcast script).",
    )
    parser.add_argument(
        "--pipeline-query",
        type=str,
        default="",
        metavar="QUERY",
        help="Optional focus query for Agent 3 (Retrieval) when using --notebook-pipeline.",
    )

    # ── Proposal Funding & Scope (Mode 4 extensions) ──────────
    parser.add_argument(
        "--funding-agency",
        type=str,
        default="None (General)",
        help="Funding agency for proposal (e.g. 'Horizon Europe', 'Vinnova', 'VR (Swedish Research Council)'). "
             "Adjusts tone, word-count targets, and required sections.",
    )
    parser.add_argument(
        "--budget",
        type=str,
        default="",
        help="Budget range for the proposal (e.g. '€500K – €1M')",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=36,
        help="Project duration in months (default: 36)",
    )
    parser.add_argument(
        "--trl",
        type=str,
        default="",
        help="Technology Readiness Level range (e.g. 'TRL 3–5')",
    )
    parser.add_argument(
        "--consortium",
        type=str,
        default="",
        help="Consortium description (e.g. '3 universities + 1 SME')",
    )

    # ── Writing Style Profiles ────────────────────────────────
    parser.add_argument(
        "--style-profile",
        type=str,
        metavar="NAME",
        default="",
        help="Activate a named writing style profile for this run "
             "(e.g. --style-profile 'Academic Writing'). "
             "Use --list-style-profiles to see available profiles.",
    )
    parser.add_argument(
        "--create-style-profile",
        action="store_true",
        help="Analyse uploaded documents and create a new writing style profile "
             "(requires --style-name and --files).",
    )
    parser.add_argument(
        "--style-name",
        type=str,
        default="",
        metavar="NAME",
        help="Name for the new style profile (used with --create-style-profile).",
    )
    parser.add_argument(
        "--list-style-profiles",
        action="store_true",
        help="List all saved writing style profiles.",
    )

    # ── Socratic Clarification ────────────────────────────────
    parser.add_argument(
        "--no-clarify",
        action="store_true",
        help="Skip the Socratic clarification step and run immediately. "
             "Useful for scripting or when you already know exactly what you want.",
    )

    # ── LLM / RAG tuning ─────────────────────────────────────
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=32768,
        help="LLM context window in tokens (default: 32768). "
             "Increase for models like mistral-nemo:12b (128k).",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model for hybrid RAG dense search (default: nomic-embed-text). "
             "Must be pulled first: ollama pull nomic-embed-text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of chunks to retrieve per query via hybrid RRF (default: 8).",
    )
    parser.add_argument(
        "--check-system",
        action="store_true",
        help="Detect hardware, list available Ollama models, and show the recommended "
             "configuration for this machine, then exit.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    # ── Sub-project selector ──────────────────────────────────
    parser.add_argument(
        "--project",
        type=str,
        choices=["mode1", "mode2", "mode3", "mode4", "mode5", "mode6"],
        metavar="MODE",
        default=None,
        help="Select a specific sub-project to run (e.g. --project mode2). "
             "If omitted for Modes 1-3, use --mode instead. "
             "For Modes 2-5, use --propose / --wisdom / --systematic-review / --notebook.",
    )

    # ── Safe shutdown utility ─────────────────────────────────
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Kill any stale processes on ports 8501 (Streamlit), 8000 (Google Search), "
             "and 11434 (Ollama), flush ChromaDB file locks, then exit. "
             "Run this before restarting if you see 'address already in use' errors.",
    )

    # ── Document parsing ──────────────────────────────────────
    parser.add_argument(
        "--docling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Docling for advanced document parsing (layout-aware PDF, tables as Markdown, "
             "PPTX, XLSX, HTML, images). On by default. Use --no-docling to use the fast "
             "pdfplumber-only parser instead.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR when using Docling (--docling required). "
             "Extracts text from scanned PDFs and images via EasyOCR.",
    )

    return parser.parse_args()


def _print_hardware_banner(ollama_base_url: str, user_model: str | None = None) -> dict:
    """Detect hardware, query Ollama, print a rich summary, and return the rec dict."""
    from config.hardware import detect_hardware, get_available_models, recommend_config

    hw = detect_hardware()
    available = get_available_models(ollama_base_url)
    rec = recommend_config(hw, available)

    gpu_labels = {
        "apple_silicon": "Apple Silicon (Metal GPU)",
        "nvidia": "NVIDIA GPU (CUDA)",
        "cpu": "CPU only",
    }

    hw_table = Table(title="System Hardware", border_style="cyan", show_header=False)
    hw_table.add_column("Key", style="bold")
    hw_table.add_column("Value")
    hw_table.add_row("CPU", hw["cpu"])
    hw_table.add_row("RAM", f"{hw['ram_gb']} GB")
    hw_table.add_row("Accelerator", gpu_labels.get(hw["gpu_type"], "Unknown"))
    hw_table.add_row("OS", f"{hw['os']} ({hw['arch']})")
    console.print(hw_table)

    if available:
        m_table = Table(title=f"Pulled Ollama Models ({len(available)})", border_style="green")
        m_table.add_column("Model", style="cyan")
        m_table.add_column("Status")
        for m in available:
            tag = "⭐ recommended" if m == rec.get("model") else ""
            m_table.add_row(m, tag)
        console.print(m_table)
    else:
        console.print("[yellow]⚠  No models found in Ollama (is it running?).[/yellow]")

    if rec["can_run"]:
        console.print(
            Panel(
                f"[bold green]Recommended model:[/bold green] {rec['model']}\n"
                f"[bold green]Recommended num_ctx:[/bold green] {rec['num_ctx']:,}\n\n"
                f"{rec['reasoning']}\n\n"
                f"[dim]{rec['hardware_note']}[/dim]",
                title="💡 Recommendation",
                border_style="green",
            )
        )
        if user_model and user_model not in available:
            console.print(
                f"[yellow]⚠  --model {user_model!r} is not in the pulled models list.\n"
                f"   Run: ollama pull {user_model}[/yellow]"
            )
        elif user_model and user_model != rec["model"]:
            console.print(
                f"[dim]ℹ  Using --model {user_model!r} (recommended: {rec['model']})[/dim]"
            )

        # ── Tight-fit: let the user choose between capable vs safe ───
        if rec.get("tight_fit") and rec.get("safe_alternative") and not user_model:
            safe_alt = rec["safe_alternative"]
            console.print(
                f"\n[bold yellow]⚠  Tight memory fit detected.[/bold yellow] "
                f"Choose which model to run:\n"
                f"  [bold cyan]1[/bold cyan]. [bold]{rec['model']}[/bold] — "
                f"higher capability, uses ≥85 % of available RAM\n"
                f"  [bold cyan]2[/bold cyan]. [bold]{safe_alt['name']}[/bold] — "
                f"{safe_alt['ram_gb']} GB, {safe_alt['note']}, more headroom"
            )
            if sys.stdin.isatty():
                choice = Prompt.ask("Your choice", choices=["1", "2"], default="1")
                if choice == "2":
                    rec = dict(rec)
                    rec["model"] = safe_alt["name"]
                    rec["num_ctx"] = safe_alt["num_ctx"]
                    rec["user_chose_model"] = True
                    console.print(f"[green]✓ Will use {safe_alt['name']}[/green]")
                else:
                    console.print(f"[green]✓ Will use {rec['model']} (tight fit)[/green]")
            else:
                console.print(
                    f"[dim]Non-interactive mode — defaulting to {rec['model']}. "
                    f"Pass --model {safe_alt['name']!r} to use the safer option.[/dim]"
                )
    else:
        console.print(
            Panel(
                f"[yellow]{rec['reasoning']}[/yellow]\n\n"
                f"{rec['hardware_note']}\n\n"
                + (f"[bold]Pull the recommended model:[/bold]\n  {rec['pull_command']}"
                   if rec["pull_command"] else ""),
                title="⚠️  No Compatible Model Available",
                border_style="yellow",
            )
        )

    return rec


def _cmd_list_docs():
    from config.settings import get_settings
    import chromadb

    cfg = get_settings()
    persist_dir = cfg.chroma_persist_dir
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        if not collections:
            console.print("[yellow]No embedding collections found in ChromaDB.[/yellow]")
            console.print(f"[dim]Store location: {persist_dir}[/dim]")
            return
        for col in collections:
            count = col.count()
            console.print(f"[cyan]{col.name}[/cyan] — {count} embedded chunks")
    except Exception as e:
        console.print(f"[red]Could not read ChromaDB store: {e}[/red]")
        console.print(f"[dim]Store location: {persist_dir}[/dim]")


def _cmd_clear_store():
    from config.settings import get_settings
    import chromadb

    cfg = get_settings()
    persist_dir = cfg.chroma_persist_dir
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        if not collections:
            console.print("[yellow]Nothing to clear — ChromaDB store is already empty.[/yellow]")
            return
        for col in collections:
            client.delete_collection(col.name)
            console.print(f"[green]✓ Deleted collection:[/green] {col.name}")
        console.print("[green]✓ ChromaDB store cleared.[/green]")
    except Exception as e:
        console.print(f"[red]Could not clear ChromaDB store: {e}[/red]")


def _ask_clarifying_questions(goal: str, mode: str, args) -> dict:
    """Interactively ask 2–3 Socratic clarifying questions in the terminal.

    Each question presents numbered options with the agent-recommended answer
    highlighted. Option N+1 is always "Other (please specify)" for custom input.
    Pressing Enter accepts the recommended option.

    Returns a dict of {key: answer} pairs, or {} if the LLM call fails.
    Respects --no-clarify to skip.
    """
    if getattr(args, "no_clarify", False):
        return {}
    from tools.clarifier import generate_clarifying_questions
    from config.settings import get_settings as _gs
    _cfg = _gs()
    console.print("\n[bold cyan]Clarifying your requirements…[/bold cyan]")
    try:
        questions = generate_clarifying_questions(
            goal=goal,
            mode=mode,
            model_name=args.model,
            ollama_base_url=_cfg.ollama_base_url,
            num_ctx=min(args.num_ctx, 4096),
        )
    except Exception as e:
        console.print(f"[yellow]Could not generate clarifying questions ({e}). Continuing.[/yellow]")
        return {}

    if not questions:
        return {}

    answers: dict = {}
    console.print("[dim]Select the option that best fits, or choose Other for a custom answer.[/dim]\n")
    for q in questions:
        question_text = q.get("question", "")
        key = q.get("key", "q")
        options: list = list(q.get("options") or [])
        recommended: str = q.get("recommended", "")

        # 1-based index of the recommended option (fallback: 1)
        default_n = 1
        if recommended and recommended in options:
            default_n = options.index(recommended) + 1

        other_n = len(options) + 1

        console.print(f"[bold]{question_text}[/bold]")
        for i, opt in enumerate(options, 1):
            if opt == recommended:
                console.print(f"  [cyan]{i}. {opt}[/cyan] [dim](Recommended)[/dim]")
            else:
                console.print(f"  {i}. {opt}")
        console.print(f"  {other_n}. Other (please specify)")

        raw = input(
            f"  → Enter number 1–{other_n} [default {default_n}]: "
        ).strip()

        if not raw:
            # Enter → accept recommended (or first option)
            answers[key] = recommended or (options[0] if options else "")
        elif raw.isdigit():
            n = int(raw)
            if n == other_n:
                custom = input("  → Please specify: ").strip()
                if custom:
                    answers[key] = custom
            elif 1 <= n <= len(options):
                answers[key] = options[n - 1]
            # out-of-range → skip silently
        else:
            # User typed free text directly
            answers[key] = raw

        console.print()

    return answers


def _cmd_list_style_profiles():
    from agents.style_memory import StyleMemory
    mem = StyleMemory()
    profiles = mem.list_profiles()
    if not profiles:
        console.print("[yellow]No style profiles saved yet.[/yellow]")
        console.print("[dim]Create one with: python main.py --create-style-profile "
                      "--style-name 'Name' --files paper.pdf[/dim]")
        return
    t = Table(title=f"Writing Style Profiles ({len(profiles)})", border_style="magenta")
    t.add_column("Profile ID", style="dim")
    t.add_column("Name", style="bold cyan")
    t.add_column("Documents", style="green")
    t.add_column("Created", style="dim")
    for p in profiles:
        docs = ", ".join(p.get("sample_documents", [])) or "—"
        if len(docs) > 50:
            docs = docs[:47] + "…"
        t.add_row(
            p["profile_id"],
            p["name"],
            docs,
            p.get("created_at", "")[:10],
        )
    console.print(t)
    console.print("\n[dim]Activate with: python main.py --style-profile 'Name' --goal \"...\"[/dim]")


def _cmd_create_style_profile(args):
    """Create a new writing style profile from uploaded documents."""
    from agents.style_memory import StyleMemory
    from config.settings import get_settings
    cfg = get_settings()

    if not args.style_name:
        console.print("[red]✗ --style-name is required with --create-style-profile.[/red]")
        console.print("[dim]Example: python main.py --create-style-profile "
                      "--style-name 'Academic Writing' --files paper.pdf[/dim]")
        return

    if not args.files:
        console.print("[red]✗ --files is required with --create-style-profile.[/red]")
        console.print("[dim]Provide 2–5 of your own writing samples.[/dim]")
        return

    console.print(f"\n[bold]Analysing writing style from {len(args.files)} document(s)…[/bold]")
    console.print("[dim]This uses the LLM — takes 30–60 seconds.[/dim]\n")

    from tools.document_tools import DocumentProcessor
    processor = DocumentProcessor(chunk_size=800, overlap=150, max_raw_chars=50_000)
    processed = []
    for f in args.files:
        try:
            doc = processor.process_file(Path(f))
            processed.append(doc)
            console.print(f"  ✓ {doc.filename} ({len(doc.raw_text):,} chars)")
        except Exception as e:
            console.print(f"  [yellow]⚠ Could not read {f}: {e}[/yellow]")

    if not processed:
        console.print("[red]✗ No documents could be processed.[/red]")
        return

    num_ctx = getattr(args, "num_ctx", 32768)
    model = getattr(args, "model", cfg.ollama_model)

    mem = StyleMemory()
    try:
        profile_id = mem.create_profile(
            name=args.style_name,
            documents=processed,
            model_name=model,
            ollama_base_url=cfg.ollama_base_url,
            num_ctx=num_ctx,
        )
        profile = mem.load(profile_id)
        console.print(
            Panel(
                f"[bold green]Profile created![/bold green]\n\n"
                f"Name: [cyan]{args.style_name}[/cyan]\n"
                f"ID:   [dim]{profile_id}[/dim]\n"
                f"Docs: {len(processed)}\n\n"
                f"[bold]Injection prompt preview:[/bold]\n"
                f"{(profile or {}).get('injection_prompt','')[:400]}…",
                title="✍️ Style Profile Created",
                border_style="magenta",
            )
        )
        console.print(
            f"\n[dim]Activate: python main.py --style-profile '{args.style_name}' "
            f"--goal \"your goal\"[/dim]"
        )
    except Exception as e:
        console.print(f"[red]✗ Style analysis failed: {e}[/red]")


def _cmd_list_proposals():
    """List saved ProposalGPT sessions from outputs/memory/proposal_gpt_*.json."""
    import glob, json as _json
    pattern = "outputs/memory/proposal_gpt_*.json"
    files = sorted(glob.glob(pattern), key=lambda f: Path(f).stat().st_mtime, reverse=True)

    if not files:
        console.print("[yellow]No saved ProposalGPT sessions found.[/yellow]")
        console.print("[dim]Generate one with: python main.py --propose --call-file funding_call.pdf[/dim]")
        return

    table = Table(title=f"ProposalGPT Sessions ({len(files)})", border_style="blue")
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Agency", max_width=20)
    table.add_column("Score", justify="center")
    table.add_column("Compliance", justify="center")
    table.add_column("Saved")

    for fpath in files[:20]:
        try:
            data = _json.loads(Path(fpath).read_text())
            table.add_row(
                data.get("session_id", Path(fpath).stem),
                data.get("funding_agency", "?")[:20],
                f"{data.get('overall_score', 0):.1f}/5.0",
                f"{data.get('compliance_score', 0)}/100",
                data.get("saved_at", "")[:16],
            )
        except Exception:
            table.add_row(Path(fpath).stem, "?", "?", "?", "?")
    console.print(table)
    console.print("\n[dim]Export: python main.py --export-proposal <session_id> --output ./my_proposal[/dim]")


def _cmd_propose_gpt(args) -> None:
    """Run the ProposalGPT 9-agent pipeline from the CLI."""
    from agents.proposal_gpt_state import create_proposal_gpt_state
    from agents.proposal_gpt_graph import run_proposal_gpt
    from tools.proposal_tools import assemble_full_proposal_md, build_proposal_docx, build_budget_csv

    # ── Process funding call ───────────────────────────────────────────────
    call_text = ""
    call_filename = "funding_call"

    if getattr(args, "call_file", None) and args.call_file:
        call_path = Path(args.call_file)
        if not call_path.exists():
            console.print(f"[red]✗ File not found: {call_path}[/red]")
            sys.exit(1)
        console.print(f"[bold]Processing funding call:[/bold] {call_path.name}")
        from tools.document_tools import DocumentProcessor
        processor = DocumentProcessor(chunk_size=1000, overlap=100, max_raw_chars=80_000)
        doc = processor.process_file(call_path)
        call_text = doc.raw_text
        call_filename = call_path.name
    elif getattr(args, "goal", None):
        # Use --goal as call context when no file provided
        call_text = f"Research Proposal Request: {args.goal}"
        call_filename = "goal_based"
    else:
        console.print("[red]✗ Provide --call-file or --goal for ProposalGPT.[/red]")
        sys.exit(1)

    # ── Process supporting files ────────────────────────────────────────────
    cv_texts: list = []
    if getattr(args, "files", None):
        from tools.document_tools import DocumentProcessor
        processor = DocumentProcessor(chunk_size=1000, overlap=100)
        for f in args.files:
            try:
                doc = processor.process_file(Path(f))
                cv_texts.append(doc.raw_text)
                console.print(f"  ✓ Loaded: {doc.filename}")
            except Exception as exc:
                console.print(f"  [yellow]⚠ Could not read {f}: {exc}[/yellow]")

    import uuid as _uuid
    session_id = str(_uuid.uuid4())[:8]
    initial_state = create_proposal_gpt_state(
        funding_call_text=call_text,
        user_ideas=getattr(args, "ideas", "") or getattr(args, "goal", "") or "",
        requirements=getattr(args, "instructions", ""),
        funding_agency=getattr(args, "funding_agency", "Generic"),
        model_name=args.model,
        num_ctx=args.num_ctx,
        session_id=session_id,
        cv_texts=cv_texts,
        institution_info=getattr(args, "institution", ""),
        funding_call_filename=call_filename,
    )

    console.print(
        Panel(
            f"[bold cyan]ProposalGPT — 9-Agent Pipeline[/bold cyan]\n\n"
            f"Funding Call: [italic]{call_filename}[/italic]\n"
            f"Agency: [bold]{initial_state['funding_agency']}[/bold]  |  "
            f"Model: [bold]{args.model}[/bold]  |  "
            f"Reviewer: [bold]{'off' if getattr(args,'no_reviewer',False) else 'on'}[/bold]",
            title="📋 ProposalGPT",
            border_style="blue",
        )
    )

    _AGENT_LABELS = {
        "funding_call_analyzer":   "Agent 1 — Analysing funding call",
        "research_planner":        "Agent 2 — Building proposal strategy",
        "literature_review_agent": "Agent 3 — Reviewing literature",
        "proposal_writer":         "Agent 4 — Writing proposal sections",
        "impact_agent":            "Agent 5 — Generating impact sections",
        "budget_agent":            "Agent 6 — Building budget",
        "compliance_agent":        "Agent 7 — Checking compliance",
        "reviewer_agent":          "Agent 8 — Simulating reviewers",
        "improvement_agent":       "Agent 9 — Improving weak sections",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("ProposalGPT", total=100)

        def _cb(node_name: str, state: dict) -> None:
            pct = state.get("progress_pct", 0)
            label = _AGENT_LABELS.get(node_name, node_name.replace("_", " ").title())
            progress.update(task, completed=pct, description=label)

        try:
            start = time.time()
            final = run_proposal_gpt(initial_state, stream_callback=_cb)
        except Exception as exc:
            console.print(f"\n[red]✗ Pipeline failed: {exc}[/red]")
            sys.exit(1)

    elapsed = time.time() - start
    console.print(f"\n[green]✓ Complete in {elapsed:.1f}s[/green]\n")

    # Print summary
    console.print(f"[bold]Compliance Score:[/bold] {final.get('compliance_score', 0)}/100")
    console.print(f"[bold]Reviewer Score:[/bold]   {final.get('overall_score', 0):.1f}/5.0")
    errors = final.get("errors", [])
    for e in errors:
        console.print(f"[yellow]⚠ {e}[/yellow]")

    # ── Save outputs ────────────────────────────────────────────────────────
    out_dir = Path(getattr(args, "output", None) or f"./outputs/proposal_{session_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Markdown proposal
    md = assemble_full_proposal_md(dict(final))
    (out_dir / "proposal.md").write_text(md, encoding="utf-8")
    console.print(f"[green]✓ Proposal (Markdown):[/green]  {out_dir / 'proposal.md'}")

    # DOCX
    try:
        docx_bytes = build_proposal_docx(dict(final))
        (out_dir / "proposal.docx").write_bytes(docx_bytes)
        console.print(f"[green]✓ Proposal (DOCX):    [/green]  {out_dir / 'proposal.docx'}")
    except Exception as exc:
        console.print(f"[yellow]⚠ DOCX failed: {exc}[/yellow]")

    # Budget CSV
    csv_bytes = build_budget_csv(dict(final))
    (out_dir / "budget.csv").write_bytes(csv_bytes)
    console.print(f"[green]✓ Budget (CSV):        [/green]  {out_dir / 'budget.csv'}")

    # Strategy report
    strategy_md = (
        f"# Win Strategy\n\n{final.get('win_strategy','')}\n\n"
        f"# SWOT Analysis\n\n{final.get('swot_analysis','')}\n\n"
        f"# Reviewer Perspective\n\n{final.get('reviewer_perspective','')}"
    )
    (out_dir / "strategy_report.md").write_text(strategy_md, encoding="utf-8")
    console.print(f"[green]✓ Strategy Report:     [/green]  {out_dir / 'strategy_report.md'}")

    # Reviewer report
    (out_dir / "reviewer_report.md").write_text(final.get("reviewer_report", ""), encoding="utf-8")
    console.print(f"[green]✓ Reviewer Report:     [/green]  {out_dir / 'reviewer_report.md'}")

    # Compliance report
    (out_dir / "compliance_report.md").write_text(final.get("compliance_report", ""), encoding="utf-8")
    console.print(f"[green]✓ Compliance Report:   [/green]  {out_dir / 'compliance_report.md'}")

    # Save session JSON for --list-proposals / --export-proposal
    import json as _json2, datetime as _dt
    session_data = dict(final)
    session_data["saved_at"] = _dt.datetime.utcnow().isoformat()
    Path("outputs/memory").mkdir(parents=True, exist_ok=True)
    session_file = Path(f"outputs/memory/proposal_gpt_{session_id}.json")
    session_file.write_text(_json2.dumps(session_data, default=str), encoding="utf-8")
    console.print(f"\n[dim]Session saved: {session_file}[/dim]")
    console.print(f"[dim]Export later: python main.py --export-proposal {session_id}[/dim]")

    # ── Interactive feedback refinement ────────────────────────────────────────
    if md:
        refined_md = _feedback_loop(
            current_output=md,
            mode="proposal",
            model_name=args.model,
            num_ctx=args.num_ctx,
            context=final.get("literature_review", "")[:1500],
        )
        if refined_md != md:
            refined_path = out_dir / "proposal_refined.md"
            refined_path.write_text(refined_md, encoding="utf-8")
            console.print(f"[green]✓ Refined proposal saved:[/green] {refined_path}")


def _cmd_export_proposal_gpt(session_id: str, args) -> None:
    """Export a saved ProposalGPT session to all formats."""
    import json as _json
    from tools.proposal_tools import assemble_full_proposal_md, build_proposal_docx, build_budget_csv

    session_file = Path(f"outputs/memory/proposal_gpt_{session_id}.json")
    if not session_file.exists():
        console.print(f"[red]✗ Session not found: {session_file}[/red]")
        console.print("[dim]List sessions: python main.py --list-proposals[/dim]")
        sys.exit(1)

    state = _json.loads(session_file.read_text())
    out_dir = Path(getattr(args, "output", None) or f"./outputs/proposal_{session_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    md = assemble_full_proposal_md(state)
    (out_dir / "proposal.md").write_text(md, encoding="utf-8")

    try:
        (out_dir / "proposal.docx").write_bytes(build_proposal_docx(state))
        console.print(f"[green]✓ DOCX:[/green] {out_dir / 'proposal.docx'}")
    except Exception as exc:
        console.print(f"[yellow]⚠ DOCX: {exc}[/yellow]")

    try:
        from tools.export_tools import build_pdf
        pdf = build_pdf(md, [])
        (out_dir / "proposal.pdf").write_bytes(pdf)
        console.print(f"[green]✓ PDF:[/green]  {out_dir / 'proposal.pdf'}")
    except Exception as exc:
        console.print(f"[yellow]⚠ PDF: {exc}[/yellow]")

    (out_dir / "budget.csv").write_bytes(build_budget_csv(state))
    (out_dir / "compliance_report.md").write_text(state.get("compliance_report", ""), encoding="utf-8")
    (out_dir / "reviewer_report.md").write_text(state.get("reviewer_report", ""), encoding="utf-8")

    console.print(f"[green]✓ Markdown:[/green] {out_dir / 'proposal.md'}")
    console.print(f"[green]✓ Budget CSV:[/green] {out_dir / 'budget.csv'}")
    console.print(f"\n[bold]All files saved to:[/bold] {out_dir}")


def _cmd_list_stories():
    """Print a table of all saved Research Partner sessions."""
    from agents.story_memory import StorytellerMemory
    memory = StorytellerMemory()
    sessions = memory.list_sessions()

    if not sessions:
        console.print("[yellow]No saved Research Partner sessions found.[/yellow]")
        console.print("[dim]Start one with: python main.py --story --topic \"your topic\"[/dim]")
        return

    table = Table(title=f"Research Partner Sessions ({len(sessions)})", border_style="cyan")
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Topic", max_width=40)
    table.add_column("Turns", justify="center")
    table.add_column("Concepts", justify="center")
    table.add_column("Last Modified")

    for s in sessions:
        table.add_row(
            s["session_id"],
            s.get("topic", "")[:40],
            str(s.get("turn_count", 0)),
            str(len(s.get("concepts_covered", []))),
            s.get("last_modified", "")[:16],
        )
    console.print(table)


def _cmd_list_wisdom():
    """Print a table of all saved Wisdom Mode sessions."""
    from agents.wisdom_memory import WisdomMemory
    memory = WisdomMemory()
    sessions = memory.list_sessions()

    if not sessions:
        console.print("[yellow]No saved Wisdom Mode sessions found.[/yellow]")
        console.print("[dim]Start one with: python main.py --wisdom --topic \"your topic\"[/dim]")
        return

    table = Table(title=f"Wisdom Mode Sessions ({len(sessions)})", border_style="magenta")
    table.add_column("Session ID", style="magenta", no_wrap=True)
    table.add_column("Topic", max_width=35)
    table.add_column("Phase")
    table.add_column("Turns", justify="center")
    table.add_column("Last Modified")

    for s in sessions:
        phase = s.get("phase", "clarifying")
        phase_label = {"clarifying": "🔵 Clarifying", "done": "✅ Done"}.get(phase, phase)
        table.add_row(
            s["session_id"],
            s.get("topic", "")[:35],
            phase_label,
            str(s.get("turn_count", 0)),
            s.get("last_modified", "")[:16],
        )
    console.print(table)


def _cmd_list_notebooks():
    """Print a table of all saved Research Notebooks."""
    from agents.notebook_memory import NotebookMemory
    memory = NotebookMemory()
    notebooks = memory.list_notebooks()

    if not notebooks:
        console.print("[yellow]No saved Research Notebooks found.[/yellow]")
        console.print("[dim]Start one with: python main.py --notebook --notebook-name \"My Notebook\"[/dim]")
        return

    table = Table(title=f"Research Notebooks ({len(notebooks)})", border_style="blue")
    table.add_column("Notebook ID", style="cyan", no_wrap=True)
    table.add_column("Name", max_width=40)
    table.add_column("Sources", justify="center")
    table.add_column("Turns", justify="center")
    table.add_column("Last Modified")

    for nb in notebooks:
        table.add_row(
            nb["notebook_id"],
            nb.get("name", "Untitled")[:40],
            str(nb.get("source_count", 0)),
            str(nb.get("turn_count", 0)),
            nb.get("last_modified", "")[:16],
        )
    console.print(table)


def _cmd_notebook_advanced(notebook_id: str, feature: str, args) -> None:
    """Run an advanced notebook feature (summary, faq, review, audio, mindmap, graph, compare)."""
    from config.settings import get_settings
    settings_cfg = get_settings()
    settings = {
        "model": args.model,
        "num_ctx": args.num_ctx,
        "embed_model": getattr(args, "embed_model", settings_cfg.embedding_model),
    }

    from agents.notebook_memory import NotebookMemory
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        console.print(f"[red]Notebook not found:[/red] {notebook_id}")
        console.print("[dim]Use --list-notebooks to see available notebooks.[/dim]")
        return

    nb_name = notebook.get("name", "Notebook")
    console.rule(f"[bold blue]{nb_name}[/bold blue]")
    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if feature == "summary":
        from agents.notebook_advanced import generate_cross_document_summary
        with console.status("[bold blue]Generating cross-document summary…[/bold blue]"):
            result, err = generate_cross_document_summary(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]")
            return
        console.print(Markdown(result))
        out_path = out_dir / f"summary_{notebook_id}.md"
        out_path.write_text(result, encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {out_path}")

    elif feature == "faq":
        from agents.notebook_advanced import generate_faq
        with console.status("[bold blue]Generating FAQ…[/bold blue]"):
            items, err = generate_faq(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]")
            return
        for i, item in enumerate(items, 1):
            console.print(f"\n[bold cyan]Q{i}: {item.get('question', '')}[/bold cyan]")
            console.print(Markdown(item.get("answer", "")))
        md = "\n\n".join(f"### {it.get('question','')}\n{it.get('answer','')}" for it in items)
        out_path = out_dir / f"faq_{notebook_id}.md"
        out_path.write_text(md, encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {out_path}")

    elif feature == "review":
        from agents.notebook_advanced import generate_literature_review
        with console.status("[bold blue]Generating literature review…[/bold blue]"):
            result, err = generate_literature_review(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]")
            return
        console.print(Markdown(result))
        out_path = out_dir / f"literature_review_{notebook_id}.md"
        out_path.write_text(result, encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {out_path}")

    elif feature == "audio":
        from agents.notebook_advanced import generate_audio_summary, synthesize_speech
        with console.status("[bold blue]Generating audio summary script…[/bold blue]"):
            result, err = generate_audio_summary(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]")
            return
        console.print(Panel(result, title="🔊 Audio Summary Script", border_style="cyan"))
        txt_path = out_dir / f"audio_summary_{notebook_id}.txt"
        txt_path.write_text(result, encoding="utf-8")
        console.print(f"[green]✓ Script saved:[/green] {txt_path}")
        # Synthesize WAV
        with console.status("[bold blue]Synthesizing speech (~30 s)…[/bold blue]"):
            wav_bytes, wav_err = synthesize_speech(result)
        if wav_err:
            console.print(f"[yellow]⚠ WAV synthesis unavailable: {wav_err}[/yellow]")
        else:
            wav_path = out_dir / f"audio_summary_{notebook_id}.wav"
            wav_path.write_bytes(wav_bytes)
            console.print(f"[green]✓ Audio saved:[/green] {wav_path}")

    elif feature == "timeline":
        from agents.notebook_advanced import extract_timeline
        with console.status("[bold blue]Extracting timeline…[/bold blue]"):
            items, err = extract_timeline(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]")
            return
        src_names = [s["filename"] for s in notebook.get("sources", [])]
        table = Table(title=f"Timeline — {nb_name}", border_style="blue")
        table.add_column("Year", no_wrap=True)
        table.add_column("Event", max_width=55)
        table.add_column("Significance", max_width=40)
        table.add_column("Source", max_width=20)
        for item in items:
            src_n = item.get("source", 0)
            src_label = (
                src_names[src_n - 1][:20]
                if isinstance(src_n, int) and 1 <= src_n <= len(src_names) else "—"
            )
            table.add_row(
                item.get("year", "n.d."),
                item.get("event", "")[:55],
                item.get("significance", "")[:40],
                src_label,
            )
        console.print(table)
        md_lines = ["| Year | Event | Significance | Source |",
                    "|------|-------|-------------|--------|"]
        for item in items:
            src_n = item.get("source", 0)
            src_label = (
                src_names[src_n - 1][:20]
                if isinstance(src_n, int) and 1 <= src_n <= len(src_names) else "—"
            )
            md_lines.append(
                f"| {item.get('year','n.d.')} | {item.get('event','')} | "
                f"{item.get('significance','')} | {src_label} |"
            )
        out_path = out_dir / f"timeline_{notebook_id}.md"
        out_path.write_text("\n".join(md_lines), encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {out_path}")

    elif feature == "study-table":
        from agents.notebook_advanced import generate_study_comparison
        with console.status("[bold blue]Generating study comparison table…[/bold blue]"):
            result, err = generate_study_comparison(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]")
            return
        console.print(Markdown(result))
        out_path = out_dir / f"study_comparison_{notebook_id}.md"
        out_path.write_text(result, encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {out_path}")

    elif feature == "mindmap":
        from agents.notebook_advanced import generate_mindmap, render_dot_bytes
        with console.status("[bold blue]Extracting mind map…[/bold blue]"):
            dot, err = generate_mindmap(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]")
            return
        base = out_dir / f"mindmap_{notebook_id}"
        base.with_suffix(".dot").write_text(dot, encoding="utf-8")
        console.print(f"[green]✓[/green] DOT: {base.with_suffix('.dot')}")
        console.print("\n[dim]── DOT source (pipe to xdot or dot -Tpng) ──────────────[/dim]")
        print(dot)
        console.print("[dim]────────────────────────────────────────────────────────[/dim]\n")
        for fmt in ("png", "svg"):
            img, img_err = render_dot_bytes(dot, fmt)
            if img:
                base.with_suffix(f".{fmt}").write_bytes(img)
                console.print(f"[green]✓[/green] {fmt.upper()}: {base.with_suffix(f'.{fmt}')}")
            else:
                console.print(f"[yellow]⚠ {fmt.upper()} render unavailable: {img_err}[/yellow]")

    elif feature == "graph":
        from agents.notebook_advanced import extract_knowledge_graph, render_dot_bytes
        with console.status("[bold blue]Extracting knowledge graph…[/bold blue]"):
            dot, err = extract_knowledge_graph(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]")
            return
        base = out_dir / f"knowledge_graph_{notebook_id}"
        base.with_suffix(".dot").write_text(dot, encoding="utf-8")
        console.print(f"[green]✓[/green] DOT: {base.with_suffix('.dot')}")
        console.print("\n[dim]── DOT source (pipe to xdot or dot -Tpng) ──────────────[/dim]")
        print(dot)
        console.print("[dim]────────────────────────────────────────────────────────[/dim]\n")
        for fmt in ("png", "svg"):
            img, img_err = render_dot_bytes(dot, fmt)
            if img:
                base.with_suffix(f".{fmt}").write_bytes(img)
                console.print(f"[green]✓[/green] {fmt.upper()}: {base.with_suffix(f'.{fmt}')}")
            else:
                console.print(f"[yellow]⚠ {fmt.upper()} render unavailable: {img_err}[/yellow]")

    elif feature == "compare":
        compare_docs = getattr(args, "compare_docs", [])
        if len(compare_docs) < 2:
            console.print("[red]--compare-docs requires exactly two source filenames.[/red]")
            console.print("[dim]Example: --compare-docs paper_0.pdf paper_1.pdf[/dim]")
            return
        # Resolve filenames to doc_ids
        sources = notebook.get("sources", [])
        name_to_id = {s["filename"]: s["doc_id"] for s in sources}
        doc_a = name_to_id.get(compare_docs[0], "")
        doc_b = name_to_id.get(compare_docs[1], "")
        if not doc_a or not doc_b:
            available = [s["filename"] for s in sources]
            console.print(f"[red]Source not found.[/red] Available: {available}")
            return
        from agents.notebook_advanced import compare_sources
        with console.status("[bold blue]Comparing sources…[/bold blue]"):
            result, err = compare_sources(notebook_id, doc_a, doc_b, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]")
            return
        console.print(Markdown(result))
        out_path = out_dir / f"comparison_{notebook_id}.md"
        out_path.write_text(result, encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {out_path}")


def _cmd_notebook_pipeline(notebook_id: str, args) -> None:
    """
    Run the full 7-agent Mode 8 pipeline and save all outputs to outputs/.

    Agents:
      1  Document Ingestion    — loads sources + chunks from notebook JSON
      2  Summarization         — per-doc + cross-document synthesis
      3  Retrieval             — hybrid RAG on focus query
      4  Citation Verification — verifies summary claims against source material
      5  Knowledge Graph       — entity–relationship graph (DOT + PNG + SVG)
      6  Study Guide           — key concepts, glossary, Q&A, summary (MD + DOCX + PDF)
      7  Podcast Script        — two-speaker dialogue transcript (TXT)
    """
    from pathlib import Path

    from agents.notebook_memory import NotebookMemory
    from agents.notebook_pipeline_graph import run_notebook_pipeline
    from agents.notebook_pipeline_state import create_pipeline_state

    notebook = NotebookMemory().load(notebook_id)
    if not notebook:
        console.print(f"[red]Notebook '{notebook_id}' not found.[/red]")
        return

    nb_name = notebook.get("name", notebook_id)
    console.print(f"\n[bold]Running 7-agent pipeline[/bold] for notebook: [cyan]{nb_name}[/cyan]")
    console.print(f"  Sources : {len(notebook.get('sources', []))}")
    console.print(f"  Chunks  : {len(notebook.get('chunks', []))}\n")

    from config.settings import get_settings as _get_cfg
    _cfg = _get_cfg()
    settings = {
        "model": args.model,
        "num_ctx": args.num_ctx,
        "embed_model": getattr(args, "embed_model", _cfg.embedding_model),
    }
    query = getattr(args, "pipeline_query", "") or ""

    initial = create_pipeline_state(
        notebook_id=notebook_id,
        settings=settings,
        query=query,
    )

    _LABELS = {
        "ingest": "Agent 1 — Document Ingestion",
        "summarize": "Agent 2 — Summarization",
        "retrieve": "Agent 3 — Retrieval",
        "verify_citations": "Agent 4 — Citation Verification",
        "build_kg": "Agent 5 — Knowledge Graph",
        "generate_study_guide": "Agent 6 — Study Guide",
        "generate_podcast": "Agent 7 — Podcast Script",
    }

    def _cb(node_name: str, partial: dict) -> None:
        pct = partial.get("progress_pct", 0)
        label = _LABELS.get(node_name, node_name)
        console.print(f"  [green]✓[/green] {label} ({pct}%)")

    try:
        final = run_notebook_pipeline(initial, stream_callback=_cb)
    except Exception as exc:
        console.print(f"[red]Pipeline failed: {exc}[/red]")
        return

    errors = final.get("errors", [])
    if errors:
        console.print(f"\n[yellow]Warnings ({len(errors)}):[/yellow]")
        for e in errors:
            console.print(f"  [yellow]·[/yellow] {e}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    safe_name = nb_name.replace(" ", "_")[:30]

    console.print("\n[bold]Saving outputs…[/bold]")

    # Summary
    if final.get("cross_summary"):
        p = out_dir / f"pipeline_summary_{safe_name}.md"
        p.write_text(final["cross_summary"], encoding="utf-8")
        console.print(f"  [green]✓[/green] Summary:          {p}")

    # Citation report
    if final.get("citation_report"):
        p = out_dir / f"pipeline_citations_{safe_name}.md"
        p.write_text(final["citation_report"], encoding="utf-8")
        console.print(f"  [green]✓[/green] Citation report:  {p}")

    # Knowledge graph — DOT + PNG + SVG
    if final.get("knowledge_graph_dot"):
        dot = final["knowledge_graph_dot"]
        base = out_dir / f"pipeline_kg_{safe_name}"
        base.with_suffix(".dot").write_text(dot, encoding="utf-8")
        console.print(f"  [green]✓[/green] KG DOT:           {base.with_suffix('.dot')}")
        console.print("\n[dim]── DOT source ────────────────────────────────────────────[/dim]")
        print(dot)
        console.print("[dim]─────────────────────────────────────────────────────────[/dim]\n")
        from agents.notebook_advanced import render_dot_bytes
        for fmt in ("png", "svg"):
            img, img_err = render_dot_bytes(dot, fmt)
            if img:
                base.with_suffix(f".{fmt}").write_bytes(img)
                console.print(f"  [green]✓[/green] KG {fmt.upper()}:          {base.with_suffix(f'.{fmt}')}")
            else:
                console.print(f"  [yellow]⚠ KG {fmt.upper()} unavailable: {img_err}[/yellow]")

    # Study guide — MD + DOCX + PDF
    if final.get("study_guide"):
        guide = final["study_guide"]
        p = out_dir / f"pipeline_study_guide_{safe_name}.md"
        p.write_text(guide, encoding="utf-8")
        console.print(f"  [green]✓[/green] Study guide (.md): {p}")
        try:
            from tools.export_tools import build_docx, build_pdf
            docx_bytes = build_docx(guide, [])
            docx_p = out_dir / f"pipeline_study_guide_{safe_name}.docx"
            docx_p.write_bytes(docx_bytes)
            console.print(f"  [green]✓[/green] Study guide (.docx): {docx_p}")
            pdf_bytes = build_pdf(guide, [])
            pdf_p = out_dir / f"pipeline_study_guide_{safe_name}.pdf"
            pdf_p.write_bytes(pdf_bytes)
            console.print(f"  [green]✓[/green] Study guide (.pdf): {pdf_p}")
        except Exception as exc:
            console.print(f"  [yellow]⚠ DOCX/PDF export: {exc}[/yellow]")

    # Podcast script
    if final.get("podcast_script"):
        p = out_dir / f"pipeline_podcast_{safe_name}.txt"
        p.write_text(final["podcast_script"], encoding="utf-8")
        console.print(f"  [green]✓[/green] Podcast script:   {p}")

    console.print("\n[bold green]Pipeline complete.[/bold green]")

    # ── Interactive feedback refinement ────────────────────────────────────────
    study_guide = final.get("study_guide", "")
    if study_guide:
        context = " ".join(
            s.get("filename", "") for s in notebook.get("sources", [])[:5]
        )
        refined = _feedback_loop(
            current_output=study_guide,
            mode="notebook",
            model_name=args.model,
            num_ctx=args.num_ctx,
            context=context,
        )
        if refined != study_guide:
            refined_p = out_dir / f"pipeline_study_guide_{safe_name}_refined.md"
            refined_p.write_text(refined, encoding="utf-8")
            console.print(f"  [green]✓[/green] Refined study guide: {refined_p}")


def _cmd_notebook(args) -> None:
    """Interactive Research Notebook (Mode 8) with grounded Q&A + slash commands."""
    from agents.notebook_memory import NotebookMemory
    from agents.notebook_state import create_notebook_state
    from agents.notebook_graph import run_notebook_turn
    from config.settings import get_settings

    settings_cfg = get_settings()
    settings = {
        "model": args.model,
        "num_ctx": args.num_ctx,
        "embed_model": getattr(args, "embed_model", settings_cfg.embedding_model),
        "chunk_size": 800,
        "chunk_overlap": 150,
    }

    memory = NotebookMemory()
    notebook_id = getattr(args, "notebook_id", "")

    if not notebook_id:
        # Show existing notebooks and let user choose or create
        notebooks = memory.list_notebooks()
        if notebooks:
            console.print("\n[bold]Existing notebooks:[/bold]")
            for i, nb in enumerate(notebooks, 1):
                console.print(
                    f"  [cyan]{i}[/cyan]. {nb['name']} "
                    f"[dim]({nb['source_count']} sources, {nb['turn_count']} turns)[/dim]"
                )
            console.print(f"  [cyan]{len(notebooks)+1}[/cyan]. [bold]Create new notebook[/bold]")
            choice = Prompt.ask(
                "Select", choices=[str(i) for i in range(1, len(notebooks) + 2)], default="1",
            )
            idx = int(choice) - 1
            if idx < len(notebooks):
                notebook_id = notebooks[idx]["notebook_id"]
            else:
                notebook_id = ""  # create new
        if not notebook_id:
            nb_name = getattr(args, "notebook_name", "") or Prompt.ask("Notebook name")
            notebook_id = memory.new_notebook(nb_name.strip() or "Untitled Notebook")
            console.print(f"[green]✓ Created notebook:[/green] {notebook_id}")

    notebook = memory.load(notebook_id)
    if not notebook:
        console.print(f"[red]Notebook not found:[/red] {notebook_id}")
        return

    # Ingest any --files passed on the command line
    if getattr(args, "files", None):
        from tools.hybrid_store import get_or_create_store
        with console.status("[bold]Processing documents…[/bold]"):
            processed = _process_files(
                list(args.files),
                use_docling=getattr(args, "docling", True),
                use_ocr=getattr(args, "ocr", False),
            )
        if processed:
            store = get_or_create_store(
                session_id=f"notebook_{notebook_id}",
                embed_model=settings["embed_model"],
                ollama_base_url=settings_cfg.ollama_base_url,
                persist_dir=settings_cfg.chroma_persist_dir,
            )
            try:
                store.add_documents(processed)
            except RuntimeError as e:
                console.print(f"[yellow]⚠ Embedding model unavailable: {e}[/yellow]")
            for doc in processed:
                if memory.add_source(notebook_id, doc, source_type="file"):
                    console.print(f"[green]✓ Added:[/green] {doc.filename}")
            notebook = memory.load(notebook_id)

    console.rule("[bold blue]Research Notebook — Mode 8[/bold blue]")
    sources = notebook.get("sources", [])
    console.print(Panel(
        f"Notebook: [bold]{notebook.get('name', 'Notebook')}[/bold]\n"
        f"ID:       [cyan]{notebook_id}[/cyan]\n"
        f"Sources:  {len(sources)}\n"
        f"Turns:    {len(notebook.get('conversation', []))}"
        + (("\nSources:  " + ", ".join(s['filename'][:25] for s in sources[:4])
            + (" …" if len(sources) > 4 else "")) if sources else ""),
        title="📓 Session Info",
        border_style="blue",
    ))

    console.print(
        "[dim]Type your question, or use a slash command:\n"
        "  /add <file>    Add a document    /url <url>      Add a web page\n"
        "  /sources       List sources      /summary        Cross-doc summary\n"
        "  /faq           Generate FAQ      /review         Literature review\n"
        "  /audio         Audio script+WAV  /mindmap        Mind map (DOT+PNG+SVG)\n"
        "  /graph         Knowledge graph   /compare        Compare two sources\n"
        "  /timeline      Extract timeline  /study-table    Study comparison table\n"
        "  /quit          Exit[/dim]\n"
    )

    # Print recent conversation history
    history = notebook.get("conversation", [])
    if history:
        console.print("[dim]Recent conversation:[/dim]\n")
        for turn in history[-4:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                console.print(f"[bold green]You[/bold green]: {content[:200]}")
            else:
                console.print(f"[bold blue]Notebook[/bold blue]: {content[:300]}"
                              + (" …" if len(content) > 300 else ""))
                citations = turn.get("citations") or []
                if citations:
                    refs = ", ".join(
                        f"[{c['n']}] {c.get('doc_name','')} p.{c.get('page','')}"
                        for c in citations
                    )
                    console.print(f"[dim]  Sources: {refs}[/dim]")
            console.print()

    last_questions: list[str] = []

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Notebook saved. Goodbye![/dim]")
            _cli_safe_exit()
            break

        stripped = user_input.strip()

        if not stripped:
            continue

        if stripped.lower() in ("quit", "exit", "bye", "q", ":q", "/quit", "/exit"):
            console.print("[dim]Notebook saved. Goodbye![/dim]")
            _cli_safe_exit()
            break

        # Shortcut: numbers 1/2/3 pick a suggested question
        if stripped in ("1", "2", "3") and last_questions:
            idx = int(stripped) - 1
            if idx < len(last_questions):
                stripped = last_questions[idx]
                console.print(f"[dim]→ {stripped}[/dim]")
            else:
                console.print("[yellow]No question at that number.[/yellow]")
                continue

        # ── Slash commands ────────────────────────────────────
        if stripped.startswith("/"):
            parts = stripped.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/sources":
                nb = memory.load(notebook_id)
                srcs = nb.get("sources", []) if nb else []
                if srcs:
                    for s in srcs:
                        icon = "🔗" if s.get("source_type") == "url" else "📄"
                        console.print(f"  {icon} {s['filename']} ({s.get('total_chunks',0)} chunks)")
                else:
                    console.print("[yellow]No sources yet.[/yellow]")

            elif cmd == "/add":
                fp = Path(arg.strip()) if arg.strip() else None
                if not fp:
                    fp = Path(Prompt.ask("File path"))
                docs = _process_files(
                    [fp],
                    use_docling=getattr(args, "docling", True),
                    use_ocr=getattr(args, "ocr", False),
                )
                if docs:
                    from tools.hybrid_store import get_or_create_store
                    store = get_or_create_store(
                        session_id=f"notebook_{notebook_id}",
                        embed_model=settings["embed_model"],
                        ollama_base_url=settings_cfg.ollama_base_url,
                        persist_dir=settings_cfg.chroma_persist_dir,
                    )
                    try:
                        store.add_documents(docs)
                    except RuntimeError as e:
                        console.print(f"[yellow]⚠ Embedding unavailable: {e}[/yellow]")
                    for doc in docs:
                        if memory.add_source(notebook_id, doc, source_type="file"):
                            console.print(f"[green]✓ Added:[/green] {doc.filename}")

            elif cmd == "/url":
                url = arg.strip() or Prompt.ask("URL")
                from tools.document_tools import DocumentProcessor
                from tools.web_loader import load_url_as_document
                processor = DocumentProcessor(
                    chunk_size=settings["chunk_size"],
                    overlap=settings["chunk_overlap"],
                )
                with console.status("Fetching…"):
                    doc, err = load_url_as_document(url, processor)
                if err:
                    console.print(f"[red]✗ {err}[/red]")
                else:
                    from tools.hybrid_store import get_or_create_store
                    store = get_or_create_store(
                        session_id=f"notebook_{notebook_id}",
                        embed_model=settings["embed_model"],
                        ollama_base_url=settings_cfg.ollama_base_url,
                        persist_dir=settings_cfg.chroma_persist_dir,
                    )
                    try:
                        store.add_documents([doc])
                    except RuntimeError as e:
                        console.print(f"[yellow]⚠ Embedding unavailable: {e}[/yellow]")
                    if memory.add_source(notebook_id, doc, source_type="url", url=url):
                        console.print(f"[green]✓ Added:[/green] {doc.filename}")

            elif cmd in ("/summary", "/faq", "/review", "/audio", "/mindmap",
                         "/graph", "/timeline", "/study-table"):
                feature_map = {
                    "/summary": "summary", "/faq": "faq", "/review": "review",
                    "/audio": "audio", "/mindmap": "mindmap", "/graph": "graph",
                    "/timeline": "timeline", "/study-table": "study-table",
                }
                _cmd_notebook_advanced(notebook_id, feature_map[cmd], args)

            elif cmd == "/compare":
                nb = memory.load(notebook_id)
                srcs = nb.get("sources", []) if nb else []
                if len(srcs) < 2:
                    console.print("[yellow]Need at least 2 sources to compare.[/yellow]")
                else:
                    console.print("Available sources:")
                    for i, s in enumerate(srcs, 1):
                        console.print(f"  {i}. {s['filename']}")
                    a = Prompt.ask("Select source A (number)")
                    b = Prompt.ask("Select source B (number)")
                    try:
                        sa = srcs[int(a) - 1]["filename"]
                        sb = srcs[int(b) - 1]["filename"]
                        _fake_args = type("A", (), {"compare_docs": [sa, sb], **vars(args)})()
                        _cmd_notebook_advanced(notebook_id, "compare", _fake_args)
                    except (ValueError, IndexError):
                        console.print("[red]Invalid selection.[/red]")
            else:
                console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
            continue

        # ── Q&A: ask a question ───────────────────────────────
        nb = memory.load(notebook_id)
        if not nb or not nb.get("sources"):
            console.print(
                "[yellow]No sources. Use /add <file> or /url <url> to add documents.[/yellow]"
            )
            continue

        state = create_notebook_state(
            user_message=stripped,
            notebook_id=notebook_id,
            model_name=args.model,
            num_ctx=args.num_ctx,
            embed_model=settings["embed_model"],
        )

        with console.status("[bold blue]Searching notebook…[/bold blue]"):
            try:
                final = run_notebook_turn(state)
            except Exception as e:
                console.print(f"[red]✗ Error: {e}[/red]")
                if getattr(args, "verbose", False):
                    import traceback
                    traceback.print_exc()
                continue

        response = final.get("assistant_response", "")
        console.print(f"\n[bold blue]Notebook[/bold blue]:\n")
        console.print(Markdown(response))

        citations = final.get("citations", [])
        if citations:
            refs = ", ".join(
                f"[{c['n']}] {c.get('doc_name','')} p.{c.get('page','')}"
                for c in citations
            )
            console.print(f"[dim]Sources: {refs}[/dim]")

        last_questions = final.get("suggested_questions", [])
        if last_questions:
            console.print("\n[dim]Follow-up questions:[/dim]")
            for i, q in enumerate(last_questions, 1):
                console.print(f"  [dim]{i}. {q}[/dim]")

        for err in final.get("errors", []):
            console.print(f"[yellow]⚠ {err}[/yellow]")
        console.print()


def _print_conversation_history(turns: list[dict], limit: int = 6) -> None:
    """Print the last N turns of a conversation to the terminal."""
    recent = turns[-limit:] if len(turns) > limit else turns
    if not recent:
        return
    skipped = len(turns) - len(recent)
    if skipped:
        console.print(f"[dim]  … {skipped} earlier turns not shown …[/dim]\n")
    for turn in recent:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            console.print(f"[bold green]You[/bold green]: {content}")
        else:
            console.print(f"[bold magenta]Partner[/bold magenta]: {content[:500]}"
                          + (" …" if len(content) > 500 else ""))
            qs = turn.get("suggested_questions") or []
            if qs:
                console.print("[dim]  Suggested: " + " | ".join(qs) + "[/dim]")
        console.print()


def _print_wisdom_history(turns: list[dict], limit: int = 6) -> None:
    """Print recent wisdom conversation turns."""
    recent = turns[-limit:] if len(turns) > limit else turns
    if not recent:
        return
    skipped = len(turns) - len(recent)
    if skipped:
        console.print(f"[dim]  … {skipped} earlier turns not shown …[/dim]\n")
    for turn in recent:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        meta = turn.get("metadata") or {}
        if role == "user":
            console.print(f"[bold green]You[/bold green]: {content}")
        else:
            label = "[bold cyan]Wisdom Agent[/bold cyan]"
            if meta.get("is_question"):
                label = "[bold yellow]Wisdom Agent (Q)[/bold yellow]"
            elif meta.get("has_wisdom"):
                label = "[bold magenta]Wisdom Agent (✅ Wisdom)[/bold magenta]"
            console.print(f"{label}: {content[:600]}" + (" …" if len(content) > 600 else ""))
        console.print()


def _print_eval_cli(eval_result: dict) -> None:
    """Print eval quality scores as a compact rich table. No-op when dict is empty."""
    if not eval_result or not eval_result.get("overall"):
        return
    overall = eval_result.get("overall", 0)
    summary = eval_result.get("summary", "")
    score_color = {5: "green", 4: "green", 3: "yellow", 2: "red", 1: "red"}.get(overall, "white")
    t = Table(title=f"Quality Score: [{score_color}]{overall}/5[/{score_color}]", border_style=score_color)
    t.add_column("Dimension")
    t.add_column("Score", style="bold")
    for key, val in eval_result.items():
        if key == "summary":
            continue
        label = key.replace("_", " ").title()
        color = {5: "green", 4: "green", 3: "yellow", 2: "red", 1: "red"}.get(val, "white") if isinstance(val, int) else "white"
        t.add_row(label, f"[{color}]{val}/5[/{color}]" if isinstance(val, int) else str(val))
    console.print(t)
    if summary:
        console.print(f"[dim]{summary}[/dim]\n")


def _print_rag_reflection_cli(rag_reflection_info) -> None:
    """Print self-reflective RAG metadata as a compact rich table. No-op when empty."""
    if not rag_reflection_info:
        return

    if isinstance(rag_reflection_info, list):
        entries = rag_reflection_info
    else:
        entries = [rag_reflection_info]

    total_retrieved = sum(e.get("total_retrieved", e.get("papers_retrieved", 0)) for e in entries)
    total_relevant = sum(e.get("total_relevant", e.get("papers_after_grading", 0)) for e in entries)
    if total_retrieved == 0:
        return

    pct = int(100 * total_relevant / total_retrieved)
    t = Table(title="Self-Reflective RAG", border_style="cyan")
    t.add_column("Metric")
    t.add_column("Value", style="bold")
    t.add_row("Retrieved", str(total_retrieved))
    t.add_row("Passed grading", f"[cyan]{total_relevant}[/cyan]")
    t.add_row("Pass rate", f"[cyan]{pct}%[/cyan]")
    for i, entry in enumerate(entries):
        cycles = entry.get("cycles")
        skipped = entry.get("grading_skipped", False)
        rewritten = entry.get("rewritten_queries", [])
        prefix = f"Q{i+1} " if len(entries) > 1 else ""
        if cycles is not None:
            t.add_row(f"{prefix}Cycles", str(cycles))
        if rewritten:
            t.add_row(f"{prefix}Rewritten query", rewritten[0][:60] + ("…" if len(rewritten[0]) > 60 else ""))
        if skipped:
            t.add_row(f"{prefix}Grading", "[yellow]skipped (all-true)[/yellow]")
    console.print(t)


def _render_wisdom_cli(state: dict) -> None:
    """Render the full wisdom output as rich panels in the terminal."""
    deep = state.get("deep_understanding", "")
    simple = state.get("simple_explanation", "")
    actions = state.get("actionable_takeaways", [])
    claims = state.get("wisdom_claims", [])
    devils = state.get("devils_advocate", "")
    confidence = state.get("overall_confidence", "")

    if deep:
        console.print(Panel(
            Markdown(deep),
            title="🔬 Scientific View",
            border_style="blue",
        ))

    if simple:
        console.print(Panel(
            Markdown(simple),
            title="🧒 Plain-Language View",
            border_style="green",
        ))

    if actions:
        action_text = "\n".join(
            f"{i}. {a}" for i, a in enumerate(actions, 1)
        ) if isinstance(actions, list) else str(actions)
        console.print(Panel(
            Markdown(action_text),
            title="🎯 Action Steps",
            border_style="yellow",
        ))

    if claims or devils:
        conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "⚪")
        val_lines = [f"**Overall confidence:** {conf_emoji} {confidence}\n"]
        for c in claims:
            lvl = c.get("confidence", "?")
            badge = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(lvl, "⚪")
            consensus = c.get("consensus", "")
            val_lines.append(
                f"{badge} **{c.get('claim', '')}**\n"
                f"   Confidence: {lvl} | {consensus}"
            )
        if devils:
            val_lines.append(f"\n**Devil's advocate:** {devils}")
        console.print(Panel(
            Markdown("\n\n".join(val_lines)),
            title="✅ Validation",
            border_style="magenta",
        ))


def _cmd_story(args) -> None:
    """Interactive Research Partner (Mode 5) chat loop."""
    from agents.story_memory import StorytellerMemory
    from agents.story_state import create_story_state
    from agents.story_graph import run_story_turn

    memory = StorytellerMemory()
    doc_context = ""
    doc_names: list[str] = []

    if args.story_session:
        session_data = memory.load(args.story_session)
        if not session_data:
            console.print(f"[red]Session not found:[/red] {args.story_session}")
            console.print("[dim]Use --list-stories to see available sessions.[/dim]")
            return
        session_id = args.story_session
        topic = session_data.get("topic", "")
        _story_clarifications: dict = {}
    else:
        topic = args.topic or ""

        if args.files:
            console.print("\n[bold]Processing documents for context…[/bold]")
            processed = _process_files(
                list(args.files), max_raw_chars=2000,
                use_docling=getattr(args, "docling", True),
                use_ocr=getattr(args, "ocr", False),
            )
            if processed:
                doc_context = "\n\n".join(d.raw_text[:2000] for d in processed)
                doc_names = [d.filename for d in processed]

        # ── Socratic clarification for new sessions ───────────
        _story_clarifications = _ask_clarifying_questions(topic, "story", args)

        # Session topic auto-inferred from first message when not specified
        session_id = memory.new_session(
            topic=topic or "New Session",
            document_context=doc_context,
            document_names=doc_names,
        )

    session_data = memory.load(session_id)
    turns = session_data.get("conversation", []) if session_data else []
    concepts = session_data.get("concepts_covered", []) if session_data else []
    # topic may be empty if not pre-specified — will be set from first message
    if not topic and session_data:
        topic = session_data.get("topic", "")

    style = args.style or "simple"
    style_labels = {
        "simple": "Simple Language",
        "analogy": "Extended Analogy",
        "walkthrough": "Step-by-Step",
        "debate": "For vs. Against",
    }

    console.rule("[bold cyan]Research Partner — Mode 5[/bold cyan]")
    console.print(Panel(
        f"Topic:   [bold]{topic}[/bold]\n"
        f"Style:   [bold]{style_labels.get(style, style)}[/bold]\n"
        f"Session: [cyan]{session_id}[/cyan]"
        + (f"\nDocs:    {', '.join(doc_names)}" if doc_names else "")
        + (f"\nConcepts covered: {', '.join(concepts[:6])}" if concepts else ""),
        title="💬 Session Info",
        border_style="cyan",
    ))

    if turns:
        console.print("[dim]Recent conversation:[/dim]\n")
        _print_conversation_history(turns)

    console.print("[dim]Type your question, type [bold]1/2/3[/bold] for a suggested question, or [bold]quit[/bold] to exit.[/dim]\n")

    turn_count = len(turns) // 2  # approx user turns already in session
    last_questions: list[str] = []  # suggested questions from previous turn

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session saved. Goodbye![/dim]")
            _cli_safe_exit()
            break

        stripped = user_input.strip()

        if stripped.lower() in ("quit", "exit", "bye", "q", ":q"):
            console.print("[dim]Session saved. Goodbye![/dim]")
            _cli_safe_exit()
            break

        if not stripped:
            continue

        # ── Shortcut: type 1/2/3 to pick a suggested question ──
        if stripped in ("1", "2", "3") and last_questions:
            idx = int(stripped) - 1
            if idx < len(last_questions):
                picked = last_questions[idx]
                console.print(f"[dim]→ {picked}[/dim]")
                user_input = picked
            else:
                console.print("[yellow]No question at that number.[/yellow]")
                continue

        # Auto-rename session from first message when topic was not pre-specified
        if turn_count == 0 and topic in ("", "New Session"):
            auto_topic = user_input.strip()[:60].rstrip("?").strip() or "Research Session"
            memory.rename(session_id, auto_topic)
            topic = auto_topic

        with console.status("[bold blue]Research Partner is thinking…[/bold blue]"):
            initial_state = create_story_state(
                user_message=user_input,
                session_id=session_id,
                topic=topic,
                model_name=args.model,
                num_ctx=args.num_ctx,
                explanation_style=style,
                clarifications=_story_clarifications,
            )
            try:
                final_state = run_story_turn(initial_state)
            except Exception as e:
                console.print(f"\n[red]✗ Error: {e}[/red]")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue

        turn_count += 1
        response = final_state.get("assistant_response", "")
        console.print(f"\n[bold magenta]Partner[/bold magenta]:\n")
        console.print(Markdown(response))

        last_questions = final_state.get("suggested_questions", [])
        if last_questions:
            console.print("\n[dim]Suggested follow-ups:[/dim]")
            for i, q in enumerate(last_questions, 1):
                console.print(f"  [dim]{i}. {q}[/dim]")

        _print_eval_cli(final_state.get("eval_result", {}))
        _print_rag_reflection_cli(final_state.get("rag_reflection_info"))

        # ── Per-turn smart recommendations ─────────────────────
        tips = recommend_story_turn(
            final_state=dict(final_state),
            turn_count=turn_count,
            current_style=style,
            session_id=session_id,
        )
        print_recommendations(tips, console, title="💡 Tip")

        for err in final_state.get("errors", []):
            console.print(f"[yellow]⚠ {err}[/yellow]")
        console.print()


def _cmd_wisdom(args) -> None:
    """Interactive Wisdom Mode (Mode 6) chat loop."""
    from agents.wisdom_memory import WisdomMemory
    from agents.wisdom_state import create_wisdom_state
    from agents.wisdom_graph import run_wisdom_turn

    memory = WisdomMemory()
    doc_context = ""
    doc_names: list[str] = []

    if args.wisdom_session:
        session_data = memory.load(args.wisdom_session)
        if not session_data:
            console.print(f"[red]Session not found:[/red] {args.wisdom_session}")
            console.print("[dim]Use --list-wisdom to see available sessions.[/dim]")
            return
        session_id = args.wisdom_session
        topic = session_data.get("topic", "")
        _wisdom_clarifications: dict = {}
    else:
        topic = args.topic or ""
        scenario = args.scenario or ""

        if args.files:
            console.print("\n[bold]Processing documents for context…[/bold]")
            processed = _process_files(
                list(args.files), max_raw_chars=2000,
                use_docling=getattr(args, "docling", True),
                use_ocr=getattr(args, "ocr", False),
            )
            if processed:
                doc_context = "\n\n".join(d.raw_text[:2000] for d in processed)
                doc_names = [d.filename for d in processed]

        # ── Socratic clarification for new sessions ───────────
        _wisdom_clarifications = _ask_clarifying_questions(
            topic or scenario, "wisdom", args
        )

        # Session topic auto-inferred from first message when not specified
        session_id = memory.new_session(
            topic=topic or "New Session",
            scenario=scenario,
            document_context=doc_context,
            document_names=doc_names,
        )

    session_data = memory.load(session_id)
    turns = session_data.get("conversation", []) if session_data else []
    phase = session_data.get("phase", "clarifying") if session_data else "clarifying"
    # topic may be empty if not pre-specified — will be set from first message
    if not topic and session_data:
        topic = session_data.get("topic", "")

    phase_label = {
        "clarifying": "🔵 Clarifying (answering questions builds context)",
        "ready_to_generate": "🟡 Ready to generate wisdom",
        "done": "✅ Wisdom generated (follow-up Q&A mode)",
    }.get(phase, phase)

    console.rule("[bold magenta]Wisdom Mode — Mode 6[/bold magenta]")
    console.print(Panel(
        f"Topic:   [bold]{topic}[/bold]\n"
        f"Phase:   {phase_label}\n"
        f"Session: [magenta]{session_id}[/magenta]"
        + (f"\nDocs:    {', '.join(doc_names)}" if doc_names else ""),
        title="🧠 Session Info",
        border_style="magenta",
    ))

    if phase == "done" and session_data and session_data.get("wisdom_output"):
        wo = session_data["wisdom_output"]
        console.print("[dim]Previously generated wisdom (summary):[/dim]")
        preview = wo.get("deep_understanding", "")[:300]
        if preview:
            console.print(Panel(Markdown(preview + " …"), border_style="dim"))

    if turns:
        console.print("[dim]Recent conversation:[/dim]\n")
        _print_wisdom_history(turns)

    console.print(
        "[dim]Respond to the agent's questions, then it will generate wisdom.\n"
        "Type [bold]quit[/bold] to exit.[/dim]\n"
    )

    # Count clarification rounds already in session
    clarification_count = sum(
        1 for t in turns
        if t.get("role") == "assistant" and (t.get("metadata") or {}).get("is_question")
    )

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session saved. Goodbye![/dim]")
            _cli_safe_exit()
            break

        if user_input.strip().lower() in ("quit", "exit", "bye", "q", ":q"):
            console.print("[dim]Session saved. Goodbye![/dim]")
            _cli_safe_exit()
            break

        if not user_input.strip():
            continue

        # Auto-rename session from first message when topic was not pre-specified
        if not turns and topic in ("", "New Session"):
            auto_topic = user_input.strip()[:60].rstrip("?").strip() or "Wisdom Session"
            memory.rename(session_id, auto_topic)
            topic = auto_topic

        with console.status("[bold magenta]Wisdom agent is thinking…[/bold magenta]"):
            initial_state = create_wisdom_state(
                user_message=user_input,
                session_id=session_id,
                topic=topic,
                model_name=args.model,
                num_ctx=args.num_ctx,
                clarifications=_wisdom_clarifications,
            )
            try:
                final_state = run_wisdom_turn(initial_state)
            except Exception as e:
                console.print(f"\n[red]✗ Error: {e}[/red]")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue

        new_phase = final_state.get("phase", "")
        response = final_state.get("assistant_response", "")

        if new_phase == "done" and final_state.get("deep_understanding"):
            console.print(f"\n[bold magenta]Wisdom Agent[/bold magenta]:\n")
            console.print(Markdown(response))
            console.print()
            _render_wisdom_cli(dict(final_state))
            _print_eval_cli(final_state.get("eval_result", {}))
            _print_rag_reflection_cli(final_state.get("rag_reflection_info"))

            # ── Interactive feedback refinement ────────────────────────────────────────
            wisdom_text = final_state.get("assistant_response", "")
            if wisdom_text:
                context = " ".join(p.get("title", "") for p in final_state.get("academic_papers", [])[:5])
                refined = _feedback_loop(
                    current_output=wisdom_text,
                    mode="wisdom",
                    model_name=args.model,
                    num_ctx=args.num_ctx,
                    context=context,
                )
                if refined != wisdom_text:
                    console.print("\n[bold]Refined Wisdom:[/bold]")
                    console.print(refined)
        else:
            console.print(f"\n[bold magenta]Wisdom Agent[/bold magenta]:\n")
            console.print(Markdown(response))
            if new_phase == "clarifying":
                clarification_count += 1

        # ── Per-turn smart recommendations ─────────────────────
        tips = recommend_wisdom_turn(
            final_state=dict(final_state),
            phase=new_phase or phase,
            clarification_count=clarification_count,
            session_id=session_id,
        )
        print_recommendations(tips, console, title="💡 Tip")

        for err in final_state.get("errors", []):
            console.print(f"[yellow]⚠ {err}[/yellow]")
        console.print()

        # Reload phase from disk so subsequent turns route correctly
        updated = memory.load(session_id)
        phase = updated.get("phase", phase) if updated else phase


def _cmd_list_grammar() -> None:
    """List saved grammar proofreading sessions."""
    from agents.grammar_memory import GrammarMemory
    sessions = GrammarMemory().list_sessions(limit=20)
    if not sessions:
        console.print("[dim]No grammar proofreading sessions found.[/dim]")
        console.print("[dim]Start one with: python main.py --grammar-check --goal \"your text\"[/dim]")
        return
    table = Table(title="Grammar Proofreading Sessions", border_style="green")
    table.add_column("Session ID", style="cyan")
    table.add_column("Context")
    table.add_column("Words", justify="right")
    table.add_column("Issues", justify="right")
    table.add_column("Text excerpt")
    table.add_column("Created")
    for s in sessions:
        table.add_row(
            s["session_id"],
            s.get("style_level", "").replace("_", " ").title(),
            str(s.get("word_count", 0)),
            str(s.get("issues_count", 0)),
            (s.get("raw_text_excerpt") or "")[:50],
            (s.get("created_at") or "")[:10],
        )
    console.print(table)


def _cmd_grammar(args) -> None:
    """Run Grammar Proofreading mode (Mode 6) from the CLI."""
    from pathlib import Path
    from agents.grammar_graph import run_grammar_check
    from agents.grammar_memory import GrammarMemory
    from agents.grammar_state import create_grammar_state

    style_level = getattr(args, "style_level", "professional_email")
    focus_areas = getattr(args, "focus", []) or []

    # ── Resolve input text ────────────────────────────────────
    raw_text = ""

    if getattr(args, "grammar_session", None):
        data = GrammarMemory().load(args.grammar_session)
        if not data:
            console.print(f"[red]Session not found:[/red] {args.grammar_session}")
            return
        raw_text = data.get("raw_text_excerpt", "")
        if not raw_text:
            console.print("[red]Session contains no text to re-process.[/red]")
            return
        style_level = data.get("style_level", style_level)

    if not raw_text and args.files:
        console.print("\n[bold]Processing file(s)…[/bold]")
        docs = _process_files(
            list(args.files), max_raw_chars=None,
            use_docling=getattr(args, "docling", True),
            use_ocr=getattr(args, "ocr", False),
        )
        if docs:
            raw_text = "\n\n".join(d.raw_text for d in docs)

    if not raw_text and args.goal:
        raw_text = args.goal

    if not raw_text:
        raw_text = Prompt.ask("[bold green]Paste text to proofread[/bold green]")

    if not raw_text.strip():
        console.print("[red]No text provided.[/red]")
        return

    # ── Create session ────────────────────────────────────────
    mem = GrammarMemory()
    session_id = mem.new_session(raw_text=raw_text, style_level=style_level)

    console.rule(f"[bold green]Grammar Proofreading — Mode 6[/bold green]")
    console.print(
        f"Context: [bold]{style_level.replace('_', ' ').title()}[/bold]  |  "
        f"Focus: {', '.join(focus_areas) or 'all areas'}  |  "
        f"Words: {len(raw_text.split()):,}"
    )
    console.print()

    initial_state = create_grammar_state(
        raw_text=raw_text,
        session_id=session_id,
        model_name=args.model,
        num_ctx=args.num_ctx,
        style_level=style_level,
        focus_areas=focus_areas,
    )

    node_labels = {
        "text_loader":      "Loading text",
        "grammar_analysis": "Detecting issues",
        "polish":           "Polishing text",
        "style_advisor":    "Style tips",
        "grammar_eval":     "Evaluating quality",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Grammar proofreading", total=100)

        def _grammar_callback(node_name: str, state: dict) -> None:
            pct = state.get("progress_pct", 0)
            label = node_labels.get(node_name, node_name)
            detail = state.get("status_detail", "")
            desc = f"{label} — {detail}" if detail else label
            progress.update(task, completed=pct, description=desc)

        start = time.time()
        try:
            final_state = run_grammar_check(initial_state, stream_callback=_grammar_callback)
        except Exception as e:
            console.print(f"\n[red]✗ Grammar check failed: {e}[/red]")
            if getattr(args, "verbose", False):
                import traceback; traceback.print_exc()
            return

    elapsed = time.time() - start
    console.print(f"\n[green]✓ Complete in {elapsed:.1f}s[/green]\n")

    for err in final_state.get("errors", []):
        console.print(f"[yellow]⚠ {err}[/yellow]")

    # ── Print detected issues ─────────────────────────────────
    issues = final_state.get("issues_found", [])
    if issues:
        issues_table = Table(title=f"Issues Found ({len(issues)})", border_style="yellow")
        issues_table.add_column("Type")
        issues_table.add_column("Severity")
        issues_table.add_column("Original")
        issues_table.add_column("Suggestion")
        issues_table.add_column("Explanation")
        for iss in issues:
            issues_table.add_row(
                iss.get("type", "?").title(),
                iss.get("severity", "?").title(),
                (iss.get("original") or "")[:40],
                (iss.get("suggestion") or "")[:40],
                (iss.get("explanation") or "")[:60],
            )
        console.print(issues_table)
    else:
        console.print("[green]No issues detected.[/green]")

    # ── Print polished text ───────────────────────────────────
    polished = final_state.get("polished_text", "")
    if polished:
        console.rule("[bold]Polished Text[/bold]")
        console.print(Markdown(polished))
        console.rule()

    # ── Print change summary ──────────────────────────────────
    change_summary = final_state.get("change_summary", "")
    if change_summary:
        console.rule("[bold]Changes Made[/bold]")
        console.print(Markdown(change_summary))
        console.rule()

    # ── Print style tips ──────────────────────────────────────
    style_tips = final_state.get("style_suggestions", [])
    if style_tips:
        tips_table = Table(title="Style Tips", border_style="blue")
        tips_table.add_column("Category")
        tips_table.add_column("Suggestion")
        tips_table.add_column("Rationale")
        for tip in style_tips:
            tips_table.add_row(
                tip.get("category", "?").title(),
                (tip.get("suggestion") or "")[:60],
                (tip.get("rationale") or "")[:60],
            )
        console.print(tips_table)

    # ── Quality eval + RAG reflection ────────────────────────
    _print_eval_cli(final_state.get("eval_result", {}))
    _print_rag_reflection_cli(final_state.get("rag_reflection_info"))

    # ── Save output ───────────────────────────────────────────
    mem.save_result(session_id, dict(final_state))

    out_path = getattr(args, "output", None) or Path(f"./outputs/grammar_{session_id}.md")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_content = f"# Proofread Output\n\n**Writing context:** {style_level}\n\n"
    if polished:
        md_content += f"## Polished Text\n\n{polished}\n\n"
    if change_summary:
        md_content += f"## Changes Made\n\n{change_summary}\n\n"
    out_path.write_text(md_content, encoding="utf-8")
    console.print(f"\n[bold green]✓ Saved:[/bold green] {out_path}")

    # ── Feedback loop ─────────────────────────────────────────
    if polished:
        from agents.feedback_agent import refine_with_feedback, MAX_FEEDBACK_ROUNDS
        current_output = polished
        for round_num in range(1, MAX_FEEDBACK_ROUNDS + 1):
            try:
                feedback = Prompt.ask(
                    f"\n[dim]Provide feedback to refine (round {round_num}/{MAX_FEEDBACK_ROUNDS}), "
                    "or press Enter to finish[/dim]"
                )
            except (KeyboardInterrupt, EOFError):
                break
            if not feedback.strip():
                break
            with console.status("[bold green]Refining…[/bold green]"):
                current_output = refine_with_feedback(
                    original_output=current_output,
                    feedback=feedback.strip(),
                    context=raw_text,
                    mode="grammar_proofreading",
                    model_name=args.model,
                    num_ctx=args.num_ctx,
                )
            console.rule(f"[bold]Refined Output (round {round_num})[/bold]")
            console.print(Markdown(current_output))
            console.rule()
            # Save refined output
            out_refined = Path(f"./outputs/grammar_{session_id}_refined_r{round_num}.md")
            out_refined.write_text(
                f"# Refined Output (Round {round_num})\n\n{current_output}", encoding="utf-8"
            )
            console.print(f"[green]✓ Saved:[/green] {out_refined}")


def _cmd_systematic_review(args) -> None:
    """Run a simplified PRISMA systematic review from the CLI."""
    from agents.systematic_review_state import create_systematic_review_state
    from agents.systematic_review_graph import run_systematic_review

    rq = args.goal
    if not rq:
        rq = Prompt.ask("[bold cyan]Systematic Review[/bold cyan] — Enter your research question")
    if not rq.strip():
        console.print("[red]A research question is required.[/red]")
        return

    inclusion = list(args.inclusion) if args.inclusion else []
    exclusion = list(args.exclusion) if args.exclusion else []

    console.print(
        Panel(
            f"[bold cyan]Systematic Review — Mode 7[/bold cyan]\n\n"
            f"Research question: [italic]{rq}[/italic]\n"
            f"Inclusion: {', '.join(inclusion) or '(auto)'}\n"
            f"Exclusion: {', '.join(exclusion) or '(auto)'}\n"
            f"Model: [bold]{args.model}[/bold]",
            title="📋 PRISMA Systematic Review",
            border_style="blue",
        )
    )

    initial_state = create_systematic_review_state(
        research_question=rq.strip(),
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion,
        model_name=args.model,
        num_ctx=args.num_ctx,
    )

    node_labels = {
        "query_generation":    "Generating search queries",
        "literature_search":   "Searching Google Scholar · arXiv · Semantic Scholar · CrossRef",
        "screening":           "Screening papers",
        "evidence_extraction": "Extracting evidence",
        "synthesis":           "Synthesising findings",
        "sr_eval":             "Evaluating review quality",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Systematic review", total=100)

        def sr_callback(node_name: str, state: dict):
            pct = state.get("progress_pct", 0)
            label = node_labels.get(node_name, node_name)
            detail = state.get("status_detail", "")
            desc = f"{label} — {detail}" if detail else label
            progress.update(task, completed=pct, description=desc)

        start = time.time()
        try:
            final_state = run_systematic_review(initial_state, stream_callback=sr_callback)
        except Exception as e:
            console.print(f"\n[red]✗ Systematic review failed: {e}[/red]")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return

    elapsed = time.time() - start
    console.print(f"\n[green]✓ Complete in {elapsed:.1f}s[/green]\n")

    # PRISMA flow
    flow = final_state.get("prisma_flow", {})
    flow_table = Table(title="PRISMA Flow", border_style="blue")
    flow_table.add_column("Stage")
    flow_table.add_column("Count", style="bold")
    for stage, count in flow.items():
        flow_table.add_row(stage.title(), str(count))
    console.print(flow_table)

    # Key themes
    themes = final_state.get("key_themes", [])
    if themes:
        console.print("\n[bold]Key Themes:[/bold]")
        for t in themes:
            console.print(f"  • {t}")

    # Evidence table
    evidence = final_state.get("evidence_table", [])
    if evidence:
        ev_table = Table(title=f"Evidence Table ({len(evidence)} papers)", border_style="green")
        ev_table.add_column("Citation", style="cyan", no_wrap=True)
        ev_table.add_column("Year", no_wrap=True)
        ev_table.add_column("Design")
        ev_table.add_column("Quality")
        ev_table.add_column("Key Finding", max_width=55)
        for row in evidence:
            qual = row.get("quality", "?")
            qual_c = {"High": "green", "Medium": "yellow", "Low": "red"}.get(qual, "white")
            ev_table.add_row(
                row.get("citation_key", "")[:25],
                str(row.get("year", "n.d.")),
                row.get("study_design", "")[:18],
                f"[{qual_c}]{qual}[/{qual_c}]",
                row.get("key_finding", "")[:55],
            )
        console.print(ev_table)

    # Synthesis
    synthesis = final_state.get("narrative_synthesis", "")
    if synthesis:
        console.print(Panel(Markdown(synthesis[:2000] + (" …" if len(synthesis) > 2000 else "")),
                            title="Narrative Synthesis", border_style="blue"))

    gaps = final_state.get("research_gaps", [])
    if gaps:
        console.print("\n[bold]Research Gaps:[/bold]")
        for g in gaps:
            console.print(f"  • {g}")

    conclusion = final_state.get("conclusion", "")
    if conclusion:
        console.print(Panel(Markdown(conclusion), title="Conclusion", border_style="green"))

    _print_eval_cli(final_state.get("eval_result", {}))
    _print_rag_reflection_cli(final_state.get("rag_reflection_info"))

    # ── Interactive feedback refinement ────────────────────────────────────────
    synthesis = final_state.get("narrative_synthesis", "")
    if synthesis:
        context = " ".join(p.get("title", "") for p in final_state.get("included_papers", [])[:5])
        refined = _feedback_loop(
            current_output=synthesis,
            mode="systematic_review",
            model_name=args.model,
            num_ctx=args.num_ctx,
            context=context,
        )
        if refined != synthesis:
            console.print("\n[bold]Refined Synthesis:[/bold]")
            console.print(refined)

    # Save Markdown
    sid = initial_state.get("session_id", "sr")
    out_path = args.output or Path(f"./outputs/systematic_review_{sid}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from ui.tabs.systematic_review import _build_sr_markdown
    md_text = _build_sr_markdown(rq, final_state)
    out_path.write_text(md_text, encoding="utf-8")
    console.print(f"\n[bold green]✓ Report saved:[/bold green] {out_path}")

    # ── Optional post-run tools ────────────────────────────────────────────────

    # DOCX report
    if getattr(args, "sr_docx", False):
        docx_path = Path(f"./outputs/prisma_report_{sid}.docx")
        docx_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from tools.prisma_report import generate_prisma_docx
            with console.status("Generating PRISMA 2020 DOCX…"):
                docx_bytes = generate_prisma_docx(
                    final_state,
                    author=getattr(args, "sr_author", ""),
                    institution=getattr(args, "sr_institution", ""),
                )
            docx_path.write_bytes(docx_bytes)
            console.print(f"[bold green]✓ DOCX saved:[/bold green] {docx_path}")
        except Exception as e:
            console.print(f"[red]DOCX generation failed: {e}[/red]")

    # PDF report
    if getattr(args, "sr_pdf", False):
        pdf_path = Path(f"./outputs/prisma_report_{sid}.pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from tools.prisma_report import generate_prisma_pdf
            with console.status("Generating PRISMA 2020 PDF…"):
                pdf_bytes = generate_prisma_pdf(
                    final_state,
                    author=getattr(args, "sr_author", ""),
                    institution=getattr(args, "sr_institution", ""),
                )
            pdf_path.write_bytes(pdf_bytes)
            console.print(f"[bold green]✓ PDF saved:[/bold green] {pdf_path}")
        except Exception as e:
            console.print(f"[red]PDF generation failed: {e}[/red]")

    # Plain-language summaries
    pls_fmt = getattr(args, "sr_plain_language", "")
    if pls_fmt:
        try:
            from tools.plain_language import (
                generate_patient_summary, generate_policy_brief,
                generate_press_release, generate_all_summaries,
            )
            formats = (
                {"patient", "policy", "press"}
                if pls_fmt == "all"
                else {pls_fmt}
            )
            with console.status(f"Generating plain-language summary ({pls_fmt})…"):
                if pls_fmt == "all":
                    summaries = generate_all_summaries(final_state, args.model, args.num_ctx)
                else:
                    fn = {"patient": generate_patient_summary,
                          "policy": generate_policy_brief,
                          "press": generate_press_release}[pls_fmt]
                    summaries = {pls_fmt: fn(final_state, args.model, args.num_ctx)}

            for fmt_key, text in summaries.items():
                summary_path = Path(f"./outputs/summary_{fmt_key}_{sid}.txt")
                summary_path.write_text(text, encoding="utf-8")
                console.print(f"[bold green]✓ {fmt_key.title()} summary saved:[/bold green] {summary_path}")
                console.print(Panel(Markdown(text[:800] + ("…" if len(text) > 800 else "")),
                                    title=f"Plain-Language Summary ({fmt_key})", border_style="cyan"))
        except Exception as e:
            console.print(f"[red]Plain-language summary failed: {e}[/red]")

    # Research trend analysis
    if getattr(args, "sr_trends", False):
        try:
            from tools.trend_analyzer import analyze_trends
            with console.status("Querying CrossRef for field-wide trend data…"):
                trend_data = analyze_trends(
                    research_question=rq,
                    search_queries=final_state.get("search_queries", []),
                    corpus_papers=final_state.get("included_papers", []),
                )
            combined = trend_data.get("combined_by_year", {})
            trend_table = Table(title="Research Publication Trend", border_style="cyan")
            trend_table.add_column("Year")
            trend_table.add_column("Field-wide (CrossRef)", style="blue")
            trend_table.add_column("This SR corpus", style="green")
            corpus_by_yr = trend_data.get("corpus_by_year", {})
            for yr in sorted(combined):
                trend_table.add_row(
                    str(yr),
                    str(combined.get(yr, 0)),
                    str(corpus_by_yr.get(yr, 0)),
                )
            console.print(trend_table)
            console.print(
                f"Trend: [bold]{trend_data.get('trend', 'unknown').upper()}[/bold]  "
                f"Peak year: [bold]{trend_data.get('peak_year', 'N/A')}[/bold]  "
                f"Total CrossRef: [bold]{trend_data.get('total_field', 0):,}[/bold]"
            )
        except Exception as e:
            console.print(f"[red]Trend analysis failed: {e}[/red]")

    # Preprint tracking
    if getattr(args, "sr_preprints", False):
        included = final_state.get("included_papers", [])
        if included:
            try:
                from tools.preprint_tracker import track_preprints, preprint_summary
                with console.status("Checking preprint status via CrossRef…"):
                    tracking = track_preprints(included)
                summary = preprint_summary(tracking)
                pt_table = Table(title="Preprint Status", border_style="yellow")
                pt_table.add_column("Paper", max_width=50)
                pt_table.add_column("Status")
                pt_table.add_column("Note", max_width=50)
                status_colors = {
                    "journal": "green", "published": "cyan",
                    "preprint": "yellow", "retracted": "red",
                }
                for r in tracking:
                    paper = r.get("paper", {})
                    status = r.get("preprint_status", "")
                    color = status_colors.get(status, "white")
                    pt_table.add_row(
                        paper.get("title", "")[:50],
                        f"[{color}]{status}[/{color}]",
                        r.get("note", "")[:50],
                    )
                console.print(pt_table)
                console.print(
                    f"Journal: [green]{summary.get('journal', 0)}[/green]  "
                    f"Published (was preprint): [cyan]{summary.get('published', 0)}[/cyan]  "
                    f"Preprint only: [yellow]{summary.get('preprint', 0)}[/yellow]  "
                    f"Retracted: [red]{summary.get('retracted', 0)}[/red]"
                )
            except Exception as e:
                console.print(f"[red]Preprint tracking failed: {e}[/red]")

    # Concept drift
    if getattr(args, "sr_concept_drift", False):
        all_papers = final_state.get("raw_papers", [])
        if all_papers:
            try:
                from tools.concept_drift import detect_concept_drift
                with console.status("Analysing vocabulary evolution…"):
                    drift = detect_concept_drift(
                        papers=all_papers,
                        model_name=args.model,
                        num_ctx=args.num_ctx,
                    )
                drift_table = Table(title="Concept Drift — Vocabulary Evolution", border_style="magenta")
                drift_table.add_column("Type")
                drift_table.add_column("Term")
                drift_table.add_column("Period")
                drift_table.add_column("Change")
                for r in drift.get("rising_terms", [])[:6]:
                    drift_table.add_row(
                        "[green]Rising[/green]", r["term"],
                        f"{r['first_bucket']} → {r['last_bucket']}", f"+{r['growth']}"
                    )
                for d in drift.get("declining_terms", [])[:6]:
                    drift_table.add_row(
                        "[red]Declining[/red]", d["term"],
                        f"{d['first_bucket']} → {d['last_bucket']}", str(d["growth"])
                    )
                console.print(drift_table)
                if drift.get("llm_analysis"):
                    console.print(Panel(
                        Markdown(drift["llm_analysis"]),
                        title="Vocabulary Shift Analysis",
                        border_style="magenta",
                    ))
            except Exception as e:
                console.print(f"[red]Concept drift analysis failed: {e}[/red]")


def _cmd_run_proposal(args) -> None:
    from agents.memory import ProposalMemory
    from agents.proposal_state import create_proposal_state
    from agents.proposal_graph import run_proposal

    if not args.goal:
        console.print("[red]Error:[/red] --goal is required for --propose.")
        return

    # Pre-run proposal recommendations
    pre_tips = recommend_proposal_pre(args.goal, args.instructions)
    print_recommendations(pre_tips, console, title="💡 Proposal Recommendations")

    memory = ProposalMemory()

    funding_agency = getattr(args, "funding_agency", "None (General)") or "None (General)"
    budget_range = getattr(args, "budget", "") or ""
    duration = getattr(args, "duration", 36) or 36
    trl = getattr(args, "trl", "") or ""
    consortium_size = getattr(args, "consortium", "") or ""

    from tools.funding_templates import get_template
    tmpl = get_template(funding_agency)
    if funding_agency != "None (General)":
        console.print(
            f"[bold magenta]Funding agency:[/bold magenta] {funding_agency} · "
            f"{tmpl.get('typical_budget_range','')} · {tmpl.get('typical_duration_months','')} months"
        )

    session_id = memory.new_session(
        goal=args.goal,
        model=args.model,
        instructions=args.instructions,
    )

    console.print(
        Panel(
            f"[bold cyan]Project Proposal Writer[/bold cyan]\n\n"
            f"Goal: [italic]{args.goal}[/italic]\n"
            f"Session: [bold]{session_id}[/bold]  |  Model: [bold]{args.model}[/bold]"
            + (f"\nFunding: [magenta]{funding_agency}[/magenta]" if funding_agency != "None (General)" else "")
            + (f"  |  Consortium: {consortium_size}" if consortium_size else ""),
            title="📝 New Proposal",
            border_style="green",
        )
    )

    # ── Socratic clarification ────────────────────────────────
    _prop_clarifications = _ask_clarifying_questions(args.goal, "proposal", args)

    initial_state = create_proposal_state(
        goal=args.goal,
        instructions=args.instructions,
        session_id=session_id,
        model_name=args.model,
        clarifications=_prop_clarifications,
        funding_agency=funding_agency,
        budget_range=budget_range,
        project_duration_months=duration,
        trl_level=trl,
        consortium_size=consortium_size,
    )

    _run_proposal_with_progress(initial_state, memory, args)


def _cmd_revise_proposal(args) -> None:
    from agents.memory import ProposalMemory
    from agents.proposal_state import create_proposal_state
    from agents.proposal_graph import run_proposal

    if not args.revision:
        console.print("[red]Error:[/red] --revision is required for --revise.")
        return

    memory = ProposalMemory()
    session_data = memory.load(args.revise)
    if not session_data:
        console.print(f"[red]Session not found:[/red] {args.revise}")
        return

    console.print(
        Panel(
            f"[bold cyan]Revising Proposal[/bold cyan]\n\n"
            f"Session: [bold]{args.revise}[/bold]\n"
            f"Instruction: [italic]{args.revision}[/italic]",
            title="🔄 Proposal Revision",
            border_style="yellow",
        )
    )

    initial_state = create_proposal_state(
        goal=session_data.get("goal", ""),
        instructions=session_data.get("instructions", ""),
        session_id=args.revise,
        model_name=args.model,
        is_revision=True,
        revision_instruction=args.revision,
        previous_proposal=session_data.get("proposal_markdown", ""),
        previous_references=session_data.get("references", []),
    )

    _run_proposal_with_progress(initial_state, memory, args)


def _run_proposal_with_progress(initial_state, memory, args) -> None:
    from agents.proposal_graph import run_proposal

    node_labels = {
        "search_literature":       "Searching academic literature",
        "plan_proposal":           "Planning title and objectives",
        "write_introduction":      "Writing Introduction",
        "write_literature_review": "Writing Literature Review",
        "write_methodology":       "Writing Methodology",
        "write_outcomes":          "Writing Outcomes and Timeline",
        "assemble_proposal":       "Finalising proposal",
        "revise_proposal":         "Applying revision",
        "proposal_memory_saver":   "Saving proposal",
        "proposal_eval":           "Evaluating quality",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating…", total=100)

        def callback(node_name: str, state: dict):
            pct = state.get("progress_pct", 0)
            label = node_labels.get(node_name, node_name)
            detail = state.get("status_detail", "")
            desc = f"{label} — {detail}" if detail else label
            progress.update(task, completed=pct, description=desc)

        start = time.time()
        try:
            final_state = run_proposal(initial_state, stream_callback=callback)
        except Exception as e:
            console.print(f"\n[red]✗ Proposal workflow failed: {e}[/red]")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return

    elapsed = time.time() - start
    console.print(f"\n[green]✓ Complete in {elapsed:.1f}s[/green]\n")

    session_id = initial_state.get("session_id", "proposal")
    proposal_md = final_state.get("proposal_markdown", "")
    references  = final_state.get("selected_references", [])
    wc = final_state.get("word_counts", {})

    if wc:
        wc_table = Table(title="Word Counts", border_style="cyan")
        wc_table.add_column("Section")
        wc_table.add_column("Words", style="bold")
        for sec, cnt in wc.items():
            wc_table.add_row(sec.replace("_", " ").title(), str(cnt))
        console.print(wc_table)

    if references:
        console.print(f"\n[bold]References ({len(references)}):[/bold]")
        for r in references:
            badge = "🟢" if r.get("source") == "semantic_scholar" else "🟡"
            console.print(f"  {badge} [{r['ref_num']}] {r['title'][:65]}…  ({r.get('year','n.d.')})")

    console.print("\n" + "─" * 80)
    console.print(Markdown(proposal_md))

    out_base = args.output or Path(f"./outputs/proposal_{session_id}")
    out_base = Path(str(out_base).rstrip(".md").rstrip(".docx").rstrip(".pdf"))
    out_base.parent.mkdir(parents=True, exist_ok=True)

    md_path = out_base.with_suffix(".md")
    md_path.write_text(proposal_md, encoding="utf-8")
    console.print(f"\n[bold green]✓ Markdown:[/bold green] {md_path}")

    try:
        from tools.export_tools import build_docx
        docx_bytes = build_docx(proposal_md, references)
        docx_path = out_base.with_suffix(".docx")
        docx_path.write_bytes(docx_bytes)
        console.print(f"[bold green]✓ Word doc:[/bold green] {docx_path}")
    except Exception as e:
        console.print(f"[yellow]⚠ DOCX export failed: {e}[/yellow]")

    try:
        from tools.export_tools import build_pdf
        pdf_bytes = build_pdf(proposal_md, references)
        pdf_path = out_base.with_suffix(".pdf")
        pdf_path.write_bytes(pdf_bytes)
        console.print(f"[bold green]✓ PDF:[/bold green] {pdf_path}")
    except Exception as e:
        console.print(f"[yellow]⚠ PDF export failed: {e}[/yellow]")

    _print_eval_cli(final_state.get("eval_result", {}))
    _print_rag_reflection_cli(final_state.get("rag_reflection_info"))

    # ── Post-run proposal recommendations ────────────────────
    post_tips = recommend_proposal_post({
        **dict(final_state),
        "session_id": session_id,
    })
    print_recommendations(post_tips, console, title="💡 Next Steps")

    console.print(
        f"\n[bold]Session ID:[/bold] [cyan]{session_id}[/cyan]\n"
        "Use this ID with [bold]--revise[/bold] to continue modifying this proposal."
    )


def _cmd_export_proposal(session_id: str, out_base: Path | None) -> None:
    from agents.memory import ProposalMemory
    from tools.export_tools import build_docx, build_pdf

    memory = ProposalMemory()
    data = memory.load(session_id)
    if not data or not data.get("proposal_markdown"):
        console.print(f"[red]No proposal found for session:[/red] {session_id}")
        return

    pm   = data["proposal_markdown"]
    refs = data.get("references", [])
    base = out_base or Path(f"./outputs/proposal_{session_id}")
    base = Path(str(base).replace(".docx", "").replace(".pdf", ""))
    base.parent.mkdir(parents=True, exist_ok=True)

    try:
        docx_bytes = build_docx(pm, refs)
        docx_path = base.with_suffix(".docx")
        docx_path.write_bytes(docx_bytes)
        console.print(f"[green]✓ DOCX:[/green] {docx_path}")
    except Exception as e:
        console.print(f"[red]✗ DOCX failed: {e}[/red]")

    try:
        pdf_bytes = build_pdf(pm, refs)
        pdf_path = base.with_suffix(".pdf")
        pdf_path.write_bytes(pdf_bytes)
        console.print(f"[green]✓ PDF:[/green] {pdf_path}")
    except Exception as e:
        console.print(f"[red]✗ PDF failed: {e}[/red]")


def _process_files(
    files: list[Path],
    chunk_size: int = 800,
    overlap: int = 150,
    max_raw_chars: int = 0,
    use_docling: bool = True,
    use_ocr: bool = False,
):
    from tools.document_tools import get_processor
    processor = get_processor(
        use_docling=use_docling,
        use_ocr=use_ocr,
        chunk_size=chunk_size,
        overlap=overlap,
        max_raw_chars=max_raw_chars,
    )
    if use_docling:
        console.print(
            f"[dim]Using Docling{'+ OCR' if use_ocr else ''} for advanced parsing[/dim]"
        )
    docs = []
    for fp in files:
        if not fp.exists():
            console.print(f"[red]✗ File not found: {fp}[/red]")
            continue
        try:
            doc = processor.process_file(fp)
            chars = len(doc.raw_text)
            console.print(
                f"[green]✓[/green] {fp.name} — {doc.total_pages} pages, "
                f"{chars:,} chars extracted"
                + (f" (capped at {max_raw_chars:,})" if max_raw_chars and chars >= max_raw_chars else "")
            )
            docs.append(doc)
        except Exception as e:
            console.print(f"[red]✗ Failed to process {fp.name}: {e}[/red]")
    return docs


def _cmd_shutdown() -> None:
    """--shutdown: free all stale ports and flush ChromaDB, then exit."""
    from tools.shutdown import safe_shutdown, is_port_in_use, ALL_PORTS

    console.rule("[bold red]Safe Shutdown[/bold red]")
    console.print(
        "Checking ports: "
        "[cyan]8501[/cyan] (Streamlit)  "
        "[cyan]8000[/cyan] (Google Search)  "
        "[cyan]11434[/cyan] (Ollama)\n"
    )
    safe_shutdown(ports=ALL_PORTS, flush_db=True, console=console)
    console.rule()


def _cli_safe_exit() -> None:
    """Flush ChromaDB and free non-Streamlit ports on interactive-mode exit."""
    from tools.shutdown import safe_shutdown, PORT_GOOGLE_SEARCH, PORT_OLLAMA
    safe_shutdown(ports=[PORT_GOOGLE_SEARCH, PORT_OLLAMA], flush_db=True, console=console)


# ─── Feedback refinement loop ─────────────────────────────────────────────────

def _feedback_loop(
    current_output: str,
    mode: str,
    model_name: str,
    num_ctx: int,
    context: str = "",
) -> str:
    """
    Interactive post-output refinement loop for all CLI modes.

    After an agent pipeline prints its results, this prompts the user for
    feedback and refines the output up to MAX_FEEDBACK_ROUNDS times.
    Returns the final (possibly refined) output string.
    """
    from agents.feedback_agent import refine_with_feedback, MAX_FEEDBACK_ROUNDS

    refined = current_output
    for round_num in range(1, MAX_FEEDBACK_ROUNDS + 1):
        console.print(
            f"\n[dim]─── Feedback round {round_num}/{MAX_FEEDBACK_ROUNDS} "
            f"— press Enter to skip ───[/dim]"
        )
        try:
            feedback = input("Feedback> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Feedback skipped.[/dim]")
            break

        if not feedback:
            break

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Refining output…"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            refined = refine_with_feedback(
                original_output=refined,
                feedback=feedback,
                context=context,
                mode=mode,
                model_name=model_name,
                num_ctx=num_ctx,
            )

        console.print("\n[bold green]Refined output:[/bold green]")
        # Print first 3000 chars as a preview
        preview = refined[:3000]
        if len(refined) > 3000:
            preview += "\n…[truncated — full output in file]"
        console.print(preview)

    return refined


def main():
    args = _parse_args()

    # ── Safe shutdown utility — runs before logging setup so --verbose
    #    never leaks third-party debug lines into the shutdown output ──
    if args.shutdown:
        _configure_logging(verbose=False)
        _cmd_shutdown()
        return

    _configure_logging(args.verbose)

    # ── Install SIGINT/SIGTERM handlers for clean Ctrl+C exit ─
    from tools.shutdown import install_signal_handlers
    install_signal_handlers(console=console)

    # ── --project flag maps to mode-specific flags ────────────────────────────
    if args.project:
        _project_map = {
            "mode1": lambda: setattr(args, "mode", "search"),
            "mode2": lambda: setattr(args, "propose", True),
            "mode3": lambda: setattr(args, "wisdom", True),
            "mode4": lambda: setattr(args, "systematic_review", True),
            "mode5": lambda: setattr(args, "notebook", True),
        }
        action = _project_map.get(args.project)
        if action:
            action()

    # ── Utility commands (no Ollama / hardware check needed) ──
    if args.list_docs:
        _cmd_list_docs()
        return

    if args.clear_store:
        _cmd_clear_store()
        return

    if args.list_proposals:
        _cmd_list_proposals()
        return

    if args.list_stories:
        _cmd_list_stories()
        return

    if args.list_wisdom:
        _cmd_list_wisdom()
        return

    if args.list_notebooks:
        _cmd_list_notebooks()
        return

    # ── Notebook advanced one-shot commands ───────────────────
    if args.notebook_summary:
        _cmd_notebook_advanced(args.notebook_summary, "summary", args)
        return
    if args.notebook_faq:
        _cmd_notebook_advanced(args.notebook_faq, "faq", args)
        return
    if args.notebook_review:
        _cmd_notebook_advanced(args.notebook_review, "review", args)
        return
    if args.notebook_audio:
        _cmd_notebook_advanced(args.notebook_audio, "audio", args)
        return
    if args.notebook_mindmap:
        _cmd_notebook_advanced(args.notebook_mindmap, "mindmap", args)
        return
    if args.notebook_graph:
        _cmd_notebook_advanced(args.notebook_graph, "graph", args)
        return
    if args.notebook_compare:
        _cmd_notebook_advanced(args.notebook_compare, "compare", args)
        return
    if args.notebook_timeline:
        _cmd_notebook_advanced(args.notebook_timeline, "timeline", args)
        return
    if args.notebook_study_table:
        _cmd_notebook_advanced(args.notebook_study_table, "study-table", args)
        return
    if args.notebook_pipeline:
        _cmd_notebook_pipeline(args.notebook_pipeline, args)
        return

    if args.list_style_profiles:
        _cmd_list_style_profiles()
        return

    if args.create_style_profile:
        _cmd_create_style_profile(args)
        return

    if args.export_proposal:
        _cmd_export_proposal_gpt(args.export_proposal, args)
        return

    # ── Hardware + model availability check ───────────────────
    from config.settings import get_settings
    ollama_url = get_settings().ollama_base_url
    console.rule("[bold cyan]System Check[/bold cyan]")
    rec = _print_hardware_banner(ollama_url, user_model=args.model)
    console.rule()

    # Apply tight-fit model choice if the user selected a different model
    if rec.get("user_chose_model"):
        args.model = rec["model"]

    if args.check_system:
        return

    # ── Research Notebook (Mode 8) ────────────────────────────
    if args.notebook or getattr(args, "notebook_id", ""):
        _cmd_notebook(args)
        return

    # ── Conversational modes (5 and 6) ────────────────────────
    if args.story or args.story_session:
        _cmd_story(args)
        return

    if args.wisdom or args.wisdom_session:
        _cmd_wisdom(args)
        return

    # ── Grammar Proofreading (Mode 6) ────────────────────────
    if getattr(args, "grammar_check", False) or getattr(args, "grammar_session", None):
        _cmd_grammar(args)
        return

    if getattr(args, "list_grammar", False):
        _cmd_list_grammar()
        return

    # ── Systematic Review (Mode 7) ────────────────────────────
    if args.systematic_review:
        _cmd_systematic_review(args)
        return

    # ── ProposalGPT (Mode 2) ──────────────────────────────────
    if args.propose:
        _cmd_propose_gpt(args)
        return

    # ── Require --goal for analysis modes (1–3) ───────────────
    if not args.goal:
        console.print(
            "[red]Error:[/red] No mode selected.\n\n"
            "Quick start:\n"
            "  [cyan]python main.py --goal \"your question\"[/cyan]                          # literature search (Mode 1)\n"
            "  [cyan]python main.py --propose --goal \"your goal\"[/cyan]                  # proposal (Mode 2)\n"
            "  [cyan]python main.py --wisdom --topic \"your topic\"[/cyan]                 # wisdom (Mode 3)\n"
            "  [cyan]python main.py --systematic-review --goal \"question\"[/cyan]        # systematic review (Mode 4)\n"
            "  [cyan]python main.py --notebook --notebook-name \"My Notebook\"[/cyan]     # notebook (Mode 5)\n"
            "  [cyan]python main.py --grammar-check --goal \"your text\"[/cyan]           # grammar proofreading (Mode 6)\n"
            "  [cyan]python main.py --help[/cyan]                                         # full usage\n"
        )
        from tools.cli_recommender import recommend_startup, print_recommendations as _pr
        startup_tips = recommend_startup()
        _pr(startup_tips, console, title="💡 Getting Started")
        sys.exit(1)

    # ── Pre-run recommendations ───────────────────────────────
    pre_tips = recommend_mode(
        goal=args.goal,
        files=list(args.files),
        current_mode=args.mode,
        web=args.web,
    )
    print_recommendations(pre_tips, console, title="💡 Pre-Run Recommendations")

    console.print(
        Panel(
            f"[bold cyan]Agentic Research Assistant[/bold cyan]\n\n"
            f"Goal: [italic]{args.goal}[/italic]\n"
            f"Mode: [bold]Literature Search (1)[/bold]  |  Model: [bold]{args.model}[/bold]  |  "
            f"Web Search: [bold]{'on' if args.web else 'off'}[/bold]\n"
            f"Embed: [bold]{args.embed_model}[/bold]  |  Top-k: [bold]{args.top_k}[/bold]",
            title="🔬 Research Session",
            border_style="cyan",
        )
    )

    num_ctx = args.num_ctx
    context_char_budget = int(num_ctx * 0.6 * 3)
    max_raw_chars = min(context_char_budget, 80_000)

    processed_docs = []
    if args.files:
        console.print(
            f"\n[bold]Processing documents…[/bold] "
            f"(context budget: {max_raw_chars:,} chars, num_ctx={num_ctx:,})"
        )
        processed_docs = _process_files(
            args.files,
            max_raw_chars=max_raw_chars,
            use_docling=getattr(args, "docling", True),
            use_ocr=getattr(args, "ocr", False),
        )

    # Note: document analysis is available in Mode 5 (Research Notebook)

    from agents.state import create_initial_state
    from agents.graph import run_research

    # ── Load style profile if specified ──────────────────────
    _style_profile = None
    if args.style_profile:
        from agents.style_memory import StyleMemory as _SM
        _style_profile = _SM().load_by_name(args.style_profile)
        if _style_profile:
            console.print(
                f"[magenta]✍ Style profile active:[/magenta] [bold]{args.style_profile}[/bold]"
            )
        else:
            console.print(
                f"[yellow]⚠ Style profile '{args.style_profile}' not found — "
                f"running without style injection.[/yellow]\n"
                f"[dim]List available profiles: python main.py --list-style-profiles[/dim]"
            )

    # ── Socratic clarification ────────────────────────────────
    _clarifications = _ask_clarifying_questions(args.goal, args.mode, args)

    initial_state = create_initial_state(
        goal=args.goal,
        uploaded_docs=processed_docs,
        mode=args.mode,
        include_web_search=args.web,
        model_name=args.model,
        num_ctx=num_ctx,
        embed_model=args.embed_model,
        style_profile=_style_profile,
        clarifications=_clarifications,
    )

    console.print("\n[bold]Running research workflow…[/bold]\n")

    step_descriptions = {
        "document_ingestion":    "Indexing documents (Hybrid RAG: FAISS + BM25)",
        "query_generation":      "Generating search queries with LLM",
        "academic_search":       "Searching arXiv + Semantic Scholar",
        "web_search":            "Searching the web (Google)",
        "document_analysis":     "Analysing documents (Hybrid RRF retrieval)",
        "reference_compilation": "Compiling references",
        "report_generation":     "Generating final report",
        "research_eval":         "Evaluating output quality",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Research workflow", total=100)

        def callback(node_name: str, state: dict):
            pct = state.get("progress_pct", 0)
            label = step_descriptions.get(node_name, node_name.replace("_", " ").title())
            detail = state.get("status_detail", "")
            desc = f"{label} — {detail}" if detail else label
            progress.update(task, completed=pct, description=desc)

        start = time.time()
        try:
            final_state = run_research(initial_state, stream_callback=callback)
        except Exception as e:
            console.print(f"\n[red]✗ Workflow failed: {e}[/red]")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    elapsed = time.time() - start

    console.print(f"\n[green]✓ Complete in {elapsed:.1f}s[/green]\n")

    for err in final_state.get("errors", []):
        console.print(f"[yellow]⚠ {err}[/yellow]")

    findings = final_state.get("key_findings", [])
    if findings:
        table = Table(title="Key Findings", show_header=False, border_style="green")
        table.add_column("", style="bold white", max_width=90)
        for i, f in enumerate(findings, 1):
            table.add_row(f"[{i}] {f}")
        console.print(table)

    refs = final_state.get("references", [])
    if refs:
        console.print(f"\n[bold]References ({len(refs)}):[/bold]")
        for r in refs:
            badge = "🟢" if r["source"] == "semantic_scholar" else "🟡"
            console.print(f"  {badge} [{r['ref_num']}] {r['title'][:70]}…  ({r.get('year', 'n.d.')})")

    report = final_state.get("report", "")
    console.print("\n" + "─" * 80)
    console.print(Markdown(report))

    session_id = final_state.get("session_id", "out")
    out_path = args.output or Path(f"./outputs/report_{session_id}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    console.print(f"\n[bold green]✓ Report saved to:[/bold green] {out_path}")

    _print_eval_cli(final_state.get("eval_result", {}))
    _print_rag_reflection_cli(final_state.get("rag_reflection_info"))

    # ── Interactive feedback refinement ────────────────────────────────────────
    context = " ".join(r.get("title", "") for r in refs[:6])
    refined_report = _feedback_loop(
        current_output=report,
        mode="literature_search",
        model_name=args.model,
        num_ctx=args.num_ctx,
        context=context,
    )
    if refined_report != report:
        refined_path = out_path.with_name(f"report_{session_id}_refined.md")
        refined_path.write_text(refined_report, encoding="utf-8")
        console.print(f"[green]✓ Refined report saved:[/green] {refined_path}")

    # ── Post-run recommendations ──────────────────────────────
    post_tips = recommend_post_research(
        final_state=dict(final_state),
        mode=args.mode,
        elapsed=elapsed,
        web=args.web,
    )
    print_recommendations(post_tips, console, title="💡 Next Steps & Recommendations")

    m_table = Table(title="Workflow Metrics", border_style="cyan")
    m_table.add_column("Metric")
    m_table.add_column("Value", style="bold")
    m_table.add_row("Documents processed", str(len(processed_docs)))
    m_table.add_row("Search queries", str(len(final_state.get("search_queries", []))))
    m_table.add_row("Papers found", str(len(final_state.get("academic_papers", []))))
    m_table.add_row("References cited", str(len(refs)))
    m_table.add_row("Time elapsed", f"{elapsed:.1f}s")
    console.print(m_table)

    # ── Optional inline Systematic Review ─────────────────────
    if getattr(args, "with_systematic_review", False):
        console.rule("[bold magenta]📋 Systematic Review (PRISMA)[/bold magenta]")
        _cmd_systematic_review_inline(args.goal, args)


def _cmd_systematic_review_inline(goal: str, args) -> None:
    """Run a PRISMA systematic review on the same goal as the main research workflow."""
    from agents.systematic_review_graph import run_systematic_review
    from agents.systematic_review_state import create_systematic_review_state

    initial_state = create_systematic_review_state(
        research_question=goal,
        model_name=args.model,
        num_ctx=args.num_ctx,
    )

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Systematic review…", total=100)

        def _cb(node_name: str, state: dict) -> None:
            progress.update(task, completed=state.get("progress_pct", 0), description=node_name.replace("_", " ").title())

        try:
            sr_final = run_systematic_review(initial_state, stream_callback=_cb)
        except Exception as exc:
            console.print(f"[red]✗ Systematic review failed: {exc}[/red]")
            return

    flow = sr_final.get("prisma_flow", {})
    flow_table = Table(title="PRISMA Flow", border_style="blue")
    flow_table.add_column("Stage")
    flow_table.add_column("Count", style="bold")
    for stage in ["identified", "screened", "eligibility", "included", "excluded"]:
        flow_table.add_row(stage.capitalize(), str(flow.get(stage, 0)))
    console.print(flow_table)

    themes = sr_final.get("key_themes", [])
    if themes:
        console.print("\n[bold]Key Themes:[/bold]")
        for t in themes:
            console.print(f"  • {t}")

    console.print("\n[bold]Narrative Synthesis:[/bold]")
    console.print(Markdown(sr_final.get("narrative_synthesis", "No synthesis generated.")))

    evidence = sr_final.get("evidence_table", [])
    if evidence:
        ev_table = Table(title=f"Evidence Table ({len(evidence)} papers)", border_style="green")
        ev_table.add_column("Citation", max_width=18)
        ev_table.add_column("Design", max_width=20)
        ev_table.add_column("Quality", max_width=8)
        ev_table.add_column("Key Finding", max_width=60)
        for row in evidence:
            ev_table.add_row(
                row.get("citation_key", ""),
                row.get("study_design", ""),
                row.get("quality", ""),
                row.get("key_finding", "")[:80],
            )
        console.print(ev_table)


if __name__ == "__main__":
    main()
