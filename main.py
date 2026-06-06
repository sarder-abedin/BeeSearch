"""
main.py — ResearchBuddy CLI
────────────────────────────
Systematic Literature Review (Mode 1) and Research Notebook (Mode 2).

Run:  streamlit run app.py         # web UI
      python main.py --help        # CLI reference

Systematic Literature Review
─────────────────────────────
# Basic PRISMA review
python main.py --systematic-review \\
  --goal "What is the effect of sleep deprivation on working memory?" \\
  --inclusion "Peer-reviewed empirical studies" "Human participants" \\
  --exclusion "Animal studies" "Review papers only"

# With DOCX + PDF reports
python main.py --systematic-review --goal "..." \\
  --sr-docx --sr-pdf --sr-author "Dr. Smith" --sr-institution "MIT"

# With plain-language summaries
python main.py --systematic-review --goal "..." \\
  --sr-plain-language all

# With trend analysis and preprint tracking
python main.py --systematic-review --goal "..." \\
  --sr-trends --sr-preprints --sr-concept-drift

Research Notebook
─────────────────
# Start a new notebook (interactive Q&A)
python main.py --notebook --notebook-name "Antibiotic Resistance"

# Continue an existing notebook
python main.py --notebook --notebook-id <notebook_id>

# Add files when opening a notebook
python main.py --notebook --notebook-id <id> --files paper.pdf notes.txt

# List all saved notebooks
python main.py --list-notebooks

# Advanced analysis (one-shot)
python main.py --notebook-summary <notebook_id>     # cross-doc summary
python main.py --notebook-faq <notebook_id>         # generate FAQ
python main.py --notebook-review <notebook_id>      # literature review
python main.py --notebook-audio <notebook_id>       # audio script + WAV
python main.py --notebook-mindmap <notebook_id>     # mind map (DOT+PNG+SVG)
python main.py --notebook-graph <notebook_id>       # knowledge graph
python main.py --notebook-compare <id> --compare-docs A.pdf B.pdf
python main.py --notebook-timeline <notebook_id>    # timeline extraction
python main.py --notebook-study-table <notebook_id> # study comparison table
python main.py --notebook-pipeline <notebook_id>    # 7-agent pipeline

Utilities
─────────
python main.py --check-system                       # hardware + model check
python main.py --shutdown                           # safe shutdown
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

console = Console()

_KNOWN_FLAGS = [
    "--goal", "--files", "--model", "--output",
    "--num-ctx", "--embed-model", "--top-k", "--check-system", "--verbose",
    "--shutdown",
    "--systematic-review", "--sr", "--inclusion", "--exclusion",
    "--sr-docx", "--sr-pdf", "--sr-plain-language", "--sr-trends",
    "--sr-preprints", "--sr-concept-drift", "--sr-author", "--sr-institution",
    "--notebook", "--notebook-id", "--notebook-name", "--list-notebooks",
    "--notebook-summary", "--notebook-faq", "--notebook-review",
    "--notebook-audio", "--notebook-mindmap", "--notebook-graph",
    "--notebook-compare", "--compare-docs",
    "--notebook-timeline", "--notebook-study-table",
    "--notebook-pipeline", "--pipeline-query",
    "-g", "-f", "-v",
]


def _configure_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class _SmartParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        console.print(f"\n[red]Error:[/red] {message}")
        match = re.search(r"(?:unrecognized argument|invalid choice)[s]?:?\s*'?(--?[\w-]+)", message)
        if match:
            bad = match.group(1)
            suggestions = difflib.get_close_matches(bad, _KNOWN_FLAGS, n=3, cutoff=0.5)
            if suggestions:
                console.print("\n[yellow]Did you mean:[/yellow]")
                for s in suggestions:
                    console.print(f"  [bold cyan]{s}[/bold cyan]")
        console.print("\nRun [bold]python main.py --help[/bold] for full usage.")
        sys.exit(2)


def _parse_args():
    parser = _SmartParser(
        prog="python main.py",
        description=(
            "ResearchBuddy — local AI for systematic literature review and research notebooks.\n\n"
            "  Mode 1 — Systematic Literature Review (--systematic-review / --sr)\n"
            "  Mode 2 — Research Notebook           (--notebook)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Common ───────────────────────────────────────────────────
    parser.add_argument("-g", "--goal", type=str, default="",
                        help="Research question or goal (required for systematic review)")
    parser.add_argument("-f", "--files", nargs="+", type=Path, default=[],
                        help="Documents to add to the notebook")
    parser.add_argument("--model", type=str, default="",
                        help="Ollama model to use (e.g. llama3.1:8b)")
    parser.add_argument("--num-ctx", type=int, default=8192,
                        help="Context window size in tokens (default: 8192)")
    parser.add_argument("--embed-model", type=str, default="nomic-embed-text",
                        help="Embedding model for Hybrid RAG (default: nomic-embed-text)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Chunks per query for Hybrid RAG (default: 8)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path for the Markdown report")
    parser.add_argument("--check-system", action="store_true",
                        help="Print hardware info and Ollama model recommendations, then exit")
    parser.add_argument("--shutdown", action="store_true",
                        help="Free all stale ports and flush ChromaDB, then exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print debug logs")

    # ── Systematic Review ────────────────────────────────────────
    sr = parser.add_argument_group("Systematic Literature Review")
    sr.add_argument(
        "--systematic-review", "--sr",
        action="store_true",
        help="Run a PRISMA systematic review. Searches Google Scholar, arXiv, "
             "Semantic Scholar, and CrossRef. Requires --goal.",
    )
    sr.add_argument(
        "--inclusion", nargs="+", default=[],
        help="Inclusion criteria (one string per criterion)",
    )
    sr.add_argument(
        "--exclusion", nargs="+", default=[],
        help="Exclusion criteria (one string per criterion)",
    )
    sr.add_argument(
        "--sr-docx", action="store_true",
        help="Generate PRISMA 2020 DOCX report (outputs/prisma_report_<id>.docx)",
    )
    sr.add_argument(
        "--sr-pdf", action="store_true",
        help="Generate PRISMA 2020 PDF report (outputs/prisma_report_<id>.pdf)",
    )
    sr.add_argument(
        "--sr-plain-language", choices=["patient", "policy", "press", "all"],
        default="", metavar="FORMAT",
        help="Plain-language summary: patient (8th-grade), policy (policy brief), "
             "press (press release), all (all three)",
    )
    sr.add_argument(
        "--sr-trends", action="store_true",
        help="Fetch field-wide publication trend data from CrossRef and print a year-by-year table",
    )
    sr.add_argument(
        "--sr-preprints", action="store_true",
        help="Check each included paper against CrossRef to flag preprints and retractions",
    )
    sr.add_argument(
        "--sr-concept-drift", action="store_true",
        help="Analyse vocabulary evolution across 5-year buckets and print rising/declining terms",
    )
    sr.add_argument(
        "--sr-author", type=str, default="", metavar="NAME",
        help="Author name for the DOCX/PDF title page",
    )
    sr.add_argument(
        "--sr-institution", type=str, default="", metavar="NAME",
        help="Institution name for the DOCX/PDF title page",
    )

    # ── Research Notebook ────────────────────────────────────────
    nb = parser.add_argument_group("Research Notebook")
    nb.add_argument(
        "--notebook", action="store_true",
        help="Open the Research Notebook in interactive Q&A mode",
    )
    nb.add_argument(
        "--notebook-id", type=str, default="", metavar="ID",
        help="Notebook ID to open (use --list-notebooks to find IDs)",
    )
    nb.add_argument(
        "--notebook-name", type=str, default="", metavar="NAME",
        help="Name for a new notebook",
    )
    nb.add_argument(
        "--list-notebooks", action="store_true",
        help="List all saved Research Notebooks",
    )
    nb.add_argument(
        "--notebook-summary", type=str, default="", metavar="NOTEBOOK_ID",
        help="Generate a cross-document summary for the given notebook",
    )
    nb.add_argument(
        "--notebook-faq", type=str, default="", metavar="NOTEBOOK_ID",
        help="Generate an FAQ for the given notebook",
    )
    nb.add_argument(
        "--notebook-review", type=str, default="", metavar="NOTEBOOK_ID",
        help="Generate a formal literature review for the given notebook",
    )
    nb.add_argument(
        "--notebook-audio", type=str, default="", metavar="NOTEBOOK_ID",
        help="Generate a spoken-word audio summary script for the given notebook",
    )
    nb.add_argument(
        "--notebook-mindmap", type=str, default="", metavar="NOTEBOOK_ID",
        help="Extract a mind map (DOT + PNG + SVG) from the given notebook",
    )
    nb.add_argument(
        "--notebook-graph", type=str, default="", metavar="NOTEBOOK_ID",
        help="Extract a knowledge graph (DOT + PNG + SVG) from the given notebook",
    )
    nb.add_argument(
        "--notebook-compare", type=str, default="", metavar="NOTEBOOK_ID",
        help="Compare two sources in the given notebook (requires --compare-docs)",
    )
    nb.add_argument(
        "--compare-docs", nargs=2, metavar=("SOURCE_A", "SOURCE_B"), default=[],
        help="Two source filenames to compare (used with --notebook-compare)",
    )
    nb.add_argument(
        "--notebook-timeline", type=str, default="", metavar="NOTEBOOK_ID",
        help="Extract a chronological timeline from the given notebook",
    )
    nb.add_argument(
        "--notebook-study-table", type=str, default="", metavar="NOTEBOOK_ID",
        help="Generate a structured study comparison table for the given notebook",
    )
    nb.add_argument(
        "--notebook-pipeline", type=str, default="", metavar="NOTEBOOK_ID",
        help="Run the full 7-agent pipeline for the given notebook "
             "(ingest → summarize → retrieve → verify_citations → build_kg → study_guide → podcast)",
    )
    nb.add_argument(
        "--pipeline-query", type=str, default="", metavar="QUERY",
        help="Optional focus query for Agent 3 (Retrieval) when using --notebook-pipeline",
    )

    args = parser.parse_args()

    # Resolve model default
    if not args.model:
        from config.settings import get_settings
        args.model = get_settings().ollama_model or "llama3.2:3b"

    return args


# ─── Hardware banner ─────────────────────────────────────────────────────────

def _print_hardware_banner(ollama_base_url: str, user_model: str | None = None) -> dict:
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
        console.print(Panel(
            f"[bold green]Recommended model:[/bold green] {rec['model']}\n"
            f"[bold green]Recommended num_ctx:[/bold green] {rec['num_ctx']:,}\n\n"
            f"{rec['reasoning']}\n\n"
            f"[dim]{rec['hardware_note']}[/dim]",
            title="💡 Recommendation", border_style="green",
        ))
        if user_model and user_model not in available:
            console.print(f"[yellow]⚠  --model {user_model!r} is not pulled.\n"
                          f"   Run: ollama pull {user_model}[/yellow]")
        elif user_model and user_model != rec["model"]:
            console.print(f"[dim]ℹ  Using --model {user_model!r} (recommended: {rec['model']})[/dim]")

        if rec.get("tight_fit") and rec.get("safe_alternative") and not user_model:
            safe_alt = rec["safe_alternative"]
            console.print(
                f"\n[bold yellow]⚠  Tight memory fit detected.[/bold yellow]\n"
                f"  [bold cyan]1[/bold cyan]. {rec['model']} — higher capability, uses ≥85% RAM\n"
                f"  [bold cyan]2[/bold cyan]. {safe_alt['name']} — {safe_alt['ram_gb']} GB, more headroom"
            )
            if sys.stdin.isatty():
                choice = Prompt.ask("Your choice", choices=["1", "2"], default="1")
                if choice == "2":
                    rec = dict(rec)
                    rec["model"] = safe_alt["name"]
                    rec["num_ctx"] = safe_alt["num_ctx"]
                    rec["user_chose_model"] = True
    else:
        console.print(Panel(
            f"[yellow]{rec['reasoning']}[/yellow]\n\n{rec['hardware_note']}\n\n"
            + (f"[bold]Pull:[/bold]\n  {rec['pull_command']}" if rec["pull_command"] else ""),
            title="⚠️  No Compatible Model", border_style="yellow",
        ))

    return rec


# ─── Output helpers ───────────────────────────────────────────────────────────

def _print_eval_cli(eval_result: dict) -> None:
    if not eval_result or not eval_result.get("overall"):
        return
    overall = eval_result.get("overall", 0)
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
    summary = eval_result.get("summary", "")
    if summary:
        console.print(f"[dim]{summary}[/dim]\n")


def _print_rag_reflection_cli(rag_reflection_info) -> None:
    if not rag_reflection_info:
        return
    entries = rag_reflection_info if isinstance(rag_reflection_info, list) else [rag_reflection_info]
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
        rewritten = entry.get("rewritten_queries", [])
        prefix = f"Q{i+1} " if len(entries) > 1 else ""
        if cycles is not None:
            t.add_row(f"{prefix}Cycles", str(cycles))
        if rewritten:
            t.add_row(f"{prefix}Rewritten query", rewritten[0][:60])
        if entry.get("grading_skipped"):
            t.add_row(f"{prefix}Grading", "[yellow]skipped (all-true)[/yellow]")
    console.print(t)


# ─── File processing ──────────────────────────────────────────────────────────

def _process_files(files, chunk_size=800, overlap=150, max_raw_chars=0,
                   use_docling=True, use_ocr=False, large_doc_page_threshold=50):
    from tools.document_tools import get_processor
    if use_docling:
        console.print(
            f"[dim]Using Docling{'+ OCR' if use_ocr else ''} for advanced parsing "
            f"(pdfplumber fallback for PDFs > {large_doc_page_threshold} pages)[/dim]"
        )
    docs = []
    for fp in files:
        if not fp.exists():
            console.print(f"[red]✗ File not found: {fp}[/red]")
            continue
        try:
            processor = get_processor(
                use_docling=use_docling, use_ocr=use_ocr,
                chunk_size=chunk_size, overlap=overlap,
                max_raw_chars=max_raw_chars,
                file_path=fp,
                large_doc_page_threshold=large_doc_page_threshold,
            )
            doc = processor.process_file(fp)
            chars = len(doc.raw_text)
            console.print(
                f"[green]✓[/green] {fp.name} — {doc.total_pages} pages, {chars:,} chars"
                + (f" (capped at {max_raw_chars:,})" if max_raw_chars and chars >= max_raw_chars else "")
            )
            docs.append(doc)
        except Exception as e:
            console.print(f"[red]✗ Failed to process {fp.name}: {e}[/red]")
    return docs


# ─── Safe exit / shutdown ─────────────────────────────────────────────────────

def _cli_safe_exit() -> None:
    from tools.shutdown import safe_shutdown, PORT_GOOGLE_SEARCH, PORT_OLLAMA
    safe_shutdown(ports=[PORT_GOOGLE_SEARCH, PORT_OLLAMA], flush_db=True, console=console)


def _cmd_shutdown() -> None:
    from tools.shutdown import safe_shutdown, is_port_in_use, ALL_PORTS
    console.rule("[bold red]Safe Shutdown[/bold red]")
    safe_shutdown(ports=ALL_PORTS, flush_db=True, console=console)
    console.rule()


# ─── Feedback refinement loop ─────────────────────────────────────────────────

def _feedback_loop(current_output: str, mode: str, model_name: str,
                   num_ctx: int, context: str = "") -> str:
    from agents.feedback_agent import refine_with_feedback, MAX_FEEDBACK_ROUNDS
    refined = current_output
    for round_num in range(1, MAX_FEEDBACK_ROUNDS + 1):
        console.print(f"\n[dim]─── Feedback round {round_num}/{MAX_FEEDBACK_ROUNDS} — press Enter to skip ───[/dim]")
        try:
            feedback = input("Feedback> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Feedback skipped.[/dim]")
            break
        if not feedback:
            break
        with Progress(SpinnerColumn(), TextColumn("[bold blue]Refining output…"),
                      console=console, transient=True) as progress:
            progress.add_task("", total=None)
            refined = refine_with_feedback(
                original_output=refined, feedback=feedback, context=context,
                mode=mode, model_name=model_name, num_ctx=num_ctx,
            )
        console.print("\n[bold green]Refined output:[/bold green]")
        preview = refined[:3000]
        if len(refined) > 3000:
            preview += "\n…[truncated — full output in file]"
        console.print(preview)
    return refined


# ─── Systematic Literature Review ────────────────────────────────────────────

def _cmd_systematic_review(args) -> None:
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

    console.print(Panel(
        f"[bold cyan]Systematic Literature Review[/bold cyan]\n\n"
        f"Research question: [italic]{rq}[/italic]\n"
        f"Inclusion: {', '.join(inclusion) or '(auto)'}\n"
        f"Exclusion: {', '.join(exclusion) or '(auto)'}\n"
        f"Model: [bold]{args.model}[/bold]",
        title="📋 PRISMA Systematic Review", border_style="blue",
    ))

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
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Systematic review", total=100)

        def sr_callback(node_name: str, state: dict):
            pct = state.get("progress_pct", 0)
            label = node_labels.get(node_name, node_name)
            detail = state.get("status_detail", "")
            progress.update(task, completed=pct, description=f"{label} — {detail}" if detail else label)

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
        console.print(Panel(
            Markdown(synthesis[:2000] + (" …" if len(synthesis) > 2000 else "")),
            title="Narrative Synthesis", border_style="blue",
        ))

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

    # Feedback refinement
    if synthesis:
        context = " ".join(p.get("title", "") for p in final_state.get("included_papers", [])[:5])
        refined = _feedback_loop(synthesis, "systematic_review", args.model, args.num_ctx, context)
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

    # ── Optional post-run tools ───────────────────────────────────────────────

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

    pls_fmt = getattr(args, "sr_plain_language", "")
    if pls_fmt:
        try:
            from tools.plain_language import (
                generate_patient_summary, generate_policy_brief,
                generate_press_release, generate_all_summaries,
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
                console.print(Panel(
                    Markdown(text[:800] + ("…" if len(text) > 800 else "")),
                    title=f"Plain-Language Summary ({fmt_key})", border_style="cyan",
                ))
        except Exception as e:
            console.print(f"[red]Plain-language summary failed: {e}[/red]")

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
            corpus_by_yr = trend_data.get("corpus_by_year", {})
            trend_table = Table(title="Research Publication Trend", border_style="cyan")
            trend_table.add_column("Year")
            trend_table.add_column("Field-wide (CrossRef)", style="blue")
            trend_table.add_column("This SR corpus", style="green")
            for yr in sorted(combined):
                trend_table.add_row(str(yr), str(combined.get(yr, 0)), str(corpus_by_yr.get(yr, 0)))
            console.print(trend_table)
            console.print(
                f"Trend: [bold]{trend_data.get('trend', 'unknown').upper()}[/bold]  "
                f"Peak year: [bold]{trend_data.get('peak_year', 'N/A')}[/bold]  "
                f"Total CrossRef: [bold]{trend_data.get('total_field', 0):,}[/bold]"
            )
        except Exception as e:
            console.print(f"[red]Trend analysis failed: {e}[/red]")

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
                status_colors = {"journal": "green", "published": "cyan",
                                 "preprint": "yellow", "retracted": "red"}
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
                    f"Published: [cyan]{summary.get('published', 0)}[/cyan]  "
                    f"Preprint only: [yellow]{summary.get('preprint', 0)}[/yellow]  "
                    f"Retracted: [red]{summary.get('retracted', 0)}[/red]"
                )
            except Exception as e:
                console.print(f"[red]Preprint tracking failed: {e}[/red]")

    if getattr(args, "sr_concept_drift", False):
        all_papers = final_state.get("raw_papers", [])
        if all_papers:
            try:
                from tools.concept_drift import detect_concept_drift
                with console.status("Analysing vocabulary evolution…"):
                    drift = detect_concept_drift(
                        papers=all_papers, model_name=args.model, num_ctx=args.num_ctx,
                    )
                drift_table = Table(title="Concept Drift — Vocabulary Evolution", border_style="magenta")
                drift_table.add_column("Type")
                drift_table.add_column("Term")
                drift_table.add_column("Period")
                drift_table.add_column("Change")
                for r in drift.get("rising_terms", [])[:6]:
                    drift_table.add_row("[green]Rising[/green]", r["term"],
                                        f"{r['first_bucket']} → {r['last_bucket']}", f"+{r['growth']}")
                for d in drift.get("declining_terms", [])[:6]:
                    drift_table.add_row("[red]Declining[/red]", d["term"],
                                        f"{d['first_bucket']} → {d['last_bucket']}", str(d["growth"]))
                console.print(drift_table)
                if drift.get("llm_analysis"):
                    console.print(Panel(
                        Markdown(drift["llm_analysis"]),
                        title="Vocabulary Shift Analysis", border_style="magenta",
                    ))
            except Exception as e:
                console.print(f"[red]Concept drift analysis failed: {e}[/red]")


# ─── Research Notebook ────────────────────────────────────────────────────────

def _cmd_list_notebooks():
    from agents.notebook_memory import NotebookMemory
    notebooks = NotebookMemory().list_notebooks()
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
            nb["notebook_id"], nb.get("name", "Untitled")[:40],
            str(nb.get("source_count", 0)), str(nb.get("turn_count", 0)),
            nb.get("last_modified", "")[:16],
        )
    console.print(table)


def _cmd_notebook_advanced(notebook_id: str, feature: str, args) -> None:
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
            console.print(f"[red]✗ {err}[/red]"); return
        console.print(Markdown(result))
        p = out_dir / f"summary_{notebook_id}.md"
        p.write_text(result, encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {p}")

    elif feature == "faq":
        from agents.notebook_advanced import generate_faq
        with console.status("[bold blue]Generating FAQ…[/bold blue]"):
            items, err = generate_faq(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]"); return
        for i, item in enumerate(items, 1):
            console.print(f"\n[bold cyan]Q{i}: {item.get('question', '')}[/bold cyan]")
            console.print(Markdown(item.get("answer", "")))
        md = "\n\n".join(f"### {it.get('question','')}\n{it.get('answer','')}" for it in items)
        p = out_dir / f"faq_{notebook_id}.md"
        p.write_text(md, encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {p}")

    elif feature == "review":
        from agents.notebook_advanced import generate_literature_review
        with console.status("[bold blue]Generating literature review…[/bold blue]"):
            result, err = generate_literature_review(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]"); return
        console.print(Markdown(result))
        p = out_dir / f"literature_review_{notebook_id}.md"
        p.write_text(result, encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {p}")

    elif feature == "audio":
        from agents.notebook_advanced import generate_audio_summary, synthesize_speech
        with console.status("[bold blue]Generating audio summary script…[/bold blue]"):
            result, err = generate_audio_summary(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]"); return
        console.print(Panel(result, title="🔊 Audio Summary Script", border_style="cyan"))
        txt_path = out_dir / f"audio_summary_{notebook_id}.txt"
        txt_path.write_text(result, encoding="utf-8")
        console.print(f"[green]✓ Script saved:[/green] {txt_path}")
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
            console.print(f"[red]✗ {err}[/red]"); return
        src_names = [s["filename"] for s in notebook.get("sources", [])]
        table = Table(title=f"Timeline — {nb_name}", border_style="blue")
        table.add_column("Year", no_wrap=True)
        table.add_column("Event", max_width=55)
        table.add_column("Significance", max_width=40)
        table.add_column("Source", max_width=20)
        for item in items:
            src_n = item.get("source", 0)
            src_label = (src_names[src_n - 1][:20]
                         if isinstance(src_n, int) and 1 <= src_n <= len(src_names) else "—")
            table.add_row(item.get("year", "n.d."), item.get("event", "")[:55],
                          item.get("significance", "")[:40], src_label)
        console.print(table)
        md_lines = ["| Year | Event | Significance | Source |", "|------|-------|-------------|--------|"]
        for item in items:
            src_n = item.get("source", 0)
            src_label = (src_names[src_n - 1][:20]
                         if isinstance(src_n, int) and 1 <= src_n <= len(src_names) else "—")
            md_lines.append(f"| {item.get('year','n.d.')} | {item.get('event','')} | "
                            f"{item.get('significance','')} | {src_label} |")
        p = out_dir / f"timeline_{notebook_id}.md"
        p.write_text("\n".join(md_lines), encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {p}")

    elif feature == "study-table":
        from agents.notebook_advanced import generate_study_comparison
        with console.status("[bold blue]Generating study comparison table…[/bold blue]"):
            result, err = generate_study_comparison(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]"); return
        console.print(Markdown(result))
        p = out_dir / f"study_comparison_{notebook_id}.md"
        p.write_text(result, encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {p}")

    elif feature in ("mindmap", "graph"):
        if feature == "mindmap":
            from agents.notebook_advanced import generate_mindmap as _gen, render_dot_bytes
            status_msg = "Extracting mind map…"
            base_name = f"mindmap_{notebook_id}"
        else:
            from agents.notebook_advanced import extract_knowledge_graph as _gen, render_dot_bytes
            status_msg = "Extracting knowledge graph…"
            base_name = f"knowledge_graph_{notebook_id}"
        with console.status(f"[bold blue]{status_msg}[/bold blue]"):
            dot, err = _gen(notebook_id, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]"); return
        base = out_dir / base_name
        base.with_suffix(".dot").write_text(dot, encoding="utf-8")
        console.print(f"[green]✓[/green] DOT: {base.with_suffix('.dot')}")
        print(dot)
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
            console.print("[red]--compare-docs requires exactly two source filenames.[/red]"); return
        sources = notebook.get("sources", [])
        name_to_id = {s["filename"]: s["doc_id"] for s in sources}
        doc_a, doc_b = name_to_id.get(compare_docs[0], ""), name_to_id.get(compare_docs[1], "")
        if not doc_a or not doc_b:
            console.print(f"[red]Source not found.[/red] Available: {[s['filename'] for s in sources]}"); return
        from agents.notebook_advanced import compare_sources
        with console.status("[bold blue]Comparing sources…[/bold blue]"):
            result, err = compare_sources(notebook_id, doc_a, doc_b, settings)
        if err:
            console.print(f"[red]✗ {err}[/red]"); return
        console.print(Markdown(result))
        p = out_dir / f"comparison_{notebook_id}.md"
        p.write_text(result, encoding="utf-8")
        console.print(f"\n[green]✓ Saved:[/green] {p}")


def _cmd_notebook_pipeline(notebook_id: str, args) -> None:
    from agents.notebook_memory import NotebookMemory
    from agents.notebook_pipeline_graph import run_notebook_pipeline
    from agents.notebook_pipeline_state import create_pipeline_state

    notebook = NotebookMemory().load(notebook_id)
    if not notebook:
        console.print(f"[red]Notebook '{notebook_id}' not found.[/red]"); return

    nb_name = notebook.get("name", notebook_id)
    console.print(f"\n[bold]Running 7-agent pipeline[/bold] for: [cyan]{nb_name}[/cyan]")
    console.print(f"  Sources : {len(notebook.get('sources', []))}")
    console.print(f"  Chunks  : {len(notebook.get('chunks', []))}\n")

    from config.settings import get_settings as _get_cfg
    _cfg = _get_cfg()
    settings = {
        "model": args.model,
        "num_ctx": args.num_ctx,
        "embed_model": getattr(args, "embed_model", _cfg.embedding_model),
    }
    initial = create_pipeline_state(
        notebook_id=notebook_id, settings=settings,
        query=getattr(args, "pipeline_query", "") or "",
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
        console.print(f"  [green]✓[/green] {_LABELS.get(node_name, node_name)} ({partial.get('progress_pct', 0)}%)")

    try:
        final = run_notebook_pipeline(initial, stream_callback=_cb)
    except Exception as exc:
        console.print(f"[red]Pipeline failed: {exc}[/red]"); return

    if final.get("errors"):
        for e in final["errors"]:
            console.print(f"  [yellow]·[/yellow] {e}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    safe_name = nb_name.replace(" ", "_")[:30]
    console.print("\n[bold]Saving outputs…[/bold]")

    if final.get("cross_summary"):
        p = out_dir / f"pipeline_summary_{safe_name}.md"
        p.write_text(final["cross_summary"], encoding="utf-8")
        console.print(f"  [green]✓[/green] Summary:          {p}")

    if final.get("citation_report"):
        p = out_dir / f"pipeline_citations_{safe_name}.md"
        p.write_text(final["citation_report"], encoding="utf-8")
        console.print(f"  [green]✓[/green] Citation report:  {p}")

    if final.get("knowledge_graph_dot"):
        dot = final["knowledge_graph_dot"]
        base = out_dir / f"pipeline_kg_{safe_name}"
        base.with_suffix(".dot").write_text(dot, encoding="utf-8")
        console.print(f"  [green]✓[/green] KG DOT:           {base.with_suffix('.dot')}")
        from agents.notebook_advanced import render_dot_bytes
        for fmt in ("png", "svg"):
            img, img_err = render_dot_bytes(dot, fmt)
            if img:
                base.with_suffix(f".{fmt}").write_bytes(img)
                console.print(f"  [green]✓[/green] KG {fmt.upper()}:          {base.with_suffix(f'.{fmt}')}")
            else:
                console.print(f"  [yellow]⚠ KG {fmt.upper()} unavailable: {img_err}[/yellow]")

    if final.get("study_guide"):
        guide = final["study_guide"]
        p = out_dir / f"pipeline_study_guide_{safe_name}.md"
        p.write_text(guide, encoding="utf-8")
        console.print(f"  [green]✓[/green] Study guide (.md): {p}")
        try:
            from tools.export_tools import build_docx, build_pdf
            docx_p = out_dir / f"pipeline_study_guide_{safe_name}.docx"
            docx_p.write_bytes(build_docx(guide, []))
            console.print(f"  [green]✓[/green] Study guide (.docx): {docx_p}")
            pdf_p = out_dir / f"pipeline_study_guide_{safe_name}.pdf"
            pdf_p.write_bytes(build_pdf(guide, []))
            console.print(f"  [green]✓[/green] Study guide (.pdf): {pdf_p}")
        except Exception as exc:
            console.print(f"  [yellow]⚠ DOCX/PDF export: {exc}[/yellow]")

    if final.get("podcast_script"):
        p = out_dir / f"pipeline_podcast_{safe_name}.txt"
        p.write_text(final["podcast_script"], encoding="utf-8")
        console.print(f"  [green]✓[/green] Podcast script:   {p}")

    console.print("\n[bold green]Pipeline complete.[/bold green]")

    if final.get("study_guide"):
        context = " ".join(s.get("filename", "") for s in notebook.get("sources", [])[:5])
        refined = _feedback_loop(final["study_guide"], "notebook", args.model, args.num_ctx, context)
        if refined != final["study_guide"]:
            refined_p = out_dir / f"pipeline_study_guide_{safe_name}_refined.md"
            refined_p.write_text(refined, encoding="utf-8")
            console.print(f"  [green]✓[/green] Refined study guide: {refined_p}")


def _cmd_notebook(args) -> None:
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
        notebooks = memory.list_notebooks()
        if notebooks:
            console.print("\n[bold]Existing notebooks:[/bold]")
            for i, nb in enumerate(notebooks, 1):
                console.print(f"  [cyan]{i}[/cyan]. {nb['name']} "
                              f"[dim]({nb['source_count']} sources, {nb['turn_count']} turns)[/dim]")
            console.print(f"  [cyan]{len(notebooks)+1}[/cyan]. [bold]Create new notebook[/bold]")
            choice = Prompt.ask("Select", choices=[str(i) for i in range(1, len(notebooks) + 2)], default="1")
            idx = int(choice) - 1
            if idx < len(notebooks):
                notebook_id = notebooks[idx]["notebook_id"]
            else:
                notebook_id = ""
        if not notebook_id:
            nb_name = getattr(args, "notebook_name", "") or Prompt.ask("Notebook name")
            notebook_id = memory.new_notebook(nb_name.strip() or "Untitled Notebook")
            console.print(f"[green]✓ Created notebook:[/green] {notebook_id}")

    notebook = memory.load(notebook_id)
    if not notebook:
        console.print(f"[red]Notebook not found:[/red] {notebook_id}"); return

    if getattr(args, "files", None):
        from tools.hybrid_store import get_or_create_store
        with console.status("[bold]Processing documents…[/bold]"):
            processed = _process_files(list(args.files))
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

    console.rule("[bold blue]Research Notebook[/bold blue]")
    sources = notebook.get("sources", [])
    console.print(Panel(
        f"Notebook: [bold]{notebook.get('name', 'Notebook')}[/bold]\n"
        f"ID:       [cyan]{notebook_id}[/cyan]\n"
        f"Sources:  {len(sources)}\n"
        f"Turns:    {len(notebook.get('conversation', []))}"
        + (("\nFiles:  " + ", ".join(s['filename'][:25] for s in sources[:4])
            + (" …" if len(sources) > 4 else "")) if sources else ""),
        title="📓 Session Info", border_style="blue",
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
                        f"[{c['n']}] {c.get('doc_name','')} p.{c.get('page','')}" for c in citations
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

        if stripped in ("1", "2", "3") and last_questions:
            idx = int(stripped) - 1
            if idx < len(last_questions):
                stripped = last_questions[idx]
                console.print(f"[dim]→ {stripped}[/dim]")
            else:
                console.print("[yellow]No question at that number.[/yellow]"); continue

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
                fp = Path(arg.strip()) if arg.strip() else Path(Prompt.ask("File path"))
                docs = _process_files([fp])
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
                processor = DocumentProcessor(chunk_size=settings["chunk_size"],
                                              overlap=settings["chunk_overlap"])
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

        nb = memory.load(notebook_id)
        if not nb or not nb.get("sources"):
            console.print("[yellow]No sources. Use /add <file> or /url <url> to add documents.[/yellow]")
            continue

        state = create_notebook_state(
            user_message=stripped, notebook_id=notebook_id,
            model_name=args.model, num_ctx=args.num_ctx,
            embed_model=settings["embed_model"],
        )

        with console.status("[bold blue]Searching notebook…[/bold blue]"):
            try:
                final = run_notebook_turn(state)
            except Exception as e:
                console.print(f"[red]✗ Error: {e}[/red]")
                if getattr(args, "verbose", False):
                    import traceback; traceback.print_exc()
                continue

        response = final.get("assistant_response", "")
        console.print(f"\n[bold blue]Notebook[/bold blue]:\n")
        console.print(Markdown(response))

        citations = final.get("citations", [])
        if citations:
            refs = ", ".join(
                f"[{c['n']}] {c.get('doc_name','')} p.{c.get('page','')}" for c in citations
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


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    if args.shutdown:
        _configure_logging(verbose=False)
        _cmd_shutdown()
        return

    _configure_logging(args.verbose)

    from tools.shutdown import install_signal_handlers
    install_signal_handlers(console=console)

    # ── Utility commands ───────────────────────────────────────────────────────
    if args.list_notebooks:
        _cmd_list_notebooks()
        return

    if args.notebook_summary:
        _cmd_notebook_advanced(args.notebook_summary, "summary", args); return
    if args.notebook_faq:
        _cmd_notebook_advanced(args.notebook_faq, "faq", args); return
    if args.notebook_review:
        _cmd_notebook_advanced(args.notebook_review, "review", args); return
    if args.notebook_audio:
        _cmd_notebook_advanced(args.notebook_audio, "audio", args); return
    if args.notebook_mindmap:
        _cmd_notebook_advanced(args.notebook_mindmap, "mindmap", args); return
    if args.notebook_graph:
        _cmd_notebook_advanced(args.notebook_graph, "graph", args); return
    if args.notebook_compare:
        _cmd_notebook_advanced(args.notebook_compare, "compare", args); return
    if args.notebook_timeline:
        _cmd_notebook_advanced(args.notebook_timeline, "timeline", args); return
    if args.notebook_study_table:
        _cmd_notebook_advanced(args.notebook_study_table, "study-table", args); return
    if args.notebook_pipeline:
        _cmd_notebook_pipeline(args.notebook_pipeline, args); return

    # ── Hardware check ─────────────────────────────────────────────────────────
    from config.settings import get_settings
    ollama_url = get_settings().ollama_base_url
    console.rule("[bold cyan]ResearchBuddy — System Check[/bold cyan]")
    rec = _print_hardware_banner(ollama_url, user_model=args.model)
    console.rule()
    if rec.get("user_chose_model"):
        args.model = rec["model"]

    if args.check_system:
        return

    # ── Mode dispatch ──────────────────────────────────────────────────────────
    if args.notebook or getattr(args, "notebook_id", ""):
        _cmd_notebook(args)
        return

    if args.systematic_review:
        _cmd_systematic_review(args)
        return

    # No mode selected
    console.print(
        "[red]Error:[/red] No mode selected.\n\n"
        "Quick start:\n"
        "  [cyan]python main.py --systematic-review --goal \"your research question\"[/cyan]\n"
        "  [cyan]python main.py --notebook --notebook-name \"My Notebook\"[/cyan]\n"
        "  [cyan]python main.py --help[/cyan]\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
