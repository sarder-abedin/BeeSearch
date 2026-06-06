#!/usr/bin/env python3
"""
cli.py — ResearchBuddy command-line interface

Usage:
  python cli.py sr   "<research question>"  [options]
  python cli.py nb   <notebook_id>  "<question>"  [options]
  python cli.py nb   --new "<name>"  "<question>"  [options]
  python cli.py nb   --list
  python cli.py bib  <notebook_id>  <file.bib>
  python cli.py gap  <notebook_id>  "<research question>"
  python cli.py hyp  <notebook_id>  "<research question>"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print('=' * 60)


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def _settings(model: str, num_ctx: int) -> dict:
    return {
        "model": model,
        "num_ctx": num_ctx,
        "embed_model": "nomic-embed-text",
        "hybrid_top_k": 10,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "use_docling": False,
    }


# ── Command: sr ──────────────────────────────────────────────────────

def cmd_sr(args: argparse.Namespace) -> None:
    """Run a full systematic review from the command line."""
    from agents.systematic_review_graph import run_systematic_review
    from agents.systematic_review_state import create_systematic_review_state

    inclusion = [c.strip() for c in (args.include or "").split(",") if c.strip()]
    exclusion = [c.strip() for c in (args.exclude or "").split(",") if c.strip()]

    _banner(f"Systematic Review: {args.question[:60]}")
    print(f"Model      : {args.model}")
    print(f"Inclusion  : {inclusion or '(none specified)'}")
    print(f"Exclusion  : {exclusion or '(none specified)'}")
    print()

    state = create_systematic_review_state(
        research_question=args.question,
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion,
        model_name=args.model,
        num_ctx=args.num_ctx,
    )

    def progress(node_name: str, s: dict) -> None:
        pct = s.get("progress_pct", 0)
        detail = s.get("status_detail", "")
        label = {
            "query_generation":    "Generating queries",
            "literature_search":   "Searching literature",
            "screening":           "Screening papers",
            "evidence_extraction": "Extracting evidence + RoB + GRADE",
            "synthesis":           "Synthesising findings",
            "sr_eval":             "Evaluating quality",
        }.get(node_name, node_name)
        bar = "#" * (pct // 5) + "." * (20 - pct // 5)
        print(f"  [{bar}] {pct:3d}%  {label}" + (f" — {detail}" if detail else ""))

    final = run_systematic_review(state, stream_callback=progress)
    print()

    # ─ PRISMA flow
    _section("PRISMA Flow")
    flow = final.get("prisma_flow", {})
    for k, v in flow.items():
        print(f"  {k.title():12s}: {v}")

    # ─ GRADE
    grade = final.get("grade_results", {})
    if grade:
        _section("GRADE Evidence Grading")
        print(f"  Overall grade : {grade.get('overall_grade', 'n/a')}")
        print(f"  Certainty     : {grade.get('certainty_statement', '')}")

    # ─ RoB summary
    rob_table = final.get("rob_table", [])
    if rob_table:
        _section(f"Risk of Bias ({len(rob_table)} papers)")
        for r in rob_table:
            print(f"  [{r.get('citation_key','')}] {r.get('title','')[:50]} -> {r.get('overall','')}")

    # ─ Contradictions
    contras = final.get("contradictions", [])
    if contras:
        _section(f"Contradictions ({len(contras)})")
        for c in contras:
            print(f"  - {c.get('claim','')}")
            print(f"    A: {c.get('position_a', {}).get('description', '')}")
            print(f"    B: {c.get('position_b', {}).get('description', '')}")

    # ─ Synthesis
    _section("Key Themes")
    for t in final.get("key_themes", []):
        print(f"  • {t}")

    _section("Narrative Synthesis (first 1500 chars)")
    print(final.get("narrative_synthesis", "")[:1500])

    _section("Research Gaps")
    for g in final.get("research_gaps", []):
        print(f"  • {g}")

    _section("Conclusion")
    print(final.get("conclusion", ""))

    # ─ Eval
    eval_result = final.get("eval_result", {})
    if eval_result:
        _section("Quality Scores")
        for k, v in eval_result.items():
            if k != "summary":
                print(f"  {k.replace('_',' ').title():30s}: {v}")
        print(f"  Summary: {eval_result.get('summary', '')}")

    # ─ Export
    if args.output:
        from ui.tabs.systematic_review import _build_sr_markdown
        md = _build_sr_markdown(args.question, final)
        output_path = Path(args.output)
        output_path.write_text(md, encoding="utf-8")
        print(f"\n✅ Saved to {output_path.resolve()}")

    if args.json_output:
        json_path = Path(args.json_output)
        export = {
            k: v for k, v in final.items()
            if isinstance(v, (str, int, float, list, dict, type(None)))
        }
        json_path.write_text(json.dumps(export, indent=2, default=str), encoding="utf-8")
        print(f"✅ JSON saved to {json_path.resolve()}")


# ── Command: nb ──────────────────────────────────────────────────────

def cmd_nb(args: argparse.Namespace) -> None:
    """Interact with a Research Notebook from the CLI."""
    from agents.notebook_memory import NotebookMemory

    memory = NotebookMemory()

    if args.list:
        _banner("Research Notebooks")
        notebooks = memory.list_notebooks()
        if not notebooks:
            print("  No notebooks yet.")
        for nb in notebooks:
            print(
                f"  [{nb['notebook_id']}] {nb['name']} — "
                f"{nb['source_count']} sources, {nb['turn_count']} turns"
            )
        return

    # Resolve notebook ID
    if args.new:
        nb_id = memory.new_notebook(args.new)
        print(f"✅ Created notebook '{args.new}' — ID: {nb_id}")
    else:
        nb_id = args.notebook_id
        if not nb_id:
            print("Error: provide --notebook-id or --new <name>")
            sys.exit(1)

    question = args.question
    if not question:
        # Interactive REPL mode
        _banner(f"Notebook REPL — {nb_id}")
        print("Type your question and press Enter. Type 'exit' to quit.\n")
        while True:
            try:
                question = input("❓ ")
            except (KeyboardInterrupt, EOFError):
                break
            if question.strip().lower() in ("exit", "quit", "q"):
                break
            _ask_notebook(nb_id, question.strip(), args)
        return

    _ask_notebook(nb_id, question, args)


def _ask_notebook(nb_id: str, question: str, args: argparse.Namespace) -> None:
    from agents.notebook_graph import run_notebook_turn
    from agents.notebook_state import create_notebook_state
    from config.settings import get_settings

    cfg = get_settings()

    state = create_notebook_state(
        user_message=question,
        notebook_id=nb_id,
        model_name=args.model,
        num_ctx=args.num_ctx,
        embed_model=cfg.embedding_model,
        top_k=10,
        include_web_search=getattr(args, "web", False),
    )

    print(f"\n🧠 Searching notebook {nb_id}…")

    def _cb(node_name: str, _s: dict) -> None:
        labels = {"retrieve": "Retrieving", "answer": "Answering", "save": "Saving"}
        print(f"  {labels.get(node_name, node_name)}…")

    final = run_notebook_turn(state, stream_callback=_cb)

    print(f"\n🤖 Answer:\n")
    print(final.get("assistant_response", ""))

    citations = final.get("citations", [])
    if citations:
        print(f"\n📎 Sources:")
        for c in citations:
            doc = c.get("doc_name", "")
            page = c.get("page", "")
            url = c.get("url", "")
            ref = f"[{c.get('n')}] {doc}" + (f" p.{page}" if page else "") + (f" {url}" if url else "")
            print(f"  {ref}")

    suggested = final.get("suggested_questions", [])
    if suggested:
        print(f"\n💡 Suggested follow-ups:")
        for i, q in enumerate(suggested, 1):
            print(f"  {i}. {q}")


# ── Command: bib ─────────────────────────────────────────────────────

def cmd_bib(args: argparse.Namespace) -> None:
    """Import a BibTeX file into a notebook."""
    from tools.zotero_importer import import_bibtex_to_notebook

    bib_path = Path(args.bib_file)
    if not bib_path.exists():
        print(f"Error: {bib_path} not found")
        sys.exit(1)

    content = bib_path.read_text(encoding="utf-8", errors="ignore")
    settings = _settings(args.model, args.num_ctx)

    print(f"Importing {bib_path.name} into notebook {args.notebook_id}…")
    added, errors = import_bibtex_to_notebook(content, args.notebook_id, settings)
    print(f"✅ Imported {added} references.")
    for err in errors:
        print(f"  ⚠️  {err}")


# ── Command: gap ─────────────────────────────────────────────────────

def cmd_gap(args: argparse.Namespace) -> None:
    """Map research gaps from a notebook."""
    from tools.research_gaps import map_research_gaps
    from tools.hybrid_store import get_or_create_store
    from config.settings import get_settings

    cfg = get_settings()

    print(f"Retrieving chunks from notebook {args.notebook_id}…")
    evidence_table = []
    try:
        store = get_or_create_store(args.notebook_id, cfg.embedding_model, cfg.chunk_size)
        chunks = store.retrieve(args.question, k=20)
        for chunk in chunks:
            evidence_table.append({
                "citation_key": chunk.get("doc_id", "")[:8],
                "study_design": "Unknown",
                "key_finding": chunk.get("text", "")[:200],
            })
    except Exception as e:
        print(f"Warning: retrieval failed ({e}). Proceeding with empty context.")

    print("Mapping research gaps…")
    gap_map = map_research_gaps(evidence_table, args.question, [],
                                args.model, args.num_ctx)

    _section("Gap Map Summary")
    print(gap_map.get("gap_map_summary", ""))

    _section("Priority Gaps")
    for g in gap_map.get("priority_gaps", []):
        print(f"  • {g}")

    for category in [
        ("population_gaps", "Population Gaps"),
        ("methodology_gaps", "Methodology Gaps"),
        ("outcome_gaps", "Outcome Gaps"),
        ("context_gaps", "Context Gaps"),
        ("temporal_gaps", "Temporal Gaps"),
    ]:
        items = gap_map.get(category[0], [])
        if items:
            _section(category[1])
            for item in items:
                icon = {"High": "[H]", "Medium": "[M]", "Low": "[L]"}.\
                    get(item.get("priority", "M"), "[M]")
                print(f"  {icon} {item.get('gap', '')}")
                print(f"       {item.get('rationale', '')}")

    if args.output:
        out = Path(args.output)
        lines = [f"# Research Gap Map\n", f"## Summary\n{gap_map.get('gap_map_summary', '')}\n"]
        for g in gap_map.get("priority_gaps", []):
            lines.append(f"- {g}")
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n✅ Saved to {out.resolve()}")


# ── Command: hyp ────────────────────────────────────────────────────

def cmd_hyp(args: argparse.Namespace) -> None:
    """Generate testable hypotheses from research gaps."""
    from tools.hypothesis_generator import generate_hypotheses

    gaps = [c.strip() for c in args.gaps.split(",") if c.strip()] if args.gaps else [
        "Longitudinal studies are lacking",
        "Under-studied populations",
        "Limited methodological diversity",
    ]

    print(f"Generating {args.n} hypotheses for: {args.question[:60]}…")
    hypotheses = generate_hypotheses(
        gaps, args.question, "",
        model_name=args.model,
        num_ctx=args.num_ctx,
        n_hypotheses=args.n,
    )

    _section(f"Generated {len(hypotheses)} Hypotheses")
    for i, h in enumerate(hypotheses, 1):
        print(f"\nH{i}: {h.get('hypothesis', '')}")
        print(f"   Design      : {h.get('suggested_design', '')}")
        print(f"   Feasibility : {h.get('feasibility', '')} — {h.get('feasibility_note', '')}")
        print(f"   Rationale   : {h.get('rationale', '')}")

    if args.output:
        out = Path(args.output)
        lines = []
        for i, h in enumerate(hypotheses, 1):
            lines += [
                f"## H{i}",
                h.get("hypothesis", ""),
                f"",
                f"**Design:** {h.get('suggested_design', '')}  ",
                f"**Rationale:** {h.get('rationale', '')}",
                "",
            ]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n✅ Saved to {out.resolve()}")


# ── Entry point ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="researchbuddy",
        description="ResearchBuddy CLI — AI-powered systematic review and research notebook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py sr "Effects of sleep on memory" --include "human participants" --output review.md
  python cli.py nb --list
  python cli.py nb --new "My Project" --question "What is attention?"
  python cli.py nb --notebook-id abc123 --question "Summarise the key findings"
  python cli.py nb --notebook-id abc123          # interactive REPL
  python cli.py bib abc123 references.bib
  python cli.py gap abc123 "sleep and memory"
  python cli.py hyp abc123 "sleep and memory" --gaps "no RCTs,small samples"
""",
    )

    # ─ Common options
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("-m", "--model", default="llama3.1:8b",
                        help="Ollama model name (default: llama3.1:8b)")
    parent.add_argument("--num-ctx", type=int, default=32768,
                        help="Context window size (default: 32768)")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ─ sr
    p_sr = sub.add_parser("sr", parents=[parent],
                          help="Run a systematic review")
    p_sr.add_argument("question", help="Research question")
    p_sr.add_argument("--include", default="",
                      help="Inclusion criteria (comma-separated)")
    p_sr.add_argument("--exclude", default="",
                      help="Exclusion criteria (comma-separated)")
    p_sr.add_argument("-o", "--output", metavar="FILE",
                      help="Save Markdown report to FILE")
    p_sr.add_argument("--json-output", metavar="FILE",
                      help="Save JSON state to FILE")

    # ─ nb
    p_nb = sub.add_parser("nb", parents=[parent],
                          help="Research Notebook Q&A (or --list notebooks)")
    p_nb.add_argument("notebook_id", nargs="?", help="Notebook ID")
    p_nb.add_argument("-q", "--question", default="", help="Question to ask")
    p_nb.add_argument("--new", metavar="NAME", help="Create a new notebook with this name")
    p_nb.add_argument("--list", action="store_true", help="List all notebooks")
    p_nb.add_argument("--web", action="store_true", help="Enable automatic web search")

    # ─ bib
    p_bib = sub.add_parser("bib", parents=[parent],
                           help="Import a BibTeX file into a notebook")
    p_bib.add_argument("notebook_id", help="Notebook ID")
    p_bib.add_argument("bib_file", help="Path to .bib file")

    # ─ gap
    p_gap = sub.add_parser("gap", parents=[parent],
                           help="Map research gaps from a notebook")
    p_gap.add_argument("notebook_id", help="Notebook ID")
    p_gap.add_argument("question", help="Research question context")
    p_gap.add_argument("-o", "--output", metavar="FILE",
                       help="Save gap map to FILE")

    # ─ hyp
    p_hyp = sub.add_parser("hyp", parents=[parent],
                           help="Generate testable hypotheses")
    p_hyp.add_argument("notebook_id", help="Notebook ID (for context)")
    p_hyp.add_argument("question", help="Research question")
    p_hyp.add_argument("--gaps", default="",
                       help="Known gaps (comma-separated). Falls back to defaults if empty.")
    p_hyp.add_argument("-n", type=int, default=5,
                       help="Number of hypotheses (default: 5)")
    p_hyp.add_argument("-o", "--output", metavar="FILE",
                       help="Save hypotheses to FILE")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "sr": cmd_sr,
        "nb": cmd_nb,
        "bib": cmd_bib,
        "gap": cmd_gap,
        "hyp": cmd_hyp,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
