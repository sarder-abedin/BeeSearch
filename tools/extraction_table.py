"""
tools/extraction_table.py
───────────────────────────
Structured PICO (Population, Intervention, Comparator, Outcome) extraction
table for the Systematic Review pipeline, with CSV and Markdown export.

For each included paper, an LLM call fills in the PICO fields plus a
key finding and limitations, on top of fields already present on the
paper dict (study_design, quality, citation_key, ...). Used to produce
the structured evidence table shown/exported alongside the narrative
synthesis.
"""
from __future__ import annotations
import csv, io, json, logging, re
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from config.settings import get_settings
logger = logging.getLogger(__name__)
cfg = get_settings()

def _llm(model_name: str, num_ctx: int) -> ChatOllama:
    """Build a low-temperature ChatOllama client for structured PICO extraction."""
    import httpx
    from config.observability import get_langfuse_callbacks
    return ChatOllama(model=model_name or cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0.1, num_predict=512, num_ctx=num_ctx or cfg.num_ctx, sync_client_kwargs={"timeout": httpx.Timeout(300.0)}, callbacks=get_langfuse_callbacks())

def extract_structured_row(paper: Dict[str, Any], research_question: str, model_name: str, num_ctx: int) -> Dict[str, str]:
    """Ask the LLM to fill in PICO fields for one paper and assemble a full table row.

    Combines the LLM's JSON output (population/intervention/comparator/
    outcome/key_finding/limitations) with fields already on `paper`
    (study_design, quality, citation_key, doi, url). If the LLM reply can't
    be parsed as JSON, the PICO fields fall back to "Not specified"/"N/A"
    placeholders rather than raising.
    """
    llm = _llm(model_name, num_ctx)
    authors = paper.get("authors", [])
    author_str = authors[0].split(",")[0] if authors else "Unknown"
    year = paper.get("year", "n.d.")
    author_year = f"{author_str} et al. ({year})" if len(authors) > 1 else f"{author_str} ({year})"
    raw = llm.invoke([SystemMessage(content=f"Extract structured PICO data for: {research_question}\nReturn ONLY valid JSON: {{\"population\": \"\", \"intervention\": \"\", \"comparator\": \"\", \"outcome\": \"\", \"key_finding\": \"\", \"limitations\": \"\"}}"), HumanMessage(content=f"Title: {paper.get('title','')}\nAbstract: {paper.get('abstract','')[:600]}\nDesign: {paper.get('study_design','')}\nSample: {paper.get('sample_size','')}")]).content.strip()
    try:
        # Grab the outermost {...} block in case the model added stray prose.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        extracted = json.loads(match.group(0)) if match else {}
    except Exception:
        extracted = {}
    return {"author_year": author_year, "study_design": paper.get("study_design", "Unknown"), "population": extracted.get("population", "Not specified"), "intervention": extracted.get("intervention", "Not specified"), "comparator": extracted.get("comparator", "N/A"), "outcome": extracted.get("outcome", "Not specified"), "key_finding": extracted.get("key_finding", paper.get("key_finding", "")), "limitations": extracted.get("limitations", "Not reported"), "quality": paper.get("quality", "Medium"), "citation_key": paper.get("citation_key", ""), "doi": paper.get("doi", ""), "url": paper.get("url", "")}

def build_extraction_table(evidence_table: List[Dict[str, Any]], research_question: str, model_name: str, num_ctx: int) -> List[Dict[str, str]]:
    """Run `extract_structured_row` over every paper in `evidence_table`.

    Skips (and logs) any paper whose extraction raises, so one failure
    doesn't drop the rest of the table.
    """
    rows = []
    for paper in evidence_table:
        try:
            rows.append(extract_structured_row(paper, research_question, model_name, num_ctx))
        except Exception as e:
            logger.warning("Extraction failed for '%s': %s", paper.get("title", "")[:40], e)
    return rows

def extraction_table_to_csv(rows: List[Dict[str, str]]) -> str:
    """Render extraction rows as a CSV string with human-readable column headers."""
    if not rows:
        return ""
    fieldnames = ["author_year", "study_design", "population", "intervention", "comparator", "outcome", "key_finding", "limitations", "quality", "citation_key", "doi"]
    human_headers = ["Author/Year", "Study Design", "Population", "Intervention", "Comparator", "Outcome", "Key Finding", "Limitations", "Quality", "Citation Key", "DOI"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    output.write(",".join(human_headers) + "\r\n")
    writer.writerows(rows)
    return output.getvalue()

def extraction_table_to_markdown(rows: List[Dict[str, str]]) -> str:
    """Render a subset of extraction columns as a Markdown table for display.

    Cell text is truncated to 60 characters and pipe characters are escaped
    so long LLM-generated fields don't break the Markdown table layout.
    """
    if not rows:
        return "*No data*"
    headers = ["Author/Year", "Design", "Population", "Intervention", "Outcome", "Key Finding", "Quality"]
    keys = ["author_year", "study_design", "population", "intervention", "outcome", "key_finding", "quality"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = [str(row.get(k, "")).replace("|", "\\|")[:60] for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
