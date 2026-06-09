"""
agents/notebook_advanced.py
───────────────────────────
Phase-2 advanced features for Mode 8 Research Notebook.

Each public function follows the same contract:
  feature(notebook_id, settings) -> (result, error_string)

An empty error_string means success.  All functions load the notebook from
NotebookMemory (JSON) and call the local Ollama LLM — no cloud dependencies.

Features
────────
  generate_cross_document_summary   Unified synthesis of all sources
  generate_faq                      Auto-generated Q&A pairs with citations
  generate_literature_review        Academic-style structured review
  generate_mindmap                  Concept tree → Graphviz DOT string
  generate_audio_summary            Spoken-word script (for TTS or playback)
  compare_sources                   Side-by-side analysis of two sources
  extract_knowledge_graph           Entity–relationship graph → DOT string
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.notebook_memory import NotebookMemory
from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

_MAX_CHARS_PER_DOC = 6_000   # chars sent to LLM per source
_MAX_TOTAL_CHARS = 20_000    # hard ceiling for the whole context block


# ── LLM factory ──────────────────────────────────────────────────────────────

def _max_predict(settings: dict) -> int:
    """Reserve 25% of context for the prompt; use the rest for output (min 4096)."""
    return max(4096, int(settings.get("num_ctx", cfg.num_ctx) * 0.75))


def _make_llm(settings: dict, temperature: float = 0.3, num_predict: int = 4096) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=settings.get("model", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=settings.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(300.0)},
    )


def _invoke(llm: ChatOllama, system: str, human: str) -> str:
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return resp.content.strip()


# ── Source context builder ────────────────────────────────────────────────────

def _sources_context(
    notebook: Dict[str, Any],
    max_chars_per_doc: int = _MAX_CHARS_PER_DOC,
) -> str:
    """
    Build a numbered source block from all stored chunks.  Each source is
    capped at *max_chars_per_doc* and the whole block is capped at
    *_MAX_TOTAL_CHARS* so we never blow the LLM context window.
    """
    sources = notebook.get("sources", [])
    chunks = notebook.get("chunks", [])

    by_doc: Dict[str, List[str]] = {}
    for ch in chunks:
        by_doc.setdefault(ch["doc_id"], []).append(ch.get("text", ""))

    parts: List[str] = []
    total_chars = 0
    for i, src in enumerate(sources, 1):
        remaining = _MAX_TOTAL_CHARS - total_chars
        if remaining <= 0:
            break
        cap = min(max_chars_per_doc, remaining)
        combined = " ".join(by_doc.get(src["doc_id"], []))
        excerpt = combined[:cap] + ("…" if len(combined) > cap else "")
        parts.append(f"[Source {i}: {src['filename']}]\n{excerpt}")
        total_chars += len(excerpt)

    return "\n\n".join(parts)


# ── DOT helpers ───────────────────────────────────────────────────────────────

def _safe_dot(text: str, maxlen: int = 40) -> str:
    """Escape / trim a string for safe use in a Graphviz DOT label."""
    cleaned = re.sub(r'["\\\n\r\t]', " ", str(text)).strip()
    return cleaned[:maxlen]


def _mindmap_to_dot(data: Dict[str, Any]) -> str:
    """Convert the LLM mind-map JSON to a Graphviz DOT string."""
    central = _safe_dot(data.get("central", "Main Topic"), 60)
    branches = data.get("branches", [])

    lines = [
        "digraph mindmap {",
        '  graph [rankdir=LR, bgcolor="#0f172a", pad=0.4];',
        '  node [style=filled, fontname="Helvetica", fontsize=11, fontcolor=white];',
        f'  "root" [label="{central}", shape=ellipse, '
        f'fillcolor="#3b82f6", fontsize=14];',
    ]
    for i, branch in enumerate(branches):
        concept = _safe_dot(branch.get("concept", f"Branch {i}"))
        bid = f"b{i}"
        lines.append(
            f'  "{bid}" [label="{concept}", shape=box, fillcolor="#1e40af"];'
        )
        lines.append(f'  "root" -> "{bid}" [color="#60a5fa"];')
        for j, sub in enumerate(branch.get("sub_concepts", [])[:4]):
            sid = f"b{i}s{j}"
            lines.append(
                f'  "{sid}" [label="{_safe_dot(sub)}", shape=box, '
                f'fillcolor="#374151", fontsize=10];'
            )
            lines.append(f'  "{bid}" -> "{sid}" [color="#6b7280"];')
    lines.append("}")
    return "\n".join(lines)


def _knowledge_graph_to_dot(data: Dict[str, Any]) -> str:
    """Convert knowledge-graph JSON to a Graphviz DOT string."""
    type_colors: Dict[str, str] = {
        "concept": "#3b82f6",
        "method": "#10b981",
        "dataset": "#f59e0b",
        "author": "#8b5cf6",
        "institution": "#ef4444",
    }
    default_color = "#6b7280"

    lines = [
        "digraph knowledge_graph {",
        '  graph [bgcolor="#0f172a", rankdir=TB, pad=0.4];',
        '  node [style=filled, fontname="Helvetica", fontsize=10, fontcolor=white];',
        '  edge [fontname="Helvetica", fontsize=9, '
        'fontcolor="#9ca3af", color="#4b5563"];',
    ]
    for node in data.get("nodes", [])[:20]:
        nid = _safe_dot(node.get("id", ""))
        label = _safe_dot(node.get("label", nid))
        color = type_colors.get(node.get("type", ""), default_color)
        lines.append(f'  "{nid}" [label="{label}", fillcolor="{color}"];')
    for edge in data.get("edges", [])[:25]:
        src = _safe_dot(edge.get("from", ""))
        dst = _safe_dot(edge.get("to", ""))
        lbl = _safe_dot(edge.get("label", ""))
        lines.append(f'  "{src}" -> "{dst}" [label="{lbl}"];')
    lines.append("}")
    return "\n".join(lines)


def _parse_json_from_llm(raw: str) -> Any:
    """Extract a JSON array from an LLM response (array-first)."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return json.loads(raw)


def _parse_json_object_from_llm(raw: str) -> Any:
    """Extract a JSON object from an LLM response (object-first, for mind map / KG)."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return json.loads(raw)


# ── Feature 1: Cross-document summary ────────────────────────────────────────

def generate_cross_document_summary(
    notebook_id: str, settings: dict
) -> Tuple[str, str]:
    """
    Synthesize all notebook sources into a unified markdown summary.

    For a single source: comprehensive summary with key points, methodology,
    and implications.
    For multiple sources: synthesis with common themes, complementary
    contributions, contradictions, and key takeaways.

    Returns (markdown_string, error_string).
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    sources = notebook.get("sources", [])
    if not sources:
        return "", "This notebook has no sources. Add documents first."

    context = _sources_context(notebook)

    if len(sources) == 1:
        system = (
            "You are a research analyst. Summarize the key points, methodology, "
            "findings, and implications of the provided source. "
            "Use clear markdown headings (##). Be thorough but concise."
        )
        human = f"SOURCE:\n{context}\n\nWrite a comprehensive summary in markdown."
    else:
        src_list = ", ".join(s["filename"] for s in sources)
        system = (
            "You are a research analyst synthesizing multiple documents.\n"
            "Write a cross-document synthesis in markdown with these sections:\n"
            "## Overview\nWhat the sources collectively cover.\n"
            "## Common Themes\nShared ideas, findings, or methods.\n"
            "## Complementary Contributions\nHow the sources add to each other.\n"
            "## Contradictions & Gaps\nWhere sources disagree or leave open questions.\n"
            "## Key Takeaways\n3–5 bullet conclusions.\n\n"
            "Attribute claims to specific sources by filename."
        )
        human = (
            f"SOURCES: {src_list}\n\n{context}\n\n"
            "Write the cross-document synthesis in markdown."
        )

    try:
        result = _invoke(_make_llm(settings, temperature=0.3, num_predict=_max_predict(settings)), system, human)
        return result, ""
    except Exception as e:
        logger.error("Cross-document summary failed: %s", e)
        return "", f"Summary generation failed: {e}"


# ── Feature 2: FAQ generation ─────────────────────────────────────────────────

def generate_faq(
    notebook_id: str,
    settings: dict,
    n_questions: int = 8,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Auto-generate FAQ items grounded in notebook sources.

    Returns (list_of_faq_dicts, error_string).
    Each dict: {"question": str, "answer": str, "sources": List[int]}.
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return [], f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return [], "No sources in this notebook."

    context = _sources_context(notebook)

    system = (
        f"You are a research expert. Based on the provided sources, generate "
        f"{n_questions} frequently asked questions with grounded answers.\n\n"
        "Output ONLY valid JSON — a JSON array with no surrounding text:\n"
        "[\n"
        '  {"question": "...", "answer": "...", "sources": [1, 2]},\n'
        "  ...\n"
        "]\n\n"
        "Rules:\n"
        "- Questions must cover the most important concepts in the sources.\n"
        "- Answers must be grounded ONLY in the provided sources.\n"
        "- 'sources' is an array of 1-based source numbers that support the answer.\n"
        "- Output ONLY the JSON array. No preamble, no code fences."
    )
    human = f"SOURCES:\n{context}\n\nGenerate {n_questions} FAQ items as a JSON array."

    raw = ""
    try:
        raw = _invoke(_make_llm(settings, temperature=0.3, num_predict=4096), system, human)
        items = _parse_json_from_llm(raw)
        if not isinstance(items, list):
            return [], "FAQ response was not a JSON array."
        return [i for i in items if isinstance(i, dict)], ""
    except json.JSONDecodeError as e:
        logger.error("FAQ JSON parse failed: %s | raw: %.200s", e, raw)
        return [], f"FAQ parsing failed: {e}"
    except Exception as e:
        logger.error("FAQ generation failed: %s", e)
        return [], f"FAQ generation failed: {e}"


# ── Feature 3: Literature review ──────────────────────────────────────────────

def generate_literature_review(
    notebook_id: str, settings: dict
) -> Tuple[str, str]:
    """
    Generate a formal academic-style literature review from notebook sources.

    Returns (review_markdown, error_string).
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return "", "No sources in this notebook."

    source_names = ", ".join(s["filename"] for s in notebook["sources"])
    context = _sources_context(notebook)

    system = (
        "You are an academic researcher writing a formal literature review.\n"
        "Structure the review in markdown with these sections:\n"
        "# Literature Review\n"
        "## 1. Introduction\n"
        "State the scope and purpose of this review.\n"
        "## 2. Background & Context\n"
        "Key background concepts established across the sources.\n"
        "## 3. Methodological Approaches\n"
        "Research methods and approaches described in the sources.\n"
        "## 4. Key Findings & Evidence\n"
        "Major findings organized thematically, attributed to source filenames.\n"
        "## 5. Critical Analysis\n"
        "Strengths, limitations, and gaps in the reviewed literature.\n"
        "## 6. Conclusion\n"
        "Synthesis of contributions and directions for future work.\n\n"
        "Use formal academic tone. Attribute claims to specific source filenames."
    )
    human = (
        f"SOURCES: {source_names}\n\n{context}\n\n"
        "Write the formal literature review in markdown."
    )

    try:
        result = _invoke(_make_llm(settings, temperature=0.2, num_predict=_max_predict(settings)), system, human)
        return result, ""
    except Exception as e:
        logger.error("Literature review generation failed: %s", e)
        return "", f"Literature review generation failed: {e}"


# ── Feature 4: Mind map ───────────────────────────────────────────────────────

def generate_mindmap(notebook_id: str, settings: dict) -> Tuple[str, str]:
    """
    Extract key concepts from notebook sources and return a Graphviz DOT string
    suitable for ``st.graphviz_chart``.

    Returns (dot_string, error_string).
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return "", "No sources in this notebook."

    # Use a smaller context for concept extraction — we need breadth, not depth.
    context = _sources_context(notebook, max_chars_per_doc=1_500)

    system = (
        "You are a knowledge analyst. Extract key concepts from the sources.\n"
        "Output ONLY valid JSON — no code fences, no other text:\n"
        '{"central": "Main Topic", "branches": [\n'
        '  {"concept": "Branch", "sub_concepts": ["Sub1", "Sub2"]}\n'
        "]}\n\n"
        "Rules:\n"
        "- Maximum 6 branches; maximum 4 sub-concepts per branch.\n"
        "- Labels: 2–5 words, no special characters.\n"
        "- Output ONLY the JSON object."
    )
    human = f"SOURCES:\n{context}\n\nExtract the mind map JSON."

    raw = ""
    try:
        raw = _invoke(_make_llm(settings, temperature=0.2, num_predict=1024), system, human)
        data = _parse_json_object_from_llm(raw)
        dot = _mindmap_to_dot(data)
        return dot, ""
    except json.JSONDecodeError as e:
        logger.error("Mind map JSON parse failed: %s | raw: %.200s", e, raw)
        return "", f"Mind map parsing failed: {e}"
    except Exception as e:
        logger.error("Mind map generation failed: %s", e)
        return "", f"Mind map generation failed: {e}"


# ── Feature 5: Audio summary ──────────────────────────────────────────────────

def generate_audio_summary(notebook_id: str, settings: dict) -> Tuple[str, str]:
    """
    Generate a spoken-word summary script — natural language optimised for
    text-to-speech playback (~300 words, ~2 minutes).

    Returns (script_text, error_string).  The text contains no markdown
    formatting so it reads cleanly when converted to audio.
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return "", "No sources in this notebook."

    context = _sources_context(notebook, max_chars_per_doc=2_000)
    nb_name = notebook.get("name", "Notebook")

    system = (
        "You are creating a spoken audio summary script. "
        "Write in clear, natural spoken language that sounds good when read aloud.\n\n"
        "Rules:\n"
        "- No markdown formatting — no #, *, _, backticks, or bullet dashes.\n"
        "- Short, complete sentences.\n"
        "- Natural spoken transitions: First, Additionally, Furthermore, Finally.\n"
        "- Approximately 280 to 320 words — about 2 minutes when read at a natural pace.\n"
        "- Start with an introduction (what this notebook covers).\n"
        "- End with a clear conclusion that ties everything together.\n"
        "- Do not list source filenames — integrate the content naturally."
    )
    human = (
        f'Create an audio summary script for a notebook called "{nb_name}".\n\n'
        f"SOURCES:\n{context}\n\n"
        "Write the spoken-word audio script now:"
    )

    try:
        result = _invoke(_make_llm(settings, temperature=0.5, num_predict=2048), system, human)
        return result, ""
    except Exception as e:
        logger.error("Audio summary generation failed: %s", e)
        return "", f"Audio summary generation failed: {e}"


# ── Feature 6: Source comparison ─────────────────────────────────────────────

def compare_sources(
    notebook_id: str,
    doc_id_a: str,
    doc_id_b: str,
    settings: dict,
) -> Tuple[str, str]:
    """
    Generate a side-by-side markdown comparison of two notebook sources.

    Returns (comparison_markdown, error_string).
    """
    if doc_id_a == doc_id_b:
        return "", "Please select two different sources to compare."

    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    src_map = {s["doc_id"]: s for s in notebook.get("sources", [])}
    src_a = src_map.get(doc_id_a)
    src_b = src_map.get(doc_id_b)
    if not src_a or not src_b:
        return "", "One or both selected sources were not found in this notebook."

    by_doc: Dict[str, List[str]] = {}
    for ch in notebook.get("chunks", []):
        by_doc.setdefault(ch["doc_id"], []).append(ch.get("text", ""))

    text_a = (" ".join(by_doc.get(doc_id_a, [])))[:_MAX_CHARS_PER_DOC]
    text_b = (" ".join(by_doc.get(doc_id_b, [])))[:_MAX_CHARS_PER_DOC]

    system = (
        "You are a research analyst comparing two documents.\n"
        "Write a detailed comparison in markdown with these sections:\n\n"
        "## Source Comparison\n\n"
        "### Overview\nOne paragraph per source describing its focus and main argument.\n\n"
        "### Common Ground\nShared themes, findings, or methods.\n\n"
        "### Unique Contributions\n"
        "Use a markdown table with columns: Aspect | Source A | Source B\n\n"
        "### Contradictions\nWhere the sources disagree or present conflicting evidence.\n\n"
        "### Synthesis\n"
        "How the two sources complement each other and what combined insight they provide."
    )
    human = (
        f"SOURCE A: {src_a['filename']}\n{text_a}\n\n"
        f"SOURCE B: {src_b['filename']}\n{text_b}\n\n"
        "Write the detailed comparison in markdown."
    )

    try:
        result = _invoke(_make_llm(settings, temperature=0.3, num_predict=_max_predict(settings)), system, human)
        return result, ""
    except Exception as e:
        logger.error("Source comparison failed: %s", e)
        return "", f"Source comparison failed: {e}"


# ── Feature 7: Knowledge graph ────────────────────────────────────────────────

def extract_knowledge_graph(notebook_id: str, settings: dict) -> Tuple[str, str]:
    """
    Extract entities and relationships from notebook sources and return a
    Graphviz DOT string suitable for ``st.graphviz_chart``.

    Returns (dot_string, error_string).
    Node types: concept | method | dataset | author | institution
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return "", "No sources in this notebook."

    context = _sources_context(notebook, max_chars_per_doc=1_500)

    system = (
        "You are a knowledge graph extractor.\n"
        "Output ONLY valid JSON — no code fences, no other text:\n"
        '{\n  "nodes": [\n'
        '    {"id": "n1", "label": "Entity", "type": "concept"}\n'
        "  ],\n"
        '  "edges": [\n'
        '    {"from": "n1", "to": "n2", "label": "relationship"}\n'
        "  ]\n}\n\n"
        "Rules:\n"
        "- Maximum 20 nodes, maximum 25 edges.\n"
        "- Node types: concept, method, dataset, author, institution.\n"
        "- Label: 2–5 words. No special characters.\n"
        "- Edge labels: short verb phrases (uses, builds on, contradicts, etc.).\n"
        "- Output ONLY the JSON object."
    )
    human = f"SOURCES:\n{context}\n\nExtract the knowledge graph JSON."

    raw = ""
    try:
        raw = _invoke(_make_llm(settings, temperature=0.2, num_predict=2048), system, human)
        data = _parse_json_object_from_llm(raw)
        dot = _knowledge_graph_to_dot(data)
        return dot, ""
    except json.JSONDecodeError as e:
        logger.error("Knowledge graph JSON parse failed: %s | raw: %.200s", e, raw)
        return "", f"Knowledge graph parsing failed: {e}"
    except Exception as e:
        logger.error("Knowledge graph extraction failed: %s", e)
        return "", f"Knowledge graph extraction failed: {e}"


# ── Utility: DOT → raster/vector render ─────────────────────────────────────

def render_dot_bytes(dot_string: str, fmt: str = "png") -> Tuple[bytes, str]:
    """
    Render a Graphviz DOT string to the requested format.

    Parameters
    ----------
    dot_string : valid Graphviz DOT source
    fmt        : "png" | "svg" | "pdf"

    Returns (image_bytes, error_string).  Requires the *graphviz* Python package
    and the graphviz system tools (``apt install graphviz``).
    """
    try:
        import graphviz as gv
        src = gv.Source(dot_string)
        return src.pipe(format=fmt), ""
    except ImportError:
        return b"", "graphviz Python package not installed (pip install graphviz)."
    except Exception as e:
        logger.error("DOT render to %s failed: %s", fmt, e)
        return b"", f"Rendering to {fmt} failed: {e}"


# ── Utility: text → WAV audio ────────────────────────────────────────────────

def synthesize_speech(text: str, rate: int = 150) -> Tuple[bytes, str]:
    """
    Convert *text* to a WAV audio file using pyttsx3 (offline TTS).

    Returns (wav_bytes, error_string).  Requires the *pyttsx3* Python package
    and the espeak-ng system package (``apt install espeak-ng``).
    """
    try:
        import os
        import tempfile

        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "tts.wav")
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                return b"", "TTS engine produced no output. Is espeak-ng installed?"
            return open(tmp_path, "rb").read(), ""
    except ImportError:
        return b"", "pyttsx3 not installed (pip install pyttsx3)."
    except Exception as e:
        logger.error("TTS synthesis failed: %s", e)
        return b"", f"Speech synthesis failed: {e}"


# ── Feature 8: Timeline extraction ───────────────────────────────────────────

def extract_timeline(notebook_id: str, settings: dict) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract a chronological timeline of events, discoveries, and milestones
    from notebook sources.

    Returns (list_of_timeline_items, error_string).
    Each item: {"year": str, "event": str, "significance": str, "source": int}
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return [], f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return [], "No sources in this notebook."

    context = _sources_context(notebook, max_chars_per_doc=2_000)

    system = (
        "You are a research historian. Extract a chronological timeline from the sources.\n"
        "Output ONLY a JSON array — no code fences, no other text:\n"
        "[\n"
        '  {"year": "2017", "event": "Brief description", '
        '"significance": "Why it matters", "source": 1},\n'
        "  ...\n"
        "]\n\n"
        "Rules:\n"
        "- Include 5–15 most important events, discoveries, or milestones.\n"
        "- 'source' is the 1-based source number where the event is mentioned.\n"
        "- Year can be approximate: '~1990', 'early 2000s', '2017–2020'.\n"
        "- Output ONLY the JSON array."
    )
    human = f"SOURCES:\n{context}\n\nExtract the chronological timeline as a JSON array."

    raw = ""
    try:
        raw = _invoke(_make_llm(settings, temperature=0.2, num_predict=2048), system, human)
        items = _parse_json_from_llm(raw)
        if not isinstance(items, list):
            return [], "Timeline response was not a JSON array."
        return [i for i in items if isinstance(i, dict)], ""
    except json.JSONDecodeError as e:
        logger.error("Timeline JSON parse failed: %s | raw: %.200s", e, raw)
        return [], f"Timeline parsing failed: {e}"
    except Exception as e:
        logger.error("Timeline extraction failed: %s", e)
        return [], f"Timeline extraction failed: {e}"


# ── Feature 9: Study comparison table ────────────────────────────────────────

def generate_study_comparison(notebook_id: str, settings: dict) -> Tuple[str, str]:
    """
    Generate a structured comparison table across all notebook sources —
    comparing research type, sample/data scope, methodology, key findings,
    and limitations.

    Returns (markdown_table_with_synthesis, error_string).
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return "", "No sources in this notebook."

    source_names = [s["filename"] for s in notebook["sources"]]
    col_headers = " | ".join(f"**{n[:20]}**" for n in source_names)
    context = _sources_context(notebook)

    system = (
        "You are a systematic review analyst. Create a comparison table across all sources.\n\n"
        "Generate a markdown table with these rows and one column per source:\n"
        "| Dimension | Source 1 | Source 2 | ... |\n"
        "|-----------|----------|----------|\n"
        "Include rows for:\n"
        "- Research / Study type\n"
        "- Sample size / Data scope\n"
        "- Key methodology\n"
        "- Primary findings\n"
        "- Limitations\n"
        "- Year / Period\n\n"
        "After the table, write a **Synthesis** paragraph:\n"
        "- Strongest points of agreement\n"
        "- Most significant differences\n"
        "- What the sources collectively establish\n\n"
        "Use the exact source filenames as column headers."
    )
    human = (
        f"SOURCES: {', '.join(source_names)}\n\n{context}\n\n"
        "Generate the structured comparison table followed by the Synthesis paragraph."
    )

    try:
        result = _invoke(_make_llm(settings, temperature=0.2, num_predict=_max_predict(settings)), system, human)
        return result, ""
    except Exception as e:
        logger.error("Study comparison table generation failed: %s", e)
        return "", f"Study comparison table generation failed: {e}"
