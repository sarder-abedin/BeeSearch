"""mcp_servers/research_tools_server.py
MCP server exposing the research assistant's tools to external clients
(Claude Code, Claude Desktop, etc.) via the Model Context Protocol.

Run:  python mcp_servers/research_tools_server.py
      mcp dev mcp_servers/research_tools_server.py  (inspector UI)
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Research Assistant Tools")


# ── Academic search ────────────────────────────────────────────────────────

@mcp.tool()
def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv for academic preprints matching the query."""
    from tools.search_tools import ArxivSearcher
    results = ArxivSearcher().search(query, max_results=max_results)
    return [r.__dict__ if hasattr(r, "__dict__") else dict(r) for r in results]


@mcp.tool()
def search_semantic_scholar(query: str, max_results: int = 10) -> list[dict]:
    """Search Semantic Scholar for peer-reviewed papers with citation data."""
    from tools.search_tools import SemanticScholarSearcher
    results = SemanticScholarSearcher().search(query, max_results=max_results)
    return [r.__dict__ if hasattr(r, "__dict__") else dict(r) for r in results]


@mcp.tool()
def search_crossref(query: str, max_results: int = 10) -> list[dict]:
    """Search CrossRef for DOI-registered publications."""
    from tools.search_tools import CrossRefResolver
    results = CrossRefResolver().search(query, max_results=max_results)
    return [r.__dict__ if hasattr(r, "__dict__") else dict(r) for r in results]


# ── Web search ─────────────────────────────────────────────────────────────

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo. Returns title, url, and snippet."""
    from ddgs import DDGS
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=max_results))
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", ""), "source": "duckduckgo"}
        for r in raw
    ]


# ── Notebook RAG query ─────────────────────────────────────────────────────

@mcp.tool()
def query_notebook(notebook_id: str, question: str, top_k: int = 8) -> dict:
    """
    Query a research notebook by ID using hybrid RAG (BM25 + dense retrieval).
    Returns the top matching chunks with their source document and page number.
    """
    from agents.notebook_memory import NotebookMemory
    from tools.hybrid_store import get_or_create_store
    from config.settings import get_settings
    cfg = get_settings()
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if notebook is None:
        return {"error": f"Notebook '{notebook_id}' not found", "chunks": []}
    stored_chunks = notebook.get("chunks", [])
    if not stored_chunks:
        return {"error": "Notebook has no indexed sources", "chunks": []}
    try:
        store = get_or_create_store(
            session_id=f"notebook_{notebook_id}",
            embed_model=cfg.embedding_model,
            ollama_base_url=cfg.ollama_base_url,
            persist_dir=cfg.chroma_persist_dir,
        )
        results = store.search_hybrid(question, top_k) if store.is_indexed() else stored_chunks[:top_k]
    except Exception:
        results = stored_chunks[:top_k]
    return {
        "notebook_id": notebook_id,
        "question": question,
        "chunks": [
            {
                "text": c.get("text", ""),
                "doc_name": c.get("doc_name", ""),
                "page_num": c.get("page_num"),
                "chunk_id": c.get("chunk_id", ""),
            }
            for c in results
        ],
    }


# ── Document ingestion ─────────────────────────────────────────────────────

@mcp.tool()
def ingest_document(
    content_base64: str,
    filename: str,
    notebook_id: str = "",
    use_docling: bool = False,
    use_ocr: bool = False,
) -> dict:
    """
    Ingest a document (PDF, DOCX, TXT, MD, etc.) into a research notebook.
    Pass the file content as a base64-encoded string.
    If notebook_id is empty, returns chunks without persisting.
    """
    import base64
    from io import BytesIO
    from pathlib import Path as P
    from tools.document_tools import get_processor

    try:
        raw = base64.b64decode(content_base64)
    except Exception as e:
        return {"error": f"Invalid base64 content: {e}", "chunks": []}

    processor = get_processor(use_docling=use_docling, use_ocr=use_ocr)
    try:
        doc = processor.process_file(P(filename), file_obj=BytesIO(raw))
    except Exception as e:
        return {"error": f"Processing failed: {e}", "chunks": []}

    if notebook_id:
        from agents.notebook_memory import NotebookMemory
        mem = NotebookMemory()
        mem.add_source(notebook_id, doc, source_type="file")

    return {
        "filename": doc.filename,
        "total_chunks": doc.total_chunks,
        "total_pages": doc.total_pages,
        "notebook_id": notebook_id or None,
        "chunks": [
            {"chunk_id": c.chunk_id, "page_num": c.page_num, "text": c.text[:200]}
            for c in doc.chunks[:5]
        ],
    }


if __name__ == "__main__":
    mcp.run()
