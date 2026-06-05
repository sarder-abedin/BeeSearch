"""
tools/document_tools.py
────────────────────────
Handles ingestion and pre-processing of user-uploaded documents.

Supported formats (standard parser)
─────────────────────────────────────
  • PDF  — pdfplumber (tables + text, layout-aware)
  • DOCX — python-docx
  • TXT / MD — plain text

Use get_processor(use_docling=True) to get a DoclingProcessor that
additionally handles PPTX, XLSX, HTML, images, and provides OCR.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, List, Optional, Union

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single piece of a document, ready for embedding."""
    chunk_id: str
    doc_id: str
    doc_name: str
    page_num: int
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """All chunks extracted from one uploaded file."""
    doc_id: str
    filename: str
    file_type: str
    total_pages: int
    total_chunks: int
    chunks: List[DocumentChunk]
    raw_text: str
    content_md5: str = ""


# ── Utilities ───────────────────────────────────────────────────────────────────

def _stable_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[str]:
    """
    Sliding-window character chunker with sentence-boundary awareness.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            search_end = min(end + 200, len(text))
            boundary = -1
            for sep in (". ", "! ", "? ", "\n\n"):
                idx = text.rfind(sep, start, search_end)
                if idx > boundary:
                    boundary = idx
            if boundary > start:
                end = boundary + 2

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ── Main Processor ──────────────────────────────────────────────────────────────────

class DocumentProcessor:
    """
    Stateless helper that converts uploaded files into DocumentChunk lists.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 150,
        max_raw_chars: int = 0,
        max_pages: int = 300,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_raw_chars = max_raw_chars
        self.max_pages = max_pages

    def process_file(
        self,
        file_path: Union[str, Path],
        file_obj: Optional[IO[bytes]] = None,
    ) -> ProcessedDocument:
        path = Path(file_path)
        ext = path.suffix.lower()
        doc_id = _stable_id(path.name)

        logger.info("Processing %s (type=%s)", path.name, ext)

        if ext == ".pdf":
            pages_text, total_pages = self._extract_pdf(path, file_obj)
        elif ext in (".docx", ".doc"):
            pages_text, total_pages = self._extract_docx(path, file_obj)
        elif ext in (".txt", ".md", ".rst"):
            pages_text, total_pages = self._extract_text(path, file_obj)
        else:
            raise ValueError(
                f"Unsupported file type '{ext}'. Accepted: PDF, DOCX, TXT, MD."
            )

        raw_text = "\n\n".join(pages_text)
        content_md5 = hashlib.md5(raw_text[:50000].encode()).hexdigest()
        chunks = self._build_chunks(pages_text, doc_id, path.name)

        return ProcessedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type=ext.lstrip(".").upper(),
            total_pages=total_pages,
            total_chunks=len(chunks),
            chunks=chunks,
            raw_text=raw_text,
            content_md5=content_md5,
        )

    def process_raw_text(self, text: str, name: str = "pasted_text") -> ProcessedDocument:
        doc_id = _stable_id(text[:200])
        content_md5 = hashlib.md5(text[:50000].encode()).hexdigest()
        pages_text = [text]
        chunks = self._build_chunks(pages_text, doc_id, name)
        return ProcessedDocument(
            doc_id=doc_id,
            filename=name,
            file_type="TXT",
            total_pages=1,
            total_chunks=len(chunks),
            chunks=chunks,
            raw_text=text,
            content_md5=content_md5,
        )

    def _extract_pdf(self, path, file_obj):
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")

        source = file_obj if file_obj is not None else str(path)
        pages_text: List[str] = []
        total_chars = 0
        try:
            with pdfplumber.open(source) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    if self.max_pages and page_num >= self.max_pages:
                        break
                    try:
                        raw = page.extract_text() or ""
                        table_txt = ""
                        if len(raw) < 10_000:
                            tables = page.extract_tables()
                            for tbl in tables:
                                for row in tbl:
                                    if row:
                                        table_txt += " | ".join(str(c or "") for c in row) + "\n"
                        combined = _clean_text(raw + "\n" + table_txt)
                    except Exception as page_err:
                        logger.warning("Skipping page %d: %s", page_num + 1, page_err)
                        combined = ""
                    finally:
                        del page
                        if page_num % 10 == 0:
                            gc.collect()

                    if combined:
                        pages_text.append(combined)
                        total_chars += len(combined)
                        if self.max_raw_chars and total_chars >= self.max_raw_chars:
                            break
        except MemoryError:
            logger.error("OOM while processing PDF '%s'", path.name)
            if not pages_text:
                raise RuntimeError(f"Not enough memory to process '{path.name}'.")

        return pages_text, len(pages_text)

    def _extract_docx(self, path, file_obj):
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")

        source = file_obj if file_obj is not None else str(path)
        doc = DocxDocument(source)
        paragraphs: List[str] = []
        current_page: List[str] = []
        total_chars = 0
        for para in doc.paragraphs:
            text = _clean_text(para.text)
            if text:
                current_page.append(text)
                total_chars += len(text)
            if len(current_page) >= 50:
                paragraphs.append("\n".join(current_page))
                current_page = []
            if self.max_raw_chars and total_chars >= self.max_raw_chars:
                break
        if current_page:
            paragraphs.append("\n".join(current_page))
        return paragraphs or [""], max(1, len(paragraphs))

    def _extract_text(self, path, file_obj):
        if file_obj is not None:
            raw = file_obj.read().decode("utf-8", errors="replace")
        else:
            raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = _clean_text(raw)
        if self.max_raw_chars and len(cleaned) > self.max_raw_chars:
            cleaned = cleaned[: self.max_raw_chars]
        sections = [s.strip() for s in re.split(r"\n\n+", cleaned) if s.strip()]
        return sections or [cleaned], len(sections) or 1

    def _build_chunks(self, pages_text, doc_id, doc_name):
        chunks: List[DocumentChunk] = []
        chunk_index = 0
        for page_num, page_text in enumerate(pages_text):
            raw_chunks = _chunk_text(page_text, self.chunk_size, self.overlap)
            for raw in raw_chunks:
                cid = _stable_id(f"{doc_id}:{chunk_index}:{raw[:50]}")
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page_num=page_num,
                        chunk_index=chunk_index,
                        text=raw,
                        metadata={"source": doc_name, "page": page_num + 1, "chunk_index": chunk_index},
                    )
                )
                chunk_index += 1
        return chunks


def get_processor(
    use_docling: bool = True,
    use_ocr: bool = False,
    chunk_size: int = 800,
    overlap: int = 150,
    max_raw_chars: int = 0,
    max_pages: int = 300,
):
    if use_docling:
        from tools.docling_processor import DoclingProcessor
        return DoclingProcessor(use_ocr=use_ocr, max_raw_chars=max_raw_chars)
    return DocumentProcessor(
        chunk_size=chunk_size, overlap=overlap,
        max_raw_chars=max_raw_chars, max_pages=max_pages,
    )
