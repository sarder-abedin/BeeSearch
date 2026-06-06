"""
tools/docling_processor.py
──────────────────────────
Docling-based document processor providing advanced PDF layout understanding,
table extraction (as Markdown), OCR for scanned documents, and support for
PPTX, XLSX, HTML, and common image formats.

Produces the same ProcessedDocument / DocumentChunk schema as DocumentProcessor
so it slots in to HybridStore and all agent pipelines without any changes there.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union

from tools.document_tools import (
    DocumentChunk,
    ProcessedDocument,
    _clean_text,
    _stable_id,
)

logger = logging.getLogger(__name__)

# ── Format support ─────────────────────────────────────────────────────────────

_DOCLING_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm",
    ".md", ".txt", ".rst",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp",
})

_CSV_EXTENSIONS = frozenset({".csv"})

SUPPORTED_EXTENSIONS: frozenset = _DOCLING_EXTENSIONS | _CSV_EXTENSIONS


# ── Converter singleton ────────────────────────────────────────────────────────

_converter_cache: Dict[tuple, Any] = {}


def _set_cache_env(models_path: Path) -> None:
    """Point HuggingFace + Docling model downloads to the project models dir."""
    models_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(models_path / "hf"))
    os.environ.setdefault("DOCLING_ARTIFACTS_PATH", str(models_path))
    os.environ.setdefault("TORCH_HOME", str(models_path / "torch"))


def _build_converter(use_ocr: bool, models_path: Path):
    """Build a Docling DocumentConverter (expensive — cached per (use_ocr, path))."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError(
            "Docling is not installed. Run: pip install docling"
        ) from exc

    _set_cache_env(models_path)

    # Try to configure PDF pipeline options with OCR
    try:
        from docling.document_converter import PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        pdf_opts = PdfPipelineOptions(
            do_ocr=use_ocr,
            do_table_structure=True,
        )
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        )
    except (ImportError, Exception):
        # Fallback: basic converter without explicit options
        return DocumentConverter()


def _get_converter(use_ocr: bool, models_path: Path):
    key = (use_ocr, str(models_path.resolve()))
    if key not in _converter_cache:
        _converter_cache[key] = _build_converter(use_ocr, models_path)
    return _converter_cache[key]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _table_md_to_plain(md: str) -> str:
    """Convert Markdown table to plain pipe-delimited rows suitable for embedding."""
    lines = []
    for line in md.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip Markdown separator rows (|---|---|)
        if stripped.startswith("|") and all(c in "|-: " for c in stripped):
            continue
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            lines.append(" | ".join(cells))
        else:
            lines.append(stripped)
    return "\n".join(lines)


def _extract_page_num(item) -> int:
    """Safely extract 0-based page number from a Docling doc item."""
    try:
        prov = getattr(item, "prov", None) or []
        if prov:
            return max(0, int(prov[0].page_no) - 1)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return 0


def _map_docling_chunks(
    raw_chunks: list,
    doc_id: str,
    doc_name: str,
) -> List[DocumentChunk]:
    """Map Docling ChunkWithMetadata objects to DocumentChunk dataclass."""
    chunks: List[DocumentChunk] = []

    for i, chunk in enumerate(raw_chunks):
        text = (getattr(chunk, "text", "") or "").strip()
        if not text:
            continue

        page_num = 0
        table_md: Optional[str] = None
        heading: str = ""
        content_type = "text"

        meta = getattr(chunk, "meta", None)
        if meta is not None:
            headings = getattr(meta, "headings", None)
            if headings:
                heading = headings[-1]

            doc_items = getattr(meta, "doc_items", None) or []
            for item in doc_items:
                page_num = _extract_page_num(item)

                # Attempt to get Markdown from table items
                try:
                    from docling_core.types.doc import TableItem
                    if isinstance(item, TableItem):
                        content_type = "table"
                        md: Optional[str] = None
                        # export_to_markdown is preferred; fall back to str
                        if hasattr(item, "export_to_markdown"):
                            md = item.export_to_markdown()
                        elif hasattr(item, "to_dataframe"):
                            try:
                                md = item.to_dataframe().to_markdown(index=False)
                            except Exception:
                                pass
                        if md:
                            table_md = md
                            # Plain rows for BM25 / embeddings
                            text = _table_md_to_plain(md) or text
                except (ImportError, Exception):
                    pass

        cid = _stable_id(f"{doc_id}:{i}:{text[:50]}")
        metadata: dict = {
            "source": doc_name,
            "page": page_num + 1,
            "chunk_index": i,
            "content_type": content_type,
        }
        if heading:
            metadata["heading"] = heading
        if table_md:
            metadata["table_md"] = table_md  # Markdown kept for UI rendering

        chunks.append(DocumentChunk(
            chunk_id=cid,
            doc_id=doc_id,
            doc_name=doc_name,
            page_num=page_num,
            chunk_index=i,
            text=_clean_text(text),
            metadata=metadata,
        ))

    return chunks


def _process_csv(
    file_obj: IO[bytes],
    path: Path,
    doc_id: str,
) -> List[DocumentChunk]:
    """Convert a CSV file into DocumentChunks (50 rows per chunk)."""
    content = file_obj.read().decode("utf-8", errors="replace")
    reader = csv.reader(content.splitlines())
    rows = [" | ".join(str(c) for c in row) for row in reader if any(str(c).strip() for c in row)]

    chunks: List[DocumentChunk] = []
    chunk_size = 50
    for start in range(0, max(len(rows), 1), chunk_size):
        chunk_text = "\n".join(rows[start:start + chunk_size])
        idx = start // chunk_size
        cid = _stable_id(f"{doc_id}:{idx}:{chunk_text[:50]}")
        chunks.append(DocumentChunk(
            chunk_id=cid,
            doc_id=doc_id,
            doc_name=path.name,
            page_num=0,
            chunk_index=idx,
            text=chunk_text,
            metadata={
                "source": path.name,
                "page": 1,
                "chunk_index": idx,
                "content_type": "table",
            },
        ))
    return chunks


# ── Main processor ─────────────────────────────────────────────────────────────

class DoclingProcessor:
    """
    Docling-backed document processor.

    Drop-in replacement for DocumentProcessor — returns the same
    ProcessedDocument / DocumentChunk schema consumed by HybridStore.

    Supported formats
    -----------------
    PDF (text + table extraction, optional OCR), DOCX, PPTX, XLSX,
    HTML, Markdown, TXT, PNG/JPG/JPEG (via OCR), CSV.
    """

    def __init__(
        self,
        use_ocr: bool = False,
        max_raw_chars: int = 0,
        models_path: Optional[Union[str, Path]] = None,
    ):
        self.use_ocr = use_ocr
        self.max_raw_chars = max_raw_chars
        self.models_path = Path(models_path or _default_models_path())

    # ── Public API ────────────────────────────────────────────

    def process_file(
        self,
        file_path: Union[str, Path],
        file_obj: Optional[IO[bytes]] = None,
    ) -> ProcessedDocument:
        path = Path(file_path)
        ext = path.suffix.lower()
        doc_id = _stable_id(path.name)

        logger.info(
            "DoclingProcessor: %s (ext=%s, ocr=%s)",
            path.name, ext, self.use_ocr,
        )

        if ext in _CSV_EXTENSIONS:
            if file_obj is None:
                with open(path, "rb") as fh:
                    buf = io.BytesIO(fh.read())
            else:
                buf = file_obj
            chunks = _process_csv(buf, path, doc_id)
            raw = "\n".join(c.text for c in chunks)
            return ProcessedDocument(
                doc_id=doc_id,
                filename=path.name,
                file_type="CSV",
                total_pages=1,
                total_chunks=len(chunks),
                chunks=chunks,
                raw_text=raw,
                content_md5=hashlib.md5(raw[:50000].encode()).hexdigest(),
            )

        if ext not in _DOCLING_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )

        return self._process_with_docling(path, file_obj, doc_id)

    def process_raw_text(self, text: str, name: str = "pasted_text") -> ProcessedDocument:
        """Plain text input — delegates to DocumentProcessor (no Docling needed)."""
        from tools.document_tools import DocumentProcessor
        return DocumentProcessor().process_raw_text(text, name)

    # ── Docling pipeline ──────────────────────────────────────

    def _process_with_docling(
        self,
        path: Path,
        file_obj: Optional[IO[bytes]],
        doc_id: str,
    ) -> ProcessedDocument:
        converter = _get_converter(self.use_ocr, self.models_path)

        # Docling requires a real file path — write in-memory streams to a temp file
        tmp_path: Optional[Path] = None
        try:
            if file_obj is not None:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=path.suffix
                ) as tmp:
                    tmp.write(file_obj.read())
                    tmp_path = Path(tmp.name)
                source = tmp_path
            else:
                source = path

            result = converter.convert(source)
            docling_doc = result.document
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        # Structure-aware chunking
        chunks = self._chunk_document(docling_doc, doc_id, path.name)

        # Export raw text as Markdown (best quality, preserves headings)
        try:
            raw_text = docling_doc.export_to_markdown()
        except Exception:
            raw_text = "\n\n".join(c.text for c in chunks)

        if self.max_raw_chars and len(raw_text) > self.max_raw_chars:
            raw_text = raw_text[: self.max_raw_chars]

        # Page count
        try:
            total_pages = len(docling_doc.pages)
        except Exception:
            total_pages = max((c.page_num for c in chunks), default=0) + 1

        content_md5 = hashlib.md5(raw_text[:50000].encode()).hexdigest()

        return ProcessedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type=path.suffix.lstrip(".").upper(),
            total_pages=total_pages,
            total_chunks=len(chunks),
            chunks=chunks,
            raw_text=raw_text,
            content_md5=content_md5,
        )

    def _chunk_document(
        self, docling_doc, doc_id: str, doc_name: str
    ) -> List[DocumentChunk]:
        """Chunk with Docling HybridChunker; fall back to char-based on failure."""
        try:
            from docling.chunking import HybridChunker
            try:
                chunker = HybridChunker(max_tokens=256)
            except TypeError:
                chunker = HybridChunker()
            raw_chunks = list(chunker.chunk(docling_doc))
            mapped = _map_docling_chunks(raw_chunks, doc_id, doc_name)
            if mapped:
                return mapped
        except Exception as exc:
            logger.warning(
                "HybridChunker failed (%s) — using text-export fallback", exc
            )
        return self._fallback_chunks(docling_doc, doc_id, doc_name)

    def _fallback_chunks(
        self, docling_doc, doc_id: str, doc_name: str
    ) -> List[DocumentChunk]:
        """Export to Markdown then split with the char-based chunker."""
        from tools.document_tools import _chunk_text

        try:
            text = docling_doc.export_to_markdown()
        except Exception:
            text = str(docling_doc)

        text = _clean_text(text)
        raw_chunks = _chunk_text(text, chunk_size=800, overlap=150)
        result: List[DocumentChunk] = []
        for i, chunk_text in enumerate(raw_chunks):
            cid = _stable_id(f"{doc_id}:{i}:{chunk_text[:50]}")
            result.append(DocumentChunk(
                chunk_id=cid,
                doc_id=doc_id,
                doc_name=doc_name,
                page_num=0,
                chunk_index=i,
                text=chunk_text,
                metadata={"source": doc_name, "page": 0, "chunk_index": i},
            ))
        return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _default_models_path() -> Path:
    try:
        from config.settings import get_settings
        return Path(get_settings().docling_models_path)
    except Exception:
        return Path("models/docling")
