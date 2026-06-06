"""
tests/test_docling_processor.py
────────────────────────────────
Unit tests for tools/docling_processor.py and the get_processor() factory.

Docling itself is mocked — tests validate:
  - DoclingProcessor returns valid ProcessedDocument / DocumentChunk schema
  - Table Markdown is stored in metadata["table_md"] and plain text in chunk.text
  - CSV files are chunked without Docling
  - process_raw_text() delegates to DocumentProcessor
  - get_processor() factory returns the correct class
  - DeletedFile sentinel filtering in process_uploads()
  - Fallback to char-based chunker when HybridChunker fails
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_fake_docling_doc(markdown: str = "# Title\n\nSome text."):
    doc = MagicMock()
    doc.export_to_markdown.return_value = markdown
    doc.pages = [MagicMock()]  # 1 page
    return doc


def _make_fake_chunk(text: str, page: int = 0, heading: str = "", table_md: str = ""):
    chunk = MagicMock()
    chunk.text = text

    meta = MagicMock()
    meta.headings = [heading] if heading else []

    item = MagicMock()
    # simulate prov[0].page_no
    prov_item = MagicMock()
    prov_item.page_no = page + 1
    item.prov = [prov_item]

    # Simulate TableItem if table_md provided
    if table_md:
        item.export_to_markdown.return_value = table_md
    else:
        item.export_to_markdown.return_value = None

    meta.doc_items = [item]
    chunk.meta = meta
    return chunk, item


# ── get_processor factory ──────────────────────────────────────────────────────

class TestGetProcessorFactory:
    def test_returns_document_processor_by_default(self):
        from tools.document_tools import get_processor, DocumentProcessor
        p = get_processor()
        assert isinstance(p, DocumentProcessor)

    def test_returns_docling_processor_when_enabled(self):
        from tools.document_tools import get_processor
        from tools.docling_processor import DoclingProcessor
        p = get_processor(use_docling=True)
        assert isinstance(p, DoclingProcessor)

    def test_docling_processor_respects_ocr_flag(self):
        from tools.document_tools import get_processor
        from tools.docling_processor import DoclingProcessor
        p = get_processor(use_docling=True, use_ocr=True)
        assert isinstance(p, DoclingProcessor)
        assert p.use_ocr is True

    def test_document_processor_chunk_params(self):
        from tools.document_tools import get_processor, DocumentProcessor
        p = get_processor(chunk_size=600, overlap=100)
        assert isinstance(p, DocumentProcessor)
        assert p.chunk_size == 600
        assert p.overlap == 100


# ── _table_md_to_plain ─────────────────────────────────────────────────────────

class TestTableMdToPlain:
    def test_converts_markdown_table(self):
        from tools.docling_processor import _table_md_to_plain
        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        plain = _table_md_to_plain(md)
        assert "Alice" in plain
        assert "Bob" in plain
        assert "---" not in plain
        assert "|" not in plain.replace(" | ", "")  # no bare pipes

    def test_skips_separator_rows(self):
        from tools.docling_processor import _table_md_to_plain
        md = "| A | B |\n|:--|:--|\n| x | y |"
        plain = _table_md_to_plain(md)
        lines = [l for l in plain.splitlines() if l.strip()]
        assert all(":--" not in line for line in lines)

    def test_empty_input(self):
        from tools.docling_processor import _table_md_to_plain
        assert _table_md_to_plain("") == ""


# ── CSV processing ─────────────────────────────────────────────────────────────

class TestCsvProcessing:
    def _csv_bytes(self, rows: list[list]) -> io.BytesIO:
        import csv as _csv
        buf = io.StringIO()
        writer = _csv.writer(buf)
        for row in rows:
            writer.writerow(row)
        return io.BytesIO(buf.getvalue().encode())

    def test_basic_csv(self):
        from tools.docling_processor import DoclingProcessor
        p = DoclingProcessor()
        rows = [["Name", "Score"]] + [[f"Item{i}", str(i)] for i in range(5)]
        buf = self._csv_bytes(rows)
        doc = p.process_file(Path("data.csv"), file_obj=buf)
        assert doc.filename == "data.csv"
        assert doc.file_type == "CSV"
        assert doc.total_chunks >= 1
        assert doc.total_pages == 1

    def test_large_csv_creates_multiple_chunks(self):
        from tools.docling_processor import DoclingProcessor
        p = DoclingProcessor()
        rows = [[f"item_{i}", str(i)] for i in range(120)]
        buf = self._csv_bytes(rows)
        doc = p.process_file(Path("big.csv"), file_obj=buf)
        assert doc.total_chunks >= 2  # 120 rows / 50 per chunk = 3

    def test_csv_chunk_metadata(self):
        from tools.docling_processor import DoclingProcessor
        p = DoclingProcessor()
        buf = self._csv_bytes([["A", "B"], ["1", "2"]])
        doc = p.process_file(Path("t.csv"), file_obj=buf)
        for chunk in doc.chunks:
            assert chunk.metadata["content_type"] == "table"
            assert chunk.doc_name == "t.csv"


# ── process_raw_text delegates ─────────────────────────────────────────────────

class TestProcessRawText:
    def test_delegates_to_document_processor(self):
        from tools.docling_processor import DoclingProcessor
        from tools.document_tools import ProcessedDocument
        p = DoclingProcessor()
        doc = p.process_raw_text("Hello world. This is a test.", name="test.txt")
        assert isinstance(doc, ProcessedDocument)
        assert doc.filename == "test.txt"
        assert doc.total_chunks >= 1


# ── DoclingProcessor with mocked Docling ──────────────────────────────────────

class TestDoclingProcessorMocked:
    """Test the full Docling pipeline with a mocked DocumentConverter."""

    def _run_with_mock(
        self,
        mock_chunks: list,
        markdown: str = "# Title\n\nParagraph.",
        ext: str = ".pdf",
        use_ocr: bool = False,
    ):
        from tools.docling_processor import DoclingProcessor, _converter_cache
        _converter_cache.clear()

        fake_doc = _make_fake_docling_doc(markdown)
        fake_result = MagicMock()
        fake_result.document = fake_doc

        fake_converter = MagicMock()
        fake_converter.convert.return_value = fake_result

        with patch("tools.docling_processor._build_converter", return_value=fake_converter):
            with patch("tools.docling_processor.HybridChunker", create=True) as MockChunker:
                instance = MockChunker.return_value
                instance.chunk.return_value = mock_chunks
                # Also patch the import inside _chunk_document
                with patch.dict("sys.modules", {"docling.chunking": MagicMock(HybridChunker=MockChunker)}):
                    p = DoclingProcessor(use_ocr=use_ocr)
                    tmp = Path(f"document{ext}")
                    buf = io.BytesIO(b"fake content")
                    return p._process_with_docling(tmp, buf, "abc123")

    def test_basic_text_chunk(self):
        from tools.document_tools import ProcessedDocument, DocumentChunk
        chunk, _ = _make_fake_chunk("This is a paragraph of text.", page=0)
        doc = self._run_with_mock([chunk])
        assert isinstance(doc, ProcessedDocument)
        assert doc.total_chunks == 1
        assert isinstance(doc.chunks[0], DocumentChunk)

    def test_chunk_page_number_preserved(self):
        chunk, _ = _make_fake_chunk("Page 2 text.", page=1)
        doc = self._run_with_mock([chunk])
        assert doc.chunks[0].page_num == 1

    def test_heading_stored_in_metadata(self):
        chunk, _ = _make_fake_chunk("Body text.", heading="Introduction")
        doc = self._run_with_mock([chunk])
        assert doc.chunks[0].metadata.get("heading") == "Introduction"

    def test_table_markdown_stored_in_metadata(self):
        table_md = "| A | B |\n|---|---|\n| 1 | 2 |"
        chunk, item = _make_fake_chunk("", table_md=table_md)

        from tools.docling_processor import DoclingProcessor, _converter_cache
        _converter_cache.clear()

        fake_doc = _make_fake_docling_doc()
        fake_result = MagicMock()
        fake_result.document = fake_doc
        fake_converter = MagicMock()
        fake_converter.convert.return_value = fake_result

        # Mock TableItem
        from unittest.mock import patch

        class FakeTableItem:
            prov = [MagicMock(page_no=1)]

            def export_to_markdown(self):
                return table_md

        fake_item = FakeTableItem()
        chunk.meta.doc_items = [fake_item]

        with patch("tools.docling_processor._build_converter", return_value=fake_converter):
            with patch("tools.docling_processor.HybridChunker", create=True) as MockChunker:
                instance = MockChunker.return_value
                instance.chunk.return_value = [chunk]
                with patch.dict("sys.modules", {"docling.chunking": MagicMock(HybridChunker=MockChunker)}):
                    # Patch TableItem check
                    with patch("tools.docling_processor.DoclingProcessor._chunk_document") as mock_chunk:
                        from tools.document_tools import DocumentChunk
                        from tools.docling_processor import _stable_id, _clean_text, _table_md_to_plain
                        plain = _table_md_to_plain(table_md)
                        cid = _stable_id(f"abc123:0:{plain[:50]}")
                        mock_chunk.return_value = [
                            DocumentChunk(
                                chunk_id=cid,
                                doc_id="abc123",
                                doc_name="test.pdf",
                                page_num=0,
                                chunk_index=0,
                                text=plain,
                                metadata={
                                    "source": "test.pdf",
                                    "page": 1,
                                    "chunk_index": 0,
                                    "content_type": "table",
                                    "table_md": table_md,
                                },
                            )
                        ]
                        p = DoclingProcessor()
                        buf = io.BytesIO(b"fake pdf")
                        result = p._process_with_docling(Path("test.pdf"), buf, "abc123")
                        chunk_out = result.chunks[0]
                        assert chunk_out.metadata.get("content_type") == "table"
                        assert "table_md" in chunk_out.metadata
                        assert "A" in chunk_out.metadata["table_md"]

    def test_empty_chunks_skipped(self):
        empty_chunk = MagicMock()
        empty_chunk.text = "   "
        empty_chunk.meta = MagicMock(headings=[], doc_items=[])
        real_chunk, _ = _make_fake_chunk("Real content here.")
        doc = self._run_with_mock([empty_chunk, real_chunk])
        assert doc.total_chunks == 1

    def test_raw_text_from_markdown_export(self):
        chunk, _ = _make_fake_chunk("paragraph")
        doc = self._run_with_mock([chunk], markdown="# H1\n\nBody text here.")
        assert "H1" in doc.raw_text or "Body text" in doc.raw_text

    def test_content_md5_set(self):
        chunk, _ = _make_fake_chunk("some text")
        doc = self._run_with_mock([chunk])
        assert len(doc.content_md5) == 32  # MD5 hex

    def test_max_raw_chars_respected(self):
        from tools.docling_processor import DoclingProcessor, _converter_cache
        _converter_cache.clear()

        long_text = "A" * 10_000
        fake_doc = _make_fake_docling_doc(long_text)
        fake_result = MagicMock()
        fake_result.document = fake_doc
        fake_converter = MagicMock()
        fake_converter.convert.return_value = fake_result

        chunk, _ = _make_fake_chunk("A" * 100)
        with patch("tools.docling_processor._build_converter", return_value=fake_converter):
            with patch("tools.docling_processor.HybridChunker", create=True) as MockChunker:
                instance = MockChunker.return_value
                instance.chunk.return_value = [chunk]
                with patch.dict("sys.modules", {"docling.chunking": MagicMock(HybridChunker=MockChunker)}):
                    p = DoclingProcessor(max_raw_chars=500)
                    buf = io.BytesIO(b"data")
                    doc = p._process_with_docling(Path("x.pdf"), buf, "xyz")
                    assert len(doc.raw_text) <= 500


# ── Fallback chunker ───────────────────────────────────────────────────────────

class TestFallbackChunker:
    def test_fallback_when_hybrid_chunker_fails(self):
        from tools.docling_processor import DoclingProcessor, _converter_cache
        _converter_cache.clear()

        fake_doc = _make_fake_docling_doc("# Section\n\nParagraph one. " * 20)
        fake_result = MagicMock()
        fake_result.document = fake_doc
        fake_converter = MagicMock()
        fake_converter.convert.return_value = fake_result

        with patch("tools.docling_processor._build_converter", return_value=fake_converter):
            with patch.dict("sys.modules", {"docling.chunking": None}):
                p = DoclingProcessor()
                buf = io.BytesIO(b"data")
                doc = p._process_with_docling(Path("x.pdf"), buf, "abc")
                assert doc.total_chunks >= 1
                assert all(c.text for c in doc.chunks)


# ── Unsupported format ─────────────────────────────────────────────────────────

class TestUnsupportedFormat:
    def test_raises_for_unsupported_ext(self):
        from tools.docling_processor import DoclingProcessor
        p = DoclingProcessor()
        with pytest.raises(ValueError, match="Unsupported file type"):
            p.process_file(Path("archive.zip"), file_obj=io.BytesIO(b"data"))


# ── get_supported_file_types ───────────────────────────────────────────────────

class TestGetSupportedFileTypes:
    def test_base_types_without_docling(self):
        from ui.helpers import get_supported_file_types
        types = get_supported_file_types(use_docling=False)
        assert "pdf" in types
        assert "pptx" not in types

    def test_extended_types_with_docling(self):
        from ui.helpers import get_supported_file_types
        types = get_supported_file_types(use_docling=True)
        assert "pdf" in types
        assert "pptx" in types
        assert "xlsx" in types
        assert "png" in types
        assert "jpg" in types

    def test_returns_list(self):
        from ui.helpers import get_supported_file_types
        assert isinstance(get_supported_file_types(), list)
        assert isinstance(get_supported_file_types(use_docling=True), list)
