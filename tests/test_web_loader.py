"""tests/test_web_loader.py — Unit tests for tools/web_loader.py"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools.document_tools import DocumentProcessor
from tools.web_loader import _title_from_url, load_url_as_document


class TestTitleFromUrl:
    def test_uses_host_and_slug(self):
        title = _title_from_url("https://www.example.com/articles/great-paper.html")
        assert "example.com" in title
        assert "great paper" in title

    def test_host_only_when_no_path(self):
        assert _title_from_url("https://example.com") == "example.com"

    def test_strips_www(self):
        assert "www." not in _title_from_url("https://www.nature.com/")


class TestLoadUrlAsDocument:
    def _processor(self):
        return DocumentProcessor(chunk_size=200, overlap=40)

    def test_fetch_error_propagates(self):
        with patch("tools.call_analyzer.fetch_call_text", return_value=("", "HTTP 404")):
            doc, err = load_url_as_document("https://x.com", self._processor())
        assert doc is None
        assert err == "HTTP 404"

    def test_empty_text_returns_error(self):
        with patch("tools.call_analyzer.fetch_call_text", return_value=("   ", "")):
            doc, err = load_url_as_document("https://x.com", self._processor())
        assert doc is None
        assert "no extractable text" in err.lower()

    def test_success_returns_processed_document(self):
        page_text = "This is a research article. " * 50
        with patch("tools.call_analyzer.fetch_call_text", return_value=(page_text, "")):
            doc, err = load_url_as_document(
                "https://example.com/research/paper", self._processor()
            )
        assert err == ""
        assert doc is not None
        assert doc.total_chunks >= 1
        assert "example.com" in doc.filename
        assert doc.raw_text.startswith("This is a research article.")
