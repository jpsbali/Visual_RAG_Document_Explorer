"""
Tests for document loaders.

Tests loading of PDF, DOCX, TXT, HTML, and JSON files.
"""

import pytest
from pathlib import Path
import tempfile
import json

from core.document_processing.loaders import DocumentLoaderFactory


class TestDocumentLoaderFactory:
    """Test document loader factory."""

    def test_supported_formats(self):
        """Test that all expected formats are supported."""
        assert ".pdf" in DocumentLoaderFactory.SUPPORTED_FORMATS
        assert ".docx" in DocumentLoaderFactory.SUPPORTED_FORMATS
        assert ".txt" in DocumentLoaderFactory.SUPPORTED_FORMATS
        assert ".html" in DocumentLoaderFactory.SUPPORTED_FORMATS
        assert ".json" in DocumentLoaderFactory.SUPPORTED_FORMATS

    def test_is_supported(self):
        """Test format support checking."""
        assert DocumentLoaderFactory.is_supported("test.pdf")
        assert DocumentLoaderFactory.is_supported("test.docx")
        assert DocumentLoaderFactory.is_supported("test.txt")
        assert DocumentLoaderFactory.is_supported("test.html")
        assert DocumentLoaderFactory.is_supported("test.json")
        assert not DocumentLoaderFactory.is_supported("test.xyz")

    def test_get_file_type(self):
        """Test file type detection."""
        assert DocumentLoaderFactory.get_file_type("test.pdf") == "pdf"
        assert DocumentLoaderFactory.get_file_type("test.docx") == "docx"
        assert DocumentLoaderFactory.get_file_type("test.txt") == "txt"
        assert DocumentLoaderFactory.get_file_type("test.html") == "html"
        assert DocumentLoaderFactory.get_file_type("test.htm") == "html"
        assert DocumentLoaderFactory.get_file_type("test.json") == "json"

    def test_get_file_type_unsupported(self):
        """Test that unsupported formats raise ValueError."""
        with pytest.raises(ValueError):
            DocumentLoaderFactory.get_file_type("test.xyz")

    def test_load_txt_file(self):
        """Test loading a TXT file."""
        # Create temporary TXT file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.\n")
            temp_path = f.name

        try:
            # Load document
            documents = DocumentLoaderFactory.load_document(temp_path)

            # Verify results
            assert len(documents) > 0
            assert documents[0].metadata["file_type"] == "txt"
            assert documents[0].metadata["filename"] == Path(temp_path).name
            assert "test document" in documents[0].page_content
        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_load_json_file(self):
        """Test loading a JSON file."""
        # Create temporary JSON file
        test_data = [
            {"title": "Document 1", "content": "This is the first document."},
            {"title": "Document 2", "content": "This is the second document."},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Load document
            documents = DocumentLoaderFactory.load_document(temp_path)

            # Verify results
            assert len(documents) > 0
            assert documents[0].metadata["file_type"] == "json"
            assert documents[0].metadata["filename"] == Path(temp_path).name
        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DocumentLoaderFactory.load_document("nonexistent.txt")

    def test_metadata_preservation(self):
        """Test that metadata is properly preserved."""
        # Create temporary TXT file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            documents = DocumentLoaderFactory.load_document(temp_path)

            # Check metadata
            assert "filename" in documents[0].metadata
            assert "file_type" in documents[0].metadata
            assert documents[0].metadata["file_type"] == "txt"
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
