"""
Document loaders for all supported file formats.

Uses LangChain document loaders to handle PDF, DOCX, TXT, HTML, and JSON files.
Automatically detects file type and preserves metadata.
"""

import os
from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    BSHTMLLoader,
    JSONLoader,
)
from langchain_core.documents import Document


class DocumentLoaderFactory:
    """Factory class for loading documents of various formats."""

    SUPPORTED_FORMATS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".txt": "txt",
        ".html": "html",
        ".htm": "html",
        ".json": "json",
    }

    @classmethod
    def load_document(cls, file_path: str) -> list[Document]:
        """
        Load a document from file path, auto-detecting format.

        Args:
            file_path: Path to the document file

        Returns:
            List of LangChain Document objects with content and metadata

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect file type from extension
        file_ext = path.suffix.lower()
        if file_ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {list(cls.SUPPORTED_FORMATS.keys())}"
            )

        file_type = cls.SUPPORTED_FORMATS[file_ext]
        filename = path.name

        # Load document based on type
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            # Add file_type and filename to metadata
            for doc in documents:
                doc.metadata["file_type"] = "pdf"
                doc.metadata["filename"] = filename
                # PyPDFLoader provides 'page' key, normalize to 'page_number'
                if "page" in doc.metadata:
                    doc.metadata["page_number"] = doc.metadata["page"] + 1  # 1-indexed
                    del doc.metadata["page"]

        elif file_type == "docx":
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["file_type"] = "docx"
                doc.metadata["filename"] = filename
                doc.metadata["page_number"] = None  # DOCX doesn't have page numbers

        elif file_type == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            for doc in documents:
                doc.metadata["file_type"] = "txt"
                doc.metadata["filename"] = filename
                doc.metadata["page_number"] = None

        elif file_type == "html":
            loader = BSHTMLLoader(file_path, open_encoding="utf-8")
            documents = loader.load()
            for doc in documents:
                doc.metadata["file_type"] = "html"
                doc.metadata["filename"] = filename
                doc.metadata["page_number"] = None

        elif file_type == "json":
            # JSONLoader requires jq_schema to extract text
            # Default: extract all text content from JSON
            loader = JSONLoader(
                file_path=file_path,
                jq_schema=".[]",  # Extract all items if JSON is array
                text_content=False,
            )
            try:
                documents = loader.load()
            except Exception:
                # If array extraction fails, try extracting as single object
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema=".",
                    text_content=False,
                )
                documents = loader.load()

            for doc in documents:
                doc.metadata["file_type"] = "json"
                doc.metadata["filename"] = filename
                doc.metadata["page_number"] = None

        return documents

    @classmethod
    def get_file_type(cls, file_path: str) -> Literal["pdf", "docx", "txt", "html", "json"]:
        """
        Get the file type from file path.

        Args:
            file_path: Path to the file

        Returns:
            File type as string literal

        Raises:
            ValueError: If file format is not supported
        """
        path = Path(file_path)
        file_ext = path.suffix.lower()

        if file_ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {list(cls.SUPPORTED_FORMATS.keys())}"
            )

        return cls.SUPPORTED_FORMATS[file_ext]

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """
        Check if a file format is supported.

        Args:
            file_path: Path to the file

        Returns:
            True if format is supported, False otherwise
        """
        path = Path(file_path)
        file_ext = path.suffix.lower()
        return file_ext in cls.SUPPORTED_FORMATS
