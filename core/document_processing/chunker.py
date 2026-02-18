"""
Adaptive document chunking with metadata generation.

Uses RecursiveCharacterTextSplitter for intelligent chunking and generates
comprehensive ChunkMetadata for each chunk.
"""

import hashlib
import uuid
from datetime import datetime
from typing import Literal

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.models import ChunkMetadata, NEREntities
from config.settings import Settings


class AdaptiveChunker:
    """Adaptive chunker that creates chunks with comprehensive metadata."""

    def __init__(self, settings: Settings):
        """
        Initialize the adaptive chunker.

        Args:
            settings: Application settings with chunking parameters
        """
        self.settings = settings
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

        # Create text splitter with settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size_max,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )

    def chunk_documents(
        self, documents: list[Document], settings: Settings = None
    ) -> list[tuple[str, ChunkMetadata]]:
        """
        Chunk documents and generate metadata for each chunk.

        Args:
            documents: List of LangChain Document objects
            settings: Optional settings override (uses instance settings if None)

        Returns:
            List of tuples (chunk_text, chunk_metadata)
        """
        if settings is None:
            settings = self.settings

        chunks_with_metadata = []

        for doc in documents:
            # Split document into chunks
            text_chunks = self.text_splitter.split_text(doc.page_content)

            # Get metadata from document
            source_file = doc.metadata.get("filename", "unknown")
            file_type = doc.metadata.get("file_type", "txt")
            page_number = doc.metadata.get("page_number")

            total_chunks = len(text_chunks)

            # Create metadata for each chunk
            for chunk_index, chunk_text in enumerate(text_chunks):
                chunk_metadata = self._create_chunk_metadata(
                    chunk_text=chunk_text,
                    source_file=source_file,
                    file_type=file_type,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                )
                chunks_with_metadata.append((chunk_text, chunk_metadata))

        return chunks_with_metadata

    def _create_chunk_metadata(
        self,
        chunk_text: str,
        source_file: str,
        file_type: Literal["pdf", "docx", "txt", "html", "json"],
        page_number: int | None,
        chunk_index: int,
        total_chunks: int,
    ) -> ChunkMetadata:
        """
        Create comprehensive metadata for a chunk.

        Args:
            chunk_text: The text content of the chunk
            source_file: Source filename
            file_type: Type of source file
            page_number: Page number (if applicable)
            chunk_index: Index of this chunk in the document
            total_chunks: Total number of chunks in the document

        Returns:
            ChunkMetadata object with all fields populated
        """
        # Generate unique chunk ID
        chunk_id = str(uuid.uuid4())

        # Calculate token count
        token_count = len(self.tokenizer.encode(chunk_text))

        # Calculate character count
        char_count = len(chunk_text)

        # Generate content hash for deduplication
        content_hash = self._generate_content_hash(chunk_text)

        # Create content preview (first 200 chars)
        content_preview = chunk_text[:200]
        if len(chunk_text) > 200:
            content_preview += "..."

        # Get current timestamp
        now = datetime.utcnow()

        # Create ChunkMetadata
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            source_file=source_file,
            file_type=file_type,
            page_number=page_number,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk_type="content",
            doc_item_type=None,
            parent_heading=None,
            hierarchy_level=None,
            chunk_method="recursive",
            chunk_size=self.settings.chunk_size_max,
            token_count=token_count,
            char_count=char_count,
            content_hash=content_hash,
            content_preview=content_preview,
            entities=NEREntities(),  # Will be populated by NER extractor
            keywords=[],  # Will be populated later if needed
            created_at=now,
            processed_at=now,
        )

        return metadata

    def _generate_content_hash(self, text: str) -> str:
        """
        Generate SHA-256 hash of normalized text for deduplication.

        Args:
            text: Text to hash

        Returns:
            Hexadecimal hash string
        """
        # Normalize text: lowercase, strip whitespace, remove extra spaces
        normalized = " ".join(text.lower().strip().split())

        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(normalized.encode("utf-8"))
        return hash_obj.hexdigest()

    def get_chunk_stats(self, chunks: list[tuple[str, ChunkMetadata]]) -> dict:
        """
        Get statistics about the chunks.

        Args:
            chunks: List of (chunk_text, chunk_metadata) tuples

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "avg_token_count": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        chunk_sizes = [metadata.char_count for _, metadata in chunks]
        token_counts = [metadata.token_count for _, metadata in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "avg_token_count": sum(token_counts) / len(token_counts),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "min_token_count": min(token_counts),
            "max_token_count": max(token_counts),
        }
