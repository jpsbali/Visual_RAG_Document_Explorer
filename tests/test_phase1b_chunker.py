"""
Tests for adaptive chunker.

Tests chunking logic and metadata generation.
"""

import pytest
from langchain_core.documents import Document

from core.document_processing.chunker import AdaptiveChunker
from config.settings import Settings
from config.models import ChunkMetadata


class TestAdaptiveChunker:
    """Test adaptive chunker."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            chunk_size_min=256,
            chunk_size_max=512,
            chunk_overlap=64,
        )

    @pytest.fixture
    def chunker(self, settings):
        """Create chunker instance."""
        return AdaptiveChunker(settings)

    def test_chunker_initialization(self, chunker, settings):
        """Test that chunker initializes correctly."""
        assert chunker.settings == settings
        assert chunker.tokenizer is not None
        assert chunker.text_splitter is not None

    def test_chunk_short_document(self, chunker):
        """Test chunking a short document."""
        # Create a short document
        doc = Document(
            page_content="This is a short test document.",
            metadata={"filename": "test.txt", "file_type": "txt"},
        )

        # Chunk document
        chunks = chunker.chunk_documents([doc])

        # Verify results
        assert len(chunks) > 0
        assert isinstance(chunks[0], tuple)
        assert isinstance(chunks[0][0], str)  # chunk text
        assert isinstance(chunks[0][1], ChunkMetadata)  # chunk metadata

    def test_chunk_long_document(self, chunker):
        """Test chunking a long document."""
        # Create a long document
        long_text = " ".join(["This is sentence number {}.".format(i) for i in range(200)])
        doc = Document(
            page_content=long_text,
            metadata={"filename": "test.txt", "file_type": "txt"},
        )

        # Chunk document
        chunks = chunker.chunk_documents([doc])

        # Verify results
        assert len(chunks) > 1  # Should be split into multiple chunks
        for chunk_text, metadata in chunks:
            assert len(chunk_text) > 0
            assert metadata.chunk_size == chunker.settings.chunk_size_max

    def test_chunk_metadata_fields(self, chunker):
        """Test that all metadata fields are populated."""
        doc = Document(
            page_content="Test document content.",
            metadata={"filename": "test.txt", "file_type": "txt", "page_number": 1},
        )

        chunks = chunker.chunk_documents([doc])
        metadata = chunks[0][1]

        # Verify all required fields
        assert metadata.chunk_id is not None
        assert metadata.source_file == "test.txt"
        assert metadata.file_type == "txt"
        assert metadata.page_number == 1
        assert metadata.chunk_index == 0
        assert metadata.total_chunks == len(chunks)
        assert metadata.chunk_type == "content"
        assert metadata.chunk_method == "recursive"
        assert metadata.chunk_size > 0
        assert metadata.token_count > 0
        assert metadata.char_count > 0
        assert metadata.content_hash is not None
        assert metadata.content_preview is not None
        assert metadata.created_at is not None
        assert metadata.processed_at is not None

    def test_content_hash_generation(self, chunker):
        """Test that content hashes are generated correctly."""
        doc1 = Document(
            page_content="Test content",
            metadata={"filename": "test.txt", "file_type": "txt"},
        )
        doc2 = Document(
            page_content="Test content",  # Same content
            metadata={"filename": "test2.txt", "file_type": "txt"},
        )
        doc3 = Document(
            page_content="Different content",
            metadata={"filename": "test3.txt", "file_type": "txt"},
        )

        chunks1 = chunker.chunk_documents([doc1])
        chunks2 = chunker.chunk_documents([doc2])
        chunks3 = chunker.chunk_documents([doc3])

        # Same content should have same hash
        assert chunks1[0][1].content_hash == chunks2[0][1].content_hash

        # Different content should have different hash
        assert chunks1[0][1].content_hash != chunks3[0][1].content_hash

    def test_content_preview(self, chunker):
        """Test content preview generation."""
        long_text = "A" * 300  # Text longer than 200 chars
        doc = Document(
            page_content=long_text,
            metadata={"filename": "test.txt", "file_type": "txt"},
        )

        chunks = chunker.chunk_documents([doc])
        preview = chunks[0][1].content_preview

        # Preview should be truncated
        assert len(preview) <= 203  # 200 chars + "..."
        assert preview.endswith("...")

    def test_chunk_stats(self, chunker):
        """Test chunk statistics calculation."""
        doc = Document(
            page_content=" ".join(["Sentence {}.".format(i) for i in range(100)]),
            metadata={"filename": "test.txt", "file_type": "txt"},
        )

        chunks = chunker.chunk_documents([doc])
        stats = chunker.get_chunk_stats(chunks)

        # Verify stats
        assert stats["total_chunks"] == len(chunks)
        assert stats["avg_chunk_size"] > 0
        assert stats["avg_token_count"] > 0
        assert stats["min_chunk_size"] > 0
        assert stats["max_chunk_size"] > 0

    def test_multiple_documents(self, chunker):
        """Test chunking multiple documents."""
        docs = [
            Document(
                page_content="First document content.",
                metadata={"filename": "doc1.txt", "file_type": "txt"},
            ),
            Document(
                page_content="Second document content.",
                metadata={"filename": "doc2.txt", "file_type": "txt"},
            ),
        ]

        chunks = chunker.chunk_documents(docs)

        # Should have chunks from both documents
        assert len(chunks) >= 2

        # Verify source files are different
        source_files = {metadata.source_file for _, metadata in chunks}
        assert "doc1.txt" in source_files
        assert "doc2.txt" in source_files


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
