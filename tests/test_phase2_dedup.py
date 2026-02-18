"""
Tests for Phase 2 deduplication service.
"""

import pytest
from datetime import datetime
from config.settings import Settings
from config.models import ChunkMetadata, NEREntities
from core.document_processing.deduplication import DeduplicationService
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(dedup_similarity_threshold=0.95)


@pytest.fixture
def sample_metadata():
    """Create sample chunk metadata."""
    return ChunkMetadata(
        chunk_id="test_chunk_1",
        source_file="test.pdf",
        file_type="pdf",
        page_number=1,
        chunk_index=0,
        total_chunks=10,
        chunk_type="content",
        chunk_method="recursive",
        chunk_size=512,
        token_count=100,
        char_count=500,
        content_hash="abc123",
        content_preview="This is a test chunk",
        entities=NEREntities(),
        keywords=[],
        created_at=datetime.now(),
        processed_at=datetime.now()
    )


@pytest.fixture
def mock_embedding_service():
    """Create mock embedding service."""
    service = Mock()
    service.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return service


class TestDeduplicationService:
    """Tests for deduplication service."""
    
    def test_initialization(self, settings):
        """Test deduplication service initialization."""
        dedup = DeduplicationService(settings)
        assert dedup.similarity_threshold == 0.95
        assert len(dedup.content_hashes) == 0
        assert len(dedup.embeddings_cache) == 0
        
    def test_compute_content_hash(self, settings):
        """Test computing content hash."""
        dedup = DeduplicationService(settings)
        
        text1 = "This is a test document"
        text2 = "This is a test document"  # Same text
        text3 = "This is a different document"
        
        hash1 = dedup.compute_content_hash(text1)
        hash2 = dedup.compute_content_hash(text2)
        hash3 = dedup.compute_content_hash(text3)
        
        assert hash1 == hash2  # Same text should have same hash
        assert hash1 != hash3  # Different text should have different hash
        
    def test_check_exact_duplicate(self, settings):
        """Test checking exact duplicates."""
        dedup = DeduplicationService(settings)
        
        content_hash = "abc123"
        
        # Initially not a duplicate
        assert not dedup.check_exact_duplicate(content_hash)
        
        # Add to cache
        dedup.add_content_hash(content_hash)
        
        # Now it's a duplicate
        assert dedup.check_exact_duplicate(content_hash)
        
    @pytest.mark.asyncio
    async def test_check_semantic_duplicate_no_match(self, settings):
        """Test checking semantic duplicates with no match."""
        dedup = DeduplicationService(settings)
        dedup.embedding_service = Mock()
        dedup.embedding_service.embed_query = AsyncMock(return_value=[0.1] * 1024)
        
        # Empty cache - no duplicates
        is_dup, similarity = await dedup.check_semantic_duplicate("test text")
        
        assert not is_dup
        assert similarity == 0.0
        
    @pytest.mark.asyncio
    async def test_check_semantic_duplicate_with_match(self, settings):
        """Test checking semantic duplicates with match."""
        dedup = DeduplicationService(settings)
        dedup.embedding_service = Mock()
        
        # Add a very similar embedding to cache
        similar_embedding = [0.1] * 1024
        dedup.add_embedding("hash1", similar_embedding)
        
        # Mock embedding service to return similar embedding
        dedup.embedding_service.embed_query = AsyncMock(return_value=[0.1] * 1024)
        
        is_dup, similarity = await dedup.check_semantic_duplicate("test text")
        
        # Should detect as duplicate (similarity ~1.0)
        assert is_dup
        assert similarity >= 0.95
        
    @pytest.mark.asyncio
    async def test_check_duplicate_exact(self, settings, sample_metadata):
        """Test checking for exact duplicate."""
        dedup = DeduplicationService(settings)
        
        # Add content hash to cache
        dedup.add_content_hash(sample_metadata.content_hash)
        
        status, similarity = await dedup.check_duplicate(
            "test text",
            sample_metadata
        )
        
        assert status == "exact_duplicate"
        assert similarity == 1.0
        
    @pytest.mark.asyncio
    async def test_check_duplicate_new(self, settings, sample_metadata):
        """Test checking for new document."""
        dedup = DeduplicationService(settings)
        dedup.embedding_service = Mock()
        dedup.embedding_service.embed_query = AsyncMock(return_value=[0.1] * 1024)
        
        status, similarity = await dedup.check_duplicate(
            "test text",
            sample_metadata
        )
        
        assert status == "new"
        assert similarity == 0.0
        
        # Should be added to cache
        assert sample_metadata.content_hash in dedup.content_hashes
        
    @pytest.mark.asyncio
    async def test_deduplicate_chunks(self, settings, sample_metadata):
        """Test deduplicating a list of chunks."""
        dedup = DeduplicationService(settings)
        dedup.embedding_service = Mock()
        dedup.embedding_service.embed_query = AsyncMock(return_value=[0.1] * 1024)
        
        # Create chunks with different hashes
        metadata1 = sample_metadata.model_copy(update={"content_hash": "hash1"})
        metadata2 = sample_metadata.model_copy(update={"content_hash": "hash2"})
        metadata3 = sample_metadata.model_copy(update={"content_hash": "hash1"})  # Duplicate
        
        chunks = [
            ("Text 1", metadata1),
            ("Text 2", metadata2),
            ("Text 1", metadata3)  # Exact duplicate
        ]
        
        deduplicated, stats = await dedup.deduplicate_chunks(chunks)
        
        assert stats["total"] == 3
        assert stats["exact_duplicates"] == 1
        assert stats["unique"] == 2
        assert len(deduplicated) == 2
        
    def test_clear_cache(self, settings):
        """Test clearing cache."""
        dedup = DeduplicationService(settings)
        
        # Add some data
        dedup.add_content_hash("hash1")
        dedup.add_embedding("hash1", [0.1] * 1024)
        
        # Clear cache
        dedup.clear_cache()
        
        assert len(dedup.content_hashes) == 0
        assert len(dedup.embeddings_cache) == 0
        
    def test_get_cache_stats(self, settings):
        """Test getting cache statistics."""
        dedup = DeduplicationService(settings)
        
        dedup.add_content_hash("hash1")
        dedup.add_embedding("hash1", [0.1] * 1024)
        
        stats = dedup.get_cache_stats()
        
        assert stats["content_hashes"] == 1
        assert stats["embeddings_cached"] == 1
        assert stats["similarity_threshold"] == 0.95
        
    def test_set_similarity_threshold(self, settings):
        """Test setting similarity threshold."""
        dedup = DeduplicationService(settings)
        
        dedup.set_similarity_threshold(0.9)
        assert dedup.similarity_threshold == 0.9
        
        # Test invalid threshold
        with pytest.raises(ValueError):
            dedup.set_similarity_threshold(1.5)
