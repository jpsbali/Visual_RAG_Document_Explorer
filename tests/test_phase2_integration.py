"""
Integration tests for Phase 1B → Phase 2 pipeline.

Tests the complete flow from document loading through to hybrid search and reranking.
"""

import pytest
from datetime import datetime
from pathlib import Path
from config.settings import Settings
from config.models import ChunkMetadata, NEREntities
from core.document_processing.loaders import DocumentLoader
from core.document_processing.chunker import AdaptiveChunker
from core.embeddings.embedding_router import get_embedding_service
from core.vectordb.router import get_vector_store
from core.search.bm25_search import BM25Search
from core.search.hybrid_search import HybridSearch
from core.document_processing.deduplication import DeduplicationService
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        default_vector_db="qdrant",
        default_embedding_model="voyage",
        dense_weight=0.5,
        sparse_weight=0.3,
        metadata_weight=0.2,
        dedup_similarity_threshold=0.95
    )


@pytest.fixture
def sample_text():
    """Create sample text for testing."""
    return """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on 
    building systems that can learn from data. Python is the most popular 
    programming language for machine learning.
    
    ## Key Concepts
    
    - Supervised Learning
    - Unsupervised Learning
    - Deep Learning
    
    Companies like Google and Microsoft use machine learning extensively.
    """


@pytest.fixture
def sample_metadata():
    """Create sample chunk metadata."""
    return ChunkMetadata(
        chunk_id="test_chunk_1",
        source_file="test.txt",
        file_type="txt",
        chunk_index=0,
        total_chunks=1,
        chunk_type="content",
        chunk_method="recursive",
        chunk_size=512,
        token_count=100,
        char_count=500,
        content_hash="abc123",
        content_preview="Introduction to Machine Learning",
        entities=NEREntities(
            organizations=["Google", "Microsoft"],
            topics=["machine learning", "artificial intelligence", "Python"]
        ),
        keywords=["machine learning", "Python", "AI"],
        created_at=datetime.now(),
        processed_at=datetime.now()
    )


class TestPhase1BToPhase2Integration:
    """Integration tests for Phase 1B → Phase 2 pipeline."""
    
    def test_chunking_to_bm25(self, sample_text, sample_metadata):
        """Test chunking documents and building BM25 index."""
        # Phase 1B: Chunk document
        chunker = AdaptiveChunker()
        chunks = chunker.chunk_text(
            text=sample_text,
            metadata={"source_file": "test.txt", "file_type": "txt"}
        )
        
        assert len(chunks) > 0
        
        # Phase 2: Build BM25 index
        bm25 = BM25Search()
        
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        bm25.build_index(documents, metadatas)
        
        # Search
        results = bm25.search("machine learning", top_k=3)
        
        assert len(results) > 0
        assert any("machine learning" in r.content.lower() for r in results)
        
    @pytest.mark.asyncio
    async def test_deduplication_pipeline(self, settings, sample_text, sample_metadata):
        """Test deduplication in the pipeline."""
        # Create deduplication service
        dedup = DeduplicationService(settings)
        dedup.embedding_service = Mock()
        dedup.embedding_service.embed_query = AsyncMock(return_value=[0.1] * 1024)
        
        # Create chunks
        chunks = [
            (sample_text, sample_metadata),
            (sample_text, sample_metadata),  # Exact duplicate
            ("Different text", sample_metadata.model_copy(update={"content_hash": "xyz789"}))
        ]
        
        # Deduplicate
        deduplicated, stats = await dedup.deduplicate_chunks(chunks)
        
        assert stats["total"] == 3
        assert stats["exact_duplicates"] == 1
        assert stats["unique"] == 2
        assert len(deduplicated) == 2
        
    @pytest.mark.asyncio
    async def test_embedding_to_vector_store(self, settings, sample_metadata):
        """Test embedding generation and vector store upsert."""
        # Mock embedding service
        mock_embedding_service = Mock()
        mock_embedding_service.embed_documents = AsyncMock(
            return_value=[[0.1] * 1024, [0.2] * 1024]
        )
        
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.create_collection = AsyncMock()
        mock_vector_store.upsert = AsyncMock(return_value=2)
        
        # Simulate Phase 1B → Phase 2 flow
        documents = [
            "First document about machine learning",
            "Second document about Python programming"
        ]
        
        # Generate embeddings (Phase 1B)
        embeddings = await mock_embedding_service.embed_documents(documents)
        
        # Upsert to vector store (Phase 2)
        count = await mock_vector_store.upsert(
            collection="test",
            ids=["doc1", "doc2"],
            embeddings=embeddings,
            documents=documents,
            metadatas=[sample_metadata.model_dump(), sample_metadata.model_dump()]
        )
        
        assert count == 2
        mock_vector_store.upsert.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_hybrid_search_pipeline(self, settings, sample_metadata):
        """Test complete hybrid search pipeline."""
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.search = AsyncMock(return_value=[
            Mock(
                content="Machine learning document",
                metadata=sample_metadata,
                score=0.9,
                search_method="dense"
            )
        ])
        
        # Create BM25 index
        bm25 = BM25Search()
        documents = [
            "Machine learning is a subset of AI",
            "Python is used for machine learning",
            "Deep learning uses neural networks"
        ]
        metadatas = [sample_metadata, sample_metadata, sample_metadata]
        bm25.build_index(documents, metadatas)
        
        # Create hybrid search
        hybrid = HybridSearch(mock_vector_store, bm25, settings)
        
        # Perform hybrid search
        results = await hybrid.search(
            query="machine learning",
            collection="test",
            top_k=5,
            search_mode="hybrid"
        )
        
        assert len(results) > 0
        assert all(r.search_method == "hybrid" for r in results)
        
    def test_metadata_filtering_integration(self, sample_metadata):
        """Test metadata filtering with NER entities."""
        from core.search.metadata_filter import MetadataFilter
        from config.models import RetrievedChunk
        
        # Create chunks with different entities
        chunks = [
            RetrievedChunk(
                content="Google uses machine learning",
                metadata=sample_metadata,
                score=0.9,
                search_method="dense"
            ),
            RetrievedChunk(
                content="Microsoft develops AI systems",
                metadata=sample_metadata.model_copy(update={
                    "entities": NEREntities(organizations=["Microsoft"])
                }),
                score=0.8,
                search_method="dense"
            ),
            RetrievedChunk(
                content="Amazon cloud services",
                metadata=sample_metadata.model_copy(update={
                    "entities": NEREntities(organizations=["Amazon"])
                }),
                score=0.7,
                search_method="dense"
            )
        ]
        
        # Filter by organization
        filtered = MetadataFilter.apply_filters(
            chunks,
            filters={"organizations": ["Google", "Microsoft"]}
        )
        
        assert len(filtered) == 2
        
    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self, settings, sample_text, sample_metadata):
        """Test full pipeline from document to search (mocked)."""
        # 1. Phase 1B: Chunk document
        chunker = AdaptiveChunker()
        chunks = chunker.chunk_text(
            text=sample_text,
            metadata={"source_file": "test.txt", "file_type": "txt"}
        )
        
        # 2. Phase 1B: Generate embeddings (mocked)
        mock_embedding_service = Mock()
        mock_embedding_service.embed_documents = AsyncMock(
            return_value=[[0.1] * 1024] * len(chunks)
        )
        embeddings = await mock_embedding_service.embed_documents(
            [chunk.content for chunk in chunks]
        )
        
        # 3. Phase 2: Build BM25 index
        bm25 = BM25Search()
        bm25.build_index(
            [chunk.content for chunk in chunks],
            [chunk.metadata for chunk in chunks]
        )
        
        # 4. Phase 2: Upsert to vector store (mocked)
        mock_vector_store = Mock()
        mock_vector_store.upsert = AsyncMock(return_value=len(chunks))
        mock_vector_store.search = AsyncMock(return_value=[
            Mock(
                content=chunks[0].content,
                metadata=chunks[0].metadata,
                score=0.9,
                search_method="dense"
            )
        ])
        
        await mock_vector_store.upsert(
            collection="test",
            ids=[chunk.metadata.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[chunk.metadata.model_dump() for chunk in chunks]
        )
        
        # 5. Phase 2: Hybrid search
        hybrid = HybridSearch(mock_vector_store, bm25, settings)
        results = await hybrid.search(
            query="machine learning",
            collection="test",
            top_k=5,
            search_mode="hybrid"
        )
        
        assert len(results) > 0
        
    def test_phase1b_metadata_to_phase2_filters(self, sample_metadata):
        """Test that Phase 1B metadata works with Phase 2 filters."""
        from core.search.metadata_filter import MetadataFilter
        from config.models import RetrievedChunk
        
        # Phase 1B produces chunks with NER entities
        chunk = RetrievedChunk(
            content="Test content",
            metadata=sample_metadata,
            score=0.9,
            search_method="dense"
        )
        
        # Phase 2 can filter by those entities
        filtered = MetadataFilter.apply_filters(
            [chunk],
            filters={"organizations": ["Google"]}
        )
        
        assert len(filtered) == 1
        
        # Filter by non-existent entity
        filtered = MetadataFilter.apply_filters(
            [chunk],
            filters={"organizations": ["NonExistent"]}
        )
        
        assert len(filtered) == 0


@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration"),
    reason="Requires external services"
)
@pytest.mark.asyncio
class TestRealIntegration:
    """Real integration tests with actual services."""
    
    async def test_real_embedding_and_search(self, settings):
        """Test with real embedding service and vector store."""
        # This test requires actual API keys and services
        # Skip in CI/CD unless explicitly enabled
        pytest.skip("Requires real API keys and services")


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services"
    )
