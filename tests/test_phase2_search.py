"""
Tests for Phase 2 search implementations.

Tests BM25, metadata filtering, and hybrid search with RRF.
"""

import pytest
from datetime import datetime
from config.settings import Settings
from config.models import ChunkMetadata, NEREntities, RetrievedChunk
from core.search.bm25_search import BM25Search
from core.search.metadata_filter import MetadataFilter
from core.search.hybrid_search import HybridSearch
from unittest.mock import Mock, AsyncMock


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
        entities=NEREntities(
            people=["John Doe", "Jane Smith"],
            organizations=["Acme Corp", "TechCo"],
            dates=["2024-01-01"],
            locations=["New York"],
            topics=["technology", "business"]
        ),
        keywords=["test", "sample"],
        created_at=datetime.now(),
        processed_at=datetime.now()
    )


@pytest.fixture
def sample_chunks(sample_metadata):
    """Create sample retrieved chunks."""
    return [
        RetrievedChunk(
            content="This is a test document about technology",
            metadata=sample_metadata,
            score=0.9,
            search_method="dense"
        ),
        RetrievedChunk(
            content="Another document about business",
            metadata=sample_metadata,
            score=0.8,
            search_method="dense"
        ),
        RetrievedChunk(
            content="Third document with different content",
            metadata=sample_metadata,
            score=0.7,
            search_method="dense"
        )
    ]


class TestBM25Search:
    """Tests for BM25 sparse search."""
    
    def test_initialization(self):
        """Test BM25 search initialization."""
        bm25 = BM25Search(k1=1.5, b=0.75)
        assert bm25.k1 == 1.5
        assert bm25.b == 0.75
        assert bm25.index is None
        
    def test_build_index(self, sample_metadata):
        """Test building BM25 index."""
        bm25 = BM25Search()
        
        documents = [
            "This is the first document",
            "This is the second document",
            "And this is the third one"
        ]
        metadatas = [sample_metadata, sample_metadata, sample_metadata]
        
        bm25.build_index(documents, metadatas)
        
        assert bm25.index is not None
        assert len(bm25.documents) == 3
        assert len(bm25.metadatas) == 3
        assert bm25.get_index_size() == 3
        
    def test_search(self, sample_metadata):
        """Test BM25 search."""
        bm25 = BM25Search()
        
        documents = [
            "Python programming language",
            "Java programming language",
            "Machine learning with Python"
        ]
        metadatas = [sample_metadata, sample_metadata, sample_metadata]
        
        bm25.build_index(documents, metadatas)
        
        results = bm25.search("Python", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert all(r.search_method == "sparse" for r in results)
        
    def test_search_with_filters(self, sample_metadata):
        """Test BM25 search with metadata filters."""
        bm25 = BM25Search()
        
        documents = ["Test document"] * 3
        metadatas = [sample_metadata, sample_metadata, sample_metadata]
        
        bm25.build_index(documents, metadatas)
        
        results = bm25.search(
            "Test",
            top_k=10,
            filters={"organizations": ["Acme Corp"]}
        )
        
        assert len(results) > 0
        
    def test_add_documents(self, sample_metadata):
        """Test adding documents to index."""
        bm25 = BM25Search()
        
        # Build initial index
        documents = ["First document"]
        metadatas = [sample_metadata]
        bm25.build_index(documents, metadatas)
        
        # Add more documents
        new_documents = ["Second document", "Third document"]
        new_metadatas = [sample_metadata, sample_metadata]
        bm25.add_documents(new_documents, new_metadatas)
        
        assert bm25.get_index_size() == 3
        
    def test_remove_documents(self, sample_metadata):
        """Test removing documents from index."""
        bm25 = BM25Search()
        
        # Build index with unique chunk IDs
        metadata1 = sample_metadata.model_copy(update={"chunk_id": "chunk_1"})
        metadata2 = sample_metadata.model_copy(update={"chunk_id": "chunk_2"})
        metadata3 = sample_metadata.model_copy(update={"chunk_id": "chunk_3"})
        
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        metadatas = [metadata1, metadata2, metadata3]
        bm25.build_index(documents, metadatas)
        
        # Remove one document
        bm25.remove_documents(["chunk_2"])
        
        assert bm25.get_index_size() == 2
        
    def test_clear_index(self, sample_metadata):
        """Test clearing the index."""
        bm25 = BM25Search()
        
        documents = ["Test document"]
        metadatas = [sample_metadata]
        bm25.build_index(documents, metadatas)
        
        bm25.clear_index()
        
        assert bm25.index is None
        assert bm25.get_index_size() == 0


class TestMetadataFilter:
    """Tests for metadata filtering."""
    
    def test_apply_filters_no_filters(self, sample_chunks):
        """Test applying no filters."""
        filtered = MetadataFilter.apply_filters(sample_chunks, filters=None)
        assert len(filtered) == len(sample_chunks)
        
    def test_filter_by_organizations(self, sample_chunks):
        """Test filtering by organizations."""
        filtered = MetadataFilter.apply_filters(
            sample_chunks,
            filters={"organizations": ["Acme Corp"]}
        )
        assert len(filtered) > 0
        
    def test_filter_by_people(self, sample_chunks):
        """Test filtering by people."""
        filtered = MetadataFilter.apply_filters(
            sample_chunks,
            filters={"people": ["John Doe"]}
        )
        assert len(filtered) > 0
        
    def test_filter_by_file_type(self, sample_chunks):
        """Test filtering by file type."""
        filtered = MetadataFilter.apply_filters(
            sample_chunks,
            filters={"file_type": ["pdf"]}
        )
        assert len(filtered) > 0
        
    def test_filter_by_date_range(self, sample_chunks):
        """Test filtering by date range."""
        filtered = MetadataFilter.apply_filters(
            sample_chunks,
            filters={
                "date_range": {
                    "start": "2020-01-01",
                    "end": "2025-12-31"
                }
            }
        )
        assert len(filtered) > 0
        
    def test_filter_by_entity_type(self, sample_chunks):
        """Test filtering by specific entity type."""
        filtered = MetadataFilter.filter_by_entity_type(
            sample_chunks,
            entity_type="organizations",
            entity_values=["Acme Corp"]
        )
        assert len(filtered) > 0
        
    def test_filter_by_file_type_helper(self, sample_chunks):
        """Test file type filter helper."""
        filtered = MetadataFilter.filter_by_file_type(
            sample_chunks,
            file_types=["pdf"]
        )
        assert len(filtered) > 0
        
    def test_build_filter_summary(self):
        """Test building filter summary."""
        filters = {
            "organizations": ["Acme Corp"],
            "file_type": ["pdf"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
        }
        
        summary = MetadataFilter.build_filter_summary(filters)
        assert "organizations" in summary
        assert "Acme Corp" in summary


@pytest.mark.asyncio
class TestHybridSearch:
    """Tests for hybrid search with RRF."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            dense_weight=0.5,
            sparse_weight=0.3,
            metadata_weight=0.2
        )
        
    @pytest.fixture
    def mock_vector_store(self, sample_chunks):
        """Create mock vector store."""
        store = Mock()
        store.search = AsyncMock(return_value=sample_chunks)
        return store
        
    @pytest.fixture
    def mock_bm25(self, sample_chunks):
        """Create mock BM25 search."""
        bm25 = Mock()
        bm25.search = Mock(return_value=sample_chunks)
        return bm25
        
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = Mock()
        service.embed_query = AsyncMock(return_value=[0.1] * 1024)
        return service
        
    async def test_initialization(self, settings, mock_vector_store, mock_bm25):
        """Test hybrid search initialization."""
        hybrid = HybridSearch(mock_vector_store, mock_bm25, settings)
        
        assert hybrid.dense_weight == 0.5
        assert hybrid.sparse_weight == 0.3
        assert hybrid.metadata_weight == 0.2
        
    async def test_dense_search(self, settings, mock_vector_store, mock_bm25):
        """Test dense-only search."""
        hybrid = HybridSearch(mock_vector_store, mock_bm25, settings)
        
        results = await hybrid.search(
            query="test query",
            collection="test_collection",
            top_k=5,
            search_mode="dense"
        )
        
        assert len(results) > 0
        mock_vector_store.search.assert_called_once()
        
    async def test_sparse_search(self, settings, mock_vector_store, mock_bm25):
        """Test sparse-only search."""
        hybrid = HybridSearch(mock_vector_store, mock_bm25, settings)
        
        results = await hybrid.search(
            query="test query",
            collection="test_collection",
            top_k=5,
            search_mode="sparse"
        )
        
        assert len(results) > 0
        mock_bm25.search.assert_called_once()
        
    async def test_hybrid_search(self, settings, mock_vector_store, mock_bm25):
        """Test hybrid search with RRF."""
        hybrid = HybridSearch(mock_vector_store, mock_bm25, settings)
        
        results = await hybrid.search(
            query="test query",
            collection="test_collection",
            top_k=5,
            search_mode="hybrid"
        )
        
        assert len(results) > 0
        assert all(r.search_method == "hybrid" for r in results)
        
    async def test_update_weights(self, settings, mock_vector_store, mock_bm25):
        """Test updating RRF weights."""
        hybrid = HybridSearch(mock_vector_store, mock_bm25, settings)
        
        hybrid.update_weights(dense_weight=0.6, sparse_weight=0.4)
        
        assert hybrid.dense_weight == 0.6
        assert hybrid.sparse_weight == 0.4
        
    async def test_get_weights(self, settings, mock_vector_store, mock_bm25):
        """Test getting current weights."""
        hybrid = HybridSearch(mock_vector_store, mock_bm25, settings)
        
        weights = hybrid.get_weights()
        
        assert "dense_weight" in weights
        assert "sparse_weight" in weights
        assert "metadata_weight" in weights
