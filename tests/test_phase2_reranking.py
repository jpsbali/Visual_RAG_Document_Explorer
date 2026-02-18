"""
Tests for Phase 2 Cohere reranking.
"""

import pytest
from datetime import datetime
from config.settings import Settings
from config.models import ChunkMetadata, NEREntities, RetrievedChunk
from core.reranking.cohere_reranker import CohereReranker
from unittest.mock import Mock, patch


@pytest.fixture
def settings():
    """Create test settings with Cohere API key."""
    return Settings(cohere_api_key="test_api_key")


@pytest.fixture
def sample_chunks():
    """Create sample retrieved chunks."""
    metadata = ChunkMetadata(
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
    
    return [
        RetrievedChunk(
            content="Python is a programming language",
            metadata=metadata,
            score=0.8,
            search_method="dense"
        ),
        RetrievedChunk(
            content="Java is also a programming language",
            metadata=metadata,
            score=0.7,
            search_method="dense"
        ),
        RetrievedChunk(
            content="Machine learning uses Python",
            metadata=metadata,
            score=0.6,
            search_method="dense"
        )
    ]


class TestCohereReranker:
    """Tests for Cohere reranker."""
    
    def test_initialization(self, settings):
        """Test reranker initialization."""
        reranker = CohereReranker(settings)
        assert reranker.model == "rerank-english-v3.0"
        
    def test_initialization_no_api_key(self):
        """Test initialization without API key."""
        settings = Settings(cohere_api_key="")
        with pytest.raises(ValueError):
            CohereReranker(settings)
            
    @patch('cohere.Client')
    @pytest.mark.asyncio
    async def test_rerank(self, mock_client, settings, sample_chunks):
        """Test reranking chunks."""
        # Mock Cohere API response
        mock_result = Mock()
        mock_result.index = 0
        mock_result.relevance_score = 0.95
        
        mock_response = Mock()
        mock_response.results = [mock_result]
        
        mock_client_instance = Mock()
        mock_client_instance.rerank = Mock(return_value=mock_response)
        mock_client.return_value = mock_client_instance
        
        reranker = CohereReranker(settings)
        reranker.client = mock_client_instance
        
        results = await reranker.rerank(
            query="Python programming",
            chunks=sample_chunks,
            top_k=1
        )
        
        assert len(results) == 1
        assert results[0].score == 0.95
        
    @patch('cohere.Client')
    @pytest.mark.asyncio
    async def test_rerank_with_threshold(self, mock_client, settings, sample_chunks):
        """Test reranking with score threshold."""
        # Mock Cohere API response with low score
        mock_result = Mock()
        mock_result.index = 0
        mock_result.relevance_score = 0.3
        
        mock_response = Mock()
        mock_response.results = [mock_result]
        
        mock_client_instance = Mock()
        mock_client_instance.rerank = Mock(return_value=mock_response)
        mock_client.return_value = mock_client_instance
        
        reranker = CohereReranker(settings)
        reranker.client = mock_client_instance
        
        results = await reranker.rerank(
            query="Python programming",
            chunks=sample_chunks,
            score_threshold=0.5
        )
        
        # Should filter out low-scoring results
        assert len(results) == 0
        
    @pytest.mark.asyncio
    async def test_rerank_empty_chunks(self, settings):
        """Test reranking with empty chunks."""
        reranker = CohereReranker(settings)
        
        results = await reranker.rerank(
            query="test query",
            chunks=[],
            top_k=5
        )
        
        assert len(results) == 0
        
    @patch('cohere.Client')
    def test_rerank_sync(self, mock_client, settings, sample_chunks):
        """Test synchronous reranking."""
        # Mock Cohere API response
        mock_result = Mock()
        mock_result.index = 0
        mock_result.relevance_score = 0.95
        
        mock_response = Mock()
        mock_response.results = [mock_result]
        
        mock_client_instance = Mock()
        mock_client_instance.rerank = Mock(return_value=mock_response)
        mock_client.return_value = mock_client_instance
        
        reranker = CohereReranker(settings)
        reranker.client = mock_client_instance
        
        results = reranker.rerank_sync(
            query="Python programming",
            chunks=sample_chunks,
            top_k=1
        )
        
        assert len(results) == 1
        
    def test_get_model_info(self, settings):
        """Test getting model info."""
        reranker = CohereReranker(settings)
        info = reranker.get_model_info()
        
        assert info["model"] == "rerank-english-v3.0"
        assert info["provider"] == "cohere"
        assert info["type"] == "cross-encoder"
        
    def test_set_model(self, settings):
        """Test setting model."""
        reranker = CohereReranker(settings)
        reranker.set_model("rerank-multilingual-v3.0")
        
        assert reranker.model == "rerank-multilingual-v3.0"
