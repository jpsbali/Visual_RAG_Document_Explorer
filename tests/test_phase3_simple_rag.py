"""
Tests for Phase 3 Simple RAG implementation.

Tests the baseline RAG strategy with retrieve → rerank → generate pipeline.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from config.settings import Settings
from config.models import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    ChunkMetadata,
    NEREntities
)
from core.rag.simple_rag import SimpleRAG
from core.search.hybrid_search import HybridSearch
from core.reranking.cohere_reranker import CohereReranker


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        openai_api_key="test-key",
        cohere_api_key="test-key",
        voyage_api_key="test-key",
        default_llm_provider="openai",
        default_embedding_model="voyage",
        temperature=0.0,
        max_tokens=2048
    )


@pytest.fixture
def mock_hybrid_search():
    """Create mock hybrid search."""
    search = Mock(spec=HybridSearch)
    search.search = AsyncMock()
    return search


@pytest.fixture
def mock_reranker():
    """Create mock reranker."""
    reranker = Mock(spec=CohereReranker)
    reranker.rerank = AsyncMock()
    return reranker


@pytest.fixture
def sample_chunks():
    """Create sample retrieved chunks."""
    chunks = []
    for i in range(5):
        metadata = ChunkMetadata(
            chunk_id=f"chunk_{i}",
            source_file=f"document_{i % 2}.pdf",
            file_type="pdf",
            page_number=i + 1,
            chunk_index=i,
            total_chunks=10,
            chunk_type="content",
            chunk_method="recursive",
            chunk_size=512,
            token_count=128,
            char_count=512,
            content_hash=f"hash_{i}",
            content_preview=f"Preview {i}",
            entities=NEREntities(),
            created_at=datetime.now(),
            processed_at=datetime.now()
        )
        
        chunk = RetrievedChunk(
            content=f"This is test content for chunk {i}. It contains relevant information about the query.",
            metadata=metadata,
            score=0.9 - (i * 0.1),
            search_method="hybrid"
        )
        chunks.append(chunk)
    
    return chunks


@pytest.mark.asyncio
async def test_simple_rag_initialization(settings, mock_hybrid_search, mock_reranker):
    """Test Simple RAG initialization."""
    rag = SimpleRAG(settings, mock_hybrid_search, mock_reranker)
    
    assert rag.strategy_name == "simple"
    assert rag.settings == settings
    assert rag.hybrid_search == mock_hybrid_search
    assert rag.reranker == mock_reranker


@pytest.mark.asyncio
async def test_simple_rag_execute_success(
    settings,
    mock_hybrid_search,
    mock_reranker,
    sample_chunks
):
    """Test successful Simple RAG execution."""
    # Setup mocks
    mock_hybrid_search.search.return_value = sample_chunks
    mock_reranker.rerank.return_value = sample_chunks[:3]
    
    # Create RAG instance
    rag = SimpleRAG(settings, mock_hybrid_search, mock_reranker)
    
    # Mock LLM generation
    with patch.object(rag, '_generate_answer', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = "This is a test answer based on the retrieved sources."
        
        # Create request
        request = QueryRequest(
            query="What is the test query?",
            mode="simple",
            search_mode="hybrid",
            top_k=3,
            enable_reranking=True
        )
        
        # Execute
        response = await rag.execute(request)
        
        # Verify response
        assert isinstance(response, QueryResponse)
        assert response.query == request.query
        assert response.answer == "This is a test answer based on the retrieved sources."
        assert response.mode == "simple"
        assert len(response.sources) == 3
        assert len(response.citations) == 3
        assert response.response_time_ms > 0
        assert response.reranking_used is True
        assert response.compression_used is False
        
        # Verify mock calls
        mock_hybrid_search.search.assert_called_once()
        mock_reranker.rerank.assert_called_once()
        mock_generate.assert_called_once()


@pytest.mark.asyncio
async def test_simple_rag_no_reranking(
    settings,
    mock_hybrid_search,
    mock_reranker,
    sample_chunks
):
    """Test Simple RAG without reranking."""
    mock_hybrid_search.search.return_value = sample_chunks
    
    rag = SimpleRAG(settings, mock_hybrid_search, mock_reranker)
    
    with patch.object(rag, '_generate_answer', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = "Test answer"
        
        request = QueryRequest(
            query="Test query",
            mode="simple",
            top_k=3,
            enable_reranking=False
        )
        
        response = await rag.execute(request)
        
        assert response.reranking_used is False
        assert len(response.sources) == 3
        mock_reranker.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_simple_rag_no_results(
    settings,
    mock_hybrid_search,
    mock_reranker
):
    """Test Simple RAG with no retrieval results."""
    mock_hybrid_search.search.return_value = []
    
    rag = SimpleRAG(settings, mock_hybrid_search, mock_reranker)
    
    request = QueryRequest(
        query="Test query",
        mode="simple",
        top_k=3
    )
    
    response = await rag.execute(request)
    
    assert "couldn't find any relevant information" in response.answer.lower()
    assert len(response.sources) == 0
    assert len(response.citations) == 0


@pytest.mark.asyncio
async def test_simple_rag_with_filters(
    settings,
    mock_hybrid_search,
    mock_reranker,
    sample_chunks
):
    """Test Simple RAG with metadata filters."""
    mock_hybrid_search.search.return_value = sample_chunks
    mock_reranker.rerank.return_value = sample_chunks[:3]
    
    rag = SimpleRAG(settings, mock_hybrid_search, mock_reranker)
    
    with patch.object(rag, '_generate_answer', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = "Test answer"
        
        filters = {"organizations": ["Acme Corp"]}
        request = QueryRequest(
            query="Test query",
            mode="simple",
            top_k=3,
            metadata_filters=filters
        )
        
        response = await rag.execute(request)
        
        # Verify filters were passed to search
        call_args = mock_hybrid_search.search.call_args
        assert call_args[1]["filters"] == filters


@pytest.mark.asyncio
async def test_simple_rag_citations_extraction(
    settings,
    mock_hybrid_search,
    mock_reranker,
    sample_chunks
):
    """Test citation extraction in Simple RAG."""
    mock_hybrid_search.search.return_value = sample_chunks
    mock_reranker.rerank.return_value = sample_chunks[:3]
    
    rag = SimpleRAG(settings, mock_hybrid_search, mock_reranker)
    
    with patch.object(rag, '_generate_answer', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = "Test answer"
        
        request = QueryRequest(
            query="Test query",
            mode="simple",
            top_k=3
        )
        
        response = await rag.execute(request)
        
        # Verify citations
        assert len(response.citations) == 3
        for i, citation in enumerate(response.citations):
            assert citation.source_file == sample_chunks[i].metadata.source_file
            assert citation.page_number == sample_chunks[i].metadata.page_number
            assert citation.chunk_index == sample_chunks[i].metadata.chunk_index
            assert citation.relevance_score == sample_chunks[i].score


@pytest.mark.asyncio
async def test_simple_rag_performance(
    settings,
    mock_hybrid_search,
    mock_reranker,
    sample_chunks
):
    """Test Simple RAG performance metrics."""
    mock_hybrid_search.search.return_value = sample_chunks
    mock_reranker.rerank.return_value = sample_chunks[:3]
    
    rag = SimpleRAG(settings, mock_hybrid_search, mock_reranker)
    
    with patch.object(rag, '_generate_answer', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = "Test answer"
        
        request = QueryRequest(
            query="Test query",
            mode="simple",
            top_k=3
        )
        
        response = await rag.execute(request)
        
        # Verify performance metrics
        assert response.response_time_ms > 0
        assert response.response_time_ms < 10000  # Should be under 10 seconds
        assert response.initial_retrieval_count == len(sample_chunks)
        assert response.final_retrieval_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
