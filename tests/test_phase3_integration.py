"""
Phase 3 Integration Tests - End-to-end RAG pipeline.

Tests the complete integration of Phase 1B → Phase 2 → Phase 3:
- Document processing (Phase 1B)
- Vector storage and search (Phase 2)
- RAG strategies (Phase 3)
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
from core.rag.corrective_rag import CorrectiveRAG
from core.rag.self_reflective_rag import SelfReflectiveRAG
from core.rag.advanced_rag import AdvancedRAG
from core.rag.rag_router import RAGRouter, get_rag_strategy, execute_rag_query
from core.rag.query_decomposer import QueryDecomposer
from core.rag.hyde import HYDEGenerator
from core.rag.contextual_compressor import ContextualCompressor
from core.rag.grounding_verifier import GroundingVerifier


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        openai_api_key="test-key",
        cohere_api_key="test-key",
        voyage_api_key="test-key",
        default_llm_provider="openai",
        default_embedding_model="voyage",
        default_rag_strategy="auto",
        temperature=0.0,
        max_tokens=2048,
        srag_max_iterations=3
    )


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
            content=f"This is test content for chunk {i}. Apple Inc. was founded in 1976 by Steve Jobs. The company is headquartered in Cupertino, California.",
            metadata=metadata,
            score=0.9 - (i * 0.1),
            search_method="hybrid"
        )
        chunks.append(chunk)
    
    return chunks


@pytest.mark.asyncio
async def test_rag_router_initialization(settings):
    """Test RAG router initialization with all strategies."""
    mock_search = Mock()
    mock_reranker = Mock()
    
    router = RAGRouter(settings, mock_search, mock_reranker)
    
    assert "simple" in router.strategies
    assert "crag" in router.strategies
    assert "srag" in router.strategies
    assert "advanced" in router.strategies
    assert isinstance(router.strategies["simple"], SimpleRAG)
    assert isinstance(router.strategies["crag"], CorrectiveRAG)
    assert isinstance(router.strategies["srag"], SelfReflectiveRAG)
    assert isinstance(router.strategies["advanced"], AdvancedRAG)


@pytest.mark.asyncio
async def test_rag_router_explicit_routing(settings):
    """Test explicit strategy routing."""
    mock_search = Mock()
    mock_reranker = Mock()
    
    router = RAGRouter(settings, mock_search, mock_reranker)
    
    # Test each mode
    for mode in ["simple", "crag", "srag", "advanced"]:
        request = QueryRequest(query="Test query", mode=mode)
        strategy = await router.route(request)
        assert strategy.strategy_name == mode


@pytest.mark.asyncio
async def test_rag_router_auto_routing(settings):
    """Test automatic strategy routing based on query complexity."""
    mock_search = Mock()
    mock_reranker = Mock()
    
    router = RAGRouter(settings, mock_search, mock_reranker)
    
    # Mock LLM response for auto routing
    with patch.object(router, '_analyze_query_complexity', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = "advanced"
        
        request = QueryRequest(query="Complex multi-aspect query", mode="auto")
        strategy = await router.route(request)
        
        assert strategy.strategy_name == "advanced"
        mock_analyze.assert_called_once_with("Complex multi-aspect query")


@pytest.mark.asyncio
async def test_get_rag_strategy_factory(settings):
    """Test RAG strategy factory function."""
    mock_search = Mock()
    mock_reranker = Mock()
    
    # Test each strategy
    for mode in ["simple", "crag", "srag", "advanced"]:
        strategy = get_rag_strategy(settings, mock_search, mock_reranker, mode)
        assert strategy.strategy_name == mode


@pytest.mark.asyncio
async def test_query_decomposer_simple_query(settings):
    """Test query decomposer with simple query."""
    decomposer = QueryDecomposer(settings)
    
    with patch.object(decomposer.llm, 'generate', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = '["What is the capital of France?"]'
        
        subqueries = await decomposer.decompose("What is the capital of France?")
        
        assert len(subqueries) == 1
        assert subqueries[0] == "What is the capital of France?"


@pytest.mark.asyncio
async def test_query_decomposer_complex_query(settings):
    """Test query decomposer with complex query."""
    decomposer = QueryDecomposer(settings)
    
    with patch.object(decomposer.llm, 'generate', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = '''[
            "What was Apple's revenue in 2023?",
            "What was Microsoft's revenue in 2023?",
            "What was Apple's market share in 2023?"
        ]'''
        
        subqueries = await decomposer.decompose(
            "Compare Apple and Microsoft revenue and market share in 2023"
        )
        
        assert len(subqueries) == 3
        assert "Apple's revenue" in subqueries[0]
        assert "Microsoft's revenue" in subqueries[1]


@pytest.mark.asyncio
async def test_hyde_generator(settings, sample_chunks):
    """Test HYDE hypothetical document generation."""
    mock_search = Mock()
    mock_search.search = AsyncMock(return_value=sample_chunks)
    
    hyde = HYDEGenerator(settings, mock_search)
    
    with patch.object(hyde.llm, 'generate', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = """1. Apple Inc. is a technology company founded in 1976.
2. The company specializes in consumer electronics and software.
3. Apple is known for products like iPhone, iPad, and Mac."""
        
        hypothetical_docs = await hyde.generate_hypothetical_documents(
            "Tell me about Apple Inc.",
            num_documents=3
        )
        
        assert len(hypothetical_docs) > 0
        assert len(hypothetical_docs) <= 3


@pytest.mark.asyncio
async def test_contextual_compressor(settings, sample_chunks):
    """Test contextual compression."""
    compressor = ContextualCompressor(settings)
    
    with patch.object(compressor.llm, 'generate', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = "Apple Inc. was founded in 1976 by Steve Jobs."
        
        compressed = await compressor.compress_chunk(
            query="When was Apple founded?",
            chunk=sample_chunks[0]
        )
        
        assert compressed is not None
        assert len(compressed.content) <= len(sample_chunks[0].content)


@pytest.mark.asyncio
async def test_grounding_verifier(settings, sample_chunks):
    """Test grounding verification."""
    verifier = GroundingVerifier(settings)
    
    answer = "Apple Inc. was founded in 1976 by Steve Jobs in Cupertino, California."
    
    with patch.object(verifier.llm, 'generate', new_callable=AsyncMock) as mock_generate:
        # Mock claim extraction
        mock_generate.side_effect = [
            '["Apple Inc. was founded in 1976", "Apple was founded by Steve Jobs"]',
            '{"status": "grounded", "supporting_chunks": ["chunk_0"], "confidence": 0.95}',
            '{"status": "grounded", "supporting_chunks": ["chunk_0"], "confidence": 0.90}'
        ]
        
        grounding_result = await verifier.verify_answer(answer, sample_chunks)
        
        assert grounding_result.total_claims >= 0
        assert 0.0 <= grounding_result.grounding_score <= 1.0


@pytest.mark.asyncio
async def test_end_to_end_simple_rag(settings, sample_chunks):
    """Test end-to-end Simple RAG pipeline."""
    mock_search = Mock()
    mock_search.search = AsyncMock(return_value=sample_chunks)
    
    mock_reranker = Mock()
    mock_reranker.rerank = AsyncMock(return_value=sample_chunks[:3])
    
    rag = SimpleRAG(settings, mock_search, mock_reranker)
    
    with patch.object(rag, '_generate_answer', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = "Apple Inc. was founded in 1976 by Steve Jobs."
        
        request = QueryRequest(
            query="When was Apple founded?",
            mode="simple",
            top_k=3
        )
        
        response = await rag.execute(request)
        
        assert isinstance(response, QueryResponse)
        assert response.mode == "simple"
        assert len(response.sources) == 3
        assert response.answer == "Apple Inc. was founded in 1976 by Steve Jobs."


@pytest.mark.asyncio
async def test_end_to_end_advanced_rag(settings, sample_chunks):
    """Test end-to-end Advanced RAG with multi-query."""
    mock_search = Mock()
    mock_search.search = AsyncMock(return_value=sample_chunks)
    
    mock_reranker = Mock()
    mock_reranker.rerank = AsyncMock(return_value=sample_chunks[:3])
    
    rag = AdvancedRAG(settings, mock_search, mock_reranker)
    
    # Mock query decomposition
    with patch.object(rag.query_decomposer, 'decompose', new_callable=AsyncMock) as mock_decompose:
        mock_decompose.return_value = [
            "When was Apple founded?",
            "Who founded Apple?",
            "Where is Apple headquartered?"
        ]
        
        with patch.object(rag, '_generate_answer', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "Apple Inc. was founded in 1976 by Steve Jobs in Cupertino."
            
            request = QueryRequest(
                query="Tell me about Apple's founding",
                mode="advanced",
                top_k=3
            )
            
            response = await rag.execute(request)
            
            assert isinstance(response, QueryResponse)
            assert response.mode == "advanced"
            assert len(response.sources) <= 3


@pytest.mark.asyncio
async def test_execute_rag_query_convenience_function(settings, sample_chunks):
    """Test convenience function for RAG query execution."""
    mock_search = Mock()
    mock_search.search = AsyncMock(return_value=sample_chunks)
    
    mock_reranker = Mock()
    mock_reranker.rerank = AsyncMock(return_value=sample_chunks[:3])
    
    request = QueryRequest(
        query="Test query",
        mode="simple",
        top_k=3
    )
    
    with patch('core.rag.rag_router.RAGRouter') as MockRouter:
        mock_router_instance = Mock()
        mock_strategy = Mock()
        mock_strategy.execute = AsyncMock(return_value=Mock(spec=QueryResponse))
        mock_router_instance.route = AsyncMock(return_value=mock_strategy)
        MockRouter.return_value = mock_router_instance
        
        response = await execute_rag_query(request, settings, mock_search, mock_reranker)
        
        MockRouter.assert_called_once()
        mock_router_instance.route.assert_called_once_with(request)
        mock_strategy.execute.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_phase_integration_document_to_answer(settings, sample_chunks):
    """Test complete pipeline from document processing to answer generation."""
    # This test simulates the full pipeline:
    # 1. Document is processed (Phase 1B)
    # 2. Chunks are stored and indexed (Phase 2)
    # 3. Query is executed with RAG (Phase 3)
    
    mock_search = Mock()
    mock_search.search = AsyncMock(return_value=sample_chunks)
    
    mock_reranker = Mock()
    mock_reranker.rerank = AsyncMock(return_value=sample_chunks[:3])
    
    # Execute RAG query
    rag = SimpleRAG(settings, mock_search, mock_reranker)
    
    with patch.object(rag, '_generate_answer', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = "Based on the documents, Apple Inc. was founded in 1976."
        
        request = QueryRequest(
            query="When was Apple founded?",
            mode="simple",
            search_mode="hybrid",
            top_k=3,
            enable_reranking=True
        )
        
        response = await rag.execute(request)
        
        # Verify complete pipeline
        assert response.query == request.query
        assert response.mode == "simple"
        assert response.search_mode == "hybrid"
        assert len(response.sources) == 3
        assert len(response.citations) == 3
        assert response.reranking_used is True
        assert response.response_time_ms > 0
        
        # Verify Phase 2 integration (search and reranking)
        mock_search.search.assert_called_once()
        mock_reranker.rerank.assert_called_once()
        
        # Verify Phase 1B integration (chunks have proper metadata)
        for source in response.sources:
            assert source.metadata.chunk_id is not None
            assert source.metadata.source_file is not None
            assert source.metadata.chunk_index >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
