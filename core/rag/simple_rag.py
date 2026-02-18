"""
Simple RAG implementation for Visual RAG Document Explorer.

Baseline RAG strategy with retrieve → rerank → generate pipeline.
"""

import time
from typing import Optional

from core.rag.base import RAGStrategy
from config.models import QueryRequest, QueryResponse


class SimpleRAG(RAGStrategy):
    """
    Simple RAG: Basic retrieve → rerank → generate pipeline.
    
    Pipeline:
    1. Retrieve top_k*2 chunks using hybrid search
    2. Rerank to top_k using Cohere
    3. Generate answer with LLM
    4. Extract citations
    5. Return response
    
    This is the baseline strategy with minimal complexity and fastest response time.
    """
    
    @property
    def strategy_name(self) -> str:
        """Return the name of this RAG strategy."""
        return "simple"
    
    async def execute(self, request: QueryRequest) -> QueryResponse:
        """
        Execute simple RAG pipeline.
        
        Args:
            request: Query request with all parameters
            
        Returns:
            Complete query response with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve more chunks for reranking
        initial_k = request.top_k * 2
        retrieved_chunks = await self._retrieve(
            query=request.query,
            top_k=initial_k,
            filters=request.metadata_filters
        )
        
        # Step 2: Rerank if enabled
        if request.enable_reranking and retrieved_chunks:
            reranked_chunks = await self._rerank(
                query=request.query,
                chunks=retrieved_chunks,
                top_k=request.top_k
            )
        else:
            reranked_chunks = retrieved_chunks[:request.top_k]
        
        # Step 3: Generate answer
        if reranked_chunks:
            answer = await self._generate_answer(
                query=request.query,
                chunks=reranked_chunks
            )
        else:
            answer = "I couldn't find any relevant information to answer your query. Please try rephrasing your question or check if documents have been uploaded."
        
        # Step 4: Extract citations
        citations = self._extract_citations(reranked_chunks)
        
        # Step 5: Create placeholder grounding
        grounding = self._create_placeholder_grounding()
        
        response_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            mode="simple",
            search_mode=request.search_mode,
            sources=reranked_chunks,
            citations=citations,
            grounding=grounding,
            response_time_ms=response_time,
            hyde_used=False,  # Simple RAG doesn't use HYDE
            reranking_used=request.enable_reranking,
            compression_used=False,  # Simple RAG doesn't use compression
            initial_retrieval_count=len(retrieved_chunks),
            final_retrieval_count=len(reranked_chunks)
        )
