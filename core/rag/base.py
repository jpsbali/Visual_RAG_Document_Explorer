"""
Base RAG interface for Visual RAG Document Explorer.

Provides abstract base class for all RAG strategies with common helper methods
for retrieval, reranking, and context formatting.
"""

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from config.models import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    Citation,
    GroundingResult,
    ClaimVerification
)
from config.settings import Settings
from core.search.hybrid_search import HybridSearch
from core.reranking.cohere_reranker import CohereReranker
from core.llm.llm_router import get_llm_provider
from core.embeddings.embedding_router import get_embedding_service


class RAGStrategy(ABC):
    """
    Abstract base class for all RAG strategies.
    
    Provides dependency injection for all required services and common helper
    methods for retrieval, reranking, and formatting. Subclasses must implement
    the execute() method to define their specific RAG pipeline.
    """
    
    def __init__(
        self,
        settings: Settings,
        hybrid_search: HybridSearch,
        reranker: CohereReranker
    ):
        """
        Initialize RAG strategy with required dependencies.
        
        Args:
            settings: Application settings
            hybrid_search: Hybrid search service for retrieval
            reranker: Cohere reranker for precision improvement
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        
        # Initialize LLM and embedding services
        self.llm = get_llm_provider(settings)
        self.embeddings = get_embedding_service(settings)
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this RAG strategy."""
        pass
    
    @abstractmethod
    async def execute(self, request: QueryRequest) -> QueryResponse:
        """
        Execute the RAG pipeline for the given query.
        
        Args:
            request: Query request with all parameters
            
        Returns:
            Complete query response with answer, sources, and metadata
        """
        pass
    
    async def _retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None
    ) -> list[RetrievedChunk]:
        """
        Common retrieval method using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            filters: Optional metadata filters
            
        Returns:
            List of retrieved chunks
        """
        return await self.hybrid_search.search(
            query=query,
            collection=self.settings.qdrant_collection if self.settings.default_vector_db == "qdrant" else self.settings.milvus_collection,
            top_k=top_k,
            search_mode=self.settings.search_mode,
            filters=filters
        )
    
    async def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int
    ) -> list[RetrievedChunk]:
        """
        Common reranking method using Cohere.
        
        Args:
            query: Original query
            chunks: Retrieved chunks to rerank
            top_k: Number of top results to keep
            
        Returns:
            Reranked chunks
        """
        if not chunks:
            return []
        
        return await self.reranker.rerank(
            query=query,
            chunks=chunks,
            top_k=top_k
        )
    
    def _format_sources_for_prompt(
        self,
        chunks: list[RetrievedChunk]
    ) -> str:
        """
        Format retrieved chunks for LLM prompt.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Formatted string for prompt
        """
        if not chunks:
            return "No sources available."
        
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(
                f"[Source {i}]\n"
                f"File: {chunk.metadata.source_file}\n"
                f"Page: {chunk.metadata.page_number or 'N/A'}\n"
                f"Content: {chunk.content}\n"
            )
        return "\n\n".join(formatted)
    
    def _extract_citations(
        self,
        chunks: list[RetrievedChunk]
    ) -> list[Citation]:
        """
        Extract citations from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            List of citations
        """
        citations = []
        for chunk in chunks:
            # Extract first 200 chars as relevant text
            relevant_text = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            
            citation = Citation(
                source_file=chunk.metadata.source_file,
                page_number=chunk.metadata.page_number,
                chunk_index=chunk.metadata.chunk_index,
                relevant_text=relevant_text,
                relevance_score=chunk.score
            )
            citations.append(citation)
        
        return citations
    
    def _create_placeholder_grounding(self) -> GroundingResult:
        """
        Create a placeholder grounding result.
        
        This is used by strategies that don't perform grounding verification.
        The GroundingVerifier component can be used separately to enhance this.
        
        Returns:
            Placeholder grounding result
        """
        return GroundingResult(
            grounding_score=1.0,
            total_claims=0,
            grounded_claims=0,
            partially_grounded_claims=0,
            ungrounded_claims=0,
            claim_details=[],
            verified_at=datetime.now()
        )
    
    async def _generate_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User query
            chunks: Retrieved chunks for context
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated answer
        """
        if not system_prompt:
            system_prompt = """You are a helpful document assistant. Answer the user query based ONLY on the provided source documents.

Rules:
1. Only use information from the provided sources
2. If the sources don't contain enough information, say so
3. Cite sources using [Source N] notation
4. Be concise but comprehensive
5. Do not make up information"""
        
        sources_text = self._format_sources_for_prompt(chunks)
        
        user_prompt = f"""Sources:
{sources_text}

Query: {query}

Answer:"""
        
        response = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens
        )
        
        return response
