"""
Advanced RAG for Visual RAG Document Explorer.

Multi-query retrieval with Reciprocal Rank Fusion (RRF) for maximum recall.
"""

import time
from typing import Optional

from core.rag.base import RAGStrategy
from core.rag.query_decomposer import QueryDecomposer
from config.models import QueryRequest, QueryResponse, RetrievedChunk


class AdvancedRAG(RAGStrategy):
    """
    Advanced RAG: Multi-query retrieval with Reciprocal Rank Fusion.
    
    Pipeline:
    1. Generate 3-5 query variants using QueryDecomposer
    2. Retrieve independently for each variant
    3. Apply Reciprocal Rank Fusion (RRF) to merge results
    4. Rerank merged results
    5. Generate answer
    
    This strategy achieves highest recall by retrieving from multiple perspectives.
    """
    
    @property
    def strategy_name(self) -> str:
        """Return the name of this RAG strategy."""
        return "advanced"
    
    def __init__(self, *args, **kwargs):
        """Initialize Advanced RAG with query decomposer."""
        super().__init__(*args, **kwargs)
        self.query_decomposer = QueryDecomposer(self.settings)
    
    async def execute(self, request: QueryRequest) -> QueryResponse:
        """
        Execute Advanced RAG pipeline.
        
        Args:
            request: Query request with all parameters
            
        Returns:
            Complete query response with merged sources
        """
        start_time = time.time()
        
        # Step 1: Generate query variants
        query_variants = await self.query_decomposer.decompose(
            query=request.query,
            max_subqueries=5
        )
        
        # If only one variant (simple query), use it
        if len(query_variants) == 1:
            query_variants = [request.query]
        
        # Step 2: Retrieve for each variant
        all_retrievals = []
        for variant in query_variants:
            chunks = await self._retrieve(
                query=variant,
                top_k=request.top_k * 2,  # Retrieve more per variant
                filters=request.metadata_filters
            )
            all_retrievals.append(chunks)
        
        # Step 3: Apply Reciprocal Rank Fusion
        merged_chunks = self._reciprocal_rank_fusion(
            retrievals=all_retrievals,
            k=60  # RRF constant
        )
        
        # Step 4: Rerank merged results
        if request.enable_reranking and merged_chunks:
            reranked_chunks = await self._rerank(
                query=request.query,
                chunks=merged_chunks,
                top_k=request.top_k
            )
        else:
            reranked_chunks = merged_chunks[:request.top_k]
        
        # Step 5: Generate answer
        if reranked_chunks:
            answer = await self._generate_answer(
                query=request.query,
                chunks=reranked_chunks
            )
        else:
            answer = "I couldn't find relevant information to answer your query, even after trying multiple query variations."
        
        # Extract citations and create grounding
        citations = self._extract_citations(reranked_chunks)
        grounding = self._create_placeholder_grounding()
        
        response_time = (time.time() - start_time) * 1000
        
        # Calculate total initial retrievals
        total_initial = sum(len(r) for r in all_retrievals)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            mode="advanced",
            search_mode=request.search_mode,
            sources=reranked_chunks,
            citations=citations,
            grounding=grounding,
            response_time_ms=response_time,
            hyde_used=False,
            reranking_used=request.enable_reranking,
            compression_used=False,
            initial_retrieval_count=total_initial,
            final_retrieval_count=len(reranked_chunks)
        )
    
    def _reciprocal_rank_fusion(
        self,
        retrievals: list[list[RetrievedChunk]],
        k: int = 60
    ) -> list[RetrievedChunk]:
        """
        Apply Reciprocal Rank Fusion to merge multiple retrieval results.
        
        RRF formula: score(chunk) = sum(1 / (k + rank_i)) for all retrievals
        where rank_i is the rank of the chunk in retrieval i.
        
        Args:
            retrievals: List of retrieval results (one per query variant)
            k: RRF constant (default 60)
            
        Returns:
            Merged and ranked chunks
        """
        # Build RRF scores
        rrf_scores = {}
        chunk_map = {}
        
        for retrieval in retrievals:
            for rank, chunk in enumerate(retrieval, start=1):
                chunk_id = chunk.metadata.chunk_id
                
                # Store chunk
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk
                
                # Accumulate RRF score
                rrf_score = 1.0 / (k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score
        
        # Sort chunks by RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True
        )
        
        # Create merged list with updated scores
        merged_chunks = []
        for chunk_id in sorted_chunk_ids:
            chunk = chunk_map[chunk_id]
            # Update score to RRF score
            merged_chunk = RetrievedChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                score=rrf_scores[chunk_id],
                search_method=chunk.search_method
            )
            merged_chunks.append(merged_chunk)
        
        return merged_chunks
    
    async def retrieve_with_multi_query(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
        num_variants: int = 3
    ) -> list[RetrievedChunk]:
        """
        Retrieve using multiple query variants with RRF fusion.
        
        This is a convenience method for using Advanced RAG's multi-query
        retrieval without the full pipeline.
        
        Args:
            query: Original query
            top_k: Number of final results
            filters: Optional metadata filters
            num_variants: Number of query variants to generate
            
        Returns:
            Merged and ranked chunks
        """
        # Generate query variants
        query_variants = await self.query_decomposer.decompose(
            query=query,
            max_subqueries=num_variants
        )
        
        # Retrieve for each variant
        all_retrievals = []
        for variant in query_variants:
            chunks = await self._retrieve(
                query=variant,
                top_k=top_k * 2,
                filters=filters
            )
            all_retrievals.append(chunks)
        
        # Apply RRF
        merged_chunks = self._reciprocal_rank_fusion(all_retrievals)
        
        return merged_chunks[:top_k]
