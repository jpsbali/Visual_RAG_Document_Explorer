"""
Hybrid search combining dense, sparse, and metadata filtering.

Uses Reciprocal Rank Fusion (RRF) to merge results from multiple search methods.
"""

from config.models import RetrievedChunk
from config.settings import Settings
from core.vectordb.base import VectorStoreBase
from core.embeddings.embedding_router import get_embedding_service
from .bm25_search import BM25Search
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Hybrid search combining dense, sparse, and metadata filtering.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results from:
    1. Dense vector search (semantic similarity)
    2. BM25 sparse search (keyword matching)
    3. Metadata filtering (entity-based)
    """
    
    def __init__(
        self,
        vector_store: VectorStoreBase,
        bm25_search: BM25Search,
        settings: Settings
    ):
        """
        Initialize hybrid search.
        
        Args:
            vector_store: Vector database instance
            bm25_search: BM25 search instance
            settings: Application settings
        """
        self.vector_store = vector_store
        self.bm25_search = bm25_search
        self.settings = settings
        self.embedding_service = get_embedding_service(settings)
        
        # RRF weights from settings
        self.dense_weight = settings.dense_weight
        self.sparse_weight = settings.sparse_weight
        self.metadata_weight = settings.metadata_weight
        
        logger.info(
            f"Initialized HybridSearch (dense={self.dense_weight}, "
            f"sparse={self.sparse_weight}, metadata={self.metadata_weight})"
        )
        
    async def search(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        search_mode: str = "hybrid",
        filters: Optional[dict] = None
    ) -> list[RetrievedChunk]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            collection: Vector DB collection name
            top_k: Number of final results
            search_mode: "dense", "sparse", or "hybrid"
            filters: Optional metadata filters
            
        Returns:
            List of RetrievedChunk objects ranked by RRF score
        """
        if search_mode == "dense":
            return await self._dense_search(query, collection, top_k, filters)
        elif search_mode == "sparse":
            return self._sparse_search(query, top_k, filters)
        else:  # hybrid
            return await self._hybrid_search(query, collection, top_k, filters)
            
    async def _dense_search(
        self,
        query: str,
        collection: str,
        top_k: int,
        filters: Optional[dict]
    ) -> list[RetrievedChunk]:
        """
        Dense vector search only.
        
        Args:
            query: Search query
            collection: Vector DB collection name
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of retrieved chunks
        """
        query_embedding = await self.embedding_service.embed_query(query)
        
        results = await self.vector_store.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        logger.info(f"Dense search returned {len(results)} results")
        return results
        
    def _sparse_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict]
    ) -> list[RetrievedChunk]:
        """
        BM25 sparse search only.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of retrieved chunks
        """
        results = self.bm25_search.search(
            query=query,
            top_k=top_k,
            filters=filters
        )
        
        logger.info(f"Sparse search returned {len(results)} results")
        return results
        
    async def _hybrid_search(
        self,
        query: str,
        collection: str,
        top_k: int,
        filters: Optional[dict]
    ) -> list[RetrievedChunk]:
        """
        Hybrid search with RRF fusion.
        
        Retrieves from both dense and sparse, then fuses using RRF.
        
        Args:
            query: Search query
            collection: Vector DB collection name
            top_k: Number of final results
            filters: Optional metadata filters
            
        Returns:
            Fused and reranked results
        """
        # Retrieve more candidates for fusion
        retrieval_k = top_k * 3
        
        # Dense search
        dense_results = await self._dense_search(
            query, collection, retrieval_k, filters
        )
        
        # Sparse search
        sparse_results = self._sparse_search(
            query, retrieval_k, filters
        )
        
        # Apply RRF fusion
        fused_results = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k
        )
        
        logger.info(
            f"Hybrid search: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse → {len(fused_results)} fused"
        )
        
        return fused_results
        
    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        top_k: int,
        k: int = 60  # RRF constant
    ) -> list[RetrievedChunk]:
        """
        Merge results using Reciprocal Rank Fusion.
        
        RRF formula: score = sum(1 / (k + rank_i)) for each result list
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            top_k: Number of final results
            k: RRF constant (default 60)
            
        Returns:
            Fused and reranked results
        """
        # Build chunk_id -> RetrievedChunk mapping
        chunk_map = {}
        rrf_scores = {}
        
        # Process dense results
        for rank, chunk in enumerate(dense_results, start=1):
            chunk_id = chunk.metadata.chunk_id
            chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] = self.dense_weight / (k + rank)
            
        # Process sparse results
        for rank, chunk in enumerate(sparse_results, start=1):
            chunk_id = chunk.metadata.chunk_id
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk
                rrf_scores[chunk_id] = 0
            rrf_scores[chunk_id] += self.sparse_weight / (k + rank)
            
        # Sort by RRF score
        sorted_chunks = sorted(
            chunk_map.items(),
            key=lambda x: rrf_scores[x[0]],
            reverse=True
        )[:top_k]
        
        # Build final results with RRF scores
        results = []
        for chunk_id, chunk in sorted_chunks:
            # Create a new chunk with updated score and search method
            chunk.score = rrf_scores[chunk_id]
            chunk.search_method = "hybrid"
            results.append(chunk)
            
        logger.info(
            f"RRF fusion: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse → {len(results)} final"
        )
        
        return results
        
    def update_weights(
        self,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        metadata_weight: Optional[float] = None
    ) -> None:
        """
        Update RRF fusion weights.
        
        Args:
            dense_weight: Weight for dense search results
            sparse_weight: Weight for sparse search results
            metadata_weight: Weight for metadata-filtered results
        """
        if dense_weight is not None:
            self.dense_weight = dense_weight
        if sparse_weight is not None:
            self.sparse_weight = sparse_weight
        if metadata_weight is not None:
            self.metadata_weight = metadata_weight
            
        logger.info(
            f"Updated RRF weights: dense={self.dense_weight}, "
            f"sparse={self.sparse_weight}, metadata={self.metadata_weight}"
        )
        
    def get_weights(self) -> dict[str, float]:
        """
        Get current RRF fusion weights.
        
        Returns:
            Dict of weight names to values
        """
        return {
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "metadata_weight": self.metadata_weight
        }
