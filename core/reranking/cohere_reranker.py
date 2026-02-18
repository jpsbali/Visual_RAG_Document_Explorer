"""
Cohere cross-encoder reranking service.

Applies rerank-v3.5 model to reorder retrieved chunks based on query-document relevance.
"""

import cohere
from config.models import RetrievedChunk
from config.settings import Settings
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CohereReranker:
    """
    Cohere cross-encoder reranking service.
    
    Applies rerank-v3.5 model to reorder retrieved chunks
    based on query-document relevance.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize Cohere reranker.
        
        Args:
            settings: Application settings with Cohere API key
            
        Raises:
            ValueError: If Cohere API key is not configured
        """
        self.settings = settings
        
        if not settings.cohere_api_key:
            raise ValueError("Cohere API key not configured")
            
        self.client = cohere.Client(api_key=settings.cohere_api_key)
        self.model = "rerank-english-v3.0"  # or rerank-multilingual-v3.0
        
        logger.info(f"Initialized CohereReranker with model: {self.model}")
        
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks using Cohere reranker.
        
        Args:
            query: Original search query
            chunks: Retrieved chunks to rerank
            top_k: Number of top results to return (None = all)
            score_threshold: Minimum relevance score (0.0-1.0)
            
        Returns:
            Reranked chunks with updated scores
        """
        if not chunks:
            logger.warning("No chunks to rerank")
            return []
            
        # Extract documents for reranking
        documents = [chunk.content for chunk in chunks]
        
        # Call Cohere rerank API
        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k if top_k else len(documents),
                return_documents=False  # We already have the documents
            )
            
            # Build reranked results
            reranked = []
            for result in response.results:
                idx = result.index
                relevance_score = result.relevance_score
                
                # Apply score threshold
                if relevance_score < score_threshold:
                    continue
                    
                # Update chunk with reranker score
                chunk = chunks[idx]
                chunk.score = relevance_score
                reranked.append(chunk)
                
            logger.info(
                f"Reranked {len(chunks)} → {len(reranked)} chunks "
                f"(threshold: {score_threshold})"
            )
            
            return reranked
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            # Fallback: return original chunks
            return chunks[:top_k] if top_k else chunks
            
    def rerank_sync(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0
    ) -> list[RetrievedChunk]:
        """
        Synchronous version of rerank for non-async contexts.
        
        Args:
            query: Original search query
            chunks: Retrieved chunks to rerank
            top_k: Number of top results to return (None = all)
            score_threshold: Minimum relevance score (0.0-1.0)
            
        Returns:
            Reranked chunks with updated scores
        """
        if not chunks:
            logger.warning("No chunks to rerank")
            return []
            
        documents = [chunk.content for chunk in chunks]
        
        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k if top_k else len(documents),
                return_documents=False
            )
            
            reranked = []
            for result in response.results:
                idx = result.index
                relevance_score = result.relevance_score
                
                if relevance_score < score_threshold:
                    continue
                    
                chunk = chunks[idx]
                chunk.score = relevance_score
                reranked.append(chunk)
                
            logger.info(
                f"Reranked {len(chunks)} → {len(reranked)} chunks "
                f"(threshold: {score_threshold})"
            )
            
            return reranked
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            return chunks[:top_k] if top_k else chunks
            
    def get_model_info(self) -> dict:
        """
        Get information about the reranker model.
        
        Returns:
            Dict with model information
        """
        return {
            "model": self.model,
            "provider": "cohere",
            "type": "cross-encoder"
        }
        
    def set_model(self, model: str) -> None:
        """
        Set the reranker model.
        
        Args:
            model: Model name (e.g., "rerank-english-v3.0", "rerank-multilingual-v3.0")
        """
        self.model = model
        logger.info(f"Switched to Cohere model: {model}")
