"""
Reranking Node - Rerank retrieved documents using Cohere.

This node uses the CohereReranker from Phase 2 to improve
the relevance ranking of retrieved documents.
"""

from typing import Any
from agents.state import AgentState
import logging

logger = logging.getLogger(__name__)


async def reranking_node(state: AgentState) -> dict[str, Any]:
    """
    Rerank retrieved documents using Cohere.
    
    Reduces from retrieved_docs to top_k final chunks.
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with reranked_docs
    """
    try:
        # Skip if reranking disabled
        if not state.get("enable_reranking", True):
            # Just take top_k
            top_k = state.get("top_k", 5)
            return {"reranked_docs": state.get("retrieved_docs", [])[:top_k]}
        
        from core.reranking.cohere_reranker import CohereReranker
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        reranker = state.get("_reranker") or CohereReranker(settings)
        
        # Rerank
        reranked = await reranker.rerank(
            query=state["query"],
            chunks=state.get("retrieved_docs", []),
            top_k=state.get("top_k", 5)
        )
        
        logger.info(f"Reranked to {len(reranked)} chunks")
        
        return {"reranked_docs": reranked}
    
    except Exception as e:
        logger.error(f"Error in reranking_node: {e}")
        # Return top_k from retrieved_docs on error
        top_k = state.get("top_k", 5)
        return {"reranked_docs": state.get("retrieved_docs", [])[:top_k]}
