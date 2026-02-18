"""
Decomposition Node - Break complex queries into focused sub-queries.

This node uses the QueryDecomposer from Phase 3 to break down
complex queries into simpler, focused sub-queries.
"""

from typing import Any
from agents.state import AgentState
import logging

logger = logging.getLogger(__name__)


async def decomposition_node(state: AgentState) -> dict[str, Any]:
    """
    Decompose complex query into focused sub-queries.
    
    Only runs if needs_decomposition is True.
    Uses QueryDecomposer from Phase 3.
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with sub_queries
    """
    try:
        # Skip if decomposition not needed
        if not state.get("needs_decomposition", False):
            return {"sub_queries": [state["query"]]}
        
        from core.rag.query_decomposer import QueryDecomposer
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        decomposer = QueryDecomposer(settings)
        
        # Decompose query
        sub_queries = await decomposer.decompose(state["query"])
        
        # Ensure we have at least the original query
        if not sub_queries:
            sub_queries = [state["query"]]
        
        # Limit to 5 sub-queries max to prevent excessive retrieval
        sub_queries = sub_queries[:5]
        
        logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
        
        return {"sub_queries": sub_queries}
    
    except Exception as e:
        logger.error(f"Error in decomposition_node: {e}")
        # Return original query as single sub-query on error
        return {"sub_queries": [state["query"]]}
