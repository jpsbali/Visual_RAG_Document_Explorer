"""
Retrieval Node - Retrieve relevant documents using hybrid search.

This node performs document retrieval using the hybrid search system,
supporting multi-query retrieval and optional HYDE enhancement.
"""

from typing import Any
from agents.state import AgentState
import logging

logger = logging.getLogger(__name__)


async def retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve relevant documents using hybrid search.
    
    Supports:
    - Simple retrieval (single query)
    - Multi-query retrieval (for decomposed queries)
    - HYDE-enhanced retrieval
    - Metadata filtering
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with retrieved_docs
    """
    try:
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        hybrid_search = state.get("_hybrid_search")
        
        if not hybrid_search:
            logger.error("HybridSearch not injected in state")
            return {"retrieved_docs": [], "retrieval_metadata": {"error": "HybridSearch not available"}}
        
        # Determine queries to search
        queries = state.get("sub_queries", [state["query"]])
        if not queries:
            queries = [state["query"]]
        
        # Retrieve for each query
        all_chunks = []
        
        for query in queries:
            # Use HYDE if enabled
            if state.get("enable_hyde", False):
                try:
                    from core.rag.hyde import HYDEGenerator
                    
                    hyde = HYDEGenerator(settings, hybrid_search)
                    chunks, hyde_result = await hyde.retrieve_with_hyde(
                        query=query,
                        collection=state.get("collection", "documents"),
                        top_k=state.get("top_k", 5) * 2  # Retrieve more for reranking
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"HYDE failed, falling back to standard retrieval: {e}")
                    # Fall back to standard retrieval
                    chunks = await hybrid_search.search(
                        query=query,
                        collection=state.get("collection", "documents"),
                        top_k=state.get("top_k", 5) * 2,
                        filters=state.get("metadata_filters")
                    )
                    all_chunks.extend(chunks)
            else:
                # Standard hybrid search
                chunks = await hybrid_search.search(
                    query=query,
                    collection=state.get("collection", "documents"),
                    top_k=state.get("top_k", 5) * 2,
                    filters=state.get("metadata_filters")
                )
                all_chunks.extend(chunks)
        
        # Deduplicate by chunk_id
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)
        
        retrieval_metadata = {
            "num_queries": len(queries),
            "total_retrieved": len(all_chunks),
            "unique_chunks": len(unique_chunks),
            "hyde_used": state.get("enable_hyde", False),
            "search_mode": "hybrid"
        }
        
        logger.info(f"Retrieved {len(unique_chunks)} unique chunks from {len(queries)} queries")
        
        return {
            "retrieved_docs": unique_chunks,
            "retrieval_metadata": retrieval_metadata
        }
    
    except Exception as e:
        logger.error(f"Error in retrieval_node: {e}")
        return {
            "retrieved_docs": [],
            "retrieval_metadata": {"error": str(e)}
        }
