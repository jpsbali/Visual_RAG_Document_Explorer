"""
Format Node - Format final response for UI.

This node compiles the final QueryResponse with all metadata.
"""

from typing import Any
from agents.state import AgentState
import logging
import time

logger = logging.getLogger(__name__)


async def format_node(state: AgentState) -> dict[str, Any]:
    """
    Format final response for UI.
    
    Compiles QueryResponse with all metadata.
    This is the final node before END.
    
    Args:
        state: Current agent state
        
    Returns:
        Empty dict (response compiled separately)
    """
    try:
        from config.models import QueryResponse
        
        # Calculate total time
        total_time_ms = (time.time() - state.get("start_time", time.time())) * 1000
        
        # Compile response
        response = QueryResponse(
            query=state["query"],
            answer=state.get("final_answer", ""),
            mode=state.get("rag_strategy", "simple"),
            search_mode=state.get("retrieval_metadata", {}).get("search_mode", "hybrid"),
            sources=state.get("compressed_docs", []),
            citations=state.get("citations", []),
            grounding=state.get("grounding_result"),
            crag_details=state.get("crag_result"),
            reflection_details=state.get("srag_result"),
            synthesis_details=state.get("synthesis_result"),
            hyde_details=None,
            response_time_ms=total_time_ms,
            hyde_used=state.get("enable_hyde", False),
            reranking_used=state.get("enable_reranking", True),
            compression_used=state.get("enable_compression", True),
            initial_retrieval_count=len(state.get("retrieved_docs", [])),
            final_retrieval_count=len(state.get("compressed_docs", [])),
            memory_used=state.get("enable_long_term_memory", False),
            context_window_usage=state.get("context_window_usage", 0.0)
        )
        
        logger.info(f"Formatted response: {total_time_ms:.0f}ms total")
        
        # Store response in state for orchestrator to return
        return {"_response": response}
    
    except Exception as e:
        logger.error(f"Error in format_node: {e}")
        return {"_response": None}
