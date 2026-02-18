"""
Compression Node - Extract relevant portions from chunks.

This node uses the ContextualCompressor from Phase 3 to extract
only the most relevant portions of retrieved chunks.
"""

from typing import Any
from agents.state import AgentState
import logging

logger = logging.getLogger(__name__)


async def compression_node(state: AgentState) -> dict[str, Any]:
    """
    Apply contextual compression to extract relevant content.
    
    Uses ContextualCompressor from Phase 3.
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with compressed_docs
    """
    try:
        # Skip if compression disabled
        if not state.get("enable_compression", True):
            return {"compressed_docs": state.get("reranked_docs", [])}
        
        from core.rag.contextual_compressor import ContextualCompressor
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        compressor = ContextualCompressor(settings)
        
        # Compress chunks
        compressed = await compressor.compress_chunks(
            query=state["query"],
            chunks=state.get("reranked_docs", []),
            preserve_top_k=2  # Keep top 2 chunks uncompressed
        )
        
        logger.info(f"Compressed {len(state.get('reranked_docs', []))} chunks")
        
        return {"compressed_docs": compressed}
    
    except Exception as e:
        logger.error(f"Error in compression_node: {e}")
        # Return uncompressed chunks on error
        return {"compressed_docs": state.get("reranked_docs", [])}
