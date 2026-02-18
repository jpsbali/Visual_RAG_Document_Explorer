"""
Citation Node - Format source citations.

This node formats citations for the generated answer,
mapping claims to their supporting sources.
"""

from typing import Any
from agents.state import AgentState
import logging

logger = logging.getLogger(__name__)


async def citation_node(state: AgentState) -> dict[str, Any]:
    """
    Format source citations.
    
    Maps grounded claims to source documents.
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with citations
    """
    try:
        from config.models import Citation
        
        citations = []
        
        # Create citation for each source chunk
        for i, chunk in enumerate(state.get("compressed_docs", [])):
            # Find claims supported by this chunk
            supported_claims = [
                detail["claim"]
                for detail in state.get("grounding_details", [])
                if detail.get("supporting_chunk") == chunk.chunk_id
            ]
            
            if supported_claims or not state.get("grounding_details"):
                citation = Citation(
                    source_file=chunk.metadata.source_file,
                    page_number=chunk.metadata.page_number,
                    chunk_id=chunk.chunk_id,
                    relevant_text=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    claims_supported=supported_claims
                )
                citations.append(citation)
        
        logger.info(f"Created {len(citations)} citations")
        
        return {"citations": citations}
    
    except Exception as e:
        logger.error(f"Error in citation_node: {e}")
        return {"citations": []}
