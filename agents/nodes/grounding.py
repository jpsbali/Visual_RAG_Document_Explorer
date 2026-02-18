"""
Grounding Node - Verify answer grounding in sources.

This node uses the GroundingVerifier from Phase 3 to verify
that the generated answer is grounded in the source documents.
"""

from typing import Any
from agents.state import AgentState
import logging

logger = logging.getLogger(__name__)


async def grounding_node(state: AgentState) -> dict[str, Any]:
    """
    Verify answer grounding in sources.
    
    Uses GroundingVerifier from Phase 3.
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with grounding results
    """
    try:
        from core.rag.grounding_verifier import GroundingVerifier
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        verifier = GroundingVerifier(settings)
        
        # Verify grounding
        grounding_result = await verifier.verify_answer(
            answer=state.get("draft_answer", ""),
            chunks=state.get("compressed_docs", [])
        )
        
        # Decide whether to modify answer
        final_answer = state.get("draft_answer", "")
        
        if grounding_result.grounding_score < 0.5:
            final_answer = f"""**Note: This answer has low grounding confidence ({grounding_result.grounding_score:.2f}). Please verify claims independently.**

{final_answer}"""
        
        logger.info(f"Grounding score: {grounding_result.grounding_score:.2f}")
        
        return {
            "final_answer": final_answer,
            "grounding_score": grounding_result.grounding_score,
            "grounding_details": [
                {
                    "claim": claim.claim,
                    "status": claim.status,
                    "supporting_chunk": claim.supporting_chunk_id,
                    "confidence": claim.confidence
                }
                for claim in grounding_result.claims
            ],
            "grounding_result": grounding_result
        }
    
    except Exception as e:
        logger.error(f"Error in grounding_node: {e}")
        # Use draft_answer as final_answer on error
        return {
            "final_answer": state.get("draft_answer", ""),
            "grounding_score": 0.5,
            "grounding_details": [],
            "grounding_result": None
        }
