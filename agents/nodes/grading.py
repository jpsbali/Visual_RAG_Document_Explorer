"""
Grading Node - Grade retrieved documents for relevance (CRAG).

This node evaluates the relevance of retrieved documents
and determines if corrective retrieval is needed.
"""

from typing import Any
from agents.state import AgentState
import logging
import json
import re

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str:
    """Extract JSON from text."""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text


async def grading_node(state: AgentState) -> dict[str, Any]:
    """
    Grade retrieved documents for relevance (CRAG).
    
    Uses LLM-as-judge to evaluate each document.
    Determines if corrective retrieval is needed.
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with grading results
    """
    try:
        # Only run for CRAG strategy
        if state.get("rag_strategy") != "crag":
            return {}
        
        from core.llm.llm_router import get_llm_provider
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        llm = get_llm_provider(settings)
        
        # Grade top 5 documents
        docs_to_grade = state.get("retrieved_docs", [])[:5]
        relevance_scores = {}
        
        for doc in docs_to_grade:
            system_prompt = """You are a relevance grader for a document retrieval system.

Evaluate whether the retrieved document is relevant to the user query.

Score the relevance on a scale of 0.0 to 1.0:
- 0.0-0.3: Irrelevant - document has no useful information
- 0.3-0.6: Ambiguous - document has some tangentially related information
- 0.6-1.0: Relevant - document directly addresses the query

Respond in JSON format:
{
  "relevance_score": 0.0-1.0,
  "relevance_label": "relevant|ambiguous|irrelevant",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}"""
            
            user_prompt = f"""Query: {state['query']}

Document:
{doc.content[:1000]}...

Evaluate relevance."""
            
            response = await llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=300
            )
            
            try:
                grading = json.loads(extract_json(response))
                score = grading.get("relevance_score", 0.5)
            except:
                score = 0.5
            
            relevance_scores[doc.chunk_id] = score
        
        # Calculate average relevance
        avg_relevance = sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0.5
        
        # Determine if correction is needed
        needs_correction = avg_relevance < settings.crag_relevance_threshold
        
        # Determine correction strategy
        correction_strategy = None
        if needs_correction:
            if avg_relevance < 0.3:
                correction_strategy = "broaden"
            elif state.get("query_type") == "complex":
                correction_strategy = "decompose"
            else:
                correction_strategy = "reformulate"
        
        logger.info(f"Average relevance: {avg_relevance:.2f}, needs_correction: {needs_correction}")
        
        return {
            "relevance_scores": relevance_scores,
            "needs_correction": needs_correction,
            "correction_strategy": correction_strategy
        }
    
    except Exception as e:
        logger.error(f"Error in grading_node: {e}")
        return {
            "relevance_scores": {},
            "needs_correction": False,
            "correction_strategy": None
        }
