"""
Routing Node - Query classification and routing.

This node analyzes the query and determines which RAG strategy to use
based on query complexity and type.
"""

from typing import Any
from agents.state import AgentState
import logging
import json
import re

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown or other formatting."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find raw JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return text


def format_chat_history(history: list[dict]) -> str:
    """Format chat history for prompt."""
    if not history:
        return "No previous conversation"
    
    formatted = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted.append(f"{role}: {content[:200]}...")
    
    return "\n".join(formatted)


async def routing_node(state: AgentState) -> dict[str, Any]:
    """
    Analyze query and determine routing strategy.
    
    Uses LLM to classify query complexity and decide:
    - Query type (simple/complex/analytical/multi_doc)
    - Whether decomposition is needed
    - Which RAG strategy to use
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with routing decisions
    """
    try:
        from core.llm.llm_router import get_llm_provider
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        llm = get_llm_provider(settings)
        
        # Classification prompt
        system_prompt = """You are a query classifier for a document retrieval system.

Classify the user query into one of four categories:

1. **simple**: Direct factual question answerable from a single passage
   - Example: "What is the company revenue?"
   - Example: "When was the product launched?"

2. **complex**: Multi-faceted question that may need corrective retrieval
   - Example: "What are the main risks mentioned in the compliance report?"
   - Example: "Explain the new policy changes"

3. **analytical**: Question requiring deep analysis and verification
   - Example: "How does the Q1 strategy compare to industry best practices?"
   - Example: "Evaluate the effectiveness of the marketing campaign"

4. **multi_doc**: Question spanning multiple documents
   - Example: "Compare revenue across all quarterly reports"
   - Example: "What are the common themes in customer feedback?"

Also determine if the query needs decomposition into sub-queries.

Respond in JSON format:
{
  "query_type": "simple|complex|analytical|multi_doc",
  "needs_decomposition": true|false,
  "reasoning": "brief explanation"
}"""
        
        user_prompt = f"""Query: {state['query']}

Recent conversation context:
{format_chat_history(state.get('chat_history', [])[-3:])}

Classify this query."""
        
        # Get classification
        response = await llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=500
        )
        
        # Parse JSON response
        try:
            classification = json.loads(extract_json(response))
        except Exception as e:
            logger.warning(f"Failed to parse classification JSON: {e}")
            # Fallback to simple if parsing fails
            classification = {
                "query_type": "simple",
                "needs_decomposition": False,
                "reasoning": "Failed to parse classification, defaulting to simple"
            }
        
        query_type = classification.get("query_type", "simple")
        needs_decomposition = classification.get("needs_decomposition", False)
        reasoning = classification.get("reasoning", "")
        
        # Map query type to RAG strategy
        strategy_map = {
            "simple": "simple",
            "complex": "crag",
            "analytical": "srag",
            "multi_doc": "advanced"
        }
        
        rag_strategy = strategy_map.get(query_type, "simple")
        
        # Set is_multi_doc flag for synthesis node
        is_multi_doc = (query_type == "multi_doc")
        
        logger.info(f"Query classified as '{query_type}' -> strategy '{rag_strategy}' (decompose: {needs_decomposition})")
        
        return {
            "query_type": query_type,
            "needs_decomposition": needs_decomposition,
            "query_analysis_reasoning": reasoning,
            "rag_strategy": rag_strategy,
            "is_multi_doc": is_multi_doc
        }
    
    except Exception as e:
        logger.error(f"Error in routing_node: {e}")
        # Fallback to simple strategy on error
        return {
            "query_type": "simple",
            "needs_decomposition": False,
            "query_analysis_reasoning": f"Error during classification: {str(e)}",
            "rag_strategy": "simple",
            "is_multi_doc": False
        }
