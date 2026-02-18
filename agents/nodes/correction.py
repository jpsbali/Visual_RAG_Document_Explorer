"""
Correction Node - Perform corrective retrieval (CRAG).

This node performs corrective retrieval when initial results
have low relevance scores.
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


def summarize_docs(docs: list) -> str:
    """Summarize documents for prompt."""
    if not docs:
        return "No documents"
    return "\n".join(f"- {doc.content[:100]}..." for doc in docs[:3])


async def correction_node(state: AgentState) -> dict[str, Any]:
    """
    Perform corrective retrieval (CRAG).
    
    Three strategies:
    1. Reformulate: Rephrase query
    2. Broaden: Relax search parameters
    3. Decompose: Break into sub-queries
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with corrected results
    """
    try:
        # Skip if no correction needed
        if not state.get("needs_correction", False):
            return {}
        
        from core.llm.llm_router import get_llm_provider
        from core.rag.query_decomposer import QueryDecomposer
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        llm = get_llm_provider(settings)
        hybrid_search = state.get("_hybrid_search")
        
        strategy = state.get("correction_strategy", "reformulate")
        original_query = state["query"]
        new_chunks = []
        reformulated_query = None
        
        if strategy == "reformulate":
            system_prompt = """The initial retrieval returned low-relevance results.
Reformulate the query to improve retrieval. You may:
1. Rephrase using different terminology
2. Focus on key concepts
3. Add context

Respond in JSON format:
{
  "reformulated_query": "new query text",
  "reasoning": "brief explanation"
}"""
            
            user_prompt = f"""Original query: {original_query}

Low-relevance documents summary:
{summarize_docs(state.get('retrieved_docs', [])[:3])}

Reformulate the query."""
            
            response = await llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=300
            )
            
            try:
                result = json.loads(extract_json(response))
                reformulated_query = result.get("reformulated_query", original_query)
            except:
                reformulated_query = original_query
            
            new_chunks = await hybrid_search.search(
                query=reformulated_query,
                collection=state.get("collection", "documents"),
                top_k=state.get("top_k", 5) * 2
            )
        
        elif strategy == "broaden":
            new_chunks = await hybrid_search.search(
                query=original_query,
                collection=state.get("collection", "documents"),
                top_k=state.get("top_k", 5) * 4,
                filters=None
            )
        
        elif strategy == "decompose":
            decomposer = QueryDecomposer(settings)
            sub_queries = await decomposer.decompose(original_query)
            
            for sub_query in sub_queries:
                chunks = await hybrid_search.search(
                    query=sub_query,
                    collection=state.get("collection", "documents"),
                    top_k=state.get("top_k", 5)
                )
                new_chunks.extend(chunks)
        
        # Merge with original results and deduplicate
        all_chunks = state.get("retrieved_docs", []) + new_chunks
        seen_ids = set()
        merged_chunks = []
        for chunk in all_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                merged_chunks.append(chunk)
        
        logger.info(f"Corrective retrieval added {len(new_chunks)} new chunks")
        
        return {
            "reformulated_query": reformulated_query,
            "retrieved_docs": merged_chunks
        }
    
    except Exception as e:
        logger.error(f"Error in correction_node: {e}")
        return {
            "reformulated_query": None,
            "retrieved_docs": state.get("retrieved_docs", [])
        }
