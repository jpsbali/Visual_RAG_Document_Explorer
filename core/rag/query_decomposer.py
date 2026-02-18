"""
Query Decomposer for Visual RAG Document Explorer.

Breaks complex queries into 2-5 sub-queries for improved retrieval coverage.
"""

import json
from typing import Optional

from config.settings import Settings
from core.llm.llm_router import get_llm_provider


class QueryDecomposer:
    """
    Query Decomposer for breaking complex queries into sub-queries.
    
    Uses LLM to analyze query complexity and generate 2-5 focused sub-queries
    that can be independently retrieved and merged. Handles single-intent queries
    by returning the original query.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize query decomposer.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.llm = get_llm_provider(settings)
    
    async def decompose(
        self,
        query: str,
        max_subqueries: int = 5
    ) -> list[str]:
        """
        Decompose a complex query into sub-queries.
        
        Args:
            query: Original user query
            max_subqueries: Maximum number of sub-queries to generate (2-5)
            
        Returns:
            List of sub-queries (includes original if single-intent)
        """
        system_prompt = """You are a query analysis expert. Your task is to analyze user queries and break them down into focused sub-queries for better information retrieval.

Rules:
1. If the query is simple and single-intent, return it as-is
2. If the query is complex with multiple aspects, break it into 2-5 focused sub-queries
3. Each sub-query should be self-contained and independently searchable
4. Sub-queries should cover all aspects of the original query
5. Return ONLY a JSON array of strings, no other text

Examples:

Query: "What is the capital of France?"
Output: ["What is the capital of France?"]

Query: "Compare the revenue, profit margins, and market share of Apple and Microsoft in 2023"
Output: [
  "What was Apple's revenue in 2023?",
  "What was Microsoft's revenue in 2023?",
  "What were Apple's profit margins in 2023?",
  "What were Microsoft's profit margins in 2023?",
  "What was Apple's market share in 2023?",
  "What was Microsoft's market share in 2023?"
]

Query: "Explain the causes and consequences of the French Revolution"
Output: [
  "What were the main causes of the French Revolution?",
  "What were the economic factors leading to the French Revolution?",
  "What were the social consequences of the French Revolution?",
  "What were the political consequences of the French Revolution?"
]"""
        
        user_prompt = f"""Query: "{query}"

Analyze this query and return a JSON array of sub-queries (maximum {max_subqueries}). Return ONLY the JSON array, no other text."""
        
        try:
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=500
            )
            
            # Parse JSON response
            response = response.strip()
            
            # Handle markdown code blocks
            if response.startswith("```"):
                # Extract JSON from code block
                lines = response.split("\n")
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                response = "\n".join(json_lines)
            
            subqueries = json.loads(response)
            
            # Validate response
            if not isinstance(subqueries, list):
                return [query]
            
            if not subqueries:
                return [query]
            
            # Ensure all items are strings
            subqueries = [str(q).strip() for q in subqueries if q]
            
            # Limit to max_subqueries
            subqueries = subqueries[:max_subqueries]
            
            # If only one sub-query and it's the same as original, return original
            if len(subqueries) == 1 and subqueries[0].lower() == query.lower():
                return [query]
            
            return subqueries if subqueries else [query]
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: return original query if decomposition fails
            return [query]
    
    async def is_complex_query(self, query: str) -> bool:
        """
        Determine if a query is complex enough to benefit from decomposition.
        
        Args:
            query: User query
            
        Returns:
            True if query is complex, False otherwise
        """
        subqueries = await self.decompose(query, max_subqueries=2)
        return len(subqueries) > 1
