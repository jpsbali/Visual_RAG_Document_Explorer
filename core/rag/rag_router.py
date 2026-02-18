"""
RAG Router for Visual RAG Document Explorer.

Routes queries to appropriate RAG strategies based on mode selection or
automatic query complexity analysis.
"""

import json
from typing import Literal, Optional

from config.settings import Settings
from config.models import QueryRequest
from core.rag.base import RAGStrategy
from core.rag.simple_rag import SimpleRAG
from core.rag.corrective_rag import CorrectiveRAG
from core.rag.self_reflective_rag import SelfReflectiveRAG
from core.rag.advanced_rag import AdvancedRAG
from core.search.hybrid_search import HybridSearch
from core.reranking.cohere_reranker import CohereReranker
from core.llm.llm_router import get_llm_provider


class RAGRouter:
    """
    RAG Router for strategy selection and routing.
    
    Routes queries to the appropriate RAG strategy based on:
    - Explicit mode selection (simple, crag, srag, advanced)
    - Automatic mode selection based on query complexity analysis
    """
    
    def __init__(
        self,
        settings: Settings,
        hybrid_search: HybridSearch,
        reranker: CohereReranker
    ):
        """
        Initialize RAG router.
        
        Args:
            settings: Application settings
            hybrid_search: Hybrid search service
            reranker: Cohere reranker
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.llm = get_llm_provider(settings)
        
        # Initialize all strategies
        self.strategies = {
            "simple": SimpleRAG(settings, hybrid_search, reranker),
            "crag": CorrectiveRAG(settings, hybrid_search, reranker),
            "srag": SelfReflectiveRAG(settings, hybrid_search, reranker),
            "advanced": AdvancedRAG(settings, hybrid_search, reranker)
        }
    
    async def route(self, request: QueryRequest) -> RAGStrategy:
        """
        Route query to appropriate RAG strategy.
        
        Args:
            request: Query request
            
        Returns:
            Selected RAG strategy
        """
        if request.mode == "auto":
            # Automatic routing based on query analysis
            selected_mode = await self._analyze_query_complexity(request.query)
        else:
            # Use explicitly specified mode
            selected_mode = request.mode
        
        return self.strategies[selected_mode]
    
    async def _analyze_query_complexity(
        self,
        query: str
    ) -> Literal["simple", "crag", "srag", "advanced"]:
        """
        Analyze query complexity and select appropriate strategy.
        
        Strategy selection criteria:
        - Simple: Single-intent, factual queries
        - CRAG: Queries that may need retrieval correction
        - SRAG: Queries requiring high accuracy and verification
        - Advanced: Multi-aspect, complex queries
        
        Args:
            query: User query
            
        Returns:
            Selected strategy name
        """
        system_prompt = """You are a query complexity analyzer. Analyze the query and recommend the best RAG strategy.

Strategies:
1. "simple" - For straightforward, single-intent factual queries
   Example: "What is the capital of France?"

2. "crag" - For queries where initial retrieval might miss relevant info
   Example: "What are the side effects of medication X?"

3. "srag" - For queries requiring high accuracy and verification
   Example: "What were the exact financial results for Q3 2023?"

4. "advanced" - For complex, multi-aspect queries
   Example: "Compare the revenue, market share, and growth strategies of Apple and Microsoft"

Return ONLY a JSON object:
{
  "strategy": "simple|crag|srag|advanced",
  "reasoning": "brief explanation"
}"""
        
        user_prompt = f"""Query: "{query}"

Analyze this query and recommend the best RAG strategy. Return ONLY the JSON object."""
        
        try:
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=200
            )
            
            response = response.strip()
            
            # Handle markdown code blocks
            if response.startswith("```"):
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
            
            result = json.loads(response)
            strategy = result.get("strategy", "simple")
            
            # Validate strategy
            if strategy not in ["simple", "crag", "srag", "advanced"]:
                strategy = "simple"
            
            return strategy
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: use simple strategy
            return "simple"
    
    def get_strategy(
        self,
        mode: Literal["simple", "crag", "srag", "advanced"]
    ) -> RAGStrategy:
        """
        Get a specific RAG strategy by name.
        
        Args:
            mode: Strategy name
            
        Returns:
            RAG strategy instance
        """
        return self.strategies[mode]


def get_rag_strategy(
    settings: Settings,
    hybrid_search: HybridSearch,
    reranker: CohereReranker,
    mode: Optional[Literal["simple", "crag", "srag", "advanced"]] = None
) -> RAGStrategy:
    """
    Factory function to get a RAG strategy.
    
    Args:
        settings: Application settings
        hybrid_search: Hybrid search service
        reranker: Cohere reranker
        mode: Optional strategy mode (uses settings default if not provided)
        
    Returns:
        RAG strategy instance
    """
    if mode is None:
        mode = settings.default_rag_strategy
        if mode == "auto":
            mode = "simple"  # Default to simple for factory function
    
    strategies = {
        "simple": SimpleRAG,
        "crag": CorrectiveRAG,
        "srag": SelfReflectiveRAG,
        "advanced": AdvancedRAG
    }
    
    strategy_class = strategies.get(mode, SimpleRAG)
    return strategy_class(settings, hybrid_search, reranker)


async def execute_rag_query(
    request: QueryRequest,
    settings: Settings,
    hybrid_search: HybridSearch,
    reranker: CohereReranker
):
    """
    Execute a RAG query with automatic routing.
    
    This is a convenience function that handles routing and execution.
    
    Args:
        request: Query request
        settings: Application settings
        hybrid_search: Hybrid search service
        reranker: Cohere reranker
        
    Returns:
        Query response
    """
    router = RAGRouter(settings, hybrid_search, reranker)
    strategy = await router.route(request)
    return await strategy.execute(request)
