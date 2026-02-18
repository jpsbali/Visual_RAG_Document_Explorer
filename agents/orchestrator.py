"""
Main orchestrator agent entry point.

Coordinates the entire RAG pipeline execution through the LangGraph agent.
"""

from typing import AsyncIterator, Optional
from config.models import QueryRequest, QueryResponse
from config.settings import Settings
from agents.graph import get_agent_graph
from agents.state import AgentState
from core.search.hybrid_search import HybridSearch
from core.reranking.cohere_reranker import CohereReranker
from core.vectordb.router import get_vector_store
from core.search.bm25_search import BM25Search
import time
import uuid
import logging

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Main orchestrator for the LangGraph agent.
    
    Manages:
    - Graph execution
    - Dependency injection
    - Streaming responses
    - Error handling
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize orchestrator.
        
        Args:
            settings: Application settings (uses default if None)
        """
        self.settings = settings or Settings()
        self.graph = get_agent_graph()
        
        # Initialize dependencies
        self.vector_store = get_vector_store(self.settings)
        self.bm25_search = BM25Search()
        self.hybrid_search = HybridSearch(self.vector_store, self.bm25_search, self.settings)
        self.reranker = CohereReranker(self.settings)
    
    async def execute(
        self,
        request: QueryRequest,
        session_id: Optional[str] = None,
        collection: str = "documents"
    ) -> QueryResponse:
        """
        Execute query through the agent graph.
        
        Args:
            request: User query request
            session_id: Session identifier for memory
            collection: Vector DB collection to search
        
        Returns:
            Complete query response
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Initialize state
        initial_state: AgentState = {
            # Input
            "query": request.query,
            "chat_history": [],
            "session_id": session_id,
            "collection": collection,
            
            # Configuration
            "enable_reranking": request.enable_reranking,
            "enable_compression": request.enable_compression,
            "enable_hyde": request.enable_hyde,
            "top_k": request.top_k,
            "metadata_filters": request.metadata_filters,
            
            # Control
            "iteration_count": 0,
            "max_iterations": self.settings.srag_max_iterations,
            "start_time": time.time(),
            "node_timings": {},
            
            # Injected dependencies (not part of TypedDict, but passed through)
            "_hybrid_search": self.hybrid_search,
            "_reranker": self.reranker,
            "_settings": self.settings
        }
        
        # Execute graph
        try:
            logger.info(f"Executing agent graph for query: {request.query[:100]}...")
            final_state = await self.graph.ainvoke(initial_state)
            
            # Extract response
            response = final_state.get("_response")
            
            if response is None:
                # Fallback: construct response from state
                response = self._construct_response_from_state(final_state)
            
            logger.info(f"Agent execution completed in {response.response_time_ms:.0f}ms")
            return response
        
        except Exception as e:
            logger.error(f"Error executing agent graph: {e}", exc_info=True)
            # Error handling
            return QueryResponse(
                query=request.query,
                answer=f"Error processing query: {str(e)}",
                mode=request.mode,
                search_mode=request.search_mode,
                sources=[],
                citations=[],
                grounding=None,
                response_time_ms=(time.time() - initial_state["start_time"]) * 1000,
                initial_retrieval_count=0,
                final_retrieval_count=0
            )
    
    async def execute_stream(
        self,
        request: QueryRequest,
        session_id: Optional[str] = None,
        collection: str = "documents"
    ) -> AsyncIterator[dict]:
        """
        Execute query with streaming updates.
        
        Yields state updates as the graph executes.
        
        Args:
            request: User query request
            session_id: Session identifier
            collection: Vector DB collection
        
        Yields:
            State updates with node progress
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Initialize state (same as execute)
        initial_state: AgentState = {
            "query": request.query,
            "chat_history": [],
            "session_id": session_id,
            "collection": collection,
            "enable_reranking": request.enable_reranking,
            "enable_compression": request.enable_compression,
            "enable_hyde": request.enable_hyde,
            "top_k": request.top_k,
            "metadata_filters": request.metadata_filters,
            "iteration_count": 0,
            "max_iterations": self.settings.srag_max_iterations,
            "start_time": time.time(),
            "node_timings": {},
            "_hybrid_search": self.hybrid_search,
            "_reranker": self.reranker,
            "_settings": self.settings
        }
        
        # Stream graph execution
        try:
            logger.info(f"Streaming agent execution for query: {request.query[:100]}...")
            async for event in self.graph.astream(initial_state):
                # Event format: {node_name: state_update}
                for node_name, state_update in event.items():
                    yield {
                        "node": node_name,
                        "update": state_update,
                        "timestamp": time.time()
                    }
        
        except Exception as e:
            logger.error(f"Error in streaming execution: {e}", exc_info=True)
            yield {
                "node": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _construct_response_from_state(self, state: AgentState) -> QueryResponse:
        """
        Construct QueryResponse from final state.
        
        Fallback if format node doesn't create _response.
        
        Args:
            state: Final agent state
        
        Returns:
            QueryResponse
        """
        return QueryResponse(
            query=state.get("query", ""),
            answer=state.get("final_answer", "No answer generated"),
            mode=state.get("rag_strategy", "simple"),
            search_mode="hybrid",
            sources=state.get("compressed_docs", []),
            citations=state.get("citations", []),
            grounding=state.get("grounding_result"),
            crag_details=state.get("crag_result"),
            reflection_details=state.get("srag_result"),
            synthesis_details=state.get("synthesis_result"),
            response_time_ms=(time.time() - state.get("start_time", time.time())) * 1000,
            hyde_used=state.get("enable_hyde", False),
            reranking_used=state.get("enable_reranking", True),
            compression_used=state.get("enable_compression", True),
            initial_retrieval_count=len(state.get("retrieved_docs", [])),
            final_retrieval_count=len(state.get("compressed_docs", [])),
            memory_used=state.get("enable_long_term_memory", False),
            context_window_usage=state.get("context_window_usage", 0.0)
        )


# Convenience function
async def execute_agent_query(
    request: QueryRequest,
    settings: Optional[Settings] = None,
    session_id: Optional[str] = None,
    collection: str = "documents"
) -> QueryResponse:
    """
    Execute a query through the agent orchestrator.
    
    Convenience function for simple usage.
    
    Args:
        request: Query request
        settings: Application settings (uses default if None)
        session_id: Session identifier
        collection: Vector DB collection
    
    Returns:
        Query response
    """
    if settings is None:
        settings = Settings()
    
    orchestrator = AgentOrchestrator(settings)
    return await orchestrator.execute(request, session_id, collection)
