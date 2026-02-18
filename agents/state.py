"""
AgentState TypedDict schema.

Defines the shared state structure for the LangGraph agent.
"""

from typing import TypedDict, List, Dict, Any, Optional
from config.models import (
    RetrievedChunk,
    Citation,
    GroundingResult,
    CRAGResult,
    SelfReflectiveResult,
    SynthesisResult
)


class AgentState(TypedDict, total=False):
    """
    State object passed between agent nodes in the LangGraph workflow.
    
    This TypedDict defines all possible fields that can be present in the agent state
    as it flows through the LangGraph nodes. Using total=False allows nodes to only
    update the fields they need without requiring all fields to be present.
    
    Fields are organized by category:
    - Input/Query fields
    - Configuration fields
    - Retrieval fields
    - Generation fields
    - Quality check fields
    - Strategy/Routing fields
    - Memory fields
    - Metadata/Timing fields
    - Injected dependencies (prefixed with _)
    """
    
    # ============= Input/Query Fields =============
    query: str                              # User's original query
    chat_history: List[Dict[str, Any]]      # Conversation history
    session_id: str                         # Session identifier for memory
    collection: str                         # Vector DB collection to search
    
    # ============= Configuration Fields =============
    enable_reranking: bool                  # Enable Cohere reranking
    enable_compression: bool                # Enable contextual compression
    enable_hyde: bool                       # Enable HYDE query expansion
    enable_long_term_memory: bool           # Enable long-term memory search
    top_k: int                              # Number of chunks to retrieve
    metadata_filters: Optional[Dict[str, List[str]]]  # Entity-based filters
    
    # ============= Routing/Strategy Fields =============
    query_type: str                         # simple/complex/analytical/multi_doc
    needs_decomposition: bool               # Whether to decompose query
    query_analysis_reasoning: str           # Reasoning for classification
    rag_strategy: str                       # simple/crag/srag/advanced
    is_multi_doc: bool                      # Whether multi-doc synthesis needed
    
    # ============= Decomposition Fields =============
    sub_queries: Optional[List[str]]        # Decomposed sub-queries
    
    # ============= Retrieval Fields =============
    retrieved_docs: List[RetrievedChunk]    # Initially retrieved chunks
    retrieval_metadata: Dict[str, Any]      # Retrieval metadata
    
    # ============= CRAG Fields =============
    needs_correction: bool                  # Whether CRAG correction needed
    crag_result: Optional[CRAGResult]       # CRAG evaluation result
    
    # ============= Reranking Fields =============
    reranked_docs: Optional[List[RetrievedChunk]]  # Reranked chunks
    reranking_metadata: Dict[str, Any]      # Reranking metadata
    
    # ============= Synthesis Fields =============
    synthesis_result: Optional[SynthesisResult]  # Multi-doc synthesis result
    
    # ============= Compression Fields =============
    compressed_docs: List[RetrievedChunk]   # Compressed/final chunks
    compression_metadata: Dict[str, Any]    # Compression metadata
    
    # ============= Generation Fields =============
    draft_answer: str                       # Initial generated answer
    generation_metadata: Dict[str, Any]     # Generation metadata
    
    # ============= SRAG Reflection Fields =============
    needs_reflection: bool                  # Whether SRAG reflection needed
    srag_result: Optional[SelfReflectiveResult]  # SRAG reflection result
    iteration_count: int                    # Current iteration number
    max_iterations: int                     # Maximum iterations allowed
    
    # ============= Grounding Fields =============
    grounding_result: Optional[GroundingResult]  # Grounding verification result
    
    # ============= Citation Fields =============
    citations: List[Citation]               # Source citations
    
    # ============= Final Answer Fields =============
    final_answer: str                       # Final formatted answer
    
    # ============= Memory Fields =============
    relevant_memory: List[Dict[str, Any]]   # Relevant past conversations
    context_window_usage: float             # Context window usage (0.0-1.0)
    needs_summarization: bool               # Whether history needs summarization
    
    # ============= Timing/Metadata Fields =============
    start_time: float                       # Execution start timestamp
    node_timings: Dict[str, float]          # Per-node execution times
    
    # ============= Response Field =============
    _response: Any                          # Final QueryResponse object
    
    # ============= Injected Dependencies (not serialized) =============
    # These are injected by the orchestrator and used by nodes
    _hybrid_search: Any                     # HybridSearch instance
    _reranker: Any                          # CohereReranker instance
    _settings: Any                          # Settings instance
