"""
Pydantic data models for Visual RAG Document Explorer.

All data flowing through the system uses these models for validation,
serialization, and type safety.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


# ============= Query Request Models =============

class QueryRequest(BaseModel):
    """User query input with all configurable options."""
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["simple", "crag", "srag", "advanced", "auto"] = Field(
        default="auto",
        description="RAG strategy: simple, crag, srag, advanced, or auto for agent routing"
    )
    search_mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid",
        description="Search mode: dense - semantic, sparse - keyword BM25, hybrid - RRF fusion"
    )
    top_k: int = Field(default=5, ge=1, le=20)
    enable_hyde: bool = Field(
        default=False,
        description="Use HYDE - Hypothetical Document Embeddings - for query expansion"
    )
    enable_reranking: bool = Field(
        default=True,
        description="Use Cohere cross-encoder reranking for improved precision"
    )
    enable_compression: bool = Field(
        default=True,
        description="Use contextual compression to extract relevant portions"
    )
    metadata_filters: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="Entity-based metadata filters e.g. {organizations: [Acme Corp]}"
    )


# ============= Document & Upload Models =============

class UploadResponse(BaseModel):
    """Response after document upload and processing."""
    file_id: str
    filename: str
    file_type: Literal["pdf", "docx", "txt", "html", "json"]
    chunks_created: int
    entities_extracted: dict[str, list[str]]  # GLiNER/spaCy entities
    summary: str                               # Document-level summary
    duplicate_status: Literal["new", "exact_duplicate", "near_duplicate"]
    status: Literal["success", "partial", "failed"]
    message: str


# ============= Chunk Models =============

class NEREntities(BaseModel):
    """Named entities extracted by GLiNER2 and/or spaCy."""
    people: list[str] = []
    organizations: list[str] = []
    dates: list[str] = []
    locations: list[str] = []
    topics: list[str] = []
    custom: dict[str, list[str]] = {}  # GLiNER2 custom entity types
    extractor: Literal["gliner", "spacy", "ensemble"] = "ensemble"
    confidence_scores: dict[str, float] = {}  # entity -> confidence


class ChunkMetadata(BaseModel):
    """Metadata attached to each document chunk."""
    chunk_id: str
    source_file: str
    file_type: Literal["pdf", "docx", "txt", "html", "json"]
    page_number: Optional[int] = None
    chunk_index: int
    total_chunks: int
    chunk_type: Literal["content", "summary"] = "content"
    doc_item_type: Optional[str] = None
    parent_heading: Optional[str] = None
    hierarchy_level: Optional[int] = None
    chunk_method: str = "recursive"
    chunk_size: int
    token_count: int
    char_count: int
    content_hash: str                          # SHA-256 for dedup
    content_preview: str                       # First 200 chars
    entities: NEREntities = NEREntities()      # Extracted entities
    keywords: list[str] = []
    created_at: datetime
    processed_at: datetime


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store with relevance score."""
    content: str
    metadata: ChunkMetadata
    score: float
    search_method: Literal["dense", "sparse", "hybrid"]


# ============= CRAG Models =============

class CRAGEvaluation(BaseModel):
    """Corrective RAG relevance evaluation result."""
    relevance_score: float = Field(ge=0.0, le=1.0)
    relevance_label: Literal["relevant", "ambiguous", "irrelevant"]
    confidence: float = Field(ge=0.0, le=1.0)
    evaluation_method: str = "llm_grader"
    needs_correction: bool
    correction_strategy: Optional[Literal["reformulate", "broaden", "decompose"]] = None
    evaluated_at: datetime


class CRAGResult(BaseModel):
    """Full CRAG pipeline result."""
    correction_applied: bool
    evaluation: CRAGEvaluation
    original_chunks: list[RetrievedChunk]
    corrected_chunks: Optional[list[RetrievedChunk]] = None
    reformulated_query: Optional[str] = None


# ============= Self-Reflective RAG Models =============

class ReflectionResult(BaseModel):
    """Self-reflection evaluation of a generated answer."""
    answer_grounded: bool
    hallucination_detected: bool
    completeness_score: float = Field(ge=0.0, le=1.0)
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    sources_cited: list[str]
    reflection_score: float = Field(ge=0.0, le=1.0)
    needs_regeneration: bool
    reflection_reason: str
    iteration: int
    reflected_at: datetime


class SelfReflectiveResult(BaseModel):
    """Full SRAG pipeline result."""
    final_answer: str
    total_iterations: int
    reflections: list[ReflectionResult]  # One per iteration
    retrieved_chunks: list[RetrievedChunk]


# ============= Grounding Verification Models =============

class ClaimVerification(BaseModel):
    """Verification result for a single claim in the answer."""
    claim: str
    is_grounded: bool
    supporting_chunk_id: Optional[str] = None  # chunk_id of supporting evidence
    supporting_chunks: list[str] = []  # chunk_ids (for backward compatibility)
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["grounded", "partially_grounded", "ungrounded"]


class GroundingResult(BaseModel):
    """Full grounding verification result."""
    grounding_score: float = Field(ge=0.0, le=1.0)
    total_claims: int
    grounded_claims: int
    partially_grounded_claims: int
    ungrounded_claims: int
    claims: list[ClaimVerification]  # Renamed from claim_details for consistency
    claim_details: list[ClaimVerification] = []  # Deprecated, kept for backward compatibility
    verified_at: datetime
    
    def __init__(self, **data):
        """Initialize with backward compatibility."""
        if 'claims' in data and 'claim_details' not in data:
            data['claim_details'] = data['claims']
        elif 'claim_details' in data and 'claims' not in data:
            data['claims'] = data['claim_details']
        super().__init__(**data)


# ============= Citation Models =============

class Citation(BaseModel):
    """A source citation for a claim in the answer."""
    source_file: str
    page_number: Optional[int] = None
    chunk_id: str
    relevant_text: str  # The specific text supporting the claim
    claims_supported: list[str] = []  # Claims supported by this citation


# ============= Multi-Document Synthesis Models =============

class Contradiction(BaseModel):
    """A detected contradiction between documents."""
    topic: str
    document_1: str
    claim_1: str
    document_2: str
    claim_2: str


class DocumentSummary(BaseModel):
    """Per-document summary for multi-doc synthesis."""
    source_file: str
    summary: str
    key_findings: list[str]
    relevant_chunks: int


class SynthesisResult(BaseModel):
    """Multi-document synthesis result."""
    query: str
    num_documents: int
    document_summaries: dict[str, str]  # source_file -> summary
    contradictions_found: bool
    contradictions: list[Contradiction]


# ============= HYDE Models =============

class HYDEResult(BaseModel):
    """Hypothetical Document Embeddings result."""
    original_query: str
    hypothetical_documents: list[str]  # LLM-generated hypothetical answers
    enhanced_retrieval: bool


# ============= Agent State Models =============

class AgentNodeTiming(BaseModel):
    """Timing information for a single node."""
    node_name: str
    start_time: float
    end_time: float
    duration_ms: float


class AgentExecutionMetadata(BaseModel):
    """Metadata about agent execution."""
    total_time_ms: float
    node_timings: list[AgentNodeTiming]
    nodes_executed: list[str]
    loops_taken: int
    strategy_used: Literal["simple", "crag", "srag", "advanced"]


# ============= Memory Models =============

class ConversationMemory(BaseModel):
    """A stored conversation for long-term memory."""
    session_id: str
    query: str
    answer: str
    timestamp: datetime
    embedding_id: str


class MemorySummary(BaseModel):
    """Summary of conversation history."""
    session_id: str
    summary_text: str
    messages_summarized: int
    timestamp: datetime


# ============= Full Query Response =============

class QueryResponse(BaseModel):
    """Complete response to a user query."""
    query: str
    answer: str
    mode: Literal["simple", "crag", "srag", "advanced", "auto"]
    search_mode: Literal["dense", "sparse", "hybrid"]
    sources: list[RetrievedChunk]
    citations: list[Citation]
    grounding: Optional[GroundingResult] = None
    crag_details: Optional[CRAGResult] = None
    reflection_details: Optional[SelfReflectiveResult] = None
    synthesis_details: Optional[SynthesisResult] = None
    hyde_details: Optional[HYDEResult] = None
    response_time_ms: float
    hyde_used: bool = False
    reranking_used: bool = True
    compression_used: bool = True
    initial_retrieval_count: int
    final_retrieval_count: int
    
    # Agent execution metadata
    agent_metadata: Optional[AgentExecutionMetadata] = None
    memory_used: bool = False
    context_window_usage: float = 0.0


# ============= Benchmark Models =============

class BenchmarkResult(BaseModel):
    """Vector DB benchmark result."""
    db_name: Literal["qdrant", "milvus"]
    operation: Literal["index", "search", "delete"]
    num_documents: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    recall_at_k: Optional[dict[int, float]] = None  # k -> recall
    timestamp: datetime
