# Implementation Guide - Visual RAG Document Explorer

This document supplements `spec.md` and `architecture.md` with concrete implementation details: interface signatures, prompt templates, and example data flows that an AI coding agent needs to execute the tasks.

## 1. Interface Signatures

### 1.1 Vector Store Base Class

```python
# core/vectordb/base.py
from abc import ABC, abstractmethod
from typing import Optional
from config.models import ChunkMetadata, RetrievedChunk


class VectorStoreBase(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    async def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection/index."""
        ...

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data."""
        ...

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> int:
        """Insert or update documents. Returns count of upserted docs."""
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Search by vector similarity with optional metadata filters."""
        ...

    @abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> int:
        """Delete documents by ID. Returns count of deleted docs."""
        ...

    @abstractmethod
    async def get(self, collection: str, ids: list[str]) -> list[dict]:
        """Get documents by ID."""
        ...

    @abstractmethod
    async def count(self, collection: str) -> int:
        """Count documents in a collection."""
        ...

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """List all collection names."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the database is reachable."""
        ...
```

### 1.2 Embedding Service Interface

```python
# core/embeddings/base.py (implicit in embedding_router.py)
from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
```

### 1.3 NER Extractor Interface

```python
# core/document_processing/ner_base.py (implicit in ner_router.py)
from abc import ABC, abstractmethod
from config.models import NEREntities


class NERExtractor(ABC):
    """Abstract interface for NER extraction."""

    @abstractmethod
    def extract(self, text: str) -> NEREntities:
        """Extract named entities from text."""
        ...

    @abstractmethod
    def extract_batch(self, texts: list[str]) -> list[NEREntities]:
        """Extract entities from multiple texts."""
        ...

    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """Return the extractor identifier."""
        ...
```

### 1.4 RAG Strategy Interface

```python
# core/rag/base.py
from abc import ABC, abstractmethod
from config.models import QueryRequest, QueryResponse


class RAGStrategy(ABC):
    """Abstract interface for RAG strategies."""

    @abstractmethod
    async def execute(self, request: QueryRequest) -> QueryResponse:
        """Execute the RAG pipeline for a given query."""
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the strategy identifier."""
        ...
```

### 1.5 LLM Provider Interface

```python
# core/llm/base.py (implicit in llm_router.py)
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response from the LLM."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        """Generate a streaming response from the LLM."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
```

## 2. LLM Prompt Templates

### 2.1 Query Classification Prompt

```
SYSTEM: You are a query classifier for a document retrieval system. Classify the user query into one of four categories based on its complexity and requirements.

Categories:
- "simple": Direct factual question answerable from a single passage. Example: "What is the company revenue?"
- "complex": Multi-faceted question that may need corrective retrieval if initial results are poor. Example: "What are the main risks mentioned in the compliance report?"
- "analytical": Question requiring deep analysis, verification, and self-reflection. Example: "How does the Q1 strategy compare to industry best practices?"
- "multi_doc": Question that explicitly or implicitly requires information from multiple documents. Example: "Compare the financial performance across all quarterly reports."

Also determine if the query needs decomposition into sub-queries (true/false).

USER: {query}

Respond in JSON format:
{
  "query_type": "simple|complex|analytical|multi_doc",
  "needs_decomposition": true|false,
  "reasoning": "brief explanation"
}
```

### 2.2 Query Decomposition Prompt

```
SYSTEM: You are a query decomposition expert. Break down the following complex query into 2-5 focused sub-queries that, when answered individually, will provide all the information needed to answer the original query.

Each sub-query should:
- Target a specific aspect of the original question
- Be self-contained and searchable
- Not overlap significantly with other sub-queries

USER: Original query: {query}

Respond in JSON format:
{
  "sub_queries": ["sub-query 1", "sub-query 2", ...],
  "reasoning": "brief explanation of decomposition strategy"
}
```

### 2.3 CRAG Relevance Grading Prompt

```
SYSTEM: You are a relevance grader for a document retrieval system. Evaluate whether the retrieved document is relevant to the user query.

Score the relevance on a scale of 0.0 to 1.0:
- 0.0-0.3: Irrelevant - document has no useful information for the query
- 0.3-0.6: Ambiguous - document has some tangentially related information
- 0.6-1.0: Relevant - document directly addresses the query

USER:
Query: {query}
Document: {document_content}

Respond in JSON format:
{
  "relevance_score": 0.0-1.0,
  "relevance_label": "relevant|ambiguous|irrelevant",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
```

### 2.4 CRAG Query Reformulation Prompt

```
SYSTEM: The initial retrieval for the following query returned low-relevance results. Reformulate the query to improve retrieval. You may:
1. Rephrase using different terminology
2. Broaden the scope slightly
3. Focus on key concepts

USER:
Original query: {query}
Low-relevance documents summary: {doc_summaries}

Respond in JSON format:
{
  "reformulated_query": "new query text",
  "strategy": "reformulate|broaden|decompose",
  "reasoning": "brief explanation"
}
```

### 2.5 Self-Reflection Prompt (SRAG)

```
SYSTEM: You are a quality evaluator for AI-generated answers. Evaluate the following answer against the source documents for:

1. **Hallucination**: Does the answer contain claims not supported by the sources?
2. **Completeness**: Does the answer address all aspects of the query?
3. **Faithfulness**: Does the answer accurately represent the source content without distortion?

USER:
Query: {query}
Generated Answer: {answer}
Source Documents:
{sources}

Respond in JSON format:
{
  "hallucination_detected": true|false,
  "hallucination_details": "specific claims not in sources, or empty",
  "completeness_score": 0.0-1.0,
  "faithfulness_score": 0.0-1.0,
  "needs_regeneration": true|false,
  "feedback": "specific suggestions for improvement",
  "overall_score": 0.0-1.0
}
```

### 2.6 Answer Generation Prompt

```
SYSTEM: You are a helpful document assistant. Answer the user query based ONLY on the provided source documents. Follow these rules:
1. Only use information from the provided sources
2. If the sources don't contain enough information, say so explicitly
3. Cite your sources using [Source: filename, Page: X] format
4. Be precise and factual
5. If sources contradict each other, note the contradiction with both sources cited

USER:
Query: {query}

Source Documents:
{formatted_sources}

Previous conversation context:
{chat_history}

Provide a comprehensive answer with citations.
```

### 2.7 Grounding Verification Prompt

```
SYSTEM: You are a fact-checker. Extract individual claims from the answer and verify each against the source documents.

For each claim, determine:
- "grounded": The claim is directly supported by a specific source passage
- "partially_grounded": The claim is partially supported but includes inference
- "ungrounded": The claim cannot be traced to any source

USER:
Answer: {answer}
Source Documents:
{sources}

Respond in JSON format:
{
  "claims": [
    {
      "claim": "extracted claim text",
      "status": "grounded|partially_grounded|ungrounded",
      "supporting_source": "source reference or null",
      "confidence": 0.0-1.0
    }
  ],
  "overall_grounding_score": 0.0-1.0
}
```

### 2.8 Contextual Compression Prompt

```
SYSTEM: Extract only the sentences from the following document that are directly relevant to answering the user query. Remove all irrelevant content. Preserve the exact wording of relevant sentences.

USER:
Query: {query}
Document: {document_content}

Return only the relevant sentences, preserving their original wording.
```

### 2.9 Document Summarization Prompt

```
SYSTEM: Generate a concise summary of the following document. The summary should:
1. Capture the main topics and key findings
2. Be 3-5 sentences long
3. Include the most important facts, figures, and conclusions
4. Be useful for understanding the document's scope at a glance

USER:
Document title: {filename}
Document content:
{content}

Provide a concise summary.
```

### 2.10 Multi-Document Synthesis Prompt

```
SYSTEM: You are synthesizing information from multiple documents to answer a query. Follow these steps:
1. Consider each document's contribution to the answer
2. Identify any contradictions between documents
3. Combine complementary information into a unified answer
4. Explicitly note contradictions with source attribution
5. Cite each claim to its specific source document

USER:
Query: {query}

Document Summaries:
{per_document_summaries}

Relevant Passages:
{relevant_passages}

Provide a synthesized answer that combines information from all sources, noting any contradictions.
```

### 2.11 HYDE Hypothetical Document Prompt

```
SYSTEM: Generate a hypothetical document passage that would perfectly answer the following query. This passage will be used to find similar real documents. Write it as if it were an excerpt from an actual document.

USER: {query}

Write a 2-3 paragraph hypothetical passage that answers this query.
```

## 3. Example Data Flow

### 3.1 End-to-End Query Flow

**User Query**: "Compare the revenue growth strategies mentioned in the Q1 and Q2 reports"

**Step 1 - Load Memory**:
- Retrieve last 10 messages from session state
- Search vector DB for relevant past conversations
- State update: `chat_history` populated

**Step 2 - Query Analysis**:
- LLM classifies as `multi_doc` with `needs_decomposition: true`
- State update: `query_type: "multi_doc"`, `is_multi_doc: true`

**Step 3 - Query Decomposition**:
- Sub-queries generated:
  1. "What revenue growth strategies are mentioned in the Q1 report?"
  2. "What revenue growth strategies are mentioned in the Q2 report?"
  3. "How do Q1 and Q2 revenue strategies differ?"
- State update: `sub_queries` populated

**Step 4 - Multi-Doc Retrieve**:
- For each sub-query, run hybrid search:
  - Dense: Voyage AI embedding → Qdrant similarity search
  - Sparse: BM25 keyword search
  - Metadata: Filter by entities containing "Q1 report" / "Q2 report"
  - RRF fusion of all three result sets
- State update: `retrieved_docs` with ~20 chunks per sub-query

**Step 5 - Rerank**:
- Cohere reranker scores all retrieved chunks against original query
- Top 10 chunks selected (5 per sub-query approximately)
- State update: `reranked_docs`

**Step 6 - Contextual Compression**:
- For each of 10 chunks, extract only revenue-strategy-relevant sentences
- State update: `compressed_docs` with reduced content

**Step 7 - Multi-Doc Synthesis**:
- Group chunks by source document (Q1 report vs Q2 report)
- Generate per-document summary of revenue strategies
- Detect contradictions (e.g., Q1 says "expand internationally", Q2 says "focus domestic")
- State update: synthesis metadata populated

**Step 8 - Generation**:
- LLM generates comparative answer using compressed docs + synthesis context
- Includes citations: [Source: Q1_Report.pdf, Page: 12], [Source: Q2_Report.pdf, Page: 8]
- State update: `draft_answer`

**Step 9 - Grounding Verification**:
- Extract 6 claims from the answer
- 5 claims grounded, 1 partially grounded
- Grounding score: 0.92
- State update: `grounding_score: 0.92`, `grounding_details` populated

**Step 10 - Citation Formatting**:
- Map each grounded claim to source file, page, chunk, relevant text
- State update: `citations` list populated

**Step 11 - Save Memory**:
- Save Q&A pair to session state
- Check if context window needs summarization

**Step 12 - Format Response**:
- Compile: answer + citations + grounding score + RAG strategy used
- Return `QueryResponse` to Streamlit UI

### 3.2 Document Ingestion Flow

**Input**: User uploads "Q1_Financial_Report.pdf"

1. **Format Detection**: `.pdf` → `PyPDFLoader`
2. **Text Extraction**: Extract text from all pages, preserve page numbers
3. **Basic Metadata**: filename, file_type, page_count, upload_timestamp
4. **GLiNER2 NER**: Extract entities from full document text
   - People: ["John Smith", "Jane Doe"]
   - Organizations: ["Acme Corp", "SEC"]
   - Dates: ["Q1 2025", "March 15, 2025"]
   - Topics: ["revenue growth", "market expansion"]
5. **spaCy NER**: Extract standard entities
   - PERSON: ["John Smith", "Jane Doe"]
   - ORG: ["Acme Corp", "SEC", "NYSE"]
   - DATE: ["Q1 2025", "March 15, 2025", "fiscal year 2024"]
6. **Ensemble Merge**: Combine, deduplicate, assign confidence scores
7. **Document Summary**: LLM generates 3-sentence summary
8. **Content Hash**: SHA-256 of normalized full text
9. **Dedup Check**: Compare hash against existing documents → "new"
10. **Adaptive Chunking**: Split into ~15 chunks (avg 512 tokens each)
11. **Per-Chunk NER**: Extract entities per chunk (subset of document entities)
12. **Embedding**: Generate Voyage AI embeddings for each chunk + summary
13. **BM25 Index**: Add chunk texts to BM25 index
14. **Vector Store Upsert**: Store embeddings + metadata in Qdrant (and/or Milvus)
15. **Return**: `UploadResponse` with chunk count, entities, summary, dedup status

## 4. Configuration Schema

```python
# config/settings.py
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    voyage_api_key: str = ""
    cohere_api_key: str = ""

    # LLM Settings
    default_llm_provider: Literal["openai", "openrouter"] = "openai"
    default_model: str = "gpt-4o"
    openrouter_model: str = "anthropic/claude-sonnet-4"
    temperature: float = 0.0
    max_tokens: int = 2048

    # Embedding Settings
    default_embedding_model: Literal["voyage", "bge-m3"] = "voyage"
    voyage_model: str = "voyage-3"
    bge_model: str = "BAAI/bge-m3"

    # Vector DB Settings
    default_vector_db: Literal["qdrant", "milvus"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    milvus_url: str = "http://localhost:19530"
    milvus_collection: str = "documents"

    # Chunking Settings
    chunk_size_min: int = 256
    chunk_size_max: int = 1024
    chunk_overlap: int = 128

    # NER Settings
    ner_mode: Literal["gliner", "spacy", "ensemble"] = "ensemble"
    gliner_model: str = "urchade/gliner_multi_pii-v1"
    spacy_model: str = "en_core_web_trf"
    custom_entity_types: list[str] = []

    # Search Settings
    search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    dense_weight: float = 0.5
    sparse_weight: float = 0.3
    metadata_weight: float = 0.2

    # RAG Settings
    default_rag_strategy: Literal["simple", "crag", "srag", "advanced", "auto"] = "auto"
    rerank_top_k: int = 20
    final_top_k: int = 5
    enable_hyde: bool = False
    enable_compression: bool = True
    srag_max_iterations: int = 3
    dedup_similarity_threshold: float = 0.95
    grounding_threshold: float = 0.7

    # Memory Settings
    short_term_memory_size: int = 20
    context_window_threshold: float = 0.6
    enable_long_term_memory: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

## 5. Docker Compose Full Configuration

```yaml
version: "3.8"

services:
  qdrant:
    image: qdrant/qdrant:v1.12.1
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: milvus-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ./data/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3
    restart: unless-stopped

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: milvus-minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ./data/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    restart: unless-stopped

  milvus:
    image: milvusdb/milvus:v2.4.0
    container_name: milvus-standalone
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./data/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
    restart: unless-stopped
```

## 6. Key Library Usage Patterns

### 6.1 LangGraph State Graph Pattern

```python
from langgraph.graph import StateGraph, END
from agents.state import AgentState

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("analyze_query", analyze_query_node)
    # ... all 15 nodes
    
    # Set entry
    graph.set_entry_point("load_memory")
    
    # Add edges
    graph.add_edge("load_memory", "analyze_query")
    
    # Conditional edges
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,  # function that returns next node name
        {
            "decompose": "decompose_query",
            "retrieve_simple": "retrieve_simple",
            "retrieve_crag": "retrieve_crag",
            "retrieve_srag": "retrieve_srag",
            "retrieve_multi": "retrieve_multi",
        }
    )
    
    # ... more edges
    
    graph.add_edge("format_response", END)
    
    return graph.compile()
```

### 6.2 Streamlit Multi-Page Pattern

```python
# app.py
import streamlit as st

st.set_page_config(page_title="Visual RAG Explorer", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "settings" not in st.session_state:
    st.session_state.settings = Settings()

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Chat", "Upload", "Explorer", "Settings", "Benchmark"]
)

# Page routing
if page == "Chat":
    from ui.pages.chat import render
    render()
elif page == "Upload":
    from ui.pages.upload import render
    render()
# ... etc
```

### 6.3 Hybrid Search with RRF Pattern

```python
def reciprocal_rank_fusion(
    results_lists: list[list[tuple[str, float]]],
    k: int = 60
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using RRF."""
    fused_scores = {}
    for results in results_lists:
        for rank, (doc_id, _score) in enumerate(results):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (k + rank + 1)
    
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```
