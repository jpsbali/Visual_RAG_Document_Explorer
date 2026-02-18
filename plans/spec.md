# Visual RAG Document Explorer - Specification

## Overview

A sophisticated document exploration system that ingests PDFs, Word docs, TXT, HTML, and JSON files into a searchable, conversational interface powered by advanced RAG strategies (CRAG, SRAG, Advanced RAG with RSF), multi-document synthesis agents via LangGraph, dual vector database support (Qdrant + Milvus), GLiNER2/spaCy NER-based metadata extraction, hybrid search, and a Streamlit web UI. Designed for local deployment with OpenAI and OpenRouter LLM providers.

## Requirements

### Functional Requirements

- FR1: System SHALL ingest documents in PDF, Word (.docx), TXT, HTML, and JSON formats, supporting up to 500 documents
- FR2: System SHALL extract text content from all supported document formats using appropriate loaders (PyPDFLoader, Docx2txtLoader, TextLoader, BSHTMLLoader, JSONLoader)
- FR3: System SHALL perform adaptive chunking with configurable minimum (default 256) and maximum (default 1024) chunk sizes and configurable overlap (default 128)
- FR4: System SHALL extract named entities from documents using GLiNER2 (zero-shot NER) and spaCy (pre-trained NER) with an ensemble/router mode
- FR5: System SHALL generate document-level summaries using the configured LLM and store them as special summary chunks in the vector database
- FR6: System SHALL generate embeddings using Voyage AI (voyage-3) as primary and BGE-M3 as fallback/alternative, with a configurable router
- FR7: System SHALL store document embeddings and metadata in both Qdrant and Milvus vector databases with an abstraction layer for transparent switching
- FR8: System SHALL perform hybrid search combining dense vector search, BM25 sparse keyword search, and metadata filtering using Reciprocal Rank Fusion
- FR9: System SHALL detect and handle duplicate documents using content-hash (SHA-256) for exact duplicates and cosine similarity threshold (default 0.95) for near-duplicates
- FR10: System SHALL support incremental indexing - adding and removing documents without full re-indexing
- FR11: System SHALL implement Simple RAG, Corrective RAG (CRAG), Self-Reflective RAG (SRAG), and Advanced RAG with Relevance Score Fusion strategies
- FR12: System SHALL apply Cohere reranker (rerank-v3.5) after initial retrieval, reducing from top-k (default 20) to final-k (default 5) results
- FR13: System SHALL support HYDE (Hypothetical Document Embeddings) as an optional query expansion technique
- FR14: System SHALL decompose complex multi-part queries into focused sub-queries for independent retrieval and merged results
- FR15: System SHALL apply contextual compression after retrieval to extract only relevant portions of each chunk
- FR16: System SHALL verify answer grounding by extracting claims and checking each against source documents, producing a grounding score
- FR17: System SHALL use LangGraph to orchestrate a 15-node agent graph with conditional routing based on query complexity
- FR18: System SHALL classify queries into simple, complex, analytical, or multi-doc types and route to the appropriate RAG strategy
- FR19: System SHALL synthesize information across multiple documents, detecting contradictions and providing balanced answers with cross-referenced citations
- FR20: System SHALL maintain conversational memory with short-term (session state, last N messages) and long-term (vector DB conversation summaries) storage
- FR21: System SHALL manage context window size using tiktoken, automatically summarizing older messages when history exceeds 60% of context window
- FR22: System SHALL track and format source citations mapping each claim to specific source documents, pages, and chunks
- FR23: System SHALL provide a Streamlit web UI with Chat, Upload, Explorer, Settings, and Benchmark pages
- FR24: System SHALL support streaming responses in the chat interface
- FR25: System SHALL display grounding scores and citation details alongside answers in the UI
- FR26: System SHALL provide a benchmark dashboard comparing Qdrant vs Milvus performance (latency p50/p95/p99, throughput, recall@k, memory usage)
- FR27: System SHALL support both OpenAI and OpenRouter as LLM providers with configurable model selection

### Non-Functional Requirements

- NFR1: System SHALL run locally without cloud dependencies (except LLM API calls)
- NFR2: System SHALL use Docker Compose for Qdrant and Milvus infrastructure
- NFR3: System SHALL use Pydantic models for all data validation and serialization
- NFR4: System SHALL store configuration via environment variables with .env file support
- NFR5: System SHALL be single-user with future extensibility for per-user vector collections
- NFR6: System SHALL handle documents up to 500 in a single collection
- NFR7: System SHALL provide configurable parameters for all major pipeline components via the Settings UI

## Design Decisions

### DD1: Dual Vector Database Architecture
- **Decision**: Support both Qdrant and Milvus behind an abstract interface
- **Rationale**: Enables performance benchmarking, allows users to choose based on their needs, and provides fallback capability
- **Implementation**: Abstract `VectorStoreBase` class with `QdrantStore` and `MilvusStore` implementations, plus a `VDBRouter` for transparent switching

### DD2: GLiNER2 + spaCy NER Ensemble
- **Decision**: Use both GLiNER2 and spaCy for named entity recognition with three modes (GLiNER only, spaCy only, ensemble)
- **Rationale**: GLiNER2 excels at zero-shot custom entity types; spaCy excels at speed and standard entity types. Ensemble maximizes recall with confidence scoring for entities found by both
- **Implementation**: `NERRouter` in `core/document_processing/ner_router.py` with configurable mode selection

### DD3: LangGraph for Agent Orchestration
- **Decision**: Use LangGraph StateGraph with 15 nodes and conditional edges rather than simple chain-based RAG
- **Rationale**: Enables complex routing (CRAG correction loops, SRAG reflection loops), multi-document synthesis, and extensible agent behavior
- **Implementation**: `AgentState` TypedDict flows through all nodes; conditional edges handle strategy-specific branching

### DD4: Hybrid Search with RRF
- **Decision**: Combine dense vector search + BM25 sparse search + metadata filtering using Reciprocal Rank Fusion
- **Rationale**: Dense search captures semantic similarity; sparse search catches exact keyword matches the dense model might miss; metadata filtering leverages NER entities for precision
- **Implementation**: `HybridSearch` class in `core/search/hybrid_search.py` merging results from all three sources

### DD5: Multi-Stage Answer Quality Pipeline
- **Decision**: Apply reranking → contextual compression → generation → grounding verification as sequential quality stages
- **Rationale**: Each stage progressively improves answer accuracy: reranking improves retrieval precision, compression reduces noise, grounding verification catches hallucinations
- **Implementation**: Separate LangGraph nodes for each stage with Pydantic models tracking quality metrics

### DD6: Pydantic Models for All Data Flow
- **Decision**: Use comprehensive Pydantic models for all request/response types, chunk metadata, evaluation results, and pipeline outputs
- **Rationale**: Provides type safety, validation, serialization, and clear API contracts between components
- **Implementation**: All models defined in `config/models.py` including QueryRequest, QueryResponse, ChunkMetadata, CRAGResult, ReflectionResult, GroundingResult, etc.

### DD7: Streamlit Multi-Page Application
- **Decision**: Use Streamlit with multi-page layout for the UI
- **Rationale**: Rapid development, built-in session state for memory, native support for streaming, and sufficient for local single-user deployment
- **Implementation**: `app.py` entry point with pages in `ui/pages/` and reusable components in `ui/components/`

## Tasks

### Phase 1: Foundation

- TASK-001: Create project scaffolding with directory structure, pyproject.toml with all dependencies, .env.example, .gitignore, docker-compose.yml for Qdrant and Milvus, and Makefile
  - Depends on: None
  - Acceptance: Project installs cleanly with `pip install -e .`, Docker services start with `docker-compose up`

- TASK-002: Implement Pydantic data models in config/models.py covering QueryRequest, UploadResponse, NEREntities, ChunkMetadata, RetrievedChunk, CRAGEvaluation, CRAGResult, ReflectionResult, SelfReflectiveResult, ClaimVerification, GroundingResult, Citation, SynthesisResult, HYDEResult, QueryResponse, BenchmarkResult
  - Depends on: TASK-001
  - Acceptance: All models instantiate correctly with valid data and reject invalid data

- TASK-003: Implement configuration management in config/settings.py using pydantic-settings with environment variable loading for all API keys, LLM settings, embedding settings, vector DB settings, chunking settings, and RAG settings
  - Depends on: TASK-001
  - Acceptance: Settings load from .env file and environment variables with proper defaults

- TASK-004: Implement document loaders in core/document_processing/loaders.py supporting PDF (PyPDFLoader/PDFPlumber), Word (Docx2txtLoader), TXT (TextLoader), HTML (BSHTMLLoader), and JSON (JSONLoader) with automatic format detection
  - Depends on: TASK-001
  - Acceptance: Each loader correctly extracts text and basic metadata from test documents

- TASK-005: Implement adaptive chunker in core/document_processing/chunker.py with configurable chunk_size_min, chunk_size_max, and chunk_overlap parameters using RecursiveCharacterTextSplitter with semantic awareness
  - Depends on: TASK-004
  - Acceptance: Chunks respect min/max size constraints, preserve metadata, and include content hash

- TASK-006: Implement GLiNER2 metadata extractor in core/document_processing/gliner_extractor.py extracting people, organizations, dates, locations, topics, and configurable custom entity types
  - Depends on: TASK-002
  - Acceptance: Extracts entities from sample text with deduplication and normalization

- TASK-007: Implement spaCy metadata extractor in core/document_processing/spacy_extractor.py using en_core_web_trf model for PERSON, ORG, DATE, GPE, MONEY, and other standard entity types plus noun chunk extraction
  - Depends on: TASK-002
  - Acceptance: Extracts standard entities from sample text matching spaCy expected output

- TASK-008: Implement NER router in core/document_processing/ner_router.py with three modes: gliner_only, spacy_only, ensemble (merge and deduplicate with confidence scoring)
  - Depends on: TASK-006, TASK-007
  - Acceptance: All three modes produce valid NEREntities output; ensemble mode merges results correctly

- TASK-009: Implement document summarizer in core/document_processing/summarizer.py using LLM-based summarization with map-reduce for long documents, storing summaries as special chunks with chunk_type=summary
  - Depends on: TASK-004, TASK-013
  - Acceptance: Generates concise summaries for documents of varying lengths

- TASK-010: Implement Voyage AI embedding service in core/embeddings/voyage_embeddings.py wrapping langchain-voyageai for voyage-3 model
  - Depends on: TASK-003
  - Acceptance: Generates 1024-dimensional embeddings for text input

- TASK-011: Implement BGE-M3 embedding service in core/embeddings/bge_m3_embeddings.py using sentence-transformers for local embedding generation
  - Depends on: TASK-001
  - Acceptance: Generates embeddings locally without API calls

- TASK-012: Implement embedding router in core/embeddings/embedding_router.py for selecting between Voyage AI and BGE-M3 based on configuration
  - Depends on: TASK-010, TASK-011
  - Acceptance: Routes to correct embedding model based on settings

- TASK-013: Implement LLM providers in core/llm/ with openai_provider.py, openrouter_provider.py, and llm_router.py for provider selection
  - Depends on: TASK-003
  - Acceptance: Both providers generate responses; router selects based on configuration

### Phase 2: Vector Database and Search Layer

- TASK-014: Implement abstract vector store interface in core/vectordb/base.py defining VectorStoreBase with methods for upsert, search, delete, get, list_collections, and health_check
  - Depends on: TASK-002
  - Acceptance: Abstract class defines complete interface with type hints

- TASK-015: Implement Qdrant store in core/vectordb/qdrant_store.py with named vectors, payload filtering, batch upsert, and collection management
  - Depends on: TASK-014, TASK-001 (Docker)
  - Acceptance: CRUD operations work against running Qdrant instance

- TASK-016: Implement Milvus store in core/vectordb/milvus_store.py with dynamic schema, partition management, and batch operations
  - Depends on: TASK-014, TASK-001 (Docker)
  - Acceptance: CRUD operations work against running Milvus instance

- TASK-017: Implement VDB router in core/vectordb/router.py for transparent switching between Qdrant and Milvus based on configuration
  - Depends on: TASK-015, TASK-016
  - Acceptance: Same operations produce equivalent results on both backends

- TASK-018: Implement BM25 search index in core/search/bm25_search.py using rank-bm25 library with index building at ingestion time
  - Depends on: TASK-005
  - Acceptance: Keyword search returns relevant results for exact term matches

- TASK-019: Implement hybrid search in core/search/hybrid_search.py combining dense vector search, BM25 sparse search, and metadata filtering with Reciprocal Rank Fusion scoring
  - Depends on: TASK-017, TASK-018
  - Acceptance: Hybrid search returns better results than any single method alone on test queries

- TASK-020: Implement deduplication engine in core/document_processing/deduplication.py with content-hash (SHA-256) exact dedup and cosine similarity semantic dedup with configurable threshold
  - Depends on: TASK-012, TASK-017
  - Acceptance: Detects exact and near-duplicate documents correctly

- TASK-021: Implement incremental indexing support in the vector store layer allowing add/remove of individual documents without full re-index
  - Depends on: TASK-017, TASK-020
  - Acceptance: Adding and removing documents updates both vector store and BM25 index correctly

- TASK-022: Implement vector DB benchmark in core/vectordb/benchmark.py measuring indexing throughput, query latency (p50/p95/p99), recall@k, and memory usage for both Qdrant and Milvus
  - Depends on: TASK-017
  - Acceptance: Produces BenchmarkResult models with valid metrics for both databases

### Phase 3: RAG Engine and Accuracy Features

- TASK-023: Implement simple RAG pipeline in core/rag/simple_rag.py with retrieve → rerank → generate flow
  - Depends on: TASK-019, TASK-013
  - Acceptance: Generates answers with source citations for simple queries

- TASK-024: Implement Cohere reranker in core/reranking/cohere_reranker.py using rerank-v3.5 with configurable top-k and score threshold
  - Depends on: TASK-003
  - Acceptance: Reranks retrieved documents and improves relevance ordering

- TASK-025: Implement HYDE query expansion in core/rag/hyde.py generating hypothetical answer documents via LLM, embedding them, and using for retrieval
  - Depends on: TASK-012, TASK-013
  - Acceptance: HYDE-enhanced retrieval returns relevant results for abstract queries

- TASK-026: Implement query decomposer in core/rag/query_decomposer.py breaking complex queries into 2-5 focused sub-queries using LLM
  - Depends on: TASK-013
  - Acceptance: Complex multi-part queries are correctly decomposed into focused sub-queries

- TASK-027: Implement contextual compressor in core/rag/contextual_compressor.py extracting only relevant sentences from retrieved chunks using LLM-based extraction
  - Depends on: TASK-013
  - Acceptance: Compressed chunks contain only query-relevant content with reduced token count

- TASK-028: Implement Corrective RAG in core/rag/corrective_rag.py with LLM-as-judge relevance grading, corrective retrieval via query reformulation, and broader search when relevance is low
  - Depends on: TASK-023, TASK-024
  - Acceptance: CRAG produces CRAGResult with correct evaluation and optional correction

- TASK-029: Implement Self-Reflective RAG in core/rag/self_reflective_rag.py with draft generation, self-evaluation for hallucination/completeness/faithfulness, and iterative refinement up to max iterations
  - Depends on: TASK-023, TASK-024
  - Acceptance: SRAG produces SelfReflectiveResult with reflection history and improved answers

- TASK-030: Implement Advanced RAG with RSF in core/rag/advanced_rag.py with multi-query retrieval, Reciprocal Rank Fusion across retrieval passes, and Cohere reranking
  - Depends on: TASK-024, TASK-026
  - Acceptance: Advanced RAG produces higher quality results than simple RAG on complex queries

- TASK-031: Implement answer grounding verifier in core/rag/grounding_verifier.py extracting claims from answers, verifying each against source chunks, and computing grounding scores
  - Depends on: TASK-013
  - Acceptance: Produces GroundingResult with per-claim verification and accurate grounding score

- TASK-032: Implement RAG strategy router in core/rag/rag_router.py selecting between simple, CRAG, SRAG, and advanced strategies based on query analysis or explicit user selection
  - Depends on: TASK-023, TASK-028, TASK-029, TASK-030
  - Acceptance: Routes to correct strategy based on query type or configuration

### Phase 4: Agent Layer

- TASK-033: Define AgentState TypedDict in agents/state.py with all fields for query analysis, retrieval, generation, quality control, metadata, and routing decisions
  - Depends on: TASK-002
  - Acceptance: State schema supports all 15 graph nodes with proper typing

- TASK-034: Implement all 15 LangGraph nodes in agents/nodes/ (memory_load, routing, decomposition, retrieval, grading, correction, reranking, compression, synthesis, generation, reflection, grounding, citation, memory_save, format)
  - Depends on: TASK-033, TASK-019, TASK-024, TASK-026, TASK-027, TASK-028, TASK-029, TASK-031
  - Acceptance: Each node correctly reads from and writes to AgentState

- TASK-035: Compile LangGraph state graph in agents/graph.py with all nodes, conditional edges for CRAG correction loop, SRAG reflection loop, query decomposition branching, and strategy-based routing
  - Depends on: TASK-034
  - Acceptance: Graph compiles and executes end-to-end for all query types

- TASK-036: Implement multi-document synthesis in agents/nodes/synthesis.py with per-document grouping, summary generation, contradiction detection, and unified answer generation
  - Depends on: TASK-034
  - Acceptance: Produces SynthesisResult with document summaries, contradictions, and synthesized answer

- TASK-037: Implement conversational memory management with short-term (session state, last N messages) and long-term (vector DB conversation summaries) storage, plus context window management via tiktoken
  - Depends on: TASK-034, TASK-017
  - Acceptance: Memory persists across turns, summarizes when context exceeds threshold

- TASK-038: Implement citation tracking in agents/nodes/citation.py mapping grounded claims to source documents with file, page, chunk, and relevant text
  - Depends on: TASK-034, TASK-031
  - Acceptance: Produces Citation list with accurate source attribution

### Phase 5: Streamlit UI

- TASK-039: Implement chat interface in ui/pages/chat.py with streaming responses, grounding score display, citation viewer, RAG strategy indicator, and conversation history
  - Depends on: TASK-035
  - Acceptance: Users can chat with documents, see streaming answers with citations and grounding scores

- TASK-040: Implement document upload page in ui/pages/upload.py with drag-and-drop, progress tracking, format validation, deduplication feedback, and entity preview
  - Depends on: TASK-004, TASK-008, TASK-020
  - Acceptance: Documents upload, process, and display extracted entities and dedup status

- TASK-041: Implement document explorer in ui/pages/explorer.py with document browsing, chunk inspection, metadata viewing, and entity-based filtering
  - Depends on: TASK-017
  - Acceptance: Users can browse documents, view chunks, and filter by metadata/entities

- TASK-042: Implement settings page in ui/pages/settings.py for configuring LLM provider, model, embedding model, vector DB backend, chunking parameters, RAG strategy, NER mode, and all other configurable options
  - Depends on: TASK-003
  - Acceptance: Settings changes are reflected in system behavior

- TASK-043: Implement benchmark dashboard in ui/pages/benchmark.py with Plotly charts comparing Qdrant vs Milvus on latency, throughput, recall@k, and memory usage
  - Depends on: TASK-022
  - Acceptance: Dashboard displays benchmark results with interactive charts

- TASK-044: Implement Streamlit app entry point in app.py with sidebar navigation, page routing, and session state initialization
  - Depends on: TASK-039, TASK-040, TASK-041, TASK-042, TASK-043
  - Acceptance: App launches with `streamlit run app.py` and all pages are accessible

### Phase 6: Testing and Documentation

- TASK-045: Write unit tests for document processing (loaders, chunker, dedup, NER extractors)
  - Depends on: TASK-004, TASK-005, TASK-006, TASK-007, TASK-020
  - Acceptance: All tests pass with >80% coverage for document processing module

- TASK-046: Write unit tests for RAG pipelines (simple, CRAG, SRAG, advanced, grounding)
  - Depends on: TASK-023, TASK-028, TASK-029, TASK-030, TASK-031
  - Acceptance: All tests pass with mocked LLM responses

- TASK-047: Write integration tests for end-to-end query flow through LangGraph agent
  - Depends on: TASK-035
  - Acceptance: Integration tests verify complete query-to-answer flow

- TASK-048: Write README.md with project overview, setup instructions, configuration guide, usage examples, and architecture overview
  - Depends on: TASK-044
  - Acceptance: New user can set up and run the system following README instructions
