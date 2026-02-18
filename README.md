# Visual RAG Document Explorer

A sophisticated document exploration system that ingests PDFs, Word docs, TXT, HTML, and JSON files into a searchable, conversational interface powered by advanced RAG strategies, multi-document synthesis agents, and dual vector database support.

## Features

### Core Capabilities
- **Multi-Format Document Processing**: PDF, DOCX, TXT, HTML, JSON
- **Advanced RAG Strategies**: Simple, CRAG, SRAG, Advanced with RSF
- **Dual Vector Database Support**: Qdrant and Milvus with side-by-side benchmarking
- **Hybrid Search**: Dense (semantic) + Sparse (BM25) + Metadata filtering with RRF fusion
- **NER-Based Metadata Extraction**: GLiNER2 and spaCy with ensemble mode
- **Answer Grounding Verification**: Claim-level verification against sources
- **Multi-Document Synthesis**: Cross-document information synthesis with contradiction detection
- **Conversational Memory**: Short-term and long-term memory with context window management

### RAG Strategies
1. **Simple RAG**: Basic retrieve → rerank → generate
2. **Corrective RAG (CRAG)**: Relevance grading with corrective retrieval
3. **Self-Reflective RAG (SRAG)**: Self-evaluation and iterative refinement
4. **Advanced RAG**: Multi-query retrieval with Relevance Score Fusion

### Architecture Highlights
- **LangGraph Agent Orchestration**: 15-node workflow with conditional routing
- **Query Decomposition**: Breaks complex queries into focused sub-queries
- **Contextual Compression**: Post-retrieval extraction of relevant content
- **Cohere Reranking**: Cross-encoder reranking for improved precision
- **HYDE Support**: Hypothetical Document Embeddings for query expansion
- **Document Summarization**: LLM-generated summaries for document-level context

## Project Structure

```
Visual_RAG_Document_Explore/
├── app.py                          # Streamlit entry point
├── config/
│   ├── settings.py                 # Global configuration with pydantic-settings
│   └── models.py                   # All Pydantic data models
├── core/
│   ├── document_processing/        # Loaders, chunking, NER, deduplication
│   ├── embeddings/                 # Voyage AI, BGE-M3, router
│   ├── search/                     # Hybrid search, BM25, metadata filtering
│   ├── vectordb/                   # Qdrant, Milvus, abstraction layer
│   ├── reranking/                  # Cohere reranker
│   ├── rag/                        # RAG strategies, compression, grounding
│   └── llm/                        # OpenAI, OpenRouter providers
├── agents/
│   ├── graph.py                    # LangGraph state graph
│   ├── state.py                    # AgentState schema
│   ├── orchestrator.py             # Main orchestrator
│   └── nodes/                      # 15 agent nodes
├── ui/
│   ├── pages/                      # Chat, Upload, Explorer, Settings, Benchmark
│   ├── components/                 # Reusable UI components
│   └── styles/                     # Custom CSS
├── data/
│   ├── uploads/                    # Uploaded documents
│   └── processed/                  # Processed document cache
├── tests/                          # Unit tests
├── plans/                          # Architecture documentation
├── docker-compose.yml              # Qdrant + Milvus + etcd + MinIO
├── pyproject.toml                  # Dependencies
├── Makefile                        # Common commands
└── .env.example                    # Environment variable template
```

## Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- API keys for: OpenAI, Voyage AI, Cohere (optional: OpenRouter)

### Installation

**Quick Install:**
```bash
# Clone and install
git clone <repository-url>
cd Visual_RAG_Document_Explore
make install

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start services and run
make docker-up
make run
```

The application will be available at `http://localhost:8501`

**📖 For detailed installation instructions, troubleshooting, and alternative methods, see [`INSTALLATION.md`](INSTALLATION.md)**

**Important:** This project uses standard Python packaging with setuptools (PEP 621), NOT Poetry. Use `pip install -e ".[dev]"` or `make install`.

### Docker Services
- **Qdrant**: `http://localhost:6333` (REST), `http://localhost:6334` (gRPC)
- **Milvus**: `http://localhost:19530`
- **MinIO**: `http://localhost:9000` (API), `http://localhost:9001` (Console)

## Using the UI

The Streamlit interface provides 5 main pages accessible via the sidebar:

### 💬 Chat Interface
- **Purpose**: Conversational document Q&A with advanced RAG strategies
- **Features**:
  - Natural language queries with streaming responses
  - Multiple RAG strategies (Simple, CRAG, SRAG, Advanced, Auto)
  - Search mode selection (Dense, Sparse, Hybrid)
  - Source citations with relevance scores
  - Grounding verification scores
  - Chat history with context
  - Advanced options (HYDE, reranking, compression)

### 📤 Document Upload
- **Purpose**: Ingest documents into the system
- **Features**:
  - Multi-file upload (drag-and-drop or browse)
  - Supported formats: PDF, DOCX, TXT, HTML, JSON
  - Real-time processing pipeline visualization
  - Configurable chunking parameters
  - NER extraction with entity display
  - Document summarization
  - Automatic deduplication
  - Upload history tracking

### 🔍 Document Explorer
- **Purpose**: Browse and search indexed documents
- **Features**:
  - Document list with metadata cards
  - Search and filter capabilities
  - Chunk viewer with detailed inspection
  - Entity-based filtering
  - Similarity search
  - Metadata filtering (date, type, entities)
  - Statistics dashboard

### ⚙️ Settings
- **Purpose**: Configure system components and parameters
- **Features**:
  - LLM provider selection (OpenAI, OpenRouter)
  - Embedding model configuration (Voyage AI, BGE-M3)
  - Vector database selection (Qdrant, Milvus)
  - RAG strategy settings
  - Search mode configuration
  - NER mode selection
  - Advanced toggles (HYDE, compression, reranking)
  - Settings import/export
  - Connection testing

### 📊 Benchmark
- **Purpose**: Compare vector database performance
- **Features**:
  - Side-by-side Qdrant vs Milvus comparison
  - Configurable benchmark parameters
  - Metrics: indexing throughput, query latency, recall@k, memory usage
  - Real-time progress tracking
  - Results visualization (charts and tables)
  - Export results as CSV/JSON
  - Historical comparison

## Configuration

All settings can be configured via environment variables or the `.env` file. See [`.env.example`](.env.example) for all available options.

### Key Settings
- **LLM Provider**: OpenAI or OpenRouter
- **Embedding Model**: Voyage AI (voyage-3) or BGE-M3 (local)
- **Vector Database**: Qdrant or Milvus
- **RAG Strategy**: Simple, CRAG, SRAG, Advanced, or Auto
- **Search Mode**: Dense, Sparse, or Hybrid
- **NER Mode**: GLiNER, spaCy, or Ensemble

## Development

### Available Commands
```bash
make install      # Install dependencies
make docker-up    # Start Docker services
make docker-down  # Stop Docker services
make run          # Run Streamlit app
make test         # Run tests
make lint         # Run linter
make format       # Format code
make clean        # Clean build artifacts
```

### Running Tests
```bash
make test
# or: pytest tests/ -v
```

### Code Quality
```bash
make lint    # Check code with ruff
make format  # Format code with ruff
```

## Implementation Phases

### ✅ Phase 1A: Foundation (Complete)
- Project scaffolding and configuration
- Docker Compose setup
- Pydantic models and settings
- Directory structure with placeholder files

### ✅ Phase 1B: Document Processing (Complete)
- Document loaders for all formats (PDF, DOCX, TXT, HTML, JSON)
- Hierarchical chunking with adaptive sizing
- GLiNER2 and spaCy NER extraction with ensemble mode
- LLM-based document summarization
- Semantic deduplication engine

### ✅ Phase 2: Vector Database & Search (Complete)
- Qdrant and Milvus implementations with abstraction layer
- Hybrid search with RRF fusion
- BM25 sparse search
- Metadata filtering
- Cohere reranking
- Performance benchmarking

### ✅ Phase 3: RAG Engine (Complete)
- Simple, CRAG, SRAG, Advanced RAG strategies
- Query decomposition for complex queries
- HYDE (Hypothetical Document Embeddings)
- Contextual compression
- Answer grounding verification with claim-level analysis

### ✅ Phase 4: Agent Layer (Complete)
- LangGraph state graph with 15 nodes
- Intelligent query routing
- Multi-document synthesis with contradiction detection
- Conversational memory (short-term and long-term)
- Streaming execution support

### ✅ Phase 5: Streamlit UI (Complete)
- Chat interface with streaming responses
- Document upload with pipeline visualization
- Document explorer with search and filtering
- Settings management with live configuration
- Benchmark dashboard with side-by-side comparison
- Custom CSS with responsive design and dark mode

### 📋 Phase 6: Production & Enhancement (Future)
- User authentication and multi-tenancy
- Advanced visualizations (graphs, embeddings)
- API endpoints and webhooks
- Mobile app
- Enhanced analytics

## Architecture

See [`plans/architecture.md`](plans/architecture.md) for detailed architecture documentation.

Key architectural decisions:
- **Abstraction Layers**: Common interfaces for vector DBs, embeddings, LLMs, and RAG strategies
- **LangGraph Orchestration**: 15-node workflow with conditional routing based on query complexity
- **Hybrid Search**: Combines dense, sparse, and metadata-based retrieval with RRF fusion
- **NER Ensemble**: Merges GLiNER2 and spaCy results for maximum entity recall
- **Grounding Verification**: Post-generation claim verification against sources

## API Keys Required

- **OpenAI**: For LLM generation (required)
- **Voyage AI**: For embeddings (required, or use BGE-M3 locally)
- **Cohere**: For reranking (required for reranking feature)
- **OpenRouter**: For alternative LLM providers (optional)

## License

MIT

## Contributing

Contributions are welcome! Please see the implementation phases above for areas that need development.

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

