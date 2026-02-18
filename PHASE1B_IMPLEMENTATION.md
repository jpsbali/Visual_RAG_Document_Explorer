# Phase 1B Implementation - Core Document Processing & Foundation Services

## Overview

Phase 1B implements the core document processing pipeline and foundation services for the Visual RAG Document Explorer. These components handle document ingestion, chunking, NER extraction, summarization, and embedding generation.

## Implemented Components

### 1. Document Loaders (`core/document_processing/loaders.py`)

**DocumentLoaderFactory** - Loads documents in 5 formats:
- **PDF**: Using `PyPDFLoader` from LangChain
- **Word (.docx)**: Using `Docx2txtLoader`
- **TXT**: Using `TextLoader`
- **HTML**: Using `BSHTMLLoader`
- **JSON**: Using `JSONLoader`

**Features**:
- Auto-detects file format from extension
- Preserves metadata (filename, file_type, page_number)
- Normalizes metadata across formats

**Usage**:
```python
from core.document_processing.loaders import DocumentLoaderFactory

documents = DocumentLoaderFactory.load_document("path/to/file.pdf")
```

### 2. Adaptive Chunker (`core/document_processing/chunker.py`)

**AdaptiveChunker** - Intelligent document chunking with metadata:
- Uses `RecursiveCharacterTextSplitter` from LangChain
- Configurable chunk size (256-1024 tokens) and overlap
- Generates comprehensive `ChunkMetadata` for each chunk

**Metadata Generated**:
- `chunk_id` (UUID)
- `source_file`, `file_type`, `page_number`
- `chunk_index`, `total_chunks`
- `token_count` (using tiktoken), `char_count`
- `content_hash` (SHA-256 for deduplication)
- `content_preview` (first 200 chars)
- Timestamps (`created_at`, `processed_at`)

**Usage**:
```python
from core.document_processing.chunker import AdaptiveChunker
from config.settings import settings

chunker = AdaptiveChunker(settings)
chunks = chunker.chunk_documents(documents)
```

### 3. LLM Providers (`core/llm/`)

#### OpenAI Provider (`openai_provider.py`)
- Uses `langchain_openai.ChatOpenAI`
- Supports both sync and streaming generation
- Configurable temperature and max_tokens

#### OpenRouter Provider (`openrouter_provider.py`)
- Uses `httpx` for API calls
- Supports streaming via SSE
- Compatible with multiple models via OpenRouter

#### LLM Router (`llm_router.py`)
- Routes to OpenAI or OpenRouter based on settings
- Validates API keys
- Returns appropriate provider instance

**Usage**:
```python
from core.llm.llm_router import get_llm_provider
from config.settings import settings

llm = get_llm_provider(settings)
response = await llm.generate("Your prompt here")
```

### 4. NER Extractors (`core/document_processing/`)

#### GLiNER Extractor (`gliner_extractor.py`)
- Uses GLiNER2 for zero-shot NER
- Extracts: people, organizations, dates, locations, topics
- Supports custom entity types
- Returns confidence scores

#### spaCy Extractor (`spacy_extractor.py`)
- Uses spaCy transformer models
- Extracts standard NER entities
- Extracts noun chunks as topics
- Efficient batch processing

#### NER Router (`ner_router.py`)
- Supports 3 modes: `gliner`, `spacy`, `ensemble`
- **Ensemble mode**: Runs both extractors and merges results
- Boosts confidence for entities found by both
- Deduplicates entities

**Usage**:
```python
from core.document_processing.ner_router import NERRouter
from config.settings import settings

ner = NERRouter(settings)
entities = ner.extract("Apple Inc. was founded by Steve Jobs.")
```

### 5. Document Summarizer (`core/document_processing/summarizer.py`)

**DocumentSummarizer** - LLM-based summarization:
- Generates 3-5 sentence summaries
- Uses map-reduce for long documents (>4000 tokens)
- Configurable via LLM provider

**Usage**:
```python
from core.document_processing.summarizer import DocumentSummarizer
from core.llm.llm_router import get_llm_provider

llm = get_llm_provider(settings)
summarizer = DocumentSummarizer(llm)
summary = await summarizer.summarize(document_text, "filename.pdf")
```

### 6. Embedding Services (`core/embeddings/`)

#### Voyage Embeddings (`voyage_embeddings.py`)
- Uses Voyage AI API via LangChain
- 1024-dimensional embeddings
- Async support

#### BGE-M3 Embeddings (`bge_m3_embeddings.py`)
- Uses `sentence-transformers` locally
- BAAI/bge-m3 model
- 1024-dimensional embeddings
- No API key required

#### Embedding Router (`embedding_router.py`)
- Routes to Voyage or BGE-M3 based on settings
- Validates API keys for Voyage

**Usage**:
```python
from core.embeddings.embedding_router import get_embedding_service
from config.settings import settings

embeddings = get_embedding_service(settings)
vectors = await embeddings.embed_documents(["text1", "text2"])
```

## Setup Requirements

### 1. Install Dependencies

All dependencies are defined in `pyproject.toml`. Install with:

```bash
pip install -e .
```

### 2. Download spaCy Model

For NER extraction with spaCy:

```bash
python -m spacy download en_core_web_trf
```

Or use a smaller model for testing:

```bash
python -m spacy download en_core_web_sm
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# LLM Providers
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key

# Embedding Services
VOYAGE_API_KEY=your_voyage_key

# Optional: Cohere for reranking (Phase 2)
COHERE_API_KEY=your_cohere_key

# Model Selection
DEFAULT_LLM_PROVIDER=openai  # or openrouter
DEFAULT_EMBEDDING_MODEL=voyage  # or bge-m3

# NER Configuration
NER_MODE=ensemble  # gliner, spacy, or ensemble
```

### 4. Model Downloads

Some models will be downloaded automatically on first use:

- **GLiNER**: `urchade/gliner_multi_pii-v1` (~500MB)
- **BGE-M3**: `BAAI/bge-m3` (~2GB)
- **spaCy**: `en_core_web_trf` (~500MB)

## Testing

Run the test suite:

```bash
# Run all Phase 1B tests
pytest tests/test_phase1b_*.py -v

# Run specific test files
pytest tests/test_phase1b_loaders.py -v
pytest tests/test_phase1b_chunker.py -v
pytest tests/test_phase1b_embeddings.py -v
pytest tests/test_phase1b_integration.py -v

# Run with coverage
pytest tests/test_phase1b_*.py --cov=core --cov-report=html
```

### Test Coverage

- **Loaders**: Tests all 5 document formats, metadata preservation
- **Chunker**: Tests chunking logic, metadata generation, hash generation
- **Embeddings**: Tests both Voyage and BGE-M3, dimension verification
- **Integration**: Tests complete pipeline from loading to embedding

## Usage Examples

### Complete Document Processing Pipeline

```python
import asyncio
from config.settings import settings
from core.document_processing.loaders import DocumentLoaderFactory
from core.document_processing.chunker import AdaptiveChunker
from core.document_processing.ner_router import NERRouter
from core.document_processing.summarizer import DocumentSummarizer
from core.llm.llm_router import get_llm_provider
from core.embeddings.embedding_router import get_embedding_service

async def process_document(file_path: str):
    # 1. Load document
    documents = DocumentLoaderFactory.load_document(file_path)
    
    # 2. Chunk document
    chunker = AdaptiveChunker(settings)
    chunks = chunker.chunk_documents(documents)
    
    # 3. Extract entities for each chunk
    ner = NERRouter(settings)
    for chunk_text, metadata in chunks:
        entities = ner.extract(chunk_text)
        metadata.entities = entities
    
    # 4. Generate document summary
    llm = get_llm_provider(settings)
    summarizer = DocumentSummarizer(llm)
    full_text = " ".join([doc.page_content for doc in documents])
    summary = await summarizer.summarize(full_text, file_path)
    
    # 5. Generate embeddings
    embeddings_service = get_embedding_service(settings)
    chunk_texts = [text for text, _ in chunks]
    embeddings = await embeddings_service.embed_documents(chunk_texts)
    
    return {
        "chunks": chunks,
        "summary": summary,
        "embeddings": embeddings,
    }

# Run pipeline
result = asyncio.run(process_document("document.pdf"))
```

## Architecture Notes

### Design Decisions

1. **LangChain Integration**: Used LangChain loaders for consistency and reliability
2. **Async/Await**: All I/O operations (LLM, embeddings) use async for better performance
3. **Pydantic Models**: All data uses Pydantic models from `config/models.py` for validation
4. **Router Pattern**: Routers allow easy switching between providers without code changes
5. **Ensemble NER**: Combines GLiNER and spaCy for better entity extraction

### Performance Considerations

- **Chunking**: Uses tiktoken for accurate token counting
- **Batch Processing**: NER and embeddings support batch operations
- **Local Models**: BGE-M3 and spaCy can run locally without API calls
- **Streaming**: LLM providers support streaming for better UX

### Error Handling

All components include:
- Input validation
- API key checking
- Graceful fallbacks
- Descriptive error messages

## Next Steps (Phase 2)

Phase 1B provides the foundation for:
- Vector database integration (Qdrant, Milvus)
- BM25 sparse search
- Hybrid search with RRF fusion
- Cohere reranking
- Deduplication services

## Dependencies

Key dependencies installed:
- `langchain` - Document loaders, text splitters
- `langchain-openai` - OpenAI integration
- `langchain-voyageai` - Voyage embeddings
- `langchain-community` - Community loaders
- `gliner` - GLiNER NER model
- `spacy` - spaCy NER
- `sentence-transformers` - BGE-M3 embeddings
- `tiktoken` - Token counting
- `httpx` - HTTP client for OpenRouter
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support

## Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_trf
   ```

2. **API key errors**:
   - Check `.env` file exists and has correct keys
   - Verify keys are not empty strings

3. **Memory issues with large models**:
   - Use smaller models for testing (e.g., `en_core_web_sm` for spaCy)
   - Use `bge-small-en-v1.5` instead of `bge-m3`

4. **Import errors**:
   - Ensure package is installed: `pip install -e .`
   - Check Python version >= 3.11

## License

See main project LICENSE file.
