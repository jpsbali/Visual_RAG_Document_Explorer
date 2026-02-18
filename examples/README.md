# Phase 1B Pipeline Examples

This directory contains example scripts demonstrating the complete Phase 1B document processing pipeline. Each example showcases different aspects of the system with clear, runnable code.

## 📋 Overview

| Example | Description | Key Features |
|---------|-------------|--------------|
| [`basic_pipeline.py`](basic_pipeline.py) | Complete pipeline walkthrough | Document loading, chunking, NER, embeddings |
| [`document_summarization.py`](document_summarization.py) | LLM-based summarization | Direct & map-reduce strategies |
| [`ner_comparison.py`](ner_comparison.py) | Compare NER extractors | GLiNER vs spaCy vs Ensemble |
| [`batch_processing.py`](batch_processing.py) | Batch processing multiple docs | Efficient bulk operations |

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**
   ```bash
   # Ensure you're in the project root
   cd /path/to/Visual_RAG_Document_Explore
   
   # Install dependencies
   pip install -e .
   ```

2. **Required Models**
   ```bash
   # Download spaCy model for NER
   python -m spacy download en_core_web_sm
   ```

3. **API Keys** (in `.env` file)
   ```bash
   # For embeddings (choose one)
   VOYAGE_API_KEY=your_voyage_key_here
   # OR use local BGE-M3 (no key needed)
   
   # For LLM/summarization (choose one)
   OPENAI_API_KEY=your_openai_key_here
   # OR
   OPENROUTER_API_KEY=your_openrouter_key_here
   ```

### Running Examples

```bash
# From project root
python examples/basic_pipeline.py
python examples/document_summarization.py
python examples/ner_comparison.py
python examples/batch_processing.py
```

## 📚 Example Details

### 1. Basic Pipeline (`basic_pipeline.py`)

**What it demonstrates:**
- Loading and processing a sample document
- Adaptive chunking with configurable parameters
- Entity extraction using ensemble mode (GLiNER + spaCy)
- Embedding generation for all chunks
- Complete pipeline with formatted output

**Expected output:**
- Document statistics (words, characters, lines)
- Chunk details with token counts
- Extracted entities with confidence scores and sources
- Embedding dimensions and sample values
- Pipeline summary with status for each stage

**Run time:** ~30-60 seconds (depending on models)

**Example output:**
```
================================================================================
                          STEP 1: DOCUMENT LOADING                            
================================================================================

┏━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric      ┃ Value ┃
┡━━━━━━━━━━━━━╇━━━━━━━┩
│ Characters  │ 1,234 │
│ Words       │ 234   │
│ Lines       │ 15    │
└─────────────┴───────┘

✓ Generated 5 chunks
✓ Extracted 42 entities
✓ Generated 5 embeddings
```

---

### 2. Document Summarization (`document_summarization.py`)

**What it demonstrates:**
- Direct summarization for short documents
- Map-reduce summarization for long documents
- Automatic strategy selection based on document length
- Token counting and cost estimation
- Customizable summary length

**Expected output:**
- Document statistics with token estimates
- Strategy used (direct vs map-reduce)
- Generated summaries with different lengths
- Comparison of compression ratios
- Processing metadata

**Run time:** ~20-40 seconds (depends on LLM API)

**Requirements:**
- OpenAI API key OR OpenRouter API key
- Internet connection for API calls

**Example output:**
```
Strategy Used: direct
┏━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metadata       ┃ Value ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Tokens Used    │ 450   │
│ Strategy       │ direct│
└────────────────┴───────┘

Summary:
┌────────────────────────────────────────┐
│ Quantum computing uses qubits that can │
│ exist in multiple states, enabling    │
│ unprecedented computational power...   │
└────────────────────────────────────────┘
```

---

### 3. NER Comparison (`ner_comparison.py`)

**What it demonstrates:**
- Entity extraction with GLiNER (zero-shot)
- Entity extraction with spaCy (traditional NER)
- Ensemble mode combining both extractors
- Side-by-side comparison of results
- Confidence score analysis
- Entity overlap and unique findings

**Expected output:**
- Entities found by each extractor
- Entity type distribution
- Overlap analysis (found by both vs unique)
- Confidence score statistics
- Recommendations for each approach

**Run time:** ~45-90 seconds (loads multiple models)

**Example output:**
```
GLiNER Results: 38 entities
spaCy Results: 32 entities
Ensemble Results: 45 entities

Entities found by BOTH extractors: 25
Found only by GLiNER: 13
Found only by spaCy: 7

Average Confidence:
  GLiNER: 0.892
  spaCy:  0.756
```

---

### 4. Batch Processing (`batch_processing.py`)

**What it demonstrates:**
- Processing multiple documents efficiently
- Batch chunking with progress tracking
- Batch entity extraction across all chunks
- Batch embedding generation
- Performance metrics and statistics
- Per-document and overall summaries

**Expected output:**
- Document overview table
- Progress bars for each processing stage
- Processing time and throughput metrics
- Entity type distribution across all documents
- Per-document summary with statistics
- Performance optimization tips

**Run time:** ~60-120 seconds (processes 5 documents)

**Example output:**
```
Processing 5 documents...

Chunking documents... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02
✓ Generated 23 chunks (11.5 chunks/sec)

Extracting entities... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:15
✓ Extracted 156 entities (10.4 entities/sec)

Generating embeddings... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03
✓ Generated 23 embeddings (7.7 embeddings/sec)
```

## 🔧 Configuration

### Chunking Parameters

Adjust in the example scripts:
```python
chunker = AdaptiveChunker(
    chunk_size=200,        # Target chunk size in tokens
    chunk_overlap=50,      # Overlap between chunks
    min_chunk_size=50      # Minimum chunk size
)
```

### NER Mode Selection

Choose the appropriate mode:
```python
# Fast, standard entities
ner_router = NERRouter(mode="spacy")

# Flexible, custom entities
ner_router = NERRouter(mode="gliner")

# Best coverage, both extractors
ner_router = NERRouter(mode="ensemble")
```

### Embedding Service

Configured in `config/settings.py`:
```python
# Voyage AI (cloud, requires API key)
EMBEDDING_SERVICE = "voyage"

# BGE-M3 (local, no API key needed)
EMBEDDING_SERVICE = "bge-m3"
```

### LLM Provider

Configured in `config/settings.py`:
```python
# OpenAI
LLM_PROVIDER = "openai"

# OpenRouter (access to multiple models)
LLM_PROVIDER = "openrouter"
```

## 🐛 Troubleshooting

### Common Issues

#### 1. **spaCy Model Not Found**
```
Error: Can't find model 'en_core_web_sm'
```
**Solution:**
```bash
python -m spacy download en_core_web_sm
```

#### 2. **Missing API Keys**
```
Error: VOYAGE_API_KEY not found
```
**Solution:**
- Copy `.env.example` to `.env`
- Add your API keys to `.env`
- Restart the script

#### 3. **Out of Memory (OOM)**
```
Error: CUDA out of memory
```
**Solution:**
- Use smaller batch sizes
- Use cloud embeddings (Voyage) instead of local (BGE-M3)
- Process fewer documents at once
- Close other applications

#### 4. **Slow Performance**
**Solutions:**
- Use GPU if available (for GLiNER and BGE-M3)
- Use cloud services (Voyage, OpenAI) for faster processing
- Reduce batch sizes for memory-constrained systems
- Use spaCy instead of ensemble mode for faster NER

#### 5. **Import Errors**
```
ModuleNotFoundError: No module named 'core'
```
**Solution:**
```bash
# Ensure you're running from project root
cd /path/to/Visual_RAG_Document_Explore
python examples/basic_pipeline.py

# Or install in development mode
pip install -e .
```

## 📊 Performance Benchmarks

Approximate processing times on a typical system:

| Operation | Time (per document) | Notes |
|-----------|---------------------|-------|
| Chunking | 0.1-0.5s | Very fast, CPU-bound |
| NER (spaCy) | 1-3s | Fast, CPU-bound |
| NER (GLiNER) | 3-8s | Slower, GPU helps |
| NER (Ensemble) | 4-10s | Combined time |
| Embeddings (Voyage) | 0.5-2s | API latency |
| Embeddings (BGE-M3) | 2-5s | GPU helps significantly |
| Summarization | 3-10s | Depends on LLM API |

*Times are for ~200-word documents on a system with 16GB RAM, modern CPU, and optional GPU.*

## 🎯 Next Steps

After running these examples:

1. **Modify the sample data** - Try your own documents
2. **Adjust parameters** - Experiment with chunk sizes, NER modes
3. **Integrate into your workflow** - Use these patterns in your application
4. **Explore the tests** - Check `tests/test_phase1b_*.py` for more examples
5. **Read the documentation** - See `PHASE1B_IMPLEMENTATION.md` for details

## 💡 Tips for Production Use

1. **Caching**: Cache embeddings and entities to avoid reprocessing
2. **Error Handling**: Add retry logic for API calls
3. **Monitoring**: Track processing times and success rates
4. **Batch Sizes**: Tune batch sizes based on your hardware
5. **Model Selection**: Choose models based on accuracy vs speed tradeoffs
6. **Cost Management**: Monitor API usage for cloud services
7. **Validation**: Validate outputs before storing in vector DB

## 📖 Additional Resources

- **Phase 1B Implementation Guide**: `PHASE1B_IMPLEMENTATION.md`
- **Architecture Documentation**: `plans/architecture.md`
- **API Documentation**: Check docstrings in source files
- **Test Suite**: `tests/test_phase1b_*.py`

## 🤝 Contributing

Found an issue or want to add more examples? Please:
1. Check existing issues
2. Create a new issue describing the problem/enhancement
3. Submit a pull request with your changes

## 📝 License

This project is part of the Visual RAG Document Explorer system.
