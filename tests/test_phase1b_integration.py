"""
Integration tests for Phase 1B components.

Tests the complete document processing pipeline.
"""

import pytest
import tempfile
from pathlib import Path

from core.document_processing.loaders import DocumentLoaderFactory
from core.document_processing.chunker import AdaptiveChunker
from core.document_processing.ner_router import NERRouter
from core.llm.llm_router import get_llm_provider
from core.embeddings.embedding_router import get_embedding_service
from config.settings import Settings


class TestDocumentProcessingPipeline:
    """Test complete document processing pipeline."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            chunk_size_min=256,
            chunk_size_max=512,
            chunk_overlap=64,
            ner_mode="spacy",  # Use spaCy for testing (no API key needed)
            spacy_model="en_core_web_sm",  # Use small model for testing
            default_embedding_model="bge-m3",  # Use local model
            bge_model="BAAI/bge-small-en-v1.5",  # Use smaller model for testing
        )

    def test_load_and_chunk_pipeline(self, settings):
        """Test loading and chunking a document."""
        # Create temporary document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document. " * 50)  # Create longer text
            temp_path = f.name

        try:
            # Load document
            documents = DocumentLoaderFactory.load_document(temp_path)
            assert len(documents) > 0

            # Chunk document
            chunker = AdaptiveChunker(settings)
            chunks = chunker.chunk_documents(documents)

            # Verify chunks
            assert len(chunks) > 0
            for chunk_text, metadata in chunks:
                assert len(chunk_text) > 0
                assert metadata.source_file == Path(temp_path).name
                assert metadata.file_type == "txt"
                assert metadata.chunk_id is not None
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_chunk_and_embed_pipeline(self, settings):
        """Test chunking and embedding documents."""
        # Create temporary document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document about machine learning and AI.")
            temp_path = f.name

        try:
            # Load and chunk
            documents = DocumentLoaderFactory.load_document(temp_path)
            chunker = AdaptiveChunker(settings)
            chunks = chunker.chunk_documents(documents)

            # Get embedding service
            embedding_service = get_embedding_service(settings)

            # Embed chunks
            chunk_texts = [text for text, _ in chunks]
            embeddings = await embedding_service.embed_documents(chunk_texts)

            # Verify embeddings
            assert len(embeddings) == len(chunks)
            for embedding in embeddings:
                assert len(embedding) == embedding_service.dimension
        finally:
            Path(temp_path).unlink()

    def test_chunk_stats(self, settings):
        """Test chunk statistics calculation."""
        # Create temporary document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test sentence. " * 100)
            temp_path = f.name

        try:
            # Load and chunk
            documents = DocumentLoaderFactory.load_document(temp_path)
            chunker = AdaptiveChunker(settings)
            chunks = chunker.chunk_documents(documents)

            # Get stats
            stats = chunker.get_chunk_stats(chunks)

            # Verify stats
            assert stats["total_chunks"] > 0
            assert stats["avg_chunk_size"] > 0
            assert stats["avg_token_count"] > 0
        finally:
            Path(temp_path).unlink()


class TestLLMRouter:
    """Test LLM provider router."""

    def test_get_openai_provider(self):
        """Test getting OpenAI provider."""
        settings = Settings(
            default_llm_provider="openai",
            openai_api_key="test_key",
        )

        provider = get_llm_provider(settings)
        assert provider is not None
        assert provider.model_name == settings.default_model

    def test_get_openrouter_provider(self):
        """Test getting OpenRouter provider."""
        settings = Settings(
            default_llm_provider="openrouter",
            openrouter_api_key="test_key",
        )

        provider = get_llm_provider(settings)
        assert provider is not None
        assert provider.model_name == settings.openrouter_model

    def test_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        settings = Settings(
            default_llm_provider="openai",
            openai_api_key="",
        )

        with pytest.raises(ValueError):
            get_llm_provider(settings)

    def test_unsupported_provider(self):
        """Test that unsupported provider raises ValueError."""
        settings = Settings(
            default_llm_provider="unsupported",
        )

        with pytest.raises(ValueError):
            get_llm_provider(settings)


class TestNERRouter:
    """Test NER router."""

    @pytest.fixture
    def settings_spacy(self):
        """Create settings for spaCy."""
        return Settings(
            ner_mode="spacy",
            spacy_model="en_core_web_sm",
        )

    def test_ner_router_initialization(self, settings_spacy):
        """Test NER router initialization."""
        router = NERRouter(settings_spacy)
        assert router.mode == "spacy"
        assert router.spacy_extractor is not None

    def test_extract_entities_spacy(self, settings_spacy):
        """Test entity extraction with spaCy."""
        router = NERRouter(settings_spacy)

        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
        entities = router.extract(text)

        # Verify entities
        assert entities.extractor == "spacy"
        # Note: Actual entities depend on spaCy model performance
        assert isinstance(entities.people, list)
        assert isinstance(entities.organizations, list)
        assert isinstance(entities.locations, list)

    def test_extract_batch(self, settings_spacy):
        """Test batch entity extraction."""
        router = NERRouter(settings_spacy)

        texts = [
            "Microsoft was founded by Bill Gates.",
            "Google is located in Mountain View.",
        ]

        results = router.extract_batch(texts)

        # Verify results
        assert len(results) == len(texts)
        for entities in results:
            assert entities.extractor == "spacy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
