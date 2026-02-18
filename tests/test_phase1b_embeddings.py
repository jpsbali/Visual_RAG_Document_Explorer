"""
Tests for embedding services.

Tests Voyage and BGE-M3 embedding generation.
"""

import pytest

from core.embeddings.embedding_router import get_embedding_service
from core.embeddings.voyage_embeddings import VoyageEmbeddings
from core.embeddings.bge_m3_embeddings import BGEM3Embeddings
from config.settings import Settings


class TestEmbeddingRouter:
    """Test embedding service router."""

    def test_get_voyage_embeddings(self):
        """Test getting Voyage embeddings service."""
        settings = Settings(
            default_embedding_model="voyage",
            voyage_api_key="test_key",
        )

        service = get_embedding_service(settings)
        assert isinstance(service, VoyageEmbeddings)

    def test_get_bge_embeddings(self):
        """Test getting BGE-M3 embeddings service."""
        settings = Settings(
            default_embedding_model="bge-m3",
        )

        service = get_embedding_service(settings)
        assert isinstance(service, BGEM3Embeddings)

    def test_unsupported_model(self):
        """Test that unsupported model raises ValueError."""
        settings = Settings(
            default_embedding_model="unsupported",
        )

        with pytest.raises(ValueError):
            get_embedding_service(settings)

    def test_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        settings = Settings(
            default_embedding_model="voyage",
            voyage_api_key="",  # Empty API key
        )

        with pytest.raises(ValueError):
            get_embedding_service(settings)


class TestBGEM3Embeddings:
    """Test BGE-M3 embeddings service."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            bge_model="BAAI/bge-small-en-v1.5",  # Use smaller model for testing
        )

    @pytest.fixture
    def embeddings(self, settings):
        """Create embeddings service."""
        return BGEM3Embeddings(settings)

    @pytest.mark.asyncio
    async def test_embed_query(self, embeddings):
        """Test embedding a single query."""
        text = "This is a test query."
        embedding = await embeddings.embed_query(text)

        # Verify embedding
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_documents(self, embeddings):
        """Test embedding multiple documents."""
        texts = [
            "First document.",
            "Second document.",
            "Third document.",
        ]

        embeddings_list = await embeddings.embed_documents(texts)

        # Verify embeddings
        assert len(embeddings_list) == len(texts)
        for embedding in embeddings_list:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embedding_consistency(self, embeddings):
        """Test that same text produces same embedding."""
        text = "Consistent text."

        embedding1 = await embeddings.embed_query(text)
        embedding2 = await embeddings.embed_query(text)

        # Embeddings should be identical
        assert len(embedding1) == len(embedding2)
        for v1, v2 in zip(embedding1, embedding2):
            assert abs(v1 - v2) < 1e-6  # Allow for floating point precision

    def test_dimension_property(self, embeddings):
        """Test dimension property."""
        assert embeddings.dimension == 1024

    def test_model_name_property(self, embeddings, settings):
        """Test model name property."""
        assert embeddings.model_name == settings.bge_model


# Note: VoyageEmbeddings tests would require a valid API key
# and would make actual API calls, so they are marked as integration tests
class TestVoyageEmbeddings:
    """Test Voyage embeddings service (requires API key)."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            voyage_api_key="test_key",
            voyage_model="voyage-3",
        )

    def test_initialization(self, settings):
        """Test that Voyage embeddings can be initialized."""
        embeddings = VoyageEmbeddings(settings)
        assert embeddings.settings == settings
        assert embeddings.embeddings is not None

    def test_dimension_property(self, settings):
        """Test dimension property."""
        embeddings = VoyageEmbeddings(settings)
        assert embeddings.dimension == 1024

    def test_model_name_property(self, settings):
        """Test model name property."""
        embeddings = VoyageEmbeddings(settings)
        assert embeddings.model_name == settings.voyage_model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
