"""
Voyage AI embeddings service.

Uses Voyage AI's embedding models via LangChain.
"""

from langchain_voyageai import VoyageAIEmbeddings

from config.settings import Settings


class VoyageEmbeddings:
    """Voyage AI embeddings service wrapper."""

    def __init__(self, settings: Settings):
        """
        Initialize Voyage embeddings service.

        Args:
            settings: Application settings with API key and model config
        """
        self.settings = settings
        self.embeddings = VoyageAIEmbeddings(
            voyage_api_key=settings.voyage_api_key,
            model=settings.voyage_model,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        # Use async method if available, otherwise use sync
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
        except AttributeError:
            # Fallback to sync method
            embeddings = self.embeddings.embed_documents(texts)

        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector (list of floats)
        """
        # Use async method if available, otherwise use sync
        try:
            embedding = await self.embeddings.aembed_query(text)
        except AttributeError:
            # Fallback to sync method
            embedding = self.embeddings.embed_query(text)

        return embedding

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return 1024  # Voyage-3 produces 1024-dimensional embeddings

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.settings.voyage_model
