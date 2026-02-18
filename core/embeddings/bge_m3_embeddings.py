"""
BGE-M3 embeddings service.

Uses BAAI/bge-m3 model via sentence-transformers for local embedding generation.
"""

from sentence_transformers import SentenceTransformer

from config.settings import Settings


class BGEM3Embeddings:
    """BGE-M3 embeddings service wrapper."""

    def __init__(self, settings: Settings):
        """
        Initialize BGE-M3 embeddings service.

        Args:
            settings: Application settings with model config
        """
        self.settings = settings
        self.model = SentenceTransformer(settings.bge_model)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        # SentenceTransformer.encode returns numpy arrays
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False,
        )

        # Convert numpy arrays to lists
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector (list of floats)
        """
        # Encode single text
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Convert numpy array to list
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return 1024  # BGE-M3 produces 1024-dimensional embeddings

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.settings.bge_model
