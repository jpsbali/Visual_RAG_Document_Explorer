"""
Abstract base interface for vector database operations.

Defines the common interface that all vector store implementations
(Qdrant, Milvus, etc.) must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional
from config.models import ChunkMetadata, RetrievedChunk


class VectorStoreBase(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    async def create_collection(
        self, 
        name: str, 
        dimension: int,
        distance_metric: str = "cosine"
    ) -> None:
        """
        Create a new collection/index.
        
        Args:
            name: Collection name
            dimension: Vector dimension (1024 for Voyage/BGE-M3)
            distance_metric: Distance metric (cosine, euclidean, dot)
        """
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """
        Delete a collection and all its data.
        
        Args:
            name: Collection name to delete
        """
        pass

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> int:
        """
        Insert or update documents.
        
        Args:
            collection: Collection name
            ids: List of chunk IDs (UUIDs)
            embeddings: List of embedding vectors
            documents: List of chunk texts
            metadatas: List of ChunkMetadata dicts
            
        Returns:
            Count of upserted documents
        """
        pass

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Search by vector similarity with optional metadata filters.
        
        Args:
            collection: Collection name
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"organizations": ["Acme Corp"]})
            
        Returns:
            List of RetrievedChunk objects with scores
        """
        pass

    @abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> int:
        """
        Delete documents by ID.
        
        Args:
            collection: Collection name
            ids: List of document IDs to delete
            
        Returns:
            Count of deleted documents
        """
        pass

    @abstractmethod
    async def get(self, collection: str, ids: list[str]) -> list[dict]:
        """
        Get documents by ID.
        
        Args:
            collection: Collection name
            ids: List of document IDs to retrieve
            
        Returns:
            List of documents with metadata
        """
        pass

    @abstractmethod
    async def count(self, collection: str) -> int:
        """
        Count documents in a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Number of documents in collection
        """
        pass

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """
        List all collection names.
        
        Returns:
            List of collection names
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the database is reachable.
        
        Returns:
            True if database is healthy, False otherwise
        """
        pass
