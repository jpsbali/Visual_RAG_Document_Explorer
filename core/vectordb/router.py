"""
Vector database router for transparent switching between backends.

Routes operations to the configured vector database (Qdrant or Milvus).
"""

from config.settings import Settings
from .base import VectorStoreBase
from .qdrant_store import QdrantStore
from .milvus_store import MilvusStore
import logging

logger = logging.getLogger(__name__)


def get_vector_store(settings: Settings) -> VectorStoreBase:
    """
    Get the configured vector store implementation.
    
    Args:
        settings: Application settings
        
    Returns:
        VectorStoreBase implementation (QdrantStore or MilvusStore)
        
    Raises:
        ValueError: If vector DB backend is not supported
    """
    backend = settings.default_vector_db
    
    if backend == "qdrant":
        logger.info("Using Qdrant vector store")
        return QdrantStore(settings)
    elif backend == "milvus":
        logger.info("Using Milvus vector store")
        return MilvusStore(settings)
    else:
        raise ValueError(
            f"Unsupported vector database: {backend}. "
            f"Supported: qdrant, milvus"
        )


class VectorDBRouter:
    """
    Router for transparent switching between vector databases.
    
    Useful for benchmarking and A/B testing different vector stores.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize router with both vector stores.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.stores = {
            "qdrant": QdrantStore(settings),
            "milvus": MilvusStore(settings)
        }
        self.active_store = settings.default_vector_db
        logger.info(f"Initialized VectorDBRouter with active store: {self.active_store}")
        
    def get_store(self, backend: str = None) -> VectorStoreBase:
        """
        Get a specific vector store or the active one.
        
        Args:
            backend: Vector store backend name (qdrant, milvus), or None for active
            
        Returns:
            VectorStoreBase implementation
            
        Raises:
            ValueError: If backend is unknown
        """
        if backend is None:
            backend = self.active_store
            
        if backend not in self.stores:
            raise ValueError(
                f"Unknown vector store: {backend}. "
                f"Available: {list(self.stores.keys())}"
            )
            
        return self.stores[backend]
        
    def switch_backend(self, backend: str) -> None:
        """
        Switch the active vector database backend.
        
        Args:
            backend: Vector store backend name (qdrant, milvus)
            
        Raises:
            ValueError: If backend is unknown
        """
        if backend not in self.stores:
            raise ValueError(
                f"Unknown vector store: {backend}. "
                f"Available: {list(self.stores.keys())}"
            )
            
        self.active_store = backend
        logger.info(f"Switched to {backend} vector store")
        
    def get_active_backend(self) -> str:
        """
        Get the name of the currently active backend.
        
        Returns:
            Active backend name
        """
        return self.active_store
        
    def get_available_backends(self) -> list[str]:
        """
        Get list of available vector store backends.
        
        Returns:
            List of backend names
        """
        return list(self.stores.keys())
