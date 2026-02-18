"""
Qdrant vector database implementation.

Provides async operations for document storage and retrieval using Qdrant.
"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchAny
)
from config.models import ChunkMetadata, RetrievedChunk
from config.settings import Settings
from .base import VectorStoreBase
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QdrantStore(VectorStoreBase):
    """Qdrant vector database implementation."""
    
    def __init__(self, settings: Settings):
        """
        Initialize Qdrant store.
        
        Args:
            settings: Application settings with Qdrant configuration
        """
        self.settings = settings
        self.client = AsyncQdrantClient(url=settings.qdrant_url, check_compatibility=False)
        self.default_collection = settings.qdrant_collection
        logger.info(f"Initialized Qdrant store at {settings.qdrant_url}")
        
    async def create_collection(
        self, 
        name: str, 
        dimension: int,
        distance_metric: str = "cosine"
    ) -> None:
        """
        Create Qdrant collection with vector configuration.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, euclidean, dot)
        """
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        
        try:
            await self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distance_map.get(distance_metric, Distance.COSINE)
                )
            )
            logger.info(f"Created Qdrant collection: {name} (dim={dimension}, metric={distance_metric})")
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection {name}: {e}")
            raise
    
    async def delete_collection(self, name: str) -> None:
        """
        Delete a Qdrant collection.
        
        Args:
            name: Collection name to delete
        """
        try:
            await self.client.delete_collection(collection_name=name)
            logger.info(f"Deleted Qdrant collection: {name}")
        except Exception as e:
            logger.error(f"Failed to delete Qdrant collection {name}: {e}")
            raise
        
    async def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> int:
        """
        Upsert documents to Qdrant using PointStruct.
        
        Args:
            collection: Collection name
            ids: List of chunk IDs
            embeddings: List of embedding vectors
            documents: List of document texts
            metadatas: List of metadata dicts
            
        Returns:
            Count of upserted documents
        """
        try:
            points = [
                PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "content": document,
                        **metadata  # Flatten metadata into payload
                    }
                )
                for chunk_id, embedding, document, metadata 
                in zip(ids, embeddings, documents, metadatas)
            ]
            
            await self.client.upsert(
                collection_name=collection,
                points=points
            )
            
            logger.info(f"Upserted {len(points)} documents to Qdrant collection: {collection}")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to upsert to Qdrant collection {collection}: {e}")
            raise
        
    async def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Search Qdrant with optional metadata filters.
        
        Args:
            collection: Collection name
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"organizations": ["Acme Corp"]})
            
        Returns:
            List of RetrievedChunk objects
        """
        try:
            # Build Qdrant filter from metadata dict
            qdrant_filter = None
            if filters:
                conditions = []
                for field, values in filters.items():
                    # Support filtering by entity types (organizations, people, etc.)
                    if field in ["organizations", "people", "dates", "locations", "topics"]:
                        conditions.append(
                            FieldCondition(
                                key=f"entities.{field}",
                                match=MatchAny(any=values)
                            )
                        )
                    else:
                        # Support other metadata fields
                        conditions.append(
                            FieldCondition(
                                key=field,
                                match=MatchAny(any=values)
                            )
                        )
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            results = await self.client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter
            )
            
            # Convert to RetrievedChunk objects
            chunks = []
            for result in results:
                # Extract metadata from payload
                payload = dict(result.payload)
                content = payload.pop("content", "")
                
                # Reconstruct ChunkMetadata
                metadata = ChunkMetadata(**payload)
                
                chunks.append(
                    RetrievedChunk(
                        content=content,
                        metadata=metadata,
                        score=result.score,
                        search_method="dense"
                    )
                )
            
            logger.info(f"Retrieved {len(chunks)} chunks from Qdrant collection: {collection}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to search Qdrant collection {collection}: {e}")
            raise
        
    async def delete(self, collection: str, ids: list[str]) -> int:
        """
        Delete documents from Qdrant by ID.
        
        Args:
            collection: Collection name
            ids: List of document IDs to delete
            
        Returns:
            Count of deleted documents
        """
        try:
            await self.client.delete(
                collection_name=collection,
                points_selector=ids
            )
            logger.info(f"Deleted {len(ids)} documents from Qdrant collection: {collection}")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to delete from Qdrant collection {collection}: {e}")
            raise
        
    async def get(self, collection: str, ids: list[str]) -> list[dict]:
        """
        Retrieve documents by ID from Qdrant.
        
        Args:
            collection: Collection name
            ids: List of document IDs to retrieve
            
        Returns:
            List of documents with metadata
        """
        try:
            results = await self.client.retrieve(
                collection_name=collection,
                ids=ids
            )
            return [result.payload for result in results]
        except Exception as e:
            logger.error(f"Failed to retrieve from Qdrant collection {collection}: {e}")
            raise
        
    async def count(self, collection: str) -> int:
        """
        Count documents in Qdrant collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Number of documents
        """
        try:
            info = await self.client.get_collection(collection_name=collection)
            return info.points_count
        except Exception as e:
            logger.error(f"Failed to count Qdrant collection {collection}: {e}")
            raise
        
    async def list_collections(self) -> list[str]:
        """
        List all Qdrant collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = await self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list Qdrant collections: {e}")
            raise
        
    async def health_check(self) -> bool:
        """
        Check Qdrant health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.client.get_collections()
            logger.info("Qdrant health check passed")
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
