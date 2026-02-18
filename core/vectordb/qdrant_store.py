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
import uuid
import hashlib

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
    
    @staticmethod
    def _string_to_uuid(string_id: str) -> str:
        """
        Convert a string ID to a valid UUID.
        
        Args:
            string_id: String identifier
            
        Returns:
            UUID string
        """
        # Create a deterministic UUID from the string using MD5 hash
        hash_bytes = hashlib.md5(string_id.encode()).digest()
        return str(uuid.UUID(bytes=hash_bytes))
        
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
            # Check if collection exists, create if it doesn't (lazy initialization)
            if not await self.collection_exists(collection):
                if not embeddings:
                    raise ValueError(f"Cannot auto-create collection '{collection}': no embeddings provided")
                
                # Detect dimension from first embedding
                dimension = len(embeddings[0])
                logger.warning(
                    f"Collection '{collection}' doesn't exist. Auto-creating with dimension={dimension}"
                )
                await self.create_collection(collection, dimension)
            
            points = [
                PointStruct(
                    id=self._string_to_uuid(chunk_id),
                    vector=embedding,
                    payload={
                        "content": document,
                        "original_id": chunk_id,  # Store original ID in payload
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
                    # Ensure values is a list
                    if not isinstance(values, list):
                        values = [values]
                    
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
            
            response = await self.client.query_points(
                collection_name=collection,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter
            )
            
            # Convert to RetrievedChunk objects
            chunks = []
            # query_points returns a QueryResponse with a points attribute
            results = response.points if hasattr(response, 'points') else response
            for result in results:
                # Extract metadata from payload
                payload = dict(result.payload)
                content = payload.pop("content", "")
                
                # Try to reconstruct ChunkMetadata, but handle cases where payload
                # doesn't match ChunkMetadata structure (e.g., conversation_memory)
                try:
                    metadata = ChunkMetadata(**payload)
                except Exception:
                    # For non-standard collections, create a minimal ChunkMetadata
                    # with required fields and store extra data in entities
                    metadata = ChunkMetadata(
                        chunk_id=payload.get("original_id", "unknown"),
                        source_file=payload.get("source_file", "memory"),
                        file_type=payload.get("file_type", "txt"),  # Use valid literal
                        chunk_index=payload.get("chunk_index", 0),
                        total_chunks=payload.get("total_chunks", 1),
                        chunk_size=payload.get("chunk_size", len(content)),
                        token_count=payload.get("token_count", 0),
                        char_count=payload.get("char_count", len(content)),
                        content_hash=payload.get("content_hash", ""),
                        content_preview=payload.get("content_preview", content[:100]),
                        created_at=payload.get("created_at", 0.0),
                        processed_at=payload.get("processed_at", 0.0),
                        entities=payload  # Store all extra fields in entities
                    )
                
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
            # Convert string IDs to UUIDs
            uuid_ids = [self._string_to_uuid(id_) for id_ in ids]
            
            await self.client.delete(
                collection_name=collection,
                points_selector=uuid_ids
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
            # Convert string IDs to UUIDs
            uuid_ids = [self._string_to_uuid(id_) for id_ in ids]
            
            results = await self.client.retrieve(
                collection_name=collection,
                ids=uuid_ids
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
    
    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.
        
        Args:
            name: Collection name to check
            
        Returns:
            True if collection exists, False otherwise
        """
        collections = await self.list_collections()
        return name in collections
        
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
