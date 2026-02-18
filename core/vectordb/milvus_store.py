"""
Milvus vector database implementation.

Provides async operations for document storage and retrieval using Milvus.
"""

from pymilvus import (
    connections, Collection, CollectionSchema, 
    FieldSchema, DataType, utility
)
from config.models import ChunkMetadata, RetrievedChunk
from config.settings import Settings
from .base import VectorStoreBase
from typing import Optional
import logging
import json

logger = logging.getLogger(__name__)


class MilvusStore(VectorStoreBase):
    """Milvus vector database implementation."""
    
    def __init__(self, settings: Settings):
        """
        Initialize Milvus store.
        
        Args:
            settings: Application settings with Milvus configuration
        """
        self.settings = settings
        self.default_collection = settings.milvus_collection
        
        # Parse Milvus URL
        url_parts = settings.milvus_url.replace("http://", "").replace("https://", "").split(":")
        self.host = url_parts[0]
        self.port = url_parts[1] if len(url_parts) > 1 else "19530"
        
        # Connect to Milvus
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
        
    async def create_collection(
        self, 
        name: str, 
        dimension: int,
        distance_metric: str = "cosine"
    ) -> None:
        """
        Create Milvus collection with schema.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, euclidean, dot)
        """
        try:
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON)  # Store metadata as JSON
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=f"Document chunks for {name}"
            )
            
            collection = Collection(name=name, schema=schema)
            
            # Create index for vector field
            metric_map = {
                "cosine": "COSINE",
                "euclidean": "L2",
                "dot": "IP"
            }
            
            index_params = {
                "metric_type": metric_map.get(distance_metric, "COSINE"),
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"Created Milvus collection: {name} (dim={dimension}, metric={distance_metric})")
        except Exception as e:
            logger.error(f"Failed to create Milvus collection {name}: {e}")
            raise
    
    async def delete_collection(self, name: str) -> None:
        """
        Delete a Milvus collection.
        
        Args:
            name: Collection name to delete
        """
        try:
            utility.drop_collection(name)
            logger.info(f"Deleted Milvus collection: {name}")
        except Exception as e:
            logger.error(f"Failed to delete Milvus collection {name}: {e}")
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
        Upsert documents to Milvus.
        
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
            coll = Collection(collection)
            
            # Prepare data
            data = [
                ids,
                embeddings,
                documents,
                [json.dumps(m) for m in metadatas]  # Serialize metadata to JSON
            ]
            
            coll.insert(data)
            coll.flush()
            
            logger.info(f"Upserted {len(ids)} documents to Milvus collection: {collection}")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to upsert to Milvus collection {collection}: {e}")
            raise
        
    async def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Search Milvus with optional metadata filters.
        
        Args:
            collection: Collection name
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"organizations": ["Acme Corp"]})
            
        Returns:
            List of RetrievedChunk objects
        """
        try:
            coll = Collection(collection)
            coll.load()
            
            # Build filter expression
            expr = None
            if filters:
                # Milvus JSON filtering syntax
                conditions = []
                for field, values in filters.items():
                    if field in ["organizations", "people", "dates", "locations", "topics"]:
                        # Example: json_contains(metadata["entities"]["organizations"], "Acme Corp")
                        value_conditions = [
                            f'json_contains(metadata["entities"]["{field}"], "{v}")'
                            for v in values
                        ]
                        conditions.append(f"({' or '.join(value_conditions)})")
                    else:
                        # Support other metadata fields
                        value_list = ", ".join([f'"{v}"' for v in values])
                        conditions.append(f'metadata["{field}"] in [{value_list}]')
                expr = " and ".join(conditions) if conditions else None
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = coll.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["content", "metadata"]
            )
            
            # Convert to RetrievedChunk objects
            chunks = []
            for hits in results:
                for hit in hits:
                    content = hit.entity.get("content")
                    metadata_json = hit.entity.get("metadata")
                    
                    # Parse metadata JSON
                    if isinstance(metadata_json, str):
                        metadata_dict = json.loads(metadata_json)
                    else:
                        metadata_dict = metadata_json
                    
                    metadata = ChunkMetadata(**metadata_dict)
                    
                    chunks.append(
                        RetrievedChunk(
                            content=content,
                            metadata=metadata,
                            score=hit.score,
                            search_method="dense"
                        )
                    )
            
            logger.info(f"Retrieved {len(chunks)} chunks from Milvus collection: {collection}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to search Milvus collection {collection}: {e}")
            raise
        
    async def delete(self, collection: str, ids: list[str]) -> int:
        """
        Delete documents from Milvus by ID.
        
        Args:
            collection: Collection name
            ids: List of document IDs to delete
            
        Returns:
            Count of deleted documents
        """
        try:
            coll = Collection(collection)
            id_list = ", ".join([f'"{id}"' for id in ids])
            expr = f'id in [{id_list}]'
            coll.delete(expr)
            logger.info(f"Deleted {len(ids)} documents from Milvus collection: {collection}")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to delete from Milvus collection {collection}: {e}")
            raise
        
    async def get(self, collection: str, ids: list[str]) -> list[dict]:
        """
        Retrieve documents by ID from Milvus.
        
        Args:
            collection: Collection name
            ids: List of document IDs to retrieve
            
        Returns:
            List of documents with metadata
        """
        try:
            coll = Collection(collection)
            coll.load()
            
            id_list = ", ".join([f'"{id}"' for id in ids])
            expr = f'id in [{id_list}]'
            results = coll.query(
                expr=expr,
                output_fields=["content", "metadata"]
            )
            
            return [
                {
                    "content": r["content"],
                    "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Failed to retrieve from Milvus collection {collection}: {e}")
            raise
        
    async def count(self, collection: str) -> int:
        """
        Count documents in Milvus collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Number of documents
        """
        try:
            coll = Collection(collection)
            return coll.num_entities
        except Exception as e:
            logger.error(f"Failed to count Milvus collection {collection}: {e}")
            raise
        
    async def list_collections(self) -> list[str]:
        """
        List all Milvus collections.
        
        Returns:
            List of collection names
        """
        try:
            return utility.list_collections()
        except Exception as e:
            logger.error(f"Failed to list Milvus collections: {e}")
            raise
        
    async def health_check(self) -> bool:
        """
        Check Milvus health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            utility.list_collections()
            logger.info("Milvus health check passed")
            return True
        except Exception as e:
            logger.error(f"Milvus health check failed: {e}")
            return False
