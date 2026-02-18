"""
BM25 sparse keyword search implementation.

Provides exact term matching to complement dense vector search.
"""

from rank_bm25 import BM25Okapi
from config.models import RetrievedChunk, ChunkMetadata
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BM25Search:
    """
    BM25 sparse keyword search for exact term matching.
    
    Complements dense vector search by catching queries with specific
    terminology that might be missed by semantic embeddings.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 search.
        
        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.k1 = k1
        self.b = b
        self.index: Optional[BM25Okapi] = None
        self.documents: list[str] = []
        self.metadatas: list[ChunkMetadata] = []
        self.chunk_ids: list[str] = []
        logger.info(f"Initialized BM25Search (k1={k1}, b={b})")
        
    def build_index(
        self, 
        documents: list[str], 
        metadatas: list[ChunkMetadata]
    ) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document texts
            metadatas: List of ChunkMetadata objects
        """
        self.documents = documents
        self.metadatas = metadatas
        self.chunk_ids = [m.chunk_id for m in metadatas]
        
        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Build BM25 index
        self.index = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        
        logger.info(f"Built BM25 index with {len(documents)} documents")
        
    def add_documents(
        self, 
        documents: list[str], 
        metadatas: list[ChunkMetadata]
    ) -> None:
        """
        Add documents to existing index (incremental indexing).
        
        Args:
            documents: List of new document texts
            metadatas: List of new ChunkMetadata objects
        """
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.chunk_ids.extend([m.chunk_id for m in metadatas])
        
        # Rebuild index (BM25Okapi doesn't support incremental updates)
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.index = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        
        logger.info(f"Added {len(documents)} documents to BM25 index (total: {len(self.documents)})")
        
    def remove_documents(self, chunk_ids: list[str]) -> None:
        """
        Remove documents from index by chunk ID.
        
        Args:
            chunk_ids: List of chunk IDs to remove
        """
        # Filter out removed documents
        chunk_ids_set = set(chunk_ids)
        indices_to_keep = [
            i for i, cid in enumerate(self.chunk_ids) 
            if cid not in chunk_ids_set
        ]
        
        self.documents = [self.documents[i] for i in indices_to_keep]
        self.metadatas = [self.metadatas[i] for i in indices_to_keep]
        self.chunk_ids = [self.chunk_ids[i] for i in indices_to_keep]
        
        # Rebuild index
        if self.documents:
            tokenized_docs = [doc.lower().split() for doc in self.documents]
            self.index = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        else:
            self.index = None
            
        logger.info(f"Removed {len(chunk_ids)} documents from BM25 index (remaining: {len(self.documents)})")
        
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        filters: Optional[dict] = None
    ) -> list[RetrievedChunk]:
        """
        Search using BM25 keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters (applied post-search)
            
        Returns:
            List of RetrievedChunk objects with BM25 scores
        """
        if self.index is None:
            logger.warning("BM25 index not built, returning empty results")
            return []
            
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.index.get_scores(tokenized_query)
        
        # Get top-k indices (retrieve more for filtering)
        retrieval_k = top_k * 2 if filters else top_k
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:retrieval_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:  # Skip zero-score results
                continue
                
            metadata = self.metadatas[idx]
            
            # Apply metadata filters if provided
            if filters:
                if not self._matches_filters(metadata, filters):
                    continue
                    
            results.append(
                RetrievedChunk(
                    content=self.documents[idx],
                    metadata=metadata,
                    score=float(scores[idx]),
                    search_method="sparse"
                )
            )
            
            if len(results) >= top_k:
                break
                
        logger.info(f"BM25 search returned {len(results)} results for query: {query[:50]}...")
        return results
        
    def _matches_filters(
        self, 
        metadata: ChunkMetadata, 
        filters: dict
    ) -> bool:
        """
        Check if metadata matches filter criteria.
        
        Args:
            metadata: Chunk metadata
            filters: Filter criteria
            
        Returns:
            True if metadata matches all filters
        """
        for field, values in filters.items():
            if field in ["organizations", "people", "dates", "locations", "topics"]:
                # Check entity fields
                entity_values = getattr(metadata.entities, field, [])
                if not any(v in entity_values for v in values):
                    return False
            elif field == "file_type":
                if metadata.file_type not in values:
                    return False
            elif field == "source_file":
                if metadata.source_file not in values:
                    return False
            elif field == "chunk_type":
                if metadata.chunk_type not in values:
                    return False
        return True
        
    def get_index_size(self) -> int:
        """
        Get number of documents in index.
        
        Returns:
            Number of indexed documents
        """
        return len(self.documents)
        
    def clear_index(self) -> None:
        """Clear the BM25 index and all cached data."""
        self.index = None
        self.documents = []
        self.metadatas = []
        self.chunk_ids = []
        logger.info("Cleared BM25 index")
