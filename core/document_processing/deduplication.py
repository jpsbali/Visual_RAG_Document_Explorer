"""
Document deduplication using content hashing and semantic similarity.

Two-stage approach:
1. Content-hash deduplication: SHA-256 for exact duplicates
2. Semantic deduplication: Cosine similarity for near-duplicates
"""

from config.models import ChunkMetadata
from config.settings import Settings
from core.embeddings.embedding_router import get_embedding_service
from typing import Literal
import hashlib
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DeduplicationService:
    """
    Document deduplication using content hashing and semantic similarity.
    
    Two-stage approach:
    1. Content-hash deduplication: SHA-256 for exact duplicates
    2. Semantic deduplication: Cosine similarity for near-duplicates
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize deduplication service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.embedding_service = get_embedding_service(settings)
        self.similarity_threshold = settings.dedup_similarity_threshold
        
        # In-memory cache of seen documents
        self.content_hashes: set[str] = set()
        self.embeddings_cache: dict[str, list[float]] = {}
        
        logger.info(
            f"Initialized DeduplicationService "
            f"(threshold={self.similarity_threshold})"
        )
        
    def compute_content_hash(self, text: str) -> str:
        """
        Compute SHA-256 hash of normalized text.
        
        Args:
            text: Document text
            
        Returns:
            Hex digest of SHA-256 hash
        """
        # Normalize: lowercase, strip whitespace
        normalized = text.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()
        
    def check_exact_duplicate(self, content_hash: str) -> bool:
        """
        Check if content hash exists in cache.
        
        Args:
            content_hash: SHA-256 hash of document
            
        Returns:
            True if exact duplicate found
        """
        return content_hash in self.content_hashes
        
    def add_content_hash(self, content_hash: str) -> None:
        """
        Add content hash to cache.
        
        Args:
            content_hash: SHA-256 hash to add
        """
        self.content_hashes.add(content_hash)
        
    async def check_semantic_duplicate(
        self,
        text: str,
        embedding: list[float] = None
    ) -> tuple[bool, float]:
        """
        Check for semantic near-duplicates using cosine similarity.
        
        Args:
            text: Document text
            embedding: Pre-computed embedding (optional)
            
        Returns:
            Tuple of (is_duplicate, max_similarity)
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = await self.embedding_service.embed_query(text)
            
        # Compare with cached embeddings
        max_similarity = 0.0
        for cached_hash, cached_embedding in self.embeddings_cache.items():
            similarity = self._cosine_similarity(embedding, cached_embedding)
            max_similarity = max(max_similarity, similarity)
            
            if similarity >= self.similarity_threshold:
                logger.info(
                    f"Semantic duplicate found (similarity: {similarity:.3f})"
                )
                return True, similarity
                
        return False, max_similarity
        
    def add_embedding(self, content_hash: str, embedding: list[float]) -> None:
        """
        Add embedding to cache for future comparisons.
        
        Args:
            content_hash: Content hash key
            embedding: Embedding vector
        """
        self.embeddings_cache[content_hash] = embedding
        
    async def check_duplicate(
        self,
        text: str,
        metadata: ChunkMetadata
    ) -> tuple[Literal["new", "exact_duplicate", "near_duplicate"], float]:
        """
        Check for both exact and semantic duplicates.
        
        Args:
            text: Document text
            metadata: Chunk metadata with content_hash
            
        Returns:
            Tuple of (status, similarity_score)
        """
        content_hash = metadata.content_hash
        
        # Check exact duplicate first (fast)
        if self.check_exact_duplicate(content_hash):
            logger.info(f"Exact duplicate found: {content_hash[:8]}...")
            return "exact_duplicate", 1.0
            
        # Check semantic duplicate (slower)
        is_semantic_dup, similarity = await self.check_semantic_duplicate(text)
        
        if is_semantic_dup:
            return "near_duplicate", similarity
            
        # Not a duplicate - add to cache
        self.add_content_hash(content_hash)
        embedding = await self.embedding_service.embed_query(text)
        self.add_embedding(content_hash, embedding)
        
        return "new", 0.0
        
    async def deduplicate_chunks(
        self,
        chunks: list[tuple[str, ChunkMetadata]]
    ) -> tuple[list[tuple[str, ChunkMetadata]], dict]:
        """
        Deduplicate a list of chunks.
        
        Args:
            chunks: List of (text, metadata) tuples
            
        Returns:
            Tuple of (deduplicated_chunks, stats_dict)
        """
        deduplicated = []
        stats = {
            "total": len(chunks),
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "unique": 0
        }
        
        for text, metadata in chunks:
            status, similarity = await self.check_duplicate(text, metadata)
            
            if status == "exact_duplicate":
                stats["exact_duplicates"] += 1
            elif status == "near_duplicate":
                stats["near_duplicates"] += 1
            else:  # new
                stats["unique"] += 1
                deduplicated.append((text, metadata))
                
        logger.info(
            f"Deduplication: {stats['total']} total → "
            f"{stats['unique']} unique "
            f"({stats['exact_duplicates']} exact, "
            f"{stats['near_duplicates']} near duplicates)"
        )
        
        return deduplicated, stats
        
    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0-1.0)
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))
        
    def clear_cache(self) -> None:
        """Clear deduplication cache."""
        self.content_hashes.clear()
        self.embeddings_cache.clear()
        logger.info("Deduplication cache cleared")
        
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        return {
            "content_hashes": len(self.content_hashes),
            "embeddings_cached": len(self.embeddings_cache),
            "similarity_threshold": self.similarity_threshold
        }
        
    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Set the similarity threshold for near-duplicate detection.
        
        Args:
            threshold: Similarity threshold (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
            
        self.similarity_threshold = threshold
        logger.info(f"Updated similarity threshold to {threshold}")
