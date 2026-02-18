"""
Contextual Compressor for Visual RAG Document Explorer.

LLM-based sentence extraction to reduce noise and token usage while preserving
relevant information.
"""

from typing import Optional

from config.settings import Settings
from config.models import RetrievedChunk, ChunkMetadata
from core.llm.llm_router import get_llm_provider


class ContextualCompressor:
    """
    Contextual Compressor for extracting relevant portions from retrieved chunks.
    
    Uses LLM to identify and extract only the sentences/passages that are relevant
    to the query, reducing noise and token usage while preserving key information.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize contextual compressor.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.llm = get_llm_provider(settings)
    
    async def compress_chunk(
        self,
        query: str,
        chunk: RetrievedChunk,
        compression_ratio: float = 0.5
    ) -> Optional[RetrievedChunk]:
        """
        Compress a single chunk by extracting relevant portions.
        
        Args:
            query: User query
            chunk: Retrieved chunk to compress
            compression_ratio: Target compression ratio (0.0-1.0)
            
        Returns:
            Compressed chunk or None if no relevant content found
        """
        system_prompt = """You are a text extraction expert. Extract ONLY the sentences or passages that are directly relevant to answering the query.

Rules:
1. Extract complete sentences, not fragments
2. Preserve the original wording exactly
3. Maintain logical flow between extracted sentences
4. If nothing is relevant, return "NONE"
5. Return only the extracted text, no explanations"""
        
        user_prompt = f"""Query: "{query}"

Source Text:
{chunk.content}

Extract the relevant portions that help answer the query. Return only the extracted text."""
        
        try:
            compressed_content = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=min(len(chunk.content), 1000)
            )
            
            compressed_content = compressed_content.strip()
            
            # Check if nothing relevant was found
            if compressed_content.upper() == "NONE" or not compressed_content:
                return None
            
            # Check if compression actually reduced content
            if len(compressed_content) >= len(chunk.content):
                # No compression achieved, return original
                return chunk
            
            # Create compressed chunk with updated content
            compressed_chunk = RetrievedChunk(
                content=compressed_content,
                metadata=chunk.metadata,
                score=chunk.score,
                search_method=chunk.search_method
            )
            
            return compressed_chunk
            
        except Exception as e:
            # Fallback: return original chunk if compression fails
            return chunk
    
    async def compress_chunks(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        compression_ratio: float = 0.5,
        preserve_top_k: int = 2
    ) -> list[RetrievedChunk]:
        """
        Compress multiple chunks by extracting relevant portions.
        
        Args:
            query: User query
            chunks: Retrieved chunks to compress
            compression_ratio: Target compression ratio (0.0-1.0)
            preserve_top_k: Number of top chunks to preserve without compression
            
        Returns:
            List of compressed chunks (excluding chunks with no relevant content)
        """
        if not chunks:
            return []
        
        compressed_chunks = []
        
        # Preserve top-k chunks without compression (they're likely most relevant)
        for i, chunk in enumerate(chunks):
            if i < preserve_top_k:
                compressed_chunks.append(chunk)
            else:
                # Compress remaining chunks
                compressed = await self.compress_chunk(
                    query=query,
                    chunk=chunk,
                    compression_ratio=compression_ratio
                )
                if compressed:
                    compressed_chunks.append(compressed)
        
        return compressed_chunks
    
    async def compress_with_context_window(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        max_tokens: int = 4000
    ) -> list[RetrievedChunk]:
        """
        Compress chunks to fit within a token budget.
        
        Args:
            query: User query
            chunks: Retrieved chunks to compress
            max_tokens: Maximum total tokens allowed
            
        Returns:
            Compressed chunks that fit within token budget
        """
        if not chunks:
            return []
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        compressed_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = estimate_tokens(chunk.content)
            
            # If chunk fits, add it
            if total_tokens + chunk_tokens <= max_tokens:
                compressed_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                # Try to compress chunk to fit
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Only compress if we have reasonable space
                    # Calculate target compression ratio
                    target_ratio = remaining_tokens / chunk_tokens
                    
                    compressed = await self.compress_chunk(
                        query=query,
                        chunk=chunk,
                        compression_ratio=target_ratio
                    )
                    
                    if compressed:
                        compressed_tokens = estimate_tokens(compressed.content)
                        if total_tokens + compressed_tokens <= max_tokens:
                            compressed_chunks.append(compressed)
                            total_tokens += compressed_tokens
                        else:
                            # Can't fit even compressed version, stop here
                            break
                else:
                    # No more space, stop
                    break
        
        return compressed_chunks
    
    def get_compression_stats(
        self,
        original_chunks: list[RetrievedChunk],
        compressed_chunks: list[RetrievedChunk]
    ) -> dict:
        """
        Calculate compression statistics.
        
        Args:
            original_chunks: Original chunks before compression
            compressed_chunks: Compressed chunks
            
        Returns:
            Dictionary with compression statistics
        """
        original_chars = sum(len(c.content) for c in original_chunks)
        compressed_chars = sum(len(c.content) for c in compressed_chunks)
        
        compression_ratio = compressed_chars / original_chars if original_chars > 0 else 0
        
        return {
            "original_chunks": len(original_chunks),
            "compressed_chunks": len(compressed_chunks),
            "original_chars": original_chars,
            "compressed_chars": compressed_chars,
            "compression_ratio": compression_ratio,
            "chars_saved": original_chars - compressed_chars,
            "chunks_removed": len(original_chunks) - len(compressed_chunks)
        }
