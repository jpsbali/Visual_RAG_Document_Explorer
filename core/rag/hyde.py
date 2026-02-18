"""
HYDE (Hypothetical Document Embeddings) for Visual RAG Document Explorer.

Generates hypothetical answers and uses their embeddings for improved retrieval.
"""

from typing import Optional

from config.settings import Settings
from config.models import RetrievedChunk, HYDEResult
from core.llm.llm_router import get_llm_provider
from core.embeddings.embedding_router import get_embedding_service
from core.search.hybrid_search import HybridSearch


class HYDEGenerator:
    """
    HYDE (Hypothetical Document Embeddings) generator.
    
    Generates 1-3 hypothetical answers to the query using LLM, embeds them,
    and retrieves using those embeddings. This improves retrieval by bridging
    the semantic gap between queries and documents.
    """
    
    def __init__(
        self,
        settings: Settings,
        hybrid_search: HybridSearch
    ):
        """
        Initialize HYDE generator.
        
        Args:
            settings: Application settings
            hybrid_search: Hybrid search service for retrieval
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.llm = get_llm_provider(settings)
        self.embeddings = get_embedding_service(settings)
    
    async def generate_hypothetical_documents(
        self,
        query: str,
        num_documents: int = 3
    ) -> list[str]:
        """
        Generate hypothetical documents that would answer the query.
        
        Args:
            query: User query
            num_documents: Number of hypothetical documents to generate (1-3)
            
        Returns:
            List of hypothetical document texts
        """
        system_prompt = """You are an expert document writer. Generate hypothetical document passages that would perfectly answer the given query.

Rules:
1. Write as if you're extracting text from an actual document
2. Be specific and detailed
3. Use formal, document-like language
4. Each passage should be 2-3 sentences
5. Make each passage slightly different in focus or detail"""
        
        user_prompt = f"""Query: "{query}"

Generate {num_documents} hypothetical document passages that would answer this query. Write each passage on a new line, numbered 1., 2., 3."""
        
        try:
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Slightly higher for diversity
                max_tokens=500
            )
            
            # Parse numbered passages
            passages = []
            lines = response.strip().split("\n")
            current_passage = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line starts with a number
                if line[0].isdigit() and (line[1] == "." or line[1] == ")"):
                    # Save previous passage
                    if current_passage:
                        passages.append(" ".join(current_passage))
                        current_passage = []
                    # Start new passage (remove number prefix)
                    current_passage.append(line[2:].strip())
                else:
                    # Continue current passage
                    current_passage.append(line)
            
            # Add last passage
            if current_passage:
                passages.append(" ".join(current_passage))
            
            # Fallback: if parsing failed, split by newlines
            if not passages:
                passages = [p.strip() for p in response.split("\n") if p.strip()]
            
            # Limit to requested number
            passages = passages[:num_documents]
            
            # Ensure we have at least one passage
            if not passages:
                passages = [f"A document that answers: {query}"]
            
            return passages
            
        except Exception as e:
            # Fallback: return query as hypothetical document
            return [f"A document that answers: {query}"]
    
    async def retrieve_with_hyde(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        num_hypothetical: int = 3,
        filters: Optional[dict] = None
    ) -> tuple[list[RetrievedChunk], HYDEResult]:
        """
        Retrieve documents using HYDE approach.
        
        Args:
            query: User query
            collection: Vector store collection name
            top_k: Number of results to retrieve per hypothetical document
            num_hypothetical: Number of hypothetical documents to generate
            filters: Optional metadata filters
            
        Returns:
            Tuple of (retrieved chunks, HYDE result with metadata)
        """
        # Generate hypothetical documents
        hypothetical_docs = await self.generate_hypothetical_documents(
            query=query,
            num_documents=num_hypothetical
        )
        
        # Retrieve using each hypothetical document
        all_chunks = []
        seen_chunk_ids = set()
        
        for hyp_doc in hypothetical_docs:
            # Use hypothetical document as query
            chunks = await self.hybrid_search.search(
                query=hyp_doc,
                collection=collection,
                top_k=top_k,
                search_mode="dense",  # HYDE works best with dense search
                filters=filters
            )
            
            # Deduplicate by chunk_id
            for chunk in chunks:
                if chunk.metadata.chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.metadata.chunk_id)
        
        # Sort by score (descending)
        all_chunks.sort(key=lambda x: x.score, reverse=True)
        
        # Create HYDE result
        hyde_result = HYDEResult(
            original_query=query,
            hypothetical_documents=hypothetical_docs,
            enhanced_retrieval=len(all_chunks) > 0
        )
        
        return all_chunks, hyde_result
    
    async def enhance_retrieval(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        filters: Optional[dict] = None
    ) -> list[RetrievedChunk]:
        """
        Enhance retrieval by combining HYDE with standard retrieval.
        
        Args:
            query: User query
            collection: Vector store collection name
            top_k: Number of final results to return
            filters: Optional metadata filters
            
        Returns:
            Enhanced retrieved chunks
        """
        # Get HYDE results
        hyde_chunks, _ = await self.retrieve_with_hyde(
            query=query,
            collection=collection,
            top_k=top_k // 2,  # Get fewer from HYDE
            num_hypothetical=2,
            filters=filters
        )
        
        # Get standard retrieval results
        standard_chunks = await self.hybrid_search.search(
            query=query,
            collection=collection,
            top_k=top_k // 2,
            search_mode=self.settings.search_mode,
            filters=filters
        )
        
        # Merge and deduplicate
        all_chunks = []
        seen_chunk_ids = set()
        
        # Add HYDE chunks first (they tend to be more relevant)
        for chunk in hyde_chunks:
            if chunk.metadata.chunk_id not in seen_chunk_ids:
                all_chunks.append(chunk)
                seen_chunk_ids.add(chunk.metadata.chunk_id)
        
        # Add standard chunks
        for chunk in standard_chunks:
            if chunk.metadata.chunk_id not in seen_chunk_ids:
                all_chunks.append(chunk)
                seen_chunk_ids.add(chunk.metadata.chunk_id)
        
        # Sort by score and limit to top_k
        all_chunks.sort(key=lambda x: x.score, reverse=True)
        return all_chunks[:top_k]
