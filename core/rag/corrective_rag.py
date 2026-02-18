"""
Corrective RAG (CRAG) for Visual RAG Document Explorer.

Grades retrieval quality and applies correction strategies when needed.
"""

import json
import time
from datetime import datetime
from typing import Optional, Literal

from core.rag.base import RAGStrategy
from core.rag.query_decomposer import QueryDecomposer
from config.models import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    CRAGEvaluation,
    CRAGResult
)


class CorrectiveRAG(RAGStrategy):
    """
    Corrective RAG (CRAG): Grades retrieval quality and applies corrections.
    
    Pipeline:
    1. Retrieve initial chunks
    2. Grade each chunk for relevance
    3. If average score < threshold, apply correction:
       - reformulate: Rephrase query for better retrieval
       - broaden: Expand query scope
       - decompose: Break into sub-queries
    4. Merge original and corrected results
    5. Rerank and generate answer
    """
    
    @property
    def strategy_name(self) -> str:
        """Return the name of this RAG strategy."""
        return "crag"
    
    def __init__(self, *args, **kwargs):
        """Initialize CRAG with query decomposer."""
        super().__init__(*args, **kwargs)
        self.query_decomposer = QueryDecomposer(self.settings)
    
    async def execute(self, request: QueryRequest) -> QueryResponse:
        """
        Execute Corrective RAG pipeline.
        
        Args:
            request: Query request with all parameters
            
        Returns:
            Complete query response with CRAG details
        """
        start_time = time.time()
        
        # Step 1: Initial retrieval
        initial_k = request.top_k * 2
        retrieved_chunks = await self._retrieve(
            query=request.query,
            top_k=initial_k,
            filters=request.metadata_filters
        )
        
        # Step 2: Grade retrieval quality
        evaluation = await self._grade_retrieval(
            query=request.query,
            chunks=retrieved_chunks[:5]  # Grade top 5
        )
        
        # Step 3: Apply correction if needed
        corrected_chunks = None
        reformulated_query = None
        
        if evaluation.needs_correction:
            corrected_chunks, reformulated_query = await self._apply_correction(
                query=request.query,
                strategy=evaluation.correction_strategy,
                filters=request.metadata_filters,
                top_k=initial_k
            )
            
            # Merge original and corrected results
            if corrected_chunks:
                merged_chunks = self._merge_chunks(retrieved_chunks, corrected_chunks)
            else:
                merged_chunks = retrieved_chunks
        else:
            merged_chunks = retrieved_chunks
        
        # Step 4: Rerank
        if request.enable_reranking and merged_chunks:
            reranked_chunks = await self._rerank(
                query=request.query,
                chunks=merged_chunks,
                top_k=request.top_k
            )
        else:
            reranked_chunks = merged_chunks[:request.top_k]
        
        # Step 5: Generate answer
        if reranked_chunks:
            answer = await self._generate_answer(
                query=request.query,
                chunks=reranked_chunks
            )
        else:
            answer = "I couldn't find relevant information to answer your query, even after applying correction strategies. Please try rephrasing your question."
        
        # Extract citations and create grounding
        citations = self._extract_citations(reranked_chunks)
        grounding = self._create_placeholder_grounding()
        
        # Create CRAG result
        crag_result = CRAGResult(
            correction_applied=evaluation.needs_correction,
            evaluation=evaluation,
            original_chunks=retrieved_chunks[:5],
            corrected_chunks=corrected_chunks[:5] if corrected_chunks else None,
            reformulated_query=reformulated_query
        )
        
        response_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            mode="crag",
            search_mode=request.search_mode,
            sources=reranked_chunks,
            citations=citations,
            grounding=grounding,
            crag_details=crag_result,
            response_time_ms=response_time,
            hyde_used=False,
            reranking_used=request.enable_reranking,
            compression_used=False,
            initial_retrieval_count=len(retrieved_chunks),
            final_retrieval_count=len(reranked_chunks)
        )
    
    async def _grade_retrieval(
        self,
        query: str,
        chunks: list[RetrievedChunk]
    ) -> CRAGEvaluation:
        """
        Grade retrieval quality using LLM.
        
        Args:
            query: User query
            chunks: Retrieved chunks to grade
            
        Returns:
            CRAG evaluation result
        """
        if not chunks:
            return CRAGEvaluation(
                relevance_score=0.0,
                relevance_label="irrelevant",
                confidence=1.0,
                evaluation_method="llm_grader",
                needs_correction=True,
                correction_strategy="reformulate",
                evaluated_at=datetime.now()
            )
        
        # Format chunks for grading
        chunks_text = "\n\n".join([
            f"[Chunk {i+1}]\n{chunk.content}"
            for i, chunk in enumerate(chunks)
        ])
        
        system_prompt = """You are a retrieval quality evaluator. Grade how relevant the retrieved chunks are for answering the query.

Return a JSON object with:
{
  "relevance_score": 0.0-1.0,
  "relevance_label": "relevant|ambiguous|irrelevant",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Grading criteria:
- relevant (0.7-1.0): Chunks contain clear, direct information to answer the query
- ambiguous (0.4-0.7): Chunks contain partial or tangential information
- irrelevant (0.0-0.4): Chunks don't help answer the query"""
        
        user_prompt = f"""Query: "{query}"

Retrieved Chunks:
{chunks_text}

Grade the relevance of these chunks. Return ONLY the JSON object."""
        
        try:
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=300
            )
            
            response = response.strip()
            
            # Handle markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                response = "\n".join(json_lines)
            
            result = json.loads(response)
            
            relevance_score = float(result.get("relevance_score", 0.0))
            relevance_label = result.get("relevance_label", "irrelevant")
            confidence = float(result.get("confidence", 0.5))
            
            # Clamp values
            relevance_score = max(0.0, min(1.0, relevance_score))
            confidence = max(0.0, min(1.0, confidence))
            
            # Validate label
            if relevance_label not in ["relevant", "ambiguous", "irrelevant"]:
                relevance_label = "irrelevant"
            
            # Determine if correction is needed
            needs_correction = relevance_score < 0.5
            
            # Choose correction strategy
            correction_strategy = None
            if needs_correction:
                if relevance_score < 0.3:
                    correction_strategy = "decompose"
                elif relevance_label == "ambiguous":
                    correction_strategy = "broaden"
                else:
                    correction_strategy = "reformulate"
            
            return CRAGEvaluation(
                relevance_score=relevance_score,
                relevance_label=relevance_label,
                confidence=confidence,
                evaluation_method="llm_grader",
                needs_correction=needs_correction,
                correction_strategy=correction_strategy,
                evaluated_at=datetime.now()
            )
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: assume correction needed
            return CRAGEvaluation(
                relevance_score=0.4,
                relevance_label="ambiguous",
                confidence=0.5,
                evaluation_method="llm_grader",
                needs_correction=True,
                correction_strategy="reformulate",
                evaluated_at=datetime.now()
            )
    
    async def _apply_correction(
        self,
        query: str,
        strategy: Optional[Literal["reformulate", "broaden", "decompose"]],
        filters: Optional[dict],
        top_k: int
    ) -> tuple[Optional[list[RetrievedChunk]], Optional[str]]:
        """
        Apply correction strategy to improve retrieval.
        
        Args:
            query: Original query
            strategy: Correction strategy to apply
            filters: Metadata filters
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (corrected chunks, reformulated query)
        """
        if not strategy:
            return None, None
        
        if strategy == "reformulate":
            # Reformulate query for better retrieval
            reformulated = await self._reformulate_query(query)
            chunks = await self._retrieve(
                query=reformulated,
                top_k=top_k,
                filters=filters
            )
            return chunks, reformulated
        
        elif strategy == "broaden":
            # Broaden query scope
            broadened = await self._broaden_query(query)
            chunks = await self._retrieve(
                query=broadened,
                top_k=top_k,
                filters=filters
            )
            return chunks, broadened
        
        elif strategy == "decompose":
            # Decompose into sub-queries
            subqueries = await self.query_decomposer.decompose(query, max_subqueries=3)
            all_chunks = []
            seen_ids = set()
            
            for subquery in subqueries:
                chunks = await self._retrieve(
                    query=subquery,
                    top_k=top_k // len(subqueries),
                    filters=filters
                )
                for chunk in chunks:
                    if chunk.metadata.chunk_id not in seen_ids:
                        all_chunks.append(chunk)
                        seen_ids.add(chunk.metadata.chunk_id)
            
            return all_chunks, " | ".join(subqueries)
        
        return None, None
    
    async def _reformulate_query(self, query: str) -> str:
        """Reformulate query for better retrieval."""
        system_prompt = "You are a query reformulation expert. Rephrase the query to improve information retrieval. Return ONLY the reformulated query, no explanations."
        
        user_prompt = f'Reformulate this query for better search results: "{query}"'
        
        try:
            reformulated = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=100
            )
            return reformulated.strip().strip('"')
        except:
            return query
    
    async def _broaden_query(self, query: str) -> str:
        """Broaden query scope."""
        system_prompt = "You are a query expansion expert. Broaden the query to include related concepts. Return ONLY the broadened query, no explanations."
        
        user_prompt = f'Broaden this query to include related concepts: "{query}"'
        
        try:
            broadened = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=100
            )
            return broadened.strip().strip('"')
        except:
            return query
    
    def _merge_chunks(
        self,
        original: list[RetrievedChunk],
        corrected: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Merge and deduplicate original and corrected chunks."""
        merged = []
        seen_ids = set()
        
        # Add all chunks, deduplicating by chunk_id
        for chunk in original + corrected:
            if chunk.metadata.chunk_id not in seen_ids:
                merged.append(chunk)
                seen_ids.add(chunk.metadata.chunk_id)
        
        # Sort by score
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged
