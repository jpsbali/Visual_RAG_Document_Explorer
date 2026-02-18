"""
Self-Reflective RAG (SRAG) for Visual RAG Document Explorer.

Generates draft answers, self-evaluates, and refines through iterative reflection.
"""

import json
import time
from datetime import datetime
from typing import Optional

from core.rag.base import RAGStrategy
from config.models import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    ReflectionResult,
    SelfReflectiveResult
)


class SelfReflectiveRAG(RAGStrategy):
    """
    Self-Reflective RAG (SRAG): Iterative refinement through self-evaluation.
    
    Pipeline:
    1. Retrieve and generate draft answer
    2. Self-evaluate for hallucination, completeness, faithfulness
    3. If unsatisfactory, refine query and re-retrieve
    4. Repeat up to max_iterations
    5. Return best answer with reflection history
    """
    
    @property
    def strategy_name(self) -> str:
        """Return the name of this RAG strategy."""
        return "srag"
    
    async def execute(self, request: QueryRequest) -> QueryResponse:
        """
        Execute Self-Reflective RAG pipeline.
        
        Args:
            request: Query request with all parameters
            
        Returns:
            Complete query response with reflection details
        """
        start_time = time.time()
        
        max_iterations = self.settings.srag_max_iterations
        reflections = []
        best_answer = None
        best_chunks = None
        best_score = 0.0
        
        current_query = request.query
        
        for iteration in range(max_iterations):
            # Step 1: Retrieve
            initial_k = request.top_k * 2
            retrieved_chunks = await self._retrieve(
                query=current_query,
                top_k=initial_k,
                filters=request.metadata_filters
            )
            
            # Step 2: Rerank
            if request.enable_reranking and retrieved_chunks:
                reranked_chunks = await self._rerank(
                    query=current_query,
                    chunks=retrieved_chunks,
                    top_k=request.top_k
                )
            else:
                reranked_chunks = retrieved_chunks[:request.top_k]
            
            # Step 3: Generate draft answer
            if reranked_chunks:
                draft_answer = await self._generate_answer(
                    query=current_query,
                    chunks=reranked_chunks
                )
            else:
                draft_answer = "I couldn't find relevant information to answer your query."
            
            # Step 4: Self-evaluate
            reflection = await self._reflect_on_answer(
                query=request.query,
                answer=draft_answer,
                chunks=reranked_chunks,
                iteration=iteration + 1
            )
            reflections.append(reflection)
            
            # Track best answer
            if reflection.reflection_score > best_score:
                best_score = reflection.reflection_score
                best_answer = draft_answer
                best_chunks = reranked_chunks
            
            # Step 5: Check if we should continue
            if not reflection.needs_regeneration:
                # Answer is satisfactory, stop
                break
            
            # Step 6: Refine query for next iteration
            if iteration < max_iterations - 1:
                current_query = await self._refine_query(
                    original_query=request.query,
                    answer=draft_answer,
                    reflection=reflection
                )
        
        # Use best answer found
        final_answer = best_answer or "I couldn't generate a satisfactory answer after multiple attempts."
        final_chunks = best_chunks or []
        
        # Extract citations and create grounding
        citations = self._extract_citations(final_chunks)
        grounding = self._create_placeholder_grounding()
        
        # Create SRAG result
        srag_result = SelfReflectiveResult(
            final_answer=final_answer,
            total_iterations=len(reflections),
            reflections=reflections,
            retrieved_chunks=final_chunks
        )
        
        response_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            query=request.query,
            answer=final_answer,
            mode="srag",
            search_mode=request.search_mode,
            sources=final_chunks,
            citations=citations,
            grounding=grounding,
            reflection_details=srag_result,
            response_time_ms=response_time,
            hyde_used=False,
            reranking_used=request.enable_reranking,
            compression_used=False,
            initial_retrieval_count=len(final_chunks),
            final_retrieval_count=len(final_chunks)
        )
    
    async def _reflect_on_answer(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
        iteration: int
    ) -> ReflectionResult:
        """
        Self-evaluate the generated answer.
        
        Args:
            query: Original query
            answer: Generated answer
            chunks: Source chunks used
            iteration: Current iteration number
            
        Returns:
            Reflection result with evaluation scores
        """
        # Format sources
        sources_text = "\n\n".join([
            f"[Source {i+1}]\n{chunk.content}"
            for i, chunk in enumerate(chunks)
        ])
        
        system_prompt = """You are an answer quality evaluator. Evaluate the generated answer for:
1. Grounding: Is the answer based on the provided sources?
2. Hallucination: Does the answer contain information not in the sources?
3. Completeness: Does the answer fully address the query?
4. Faithfulness: Is the answer faithful to the source information?

Return a JSON object:
{
  "answer_grounded": true/false,
  "hallucination_detected": true/false,
  "completeness_score": 0.0-1.0,
  "faithfulness_score": 0.0-1.0,
  "sources_cited": ["source1", "source2"],
  "needs_regeneration": true/false,
  "reflection_reason": "brief explanation"
}"""
        
        user_prompt = f"""Query: "{query}"

Sources:
{sources_text}

Generated Answer:
{answer}

Evaluate this answer. Return ONLY the JSON object."""
        
        try:
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=500
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
            
            answer_grounded = bool(result.get("answer_grounded", False))
            hallucination_detected = bool(result.get("hallucination_detected", True))
            completeness_score = float(result.get("completeness_score", 0.5))
            faithfulness_score = float(result.get("faithfulness_score", 0.5))
            sources_cited = result.get("sources_cited", [])
            needs_regeneration = bool(result.get("needs_regeneration", True))
            reflection_reason = result.get("reflection_reason", "Evaluation completed")
            
            # Clamp scores
            completeness_score = max(0.0, min(1.0, completeness_score))
            faithfulness_score = max(0.0, min(1.0, faithfulness_score))
            
            # Calculate overall reflection score
            # Higher is better: grounded + no hallucination + high completeness + high faithfulness
            reflection_score = (
                (1.0 if answer_grounded else 0.0) * 0.3 +
                (0.0 if hallucination_detected else 1.0) * 0.3 +
                completeness_score * 0.2 +
                faithfulness_score * 0.2
            )
            
            return ReflectionResult(
                answer_grounded=answer_grounded,
                hallucination_detected=hallucination_detected,
                completeness_score=completeness_score,
                faithfulness_score=faithfulness_score,
                sources_cited=sources_cited,
                reflection_score=reflection_score,
                needs_regeneration=needs_regeneration,
                reflection_reason=reflection_reason,
                iteration=iteration,
                reflected_at=datetime.now()
            )
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: assume needs regeneration
            return ReflectionResult(
                answer_grounded=False,
                hallucination_detected=True,
                completeness_score=0.5,
                faithfulness_score=0.5,
                sources_cited=[],
                reflection_score=0.3,
                needs_regeneration=True,
                reflection_reason="Evaluation failed, assuming regeneration needed",
                iteration=iteration,
                reflected_at=datetime.now()
            )
    
    async def _refine_query(
        self,
        original_query: str,
        answer: str,
        reflection: ReflectionResult
    ) -> str:
        """
        Refine query based on reflection feedback.
        
        Args:
            original_query: Original user query
            answer: Generated answer
            reflection: Reflection result
            
        Returns:
            Refined query
        """
        system_prompt = """You are a query refinement expert. Based on the reflection feedback, refine the query to improve retrieval in the next iteration.

Focus on:
- If hallucination detected: Make query more specific
- If low completeness: Broaden query scope
- If low faithfulness: Rephrase for better source matching

Return ONLY the refined query, no explanations."""
        
        user_prompt = f"""Original Query: "{original_query}"

Generated Answer: "{answer}"

Reflection Feedback:
- Grounded: {reflection.answer_grounded}
- Hallucination: {reflection.hallucination_detected}
- Completeness: {reflection.completeness_score}
- Faithfulness: {reflection.faithfulness_score}
- Reason: {reflection.reflection_reason}

Refine the query for better retrieval. Return ONLY the refined query."""
        
        try:
            refined = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=100
            )
            return refined.strip().strip('"')
        except:
            return original_query
