"""
Grounding Verifier for Visual RAG Document Explorer.

Verifies that generated answers are grounded in source documents by extracting
and verifying individual claims.
"""

import json
from datetime import datetime
from typing import Optional

from config.settings import Settings
from config.models import (
    RetrievedChunk,
    GroundingResult,
    ClaimVerification
)
from core.llm.llm_router import get_llm_provider


class GroundingVerifier:
    """
    Grounding Verifier for validating answer faithfulness to sources.
    
    Extracts individual claims from the generated answer and verifies each claim
    against the source chunks. Provides three-level grounding assessment:
    grounded, partially_grounded, and ungrounded.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize grounding verifier.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.llm = get_llm_provider(settings)
    
    async def extract_claims(self, answer: str) -> list[str]:
        """
        Extract individual factual claims from an answer.
        
        Args:
            answer: Generated answer text
            
        Returns:
            List of individual claims
        """
        system_prompt = """You are a claim extraction expert. Break down the given answer into individual, atomic factual claims.

Rules:
1. Each claim should be a single, verifiable statement
2. Claims should be self-contained (include necessary context)
3. Don't include opinions or subjective statements
4. Return ONLY a JSON array of claim strings
5. If no factual claims exist, return an empty array []

Example:
Answer: "Apple was founded in 1976 by Steve Jobs and Steve Wozniak. The company is headquartered in Cupertino, California."
Output: [
  "Apple was founded in 1976",
  "Apple was founded by Steve Jobs and Steve Wozniak",
  "Apple is headquartered in Cupertino, California"
]"""
        
        user_prompt = f"""Answer: "{answer}"

Extract all factual claims as a JSON array. Return ONLY the JSON array."""
        
        try:
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=1000
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
            
            claims = json.loads(response)
            
            if not isinstance(claims, list):
                return []
            
            return [str(c).strip() for c in claims if c]
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: split by sentences
            sentences = answer.split(". ")
            return [s.strip() + "." for s in sentences if s.strip()]
    
    async def verify_claim(
        self,
        claim: str,
        chunks: list[RetrievedChunk]
    ) -> ClaimVerification:
        """
        Verify a single claim against source chunks.
        
        Args:
            claim: Claim to verify
            chunks: Source chunks to verify against
            
        Returns:
            Claim verification result
        """
        # Format sources for verification
        sources_text = "\n\n".join([
            f"[Source {i+1} - {chunk.metadata.chunk_id}]\n{chunk.content}"
            for i, chunk in enumerate(chunks)
        ])
        
        system_prompt = """You are a fact verification expert. Verify if the given claim is supported by the provided sources.

Rules:
1. Return "grounded" if the claim is fully supported by the sources
2. Return "partially_grounded" if the claim is partially supported or implied
3. Return "ungrounded" if the claim is not supported by the sources
4. List the source IDs (chunk_ids) that support the claim
5. Provide a confidence score (0.0-1.0)

Return ONLY a JSON object with this format:
{
  "status": "grounded|partially_grounded|ungrounded",
  "supporting_chunks": ["chunk_id1", "chunk_id2"],
  "confidence": 0.95
}"""
        
        user_prompt = f"""Sources:
{sources_text}

Claim: "{claim}"

Verify this claim against the sources. Return ONLY the JSON object."""
        
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
            
            status = result.get("status", "ungrounded")
            supporting_chunks = result.get("supporting_chunks", [])
            confidence = float(result.get("confidence", 0.0))
            
            # Validate status
            if status not in ["grounded", "partially_grounded", "ungrounded"]:
                status = "ungrounded"
            
            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))
            
            return ClaimVerification(
                claim=claim,
                is_grounded=(status == "grounded"),
                supporting_chunks=supporting_chunks,
                confidence=confidence,
                status=status
            )
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: mark as ungrounded
            return ClaimVerification(
                claim=claim,
                is_grounded=False,
                supporting_chunks=[],
                confidence=0.0,
                status="ungrounded"
            )
    
    async def verify_answer(
        self,
        answer: str,
        chunks: list[RetrievedChunk]
    ) -> GroundingResult:
        """
        Verify that an answer is grounded in source chunks.
        
        Args:
            answer: Generated answer to verify
            chunks: Source chunks used to generate the answer
            
        Returns:
            Complete grounding verification result
        """
        # Extract claims from answer
        claims = await self.extract_claims(answer)
        
        if not claims:
            # No claims to verify
            return GroundingResult(
                grounding_score=1.0,
                total_claims=0,
                grounded_claims=0,
                partially_grounded_claims=0,
                ungrounded_claims=0,
                claim_details=[],
                verified_at=datetime.now()
            )
        
        # Verify each claim
        claim_verifications = []
        for claim in claims:
            verification = await self.verify_claim(claim, chunks)
            claim_verifications.append(verification)
        
        # Calculate statistics
        total_claims = len(claim_verifications)
        grounded_claims = sum(1 for v in claim_verifications if v.status == "grounded")
        partially_grounded_claims = sum(1 for v in claim_verifications if v.status == "partially_grounded")
        ungrounded_claims = sum(1 for v in claim_verifications if v.status == "ungrounded")
        
        # Calculate overall grounding score
        # Grounded = 1.0, Partially = 0.5, Ungrounded = 0.0
        score_sum = sum(
            1.0 if v.status == "grounded" else 0.5 if v.status == "partially_grounded" else 0.0
            for v in claim_verifications
        )
        grounding_score = score_sum / total_claims if total_claims > 0 else 0.0
        
        return GroundingResult(
            grounding_score=grounding_score,
            total_claims=total_claims,
            grounded_claims=grounded_claims,
            partially_grounded_claims=partially_grounded_claims,
            ungrounded_claims=ungrounded_claims,
            claim_details=claim_verifications,
            verified_at=datetime.now()
        )
    
    def is_answer_trustworthy(
        self,
        grounding_result: GroundingResult,
        threshold: float = 0.7
    ) -> bool:
        """
        Determine if an answer is trustworthy based on grounding score.
        
        Args:
            grounding_result: Grounding verification result
            threshold: Minimum grounding score for trustworthiness
            
        Returns:
            True if answer is trustworthy, False otherwise
        """
        return grounding_result.grounding_score >= threshold
