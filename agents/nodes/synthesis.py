"""
Synthesis Node - Synthesize information across multiple documents.

This node performs multi-document synthesis with contradiction detection.
"""

from typing import Any
from agents.state import AgentState
import logging
import json
import re

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str:
    """Extract JSON from text."""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text


async def synthesis_node(state: AgentState) -> dict[str, Any]:
    """
    Synthesize information across multiple documents.
    
    Steps:
    1. Group chunks by source document
    2. Generate per-document summaries
    3. Detect contradictions
    4. Create synthesis context for generation
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with synthesis results
    """
    try:
        # Skip if not multi-doc
        if not state.get("is_multi_doc", False):
            return {}
        
        from core.llm.llm_router import get_llm_provider
        from config.settings import Settings
        from config.models import SynthesisResult, Contradiction
        
        settings = state.get("_settings") or Settings()
        llm = get_llm_provider(settings)
        
        # 1. Group chunks by source document
        document_groups = {}
        for chunk in state.get("compressed_docs", []):
            source = chunk.metadata.source_file
            if source not in document_groups:
                document_groups[source] = []
            document_groups[source].append(chunk)
        
        # 2. Generate per-document summaries
        document_summaries = {}
        for source, chunks in document_groups.items():
            combined_content = "\n\n".join(c.content for c in chunks[:5])
            
            system_prompt = f"""Summarize the key information from {source} relevant to the query."""
            user_prompt = f"""Query: {state['query']}

Content from {source}:
{combined_content}

Provide a concise summary of the key points."""
            
            summary = await llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=300
            )
            
            document_summaries[source] = summary
        
        # 3. Detect contradictions
        contradictions = []
        if len(document_summaries) > 1:
            system_prompt = """Identify any contradictions between the document summaries.

Respond in JSON format:
{
  "contradictions": [
    {
      "topic": "what the contradiction is about",
      "document_1": "source file 1",
      "claim_1": "what document 1 says",
      "document_2": "source file 2",
      "claim_2": "what document 2 says"
    }
  ]
}"""
            
            summaries_text = "\n\n".join(
                f"**{source}**: {summary}"
                for source, summary in document_summaries.items()
            )
            
            user_prompt = f"""Query: {state['query']}

Document Summaries:
{summaries_text}

Identify contradictions."""
            
            response = await llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=500
            )
            
            try:
                result = json.loads(extract_json(response))
                contradictions_data = result.get("contradictions", [])
                contradictions = [Contradiction(**c) for c in contradictions_data]
            except:
                contradictions = []
        
        # 4. Create synthesis result
        synthesis_result = SynthesisResult(
            query=state["query"],
            num_documents=len(document_groups),
            document_summaries=document_summaries,
            contradictions_found=len(contradictions) > 0,
            contradictions=contradictions
        )
        
        logger.info(f"Synthesized {len(document_groups)} documents, found {len(contradictions)} contradictions")
        
        return {
            "document_groups": document_groups,
            "document_summaries": document_summaries,
            "contradictions": [c.dict() for c in contradictions],
            "synthesis_result": synthesis_result
        }
    
    except Exception as e:
        logger.error(f"Error in synthesis_node: {e}")
        return {
            "document_groups": {},
            "document_summaries": {},
            "contradictions": [],
            "synthesis_result": None
        }
