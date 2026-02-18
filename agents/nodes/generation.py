"""
Generation Node - Generate answer using LLM.

This node generates the final answer using the LLM with retrieved context.
"""

from typing import Any
from agents.state import AgentState
import logging

logger = logging.getLogger(__name__)


def format_dict(d: dict) -> str:
    """Format dictionary for display."""
    return "\n".join(f"- {k}: {v}" for k, v in d.items())


def format_contradictions(contradictions: list) -> str:
    """Format contradictions for display."""
    if not contradictions:
        return "None"
    
    formatted = []
    for c in contradictions:
        formatted.append(f"- {c.get('topic', 'Unknown')}: {c.get('document_1', '')} vs {c.get('document_2', '')}")
    return "\n".join(formatted)


async def generation_node(state: AgentState) -> dict[str, Any]:
    """
    Generate answer using LLM.
    
    Constructs prompt with:
    - Query
    - Retrieved sources
    - Chat history
    - Synthesis context (if multi-doc)
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with draft_answer
    """
    try:
        from core.llm.llm_router import get_llm_provider
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        llm = get_llm_provider(settings)
        
        # Format sources
        sources_text = "\n\n".join(
            f"[Source {i+1}: {chunk.metadata.source_file}, Page {chunk.metadata.page_number}]\n{chunk.content}"
            for i, chunk in enumerate(state.get("compressed_docs", []))
        )
        
        # Format chat history
        history_text = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in state.get("chat_history", [])[-5:]
        )
        
        # Add synthesis context if multi-doc
        synthesis_context = ""
        if state.get("synthesis_result"):
            synth = state["synthesis_result"]
            synthesis_context = f"""
Multi-Document Synthesis:
- {synth.num_documents} documents analyzed
- Document summaries:
{format_dict(synth.document_summaries)}

"""
            if synth.contradictions:
                synthesis_context += f"""
Contradictions detected:
{format_contradictions(synth.contradictions)}
"""
        
        # System prompt
        system_prompt = """You are a helpful document assistant. Answer the user query based ONLY on the provided source documents.

Rules:
1. Only use information from the provided sources
2. If sources don't contain enough information, say so explicitly
3. Cite your sources using [Source: filename, Page: X] format
4. Be precise and factual
5. If sources contradict each other, note the contradiction with both sources cited
6. Do not make up information or use external knowledge"""
        
        # User prompt
        user_prompt = f"""Query: {state['query']}

{synthesis_context}

Source Documents:
{sources_text}

Previous conversation context:
{history_text}

Provide a comprehensive answer with citations."""
        
        # Generate answer
        answer = await llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=2048
        )
        
        generation_metadata = {
            "model": settings.default_model,
            "temperature": 0.0,
            "max_tokens": 2048,
            "num_sources": len(state.get("compressed_docs", []))
        }
        
        logger.info(f"Generated answer ({len(answer)} chars)")
        
        return {
            "draft_answer": answer,
            "generation_metadata": generation_metadata
        }
    
    except Exception as e:
        logger.error(f"Error in generation_node: {e}")
        return {
            "draft_answer": f"Error generating answer: {str(e)}",
            "generation_metadata": {"error": str(e)}
        }
