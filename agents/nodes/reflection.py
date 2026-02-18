"""
Reflection Node - Self-evaluate answer quality (SRAG).

This node evaluates the generated answer for hallucination,
completeness, and faithfulness, deciding if refinement is needed.
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


async def reflection_node(state: AgentState) -> dict[str, Any]:
    """
    Self-evaluate answer quality (SRAG).
    
    Checks for:
    - Hallucination
    - Completeness
    - Faithfulness
    
    Decides whether to iterate or proceed.
    
    Args:
        state: Current agent state
        
    Returns:
        Partial state update with reflection results
    """
    try:
        # Only run for SRAG strategy
        if state.get("rag_strategy") != "srag":
            return {"needs_reflection": False}
        
        from core.llm.llm_router import get_llm_provider
        from config.settings import Settings
        
        settings = state.get("_settings") or Settings()
        llm = get_llm_provider(settings)
        
        # Format sources
        sources_text = "\n\n".join(
            f"[Source {i+1}]\n{chunk.content}"
            for i, chunk in enumerate(state.get("compressed_docs", []))
        )
        
        # Reflection prompt
        system_prompt = """You are a quality evaluator for AI-generated answers.

Evaluate the following answer against the source documents for:

1. **Hallucination**: Does the answer contain claims not supported by the sources?
2. **Completeness**: Does the answer address all aspects of the query?
3. **Faithfulness**: Does the answer accurately represent the source content?

Respond in JSON format:
{
  "hallucination_detected": true|false,
  "hallucination_details": "specific claims not in sources, or empty",
  "completeness_score": 0.0-1.0,
  "faithfulness_score": 0.0-1.0,
  "needs_regeneration": true|false,
  "feedback": "specific suggestions for improvement",
  "overall_score": 0.0-1.0
}"""
        
        user_prompt = f"""Query: {state['query']}

Generated Answer:
{state.get('draft_answer', '')}

Source Documents:
{sources_text}

Evaluate the answer."""
        
        response = await llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=800
        )
        
        try:
            reflection = json.loads(extract_json(response))
        except:
            reflection = {
                "hallucination_detected": False,
                "completeness_score": 0.8,
                "faithfulness_score": 0.8,
                "needs_regeneration": False,
                "feedback": "Parsing failed, accepting answer",
                "overall_score": 0.8
            }
        
        # Add to reflection history
        reflection_history = state.get("reflection_history", [])
        reflection_history.append({
            "iteration": state.get("iteration_count", 0),
            "reflection": reflection
        })
        
        # Decide whether to iterate
        current_iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        needs_reflection = (
            reflection.get("needs_regeneration", False) and
            current_iteration < max_iterations and
            reflection.get("overall_score", 1.0) < settings.srag_quality_threshold
        )
        
        logger.info(f"Reflection iteration {current_iteration}: score={reflection.get('overall_score', 0):.2f}, needs_reflection={needs_reflection}")
        
        return {
            "needs_reflection": needs_reflection,
            "reflection_feedback": reflection.get("feedback", ""),
            "reflection_history": reflection_history,
            "iteration_count": current_iteration + 1 if needs_reflection else current_iteration
        }
    
    except Exception as e:
        logger.error(f"Error in reflection_node: {e}")
        return {
            "needs_reflection": False,
            "reflection_feedback": "",
            "reflection_history": state.get("reflection_history", []),
            "iteration_count": state.get("iteration_count", 0)
        }
