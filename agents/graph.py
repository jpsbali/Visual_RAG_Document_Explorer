"""
LangGraph state graph definition and compilation.

Defines the agent workflow graph with all nodes and edges.
"""

from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (
    memory_load_node,
    routing_node,
    decomposition_node,
    retrieval_node,
    grading_node,
    correction_node,
    reranking_node,
    compression_node,
    synthesis_node,
    generation_node,
    reflection_node,
    grounding_node,
    citation_node,
    memory_save_node,
    format_node
)


def create_agent_graph() -> StateGraph:
    """
    Create and compile the LangGraph agent workflow.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph
    graph = StateGraph(AgentState)
    
    # ============= Add Nodes =============
    graph.add_node("memory_load", memory_load_node)
    graph.add_node("routing", routing_node)
    graph.add_node("decomposition", decomposition_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("grading", grading_node)
    graph.add_node("correction", correction_node)
    graph.add_node("reranking", reranking_node)
    graph.add_node("compression", compression_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("generation", generation_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("grounding", grounding_node)
    graph.add_node("citation", citation_node)
    graph.add_node("memory_save", memory_save_node)
    graph.add_node("format", format_node)
    
    # ============= Entry Point =============
    graph.set_entry_point("memory_load")
    
    # ============= Linear Edges =============
    graph.add_edge("memory_load", "routing")
    
    # ============= Conditional: Decomposition =============
    def should_decompose(state: AgentState) -> str:
        """Decide if query needs decomposition."""
        if state.get("needs_decomposition", False):
            return "decomposition"
        return "retrieval"
    
    graph.add_conditional_edges(
        "routing",
        should_decompose,
        {
            "decomposition": "decomposition",
            "retrieval": "retrieval"
        }
    )
    
    graph.add_edge("decomposition", "retrieval")
    
    # ============= Conditional: CRAG Grading =============
    def should_grade(state: AgentState) -> str:
        """Decide if CRAG grading is needed."""
        if state.get("rag_strategy") == "crag":
            return "grading"
        return "reranking"
    
    graph.add_conditional_edges(
        "retrieval",
        should_grade,
        {
            "grading": "grading",
            "reranking": "reranking"
        }
    )
    
    # ============= Conditional: CRAG Correction =============
    def should_correct(state: AgentState) -> str:
        """Decide if CRAG correction is needed."""
        if state.get("needs_correction", False):
            return "correction"
        return "reranking"
    
    graph.add_conditional_edges(
        "grading",
        should_correct,
        {
            "correction": "correction",
            "reranking": "reranking"
        }
    )
    
    graph.add_edge("correction", "reranking")
    
    # ============= Conditional: Multi-Doc Synthesis =============
    def should_synthesize(state: AgentState) -> str:
        """Decide if multi-doc synthesis is needed."""
        if state.get("is_multi_doc", False):
            return "synthesis"
        return "compression"
    
    graph.add_conditional_edges(
        "reranking",
        should_synthesize,
        {
            "synthesis": "synthesis",
            "compression": "compression"
        }
    )
    
    graph.add_edge("synthesis", "compression")
    graph.add_edge("compression", "generation")
    
    # ============= Conditional: SRAG Reflection =============
    def should_reflect(state: AgentState) -> str:
        """Decide if SRAG reflection is needed."""
        if state.get("rag_strategy") == "srag":
            return "reflection"
        return "grounding"
    
    graph.add_conditional_edges(
        "generation",
        should_reflect,
        {
            "reflection": "reflection",
            "grounding": "grounding"
        }
    )
    
    # ============= Conditional: SRAG Iteration =============
    def should_iterate(state: AgentState) -> str:
        """Decide if SRAG should iterate."""
        if (state.get("needs_reflection", False) and
            state.get("iteration_count", 0) < state.get("max_iterations", 3)):
            return "retrieval"  # Loop back to retrieval
        return "grounding"
    
    graph.add_conditional_edges(
        "reflection",
        should_iterate,
        {
            "retrieval": "retrieval",
            "grounding": "grounding"
        }
    )
    
    # ============= Final Path =============
    graph.add_edge("grounding", "citation")
    graph.add_edge("citation", "memory_save")
    graph.add_edge("memory_save", "format")
    graph.add_edge("format", END)
    
    # ============= Compile =============
    return graph.compile()


# Singleton instance
_compiled_graph = None


def get_agent_graph() -> StateGraph:
    """
    Get the compiled agent graph (singleton).
    
    Returns:
        Compiled StateGraph
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_agent_graph()
    return _compiled_graph
