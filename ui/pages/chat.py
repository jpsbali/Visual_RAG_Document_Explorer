"""
Chat Interface page for the Streamlit application.

Main conversational interface for document Q&A with streaming support.
"""

import streamlit as st
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from agents.orchestrator import AgentOrchestrator
from config.models import QueryRequest, QueryResponse, Citation
from config.settings import Settings
from ui.components import (
    render_sidebar,
    render_citation_list,
    render_retrieved_chunks_as_citations,
    get_relevance_color,
    get_relevance_emoji
)


def initialize_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "orchestrator" not in st.session_state:
        try:
            settings = Settings()
            st.session_state.orchestrator = AgentOrchestrator(settings)
            st.session_state.orchestrator_error = None
        except Exception as e:
            st.session_state.orchestrator = None
            st.session_state.orchestrator_error = str(e)
    
    # Query configuration defaults
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "hybrid"
    if "enable_hyde" not in st.session_state:
        st.session_state.enable_hyde = False
    if "enable_reranking" not in st.session_state:
        st.session_state.enable_reranking = True
    if "enable_compression" not in st.session_state:
        st.session_state.enable_compression = True
    if "rag_strategy" not in st.session_state:
        st.session_state.rag_strategy = "auto"
    if "crag_threshold" not in st.session_state:
        st.session_state.crag_threshold = 0.5
    if "srag_max_iterations" not in st.session_state:
        st.session_state.srag_max_iterations = 3
    if "enable_short_term_memory" not in st.session_state:
        st.session_state.enable_short_term_memory = True
    if "short_term_memory_size" not in st.session_state:
        st.session_state.short_term_memory_size = 5
    if "enable_long_term_memory" not in st.session_state:
        st.session_state.enable_long_term_memory = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


def render_query_config_sidebar():
    """Render the query configuration sidebar."""
    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Query Configuration")
        
        # Search Settings
        with st.expander("🔍 Search Settings", expanded=True):
            st.session_state.top_k = st.slider(
                "Top K Results",
                min_value=1,
                max_value=20,
                value=st.session_state.top_k,
                help="Number of chunks to retrieve"
            )
            
            st.session_state.search_mode = st.selectbox(
                "Search Mode",
                options=["dense", "sparse", "hybrid"],
                index=["dense", "sparse", "hybrid"].index(st.session_state.search_mode),
                help="dense: semantic, sparse: keyword BM25, hybrid: RRF fusion"
            )
            
            st.session_state.enable_hyde = st.checkbox(
                "Enable HyDE",
                value=st.session_state.enable_hyde,
                help="Hypothetical Document Embeddings for query expansion"
            )
            
            st.session_state.enable_reranking = st.checkbox(
                "Enable Reranking",
                value=st.session_state.enable_reranking,
                help="Use Cohere cross-encoder reranking"
            )
            
            st.session_state.enable_compression = st.checkbox(
                "Enable Compression",
                value=st.session_state.enable_compression,
                help="Extract relevant portions from chunks"
            )
        
        # RAG Strategy
        with st.expander("🧠 RAG Strategy", expanded=True):
            st.session_state.rag_strategy = st.selectbox(
                "Strategy",
                options=["auto", "simple", "crag", "srag", "advanced"],
                index=["auto", "simple", "crag", "srag", "advanced"].index(st.session_state.rag_strategy),
                help="auto: agent routing, simple: basic RAG, crag: corrective, srag: self-reflective, advanced: multi-doc synthesis"
            )
            
            # CRAG-specific settings
            if st.session_state.rag_strategy in ["crag", "auto"]:
                st.session_state.crag_threshold = st.slider(
                    "CRAG Relevance Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.crag_threshold,
                    step=0.05,
                    help="Minimum relevance score for CRAG"
                )
            
            # SRAG-specific settings
            if st.session_state.rag_strategy in ["srag", "auto"]:
                st.session_state.srag_max_iterations = st.slider(
                    "SRAG Max Iterations",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.srag_max_iterations,
                    help="Maximum self-reflection iterations"
                )
        
        # Memory Settings
        with st.expander("💾 Memory Settings", expanded=False):
            st.session_state.enable_short_term_memory = st.checkbox(
                "Enable Short-Term Memory",
                value=st.session_state.enable_short_term_memory,
                help="Keep recent conversation context"
            )
            
            if st.session_state.enable_short_term_memory:
                st.session_state.short_term_memory_size = st.slider(
                    "Short-Term Memory Size",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.short_term_memory_size,
                    help="Number of recent messages to remember"
                )
            
            st.session_state.enable_long_term_memory = st.checkbox(
                "Enable Long-Term Memory",
                value=st.session_state.enable_long_term_memory,
                help="Store and retrieve from conversation history"
            )


def get_strategy_badge(mode: str) -> str:
    """Get a colored badge for the RAG strategy."""
    badges = {
        "simple": "🟢 Simple RAG",
        "crag": "🟡 CRAG",
        "srag": "🟠 SRAG",
        "advanced": "🔵 Advanced",
        "auto": "🟣 Auto"
    }
    return badges.get(mode, f"⚪ {mode.upper()}")


def get_grounding_badge(score: float) -> str:
    """Get a colored badge for grounding score."""
    emoji = get_relevance_emoji(score)
    color = get_relevance_color(score)
    return f"{emoji} Grounding: {score:.2f}"


def render_message_metadata(message: Dict[str, Any]):
    """Render metadata for an assistant message."""
    if "metadata" not in message:
        return
    
    metadata = message["metadata"]
    
    # Create columns for metadata badges
    cols = st.columns([2, 2, 2, 2])
    
    with cols[0]:
        if "mode" in metadata:
            st.caption(get_strategy_badge(metadata["mode"]))
    
    with cols[1]:
        if "grounding_score" in metadata:
            score = metadata["grounding_score"]
            st.caption(get_grounding_badge(score))
    
    with cols[2]:
        if "response_time_ms" in metadata:
            time_ms = metadata["response_time_ms"]
            st.caption(f"⏱️ {time_ms:.0f}ms")
    
    with cols[3]:
        if "sources_count" in metadata:
            count = metadata["sources_count"]
            st.caption(f"📚 {count} sources")


def export_chat_history():
    """Export chat history as JSON."""
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return
    
    export_data = {
        "session_id": st.session_state.session_id,
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.chat_history
    }
    
    json_str = json.dumps(export_data, indent=2, default=str)
    
    st.download_button(
        label="📥 Download Chat History",
        data=json_str,
        file_name=f"chat_history_{st.session_state.session_id}.json",
        mime="application/json",
        use_container_width=True
    )


def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.rerun()


async def execute_query_async(query: str) -> Optional[QueryResponse]:
    """
    Execute query asynchronously with streaming.
    
    Args:
        query: User query string
    
    Returns:
        QueryResponse or None if error
    """
    if st.session_state.orchestrator is None:
        st.error("Orchestrator not initialized. Please check your configuration.")
        return None
    
    # Create query request
    query_request = QueryRequest(
        query=query,
        mode=st.session_state.rag_strategy,
        search_mode=st.session_state.search_mode,
        top_k=st.session_state.top_k,
        enable_hyde=st.session_state.enable_hyde,
        enable_reranking=st.session_state.enable_reranking,
        enable_compression=st.session_state.enable_compression
    )
    
    # Create placeholders for streaming
    response_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Track streaming state
        current_answer = ""
        final_state = None
        nodes_executed = []
        
        # Stream execution
        status_placeholder.info("🔄 Processing query...")
        
        async for event in st.session_state.orchestrator.execute_stream(
            query_request,
            session_id=st.session_state.session_id
        ):
            if "error" in event:
                status_placeholder.error(f"Error: {event['error']}")
                return None
            
            node_name = event.get("node", "unknown")
            nodes_executed.append(node_name)
            update = event.get("update", {})
            
            # Update status
            status_placeholder.info(f"🔄 {node_name}...")
            
            # Check for partial answer
            if "final_answer" in update:
                current_answer = update["final_answer"]
                response_placeholder.markdown(current_answer)
            
            # Store final state
            final_state = update
        
        # Clear status
        status_placeholder.empty()
        
        # Execute full query to get complete response
        # (streaming gives us updates, but we need the full response object)
        response = await st.session_state.orchestrator.execute(
            query_request,
            session_id=st.session_state.session_id
        )
        
        return response
    
    except Exception as e:
        status_placeholder.error(f"Error executing query: {str(e)}")
        return None


def render_chat_page():
    """Main chat page render function."""
    # Initialize session state
    initialize_session_state()
    
    # Get system status for sidebar
    system_status = {
        "vector_db_connected": st.session_state.orchestrator is not None,
        "vector_db_type": "qdrant",  # Could be dynamic
        "llm_provider": "openai",
        "llm_model": "gpt-4"
    }
    
    stats = {
        "documents_indexed": 0,  # Could query from vector DB
        "total_chunks": 0,
        "total_entities": 0
    }
    
    # Render main sidebar
    render_sidebar(
        current_page="chat",
        system_status=system_status,
        stats=stats
    )
    
    # Render query configuration sidebar
    render_query_config_sidebar()
    
    # Main chat area
    st.title("💬 Document Q&A Chat")
    st.markdown("Ask questions about your documents and get AI-powered answers with citations.")
    
    # Check for orchestrator errors
    if st.session_state.orchestrator_error:
        st.error(f"⚠️ Failed to initialize orchestrator: {st.session_state.orchestrator_error}")
        st.info("Please check your configuration in Settings and ensure all required API keys are set.")
        return
    
    # Chat controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.caption(f"Session ID: `{st.session_state.session_id}`")
    
    with col2:
        if st.button("🗑️ New Chat", use_container_width=True):
            clear_chat_history()
    
    with col3:
        if st.session_state.chat_history:
            export_chat_history()
    
    st.divider()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Render metadata for assistant messages
            if message["role"] == "assistant":
                render_message_metadata(message)
                
                # Render citations if available
                if "citations" in message and message["citations"]:
                    st.divider()
                    render_citation_list(
                        message["citations"],
                        relevance_scores=message.get("relevance_scores"),
                        max_display=5
                    )
                elif "sources" in message and message["sources"]:
                    st.divider()
                    render_retrieved_chunks_as_citations(
                        message["sources"],
                        max_display=5
                    )
    
    # User input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Validate input
        if not prompt.strip():
            st.warning("Please enter a valid question")
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Execute query and display response
        with st.chat_message("assistant"):
            # Run async query
            response = asyncio.run(execute_query_async(prompt))
            
            if response:
                # Display answer
                st.markdown(response.answer)
                
                # Prepare metadata
                metadata = {
                    "mode": response.mode,
                    "response_time_ms": response.response_time_ms,
                    "sources_count": len(response.sources)
                }
                
                if response.grounding:
                    metadata["grounding_score"] = response.grounding.grounding_score
                
                # Render metadata
                render_message_metadata({"metadata": metadata})
                
                # Prepare relevance scores
                relevance_scores = [chunk.score for chunk in response.sources] if response.sources else None
                
                # Display citations
                if response.citations:
                    st.divider()
                    render_citation_list(
                        response.citations,
                        relevance_scores=relevance_scores,
                        max_display=5
                    )
                elif response.sources:
                    st.divider()
                    render_retrieved_chunks_as_citations(
                        response.sources,
                        max_display=5
                    )
                
                # Add assistant message to chat history
                assistant_message = {
                    "role": "assistant",
                    "content": response.answer,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata
                }
                
                if response.citations:
                    assistant_message["citations"] = response.citations
                    assistant_message["relevance_scores"] = relevance_scores
                elif response.sources:
                    assistant_message["sources"] = response.sources
                
                st.session_state.chat_history.append(assistant_message)
            else:
                error_message = "Failed to generate response. Please try again."
                st.error(error_message)
                
                # Add error to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_message,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"error": True}
                })


# Entry point
if __name__ == "__main__":
    render_chat_page()
