"""
Sidebar navigation component.

Provides navigation between pages and displays system status.
"""

import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime


def render_sidebar(
    current_page: str,
    system_status: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, int]] = None
) -> str:
    """
    Render the sidebar with navigation, system status, and quick stats.
    
    Args:
        current_page: The currently active page name
        system_status: Dictionary containing system status info:
            - vector_db_connected: bool
            - vector_db_type: str (e.g., "qdrant", "milvus")
            - llm_provider: str (e.g., "openai", "openrouter")
            - llm_model: str
        stats: Dictionary containing quick stats:
            - documents_indexed: int
            - total_chunks: int
            - total_entities: int (optional)
    
    Returns:
        str: The selected page name
    """
    with st.sidebar:
        # Project header
        st.title("📚 Visual RAG")
        st.markdown("*Document Explorer*")
        st.divider()
        
        # Navigation
        st.subheader("Navigation")
        
        pages = {
            "💬 Chat": "chat",
            "📤 Upload": "upload",
            "🔍 Explorer": "explorer",
            "⚙️ Settings": "settings",
            "📊 Benchmark": "benchmark"
        }
        
        selected_page = current_page
        for label, page_id in pages.items():
            if st.button(
                label,
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if page_id == current_page else "secondary"
            ):
                selected_page = page_id
        
        st.divider()
        
        # System Status
        st.subheader("System Status")
        
        if system_status:
            # Vector DB status
            db_connected = system_status.get("vector_db_connected", False)
            db_type = system_status.get("vector_db_type", "unknown")
            
            if db_connected:
                st.success(f"✅ Vector DB: {db_type.capitalize()}")
            else:
                st.error(f"❌ Vector DB: Disconnected")
            
            # LLM provider status
            llm_provider = system_status.get("llm_provider", "unknown")
            llm_model = system_status.get("llm_model", "")
            
            if llm_provider != "unknown":
                st.info(f"🤖 LLM: {llm_provider.capitalize()}")
                if llm_model:
                    st.caption(f"Model: {llm_model}")
            else:
                st.warning("⚠️ LLM: Not configured")
        else:
            st.info("System status unavailable")
        
        st.divider()
        
        # Quick Stats
        st.subheader("Quick Stats")
        
        if stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Documents",
                    stats.get("documents_indexed", 0)
                )
            
            with col2:
                st.metric(
                    "Chunks",
                    stats.get("total_chunks", 0)
                )
            
            if "total_entities" in stats:
                st.metric(
                    "Entities Extracted",
                    stats.get("total_entities", 0)
                )
        else:
            st.info("No documents indexed yet")
        
        st.divider()
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Visual RAG Document Explorer**
            
            A production-grade RAG system with:
            - 🔍 Multi-modal search (dense, sparse, hybrid)
            - 🧠 Advanced RAG strategies (CRAG, SRAG)
            - 🎯 Entity extraction & filtering
            - 📊 Vector DB benchmarking
            - 🤖 Agentic orchestration
            
            **Version:** 1.0.0
            """)
            
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    return selected_page
