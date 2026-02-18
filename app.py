"""
Visual RAG Document Explorer - Streamlit Entry Point

Main application entry point for the Streamlit UI.
Provides multi-page navigation for chat, upload, explorer, settings, and benchmark.
"""

import streamlit as st
from pathlib import Path
from config.settings import Settings

# Import page render functions
from ui.pages.chat import render_chat_page
from ui.pages.upload import render_upload_page

# Import page modules for explorer, settings, and benchmark
import ui.pages.explorer as explorer_module
import ui.pages.settings as settings_module
import ui.pages.benchmark as benchmark_module

# Page configuration
st.set_page_config(
    page_title="Visual RAG Document Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS styling for the application."""
    css_file = Path("ui/styles/custom.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom CSS file not found. Using default Streamlit styling.")


# Wrapper functions for pages that have main() instead of render_*_page()
# These wrappers temporarily disable set_page_config to avoid conflicts
def render_explorer_page():
    """Wrapper for explorer page that skips set_page_config."""
    # Temporarily replace set_page_config with a no-op
    original_set_page_config = st.set_page_config
    st.set_page_config = lambda **kwargs: None
    try:
        explorer_module.main()
    finally:
        # Restore original function
        st.set_page_config = original_set_page_config


def render_settings_page():
    """Wrapper for settings page that skips set_page_config."""
    original_set_page_config = st.set_page_config
    st.set_page_config = lambda **kwargs: None
    try:
        settings_module.main()
    finally:
        st.set_page_config = original_set_page_config


def render_benchmark_page():
    """Wrapper for benchmark page that skips set_page_config."""
    original_set_page_config = st.set_page_config
    st.set_page_config = lambda **kwargs: None
    try:
        benchmark_module.main()
    finally:
        st.set_page_config = original_set_page_config


# Apply custom styling
load_css()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "settings" not in st.session_state:
    st.session_state.settings = Settings()

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

# Sidebar navigation
st.sidebar.title("📚 Visual RAG Explorer")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["💬 Chat", "📤 Upload", "🔍 Explorer", "⚙️ Settings", "📊 Benchmark"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Advanced RAG document exploration system with multi-strategy retrieval, "
    "NER-based metadata extraction, and dual vector database support."
)

# Page routing - render the appropriate page based on navigation
if page == "💬 Chat":
    render_chat_page()

elif page == "📤 Upload":
    render_upload_page()

elif page == "🔍 Explorer":
    render_explorer_page()

elif page == "⚙️ Settings":
    render_settings_page()

elif page == "📊 Benchmark":
    render_benchmark_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Status:** Phase 5 Complete ✅")
st.sidebar.markdown("**All Features:** Fully Operational")
