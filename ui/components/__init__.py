"""
Shared UI components for the Streamlit application.

This module provides reusable components for building the Visual RAG UI:
- sidebar: Navigation and system status
- document_card: Document display cards
- citation_viewer: Citation display with expandable details
- chunk_inspector: Detailed chunk visualization
"""

from ui.components.sidebar import render_sidebar
from ui.components.document_card import (
    render_document_card,
    render_document_grid,
    get_file_icon,
    format_file_size
)
from ui.components.citation_viewer import (
    render_citation,
    render_citation_list,
    render_retrieved_chunks_as_citations,
    render_citation_summary,
    get_relevance_color,
    get_relevance_emoji
)
from ui.components.chunk_inspector import (
    render_chunk_inspector,
    render_retrieved_chunk,
    render_chunk_comparison
)

__all__ = [
    # Sidebar
    "render_sidebar",
    
    # Document Card
    "render_document_card",
    "render_document_grid",
    "get_file_icon",
    "format_file_size",
    
    # Citation Viewer
    "render_citation",
    "render_citation_list",
    "render_retrieved_chunks_as_citations",
    "render_citation_summary",
    "get_relevance_color",
    "get_relevance_emoji",
    
    # Chunk Inspector
    "render_chunk_inspector",
    "render_retrieved_chunk",
    "render_chunk_comparison",
]
