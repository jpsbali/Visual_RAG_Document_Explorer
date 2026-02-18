"""
Document display card component.

Displays document metadata, entities, and summary.
"""

import streamlit as st
from typing import Dict, List, Optional, Callable
from datetime import datetime


def get_file_icon(file_type: str) -> str:
    """
    Get an emoji icon for the file type.
    
    Args:
        file_type: File extension (pdf, docx, txt, html, json)
    
    Returns:
        str: Emoji icon
    """
    icons = {
        "pdf": "📕",
        "docx": "📘",
        "txt": "📄",
        "html": "🌐",
        "json": "📋"
    }
    return icons.get(file_type.lower(), "📄")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
    
    Returns:
        str: Formatted size (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def render_document_card(
    filename: str,
    file_type: str,
    file_size: int,
    upload_date: datetime,
    num_chunks: int,
    entities: Dict[str, List[str]],
    file_id: Optional[str] = None,
    summary: Optional[str] = None,
    on_click: Optional[Callable[[str], None]] = None
) -> None:
    """
    Render a document card with metadata and entities.
    
    Args:
        filename: Name of the document file
        file_type: File extension (pdf, docx, txt, html, json)
        file_size: File size in bytes
        upload_date: When the document was uploaded
        num_chunks: Number of chunks created from this document
        entities: Dictionary of extracted entities:
            - people: List[str]
            - organizations: List[str]
            - locations: List[str]
            - dates: List[str]
        file_id: Optional unique identifier for the document
        summary: Optional document summary
        on_click: Optional callback function when card is clicked
    """
    with st.container():
        # Create a bordered container using columns
        col1, col2 = st.columns([1, 4])
        
        with col1:
            # File icon
            st.markdown(f"<div style='font-size: 48px; text-align: center;'>{get_file_icon(file_type)}</div>", 
                       unsafe_allow_html=True)
        
        with col2:
            # Document header
            st.markdown(f"**{filename}**")
            
            # Metadata row
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            
            with meta_col1:
                st.caption(f"📦 {format_file_size(file_size)}")
            
            with meta_col2:
                st.caption(f"📅 {upload_date.strftime('%Y-%m-%d')}")
            
            with meta_col3:
                st.caption(f"🧩 {num_chunks} chunks")
        
        # Summary if available
        if summary:
            with st.expander("📝 Summary"):
                st.write(summary)
        
        # Entity badges
        if entities:
            st.markdown("**Entities:**")
            
            # Create badge columns
            badge_cols = st.columns(4)
            
            entity_types = [
                ("organizations", "🏢", entities.get("organizations", [])),
                ("people", "👤", entities.get("people", [])),
                ("locations", "📍", entities.get("locations", [])),
                ("dates", "📅", entities.get("dates", []))
            ]
            
            for idx, (entity_type, icon, entity_list) in enumerate(entity_types):
                with badge_cols[idx % 4]:
                    count = len(entity_list)
                    if count > 0:
                        st.metric(
                            label=f"{icon} {entity_type.capitalize()}",
                            value=count
                        )
                        # Show first few entities in tooltip
                        if entity_list:
                            preview = ", ".join(entity_list[:3])
                            if len(entity_list) > 3:
                                preview += f" (+{len(entity_list) - 3} more)"
                            st.caption(preview)
        
        # Action button
        if on_click and file_id:
            if st.button("View Details", key=f"view_{file_id}", use_container_width=True):
                on_click(file_id)
        
        # Divider between cards
        st.divider()


def render_document_grid(
    documents: List[Dict],
    on_click: Optional[Callable[[str], None]] = None,
    columns: int = 2
) -> None:
    """
    Render multiple document cards in a grid layout.
    
    Args:
        documents: List of document dictionaries, each containing:
            - filename: str
            - file_type: str
            - file_size: int
            - upload_date: datetime
            - num_chunks: int
            - entities: Dict[str, List[str]]
            - file_id: str (optional)
            - summary: str (optional)
        on_click: Optional callback function when a card is clicked
        columns: Number of columns in the grid (default: 2)
    """
    if not documents:
        st.info("No documents found")
        return
    
    # Create grid layout
    for i in range(0, len(documents), columns):
        cols = st.columns(columns)
        
        for j in range(columns):
            idx = i + j
            if idx < len(documents):
                doc = documents[idx]
                
                with cols[j]:
                    render_document_card(
                        filename=doc["filename"],
                        file_type=doc["file_type"],
                        file_size=doc["file_size"],
                        upload_date=doc["upload_date"],
                        num_chunks=doc["num_chunks"],
                        entities=doc["entities"],
                        file_id=doc.get("file_id"),
                        summary=doc.get("summary"),
                        on_click=on_click
                    )
