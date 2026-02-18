"""
Document Explorer page for Visual RAG Document Explorer.

Provides document exploration interface with filtering, search, and chunk inspection.
"""

import streamlit as st
from datetime import datetime
from typing import Optional, Dict, List, Any
import asyncio
from pathlib import Path

from config.settings import Settings
from config.models import RetrievedChunk, ChunkMetadata
from core.vectordb.router import get_vector_store
from core.search.metadata_filter import MetadataFilter
from ui.components import render_sidebar, render_document_grid, render_chunk_inspector


def initialize_session_state() -> None:
    """Initialize session state variables for explorer page."""
    if "explorer_selected_doc" not in st.session_state:
        st.session_state.explorer_selected_doc = None
    if "explorer_view_mode" not in st.session_state:
        st.session_state.explorer_view_mode = "grid"  # grid or details
    if "explorer_filters" not in st.session_state:
        st.session_state.explorer_filters = {}
    if "explorer_search_query" not in st.session_state:
        st.session_state.explorer_search_query = ""
    if "explorer_page" not in st.session_state:
        st.session_state.explorer_page = 0


async def get_all_documents(
    vector_store,
    collection: str,
    filters: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve all documents from the vector store.
    
    Args:
        vector_store: Vector store instance
        collection: Collection name
        filters: Optional metadata filters
        
    Returns:
        List of document dictionaries with metadata
    """
    try:
        # Get total count
        total_count = await vector_store.count(collection)
        
        if total_count == 0:
            return []
        
        # For exploration, we'll retrieve a sample by doing a broad search
        # Create a dummy embedding (all zeros) to get documents
        dummy_embedding = [0.0] * 1024  # Assuming 1024 dimensions
        
        # Retrieve all chunks (or a large sample)
        chunks = await vector_store.search(
            collection=collection,
            query_embedding=dummy_embedding,
            top_k=min(total_count, 1000),  # Limit to 1000 for performance
            filters=filters
        )
        
        # Group chunks by source file
        documents_map = {}
        for chunk in chunks:
            source_file = chunk.metadata.source_file
            
            if source_file not in documents_map:
                documents_map[source_file] = {
                    "filename": source_file,
                    "file_type": chunk.metadata.file_type,
                    "file_size": chunk.metadata.char_count * chunk.metadata.total_chunks,  # Estimate
                    "upload_date": chunk.metadata.created_at,
                    "num_chunks": chunk.metadata.total_chunks,
                    "entities": {
                        "people": [],
                        "organizations": [],
                        "locations": [],
                        "dates": []
                    },
                    "file_id": source_file,
                    "chunks": [],
                    "summary": None
                }
            
            # Aggregate entities
            doc = documents_map[source_file]
            doc["chunks"].append(chunk)
            
            # Collect unique entities
            for person in chunk.metadata.entities.people:
                if person not in doc["entities"]["people"]:
                    doc["entities"]["people"].append(person)
            for org in chunk.metadata.entities.organizations:
                if org not in doc["entities"]["organizations"]:
                    doc["entities"]["organizations"].append(org)
            for loc in chunk.metadata.entities.locations:
                if loc not in doc["entities"]["locations"]:
                    doc["entities"]["locations"].append(loc)
            for date in chunk.metadata.entities.dates:
                if date not in doc["entities"]["dates"]:
                    doc["entities"]["dates"].append(date)
        
        return list(documents_map.values())
        
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []


def render_filters_sidebar() -> Dict[str, Any]:
    """
    Render filter controls in the sidebar.
    
    Returns:
        Dictionary of active filters
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")
    
    filters = {}
    
    # Search by filename
    search_query = st.sidebar.text_input(
        "Search filename",
        value=st.session_state.explorer_search_query,
        placeholder="Enter filename..."
    )
    if search_query:
        filters["filename_search"] = search_query
        st.session_state.explorer_search_query = search_query
    
    # File type filter
    file_types = st.sidebar.multiselect(
        "File Type",
        options=["pdf", "docx", "txt", "html", "json"],
        default=[]
    )
    if file_types:
        filters["file_type"] = file_types
    
    # Entity filters
    with st.sidebar.expander("Entity Filters"):
        organizations = st.text_input(
            "Organizations (comma-separated)",
            placeholder="e.g., Acme Corp, TechCo"
        )
        if organizations:
            filters["organizations"] = [org.strip() for org in organizations.split(",")]
        
        people = st.text_input(
            "People (comma-separated)",
            placeholder="e.g., John Doe, Jane Smith"
        )
        if people:
            filters["people"] = [person.strip() for person in people.split(",")]
        
        locations = st.text_input(
            "Locations (comma-separated)",
            placeholder="e.g., New York, London"
        )
        if locations:
            filters["locations"] = [loc.strip() for loc in locations.split(",")]
    
    # Date range filter
    with st.sidebar.expander("Date Range"):
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", value=None)
        with col2:
            end_date = st.date_input("To", value=None)
        
        if start_date or end_date:
            filters["date_range"] = {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            }
    
    # Clear filters button
    if st.sidebar.button("Clear All Filters", use_container_width=True):
        st.session_state.explorer_filters = {}
        st.session_state.explorer_search_query = ""
        st.rerun()
    
    return filters


def apply_document_filters(
    documents: List[Dict[str, Any]],
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Apply filters to document list.
    
    Args:
        documents: List of document dictionaries
        filters: Filter criteria
        
    Returns:
        Filtered list of documents
    """
    if not filters:
        return documents
    
    filtered = []
    
    for doc in documents:
        # Filename search
        if "filename_search" in filters:
            if filters["filename_search"].lower() not in doc["filename"].lower():
                continue
        
        # File type filter
        if "file_type" in filters:
            if doc["file_type"] not in filters["file_type"]:
                continue
        
        # Entity filters
        if "organizations" in filters:
            if not any(org in doc["entities"]["organizations"] for org in filters["organizations"]):
                continue
        
        if "people" in filters:
            if not any(person in doc["entities"]["people"] for person in filters["people"]):
                continue
        
        if "locations" in filters:
            if not any(loc in doc["entities"]["locations"] for loc in filters["locations"]):
                continue
        
        # Date range filter
        if "date_range" in filters:
            date_range = filters["date_range"]
            doc_date = doc["upload_date"].date()
            
            if date_range.get("start") and doc_date < datetime.fromisoformat(date_range["start"]).date():
                continue
            if date_range.get("end") and doc_date > datetime.fromisoformat(date_range["end"]).date():
                continue
        
        filtered.append(doc)
    
    return filtered


def render_document_details(document: Dict[str, Any]) -> None:
    """
    Render detailed view of a single document.
    
    Args:
        document: Document dictionary with metadata and chunks
    """
    st.markdown(f"# 📄 {document['filename']}")
    
    # Back button
    if st.button("← Back to Documents"):
        st.session_state.explorer_selected_doc = None
        st.session_state.explorer_view_mode = "grid"
        st.rerun()
    
    st.divider()
    
    # Document metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("File Type", document["file_type"].upper())
    
    with col2:
        st.metric("Chunks", document["num_chunks"])
    
    with col3:
        st.metric("Upload Date", document["upload_date"].strftime("%Y-%m-%d"))
    
    with col4:
        total_entities = sum(len(entities) for entities in document["entities"].values())
        st.metric("Total Entities", total_entities)
    
    st.divider()
    
    # Entity statistics
    st.subheader("🏷️ Entity Statistics")
    
    entity_cols = st.columns(4)
    
    with entity_cols[0]:
        st.metric("👤 People", len(document["entities"]["people"]))
        if document["entities"]["people"]:
            with st.expander("View People"):
                for person in document["entities"]["people"]:
                    st.markdown(f"- {person}")
    
    with entity_cols[1]:
        st.metric("🏢 Organizations", len(document["entities"]["organizations"]))
        if document["entities"]["organizations"]:
            with st.expander("View Organizations"):
                for org in document["entities"]["organizations"]:
                    st.markdown(f"- {org}")
    
    with entity_cols[2]:
        st.metric("📍 Locations", len(document["entities"]["locations"]))
        if document["entities"]["locations"]:
            with st.expander("View Locations"):
                for loc in document["entities"]["locations"]:
                    st.markdown(f"- {loc}")
    
    with entity_cols[3]:
        st.metric("📅 Dates", len(document["entities"]["dates"]))
        if document["entities"]["dates"]:
            with st.expander("View Dates"):
                for date in document["entities"]["dates"]:
                    st.markdown(f"- {date}")
    
    st.divider()
    
    # Chunk viewer
    st.subheader("🧩 Document Chunks")
    
    chunks = document.get("chunks", [])
    
    if not chunks:
        st.info("No chunks available for this document")
        return
    
    # Pagination controls
    chunks_per_page = 10
    total_pages = (len(chunks) + chunks_per_page - 1) // chunks_per_page
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Previous", disabled=st.session_state.explorer_page == 0):
            st.session_state.explorer_page -= 1
            st.rerun()
    
    with col2:
        st.markdown(f"<div style='text-align: center;'>Page {st.session_state.explorer_page + 1} of {total_pages}</div>", 
                   unsafe_allow_html=True)
    
    with col3:
        if st.button("Next →", disabled=st.session_state.explorer_page >= total_pages - 1):
            st.session_state.explorer_page += 1
            st.rerun()
    
    # Display chunks for current page
    start_idx = st.session_state.explorer_page * chunks_per_page
    end_idx = min(start_idx + chunks_per_page, len(chunks))
    page_chunks = chunks[start_idx:end_idx]
    
    for idx, chunk in enumerate(page_chunks):
        with st.expander(f"Chunk {start_idx + idx + 1} - {chunk.metadata.content_preview[:50]}..."):
            render_chunk_inspector(
                chunk_content=chunk.content,
                metadata=chunk.metadata,
                relevance_score=None,
                show_full_text=False
            )


def main() -> None:
    """Main function for the Document Explorer page."""
    st.set_page_config(
        page_title="Document Explorer - Visual RAG",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load settings
    settings = Settings()
    
    # Render sidebar with filters
    system_status = {
        "vector_db_connected": True,
        "vector_db_type": settings.default_vector_db,
        "llm_provider": settings.default_llm_provider,
        "llm_model": settings.default_model
    }
    
    # Get filters before rendering sidebar
    filters = render_filters_sidebar()
    
    # Render main sidebar navigation
    selected_page = render_sidebar(
        current_page="explorer",
        system_status=system_status,
        stats=None  # Will be populated after loading documents
    )
    
    # Handle page navigation
    if selected_page != "explorer":
        st.switch_page(f"ui/pages/{selected_page}.py")
    
    # Main content
    if st.session_state.explorer_view_mode == "details" and st.session_state.explorer_selected_doc:
        # Show document details
        render_document_details(st.session_state.explorer_selected_doc)
    else:
        # Show document grid
        st.title("🔍 Document Explorer")
        st.markdown("Browse and explore your indexed documents")
        
        # Load documents
        try:
            vector_store = get_vector_store(settings)
            
            # Check if vector store is healthy
            with st.spinner("Connecting to vector database..."):
                is_healthy = asyncio.run(vector_store.health_check())
            
            if not is_healthy:
                st.error("❌ Vector database is not available. Please check your connection settings.")
                return
            
            # Get all documents
            with st.spinner("Loading documents..."):
                # Convert entity filters to metadata filters format
                metadata_filters = {}
                if "organizations" in filters:
                    metadata_filters["organizations"] = filters["organizations"]
                if "people" in filters:
                    metadata_filters["people"] = filters["people"]
                if "locations" in filters:
                    metadata_filters["locations"] = filters["locations"]
                
                documents = asyncio.run(get_all_documents(
                    vector_store,
                    settings.qdrant_collection if settings.default_vector_db == "qdrant" else settings.milvus_collection,
                    metadata_filters if metadata_filters else None
                ))
            
            # Apply additional filters (filename, file type, date range)
            documents = apply_document_filters(documents, filters)
            
            # Update stats in sidebar
            if documents:
                total_chunks = sum(doc["num_chunks"] for doc in documents)
                total_entities = sum(
                    sum(len(entities) for entities in doc["entities"].values())
                    for doc in documents
                )
                
                st.sidebar.markdown("---")
                st.sidebar.subheader("Quick Stats")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric("Documents", len(documents))
                with col2:
                    st.metric("Chunks", total_chunks)
                st.sidebar.metric("Entities", total_entities)
            
            # Display results
            if not documents:
                st.info("📭 No documents found. Upload some documents to get started!")
            else:
                st.success(f"Found {len(documents)} document(s)")
                
                # View mode selector
                view_mode = st.radio(
                    "View Mode",
                    options=["Grid", "List"],
                    horizontal=True,
                    index=0
                )
                
                st.divider()
                
                # Render documents
                def on_document_click(file_id: str):
                    """Handle document card click."""
                    # Find the document
                    for doc in documents:
                        if doc["file_id"] == file_id:
                            st.session_state.explorer_selected_doc = doc
                            st.session_state.explorer_view_mode = "details"
                            st.session_state.explorer_page = 0
                            st.rerun()
                            break
                
                if view_mode == "Grid":
                    render_document_grid(
                        documents=documents,
                        on_click=on_document_click,
                        columns=2
                    )
                else:
                    # List view - one column
                    render_document_grid(
                        documents=documents,
                        on_click=on_document_click,
                        columns=1
                    )
        
        except Exception as e:
            st.error(f"❌ Error loading documents: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
