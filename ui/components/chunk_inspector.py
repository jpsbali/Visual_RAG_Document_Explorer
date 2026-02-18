"""
Chunk visualization component.

Displays chunk content, metadata, and entities.
"""

import streamlit as st
from typing import Optional
from config.models import ChunkMetadata, RetrievedChunk
import pandas as pd


def render_chunk_inspector(
    chunk_content: str,
    metadata: ChunkMetadata,
    relevance_score: Optional[float] = None,
    show_full_text: bool = True
) -> None:
    """
    Render a detailed chunk inspection view with content and metadata.
    
    Args:
        chunk_content: The full text content of the chunk
        metadata: ChunkMetadata object containing all chunk metadata
        relevance_score: Optional relevance score (0.0 to 1.0)
        show_full_text: Whether to show full text by default (default: True)
    """
    # Header with chunk ID
    st.markdown(f"### 🧩 Chunk Inspector")
    st.caption(f"Chunk ID: `{metadata.chunk_id}`")
    
    # Relevance score if available
    if relevance_score is not None:
        score_color = "green" if relevance_score >= 0.8 else "orange" if relevance_score >= 0.5 else "red"
        st.markdown(f"**Relevance Score:** :{score_color}[{relevance_score:.3f}]")
    
    st.divider()
    
    # Chunk content
    st.markdown("#### 📄 Content")
    
    if show_full_text:
        st.text_area(
            "Full Text",
            chunk_content,
            height=300,
            disabled=True,
            key=f"chunk_text_{metadata.chunk_id}"
        )
    else:
        # Show preview with expand option
        preview = chunk_content[:500]
        if len(chunk_content) > 500:
            preview += "..."
        
        st.info(preview)
        
        if st.checkbox("Show full text", key=f"expand_{metadata.chunk_id}"):
            st.text_area(
                "Full Text",
                chunk_content,
                height=300,
                disabled=True,
                key=f"chunk_text_full_{metadata.chunk_id}"
            )
    
    st.divider()
    
    # Basic metadata
    st.markdown("#### 📊 Basic Metadata")
    
    basic_cols = st.columns(3)
    
    with basic_cols[0]:
        st.metric("Chunk Index", f"{metadata.chunk_index + 1}/{metadata.total_chunks}")
    
    with basic_cols[1]:
        st.metric("Token Count", metadata.token_count)
    
    with basic_cols[2]:
        st.metric("Character Count", metadata.char_count)
    
    # Source information
    st.markdown("#### 📁 Source Information")
    
    source_data = {
        "Source File": metadata.source_file,
        "File Type": metadata.file_type.upper(),
        "Page Number": metadata.page_number if metadata.page_number else "N/A",
        "Chunk Type": metadata.chunk_type.capitalize(),
        "Chunk Method": metadata.chunk_method.capitalize()
    }
    
    if metadata.doc_item_type:
        source_data["Document Item Type"] = metadata.doc_item_type
    
    source_df = pd.DataFrame(list(source_data.items()), columns=["Property", "Value"])
    st.dataframe(source_df, hide_index=True, use_container_width=True)
    
    # Hierarchy information
    if metadata.parent_heading or metadata.hierarchy_level is not None:
        st.markdown("#### 🌳 Hierarchy")
        
        hierarchy_data = {}
        
        if metadata.hierarchy_level is not None:
            hierarchy_data["Hierarchy Level"] = metadata.hierarchy_level
        
        if metadata.parent_heading:
            hierarchy_data["Parent Heading"] = metadata.parent_heading
        
        hierarchy_df = pd.DataFrame(list(hierarchy_data.items()), columns=["Property", "Value"])
        st.dataframe(hierarchy_df, hide_index=True, use_container_width=True)
    
    st.divider()
    
    # Entities
    st.markdown("#### 🏷️ Extracted Entities")
    
    entities = metadata.entities
    has_entities = any([
        entities.people,
        entities.organizations,
        entities.dates,
        entities.locations,
        entities.topics
    ])
    
    if has_entities:
        # Entity counts
        entity_cols = st.columns(5)
        
        with entity_cols[0]:
            st.metric("👤 People", len(entities.people))
        
        with entity_cols[1]:
            st.metric("🏢 Organizations", len(entities.organizations))
        
        with entity_cols[2]:
            st.metric("📅 Dates", len(entities.dates))
        
        with entity_cols[3]:
            st.metric("📍 Locations", len(entities.locations))
        
        with entity_cols[4]:
            st.metric("🏷️ Topics", len(entities.topics))
        
        # Entity details in expandable sections
        if entities.people:
            with st.expander(f"👤 People ({len(entities.people)})"):
                for person in entities.people:
                    confidence = entities.confidence_scores.get(person)
                    if confidence:
                        st.markdown(f"- {person} `({confidence:.2f})`")
                    else:
                        st.markdown(f"- {person}")
        
        if entities.organizations:
            with st.expander(f"🏢 Organizations ({len(entities.organizations)})"):
                for org in entities.organizations:
                    confidence = entities.confidence_scores.get(org)
                    if confidence:
                        st.markdown(f"- {org} `({confidence:.2f})`")
                    else:
                        st.markdown(f"- {org}")
        
        if entities.dates:
            with st.expander(f"📅 Dates ({len(entities.dates)})"):
                for date in entities.dates:
                    st.markdown(f"- {date}")
        
        if entities.locations:
            with st.expander(f"📍 Locations ({len(entities.locations)})"):
                for location in entities.locations:
                    confidence = entities.confidence_scores.get(location)
                    if confidence:
                        st.markdown(f"- {location} `({confidence:.2f})`")
                    else:
                        st.markdown(f"- {location}")
        
        if entities.topics:
            with st.expander(f"🏷️ Topics ({len(entities.topics)})"):
                for topic in entities.topics:
                    st.markdown(f"- {topic}")
        
        # Custom entities if available
        if entities.custom:
            with st.expander(f"🔖 Custom Entities ({sum(len(v) for v in entities.custom.values())})"):
                for entity_type, entity_list in entities.custom.items():
                    st.markdown(f"**{entity_type.capitalize()}:**")
                    for entity in entity_list:
                        st.markdown(f"- {entity}")
        
        # Extractor info
        st.caption(f"Extracted using: {entities.extractor}")
    else:
        st.info("No entities extracted from this chunk")
    
    # Keywords
    if metadata.keywords:
        st.markdown("#### 🔑 Keywords")
        
        # Display keywords as tags
        keyword_html = " ".join([
            f'<span style="background-color: #e0e0e0; padding: 4px 8px; margin: 2px; border-radius: 4px; display: inline-block;">{kw}</span>'
            for kw in metadata.keywords
        ])
        st.markdown(keyword_html, unsafe_allow_html=True)
    
    st.divider()
    
    # Timestamps
    st.markdown("#### ⏰ Timestamps")
    
    timestamp_cols = st.columns(2)
    
    with timestamp_cols[0]:
        st.caption("Created At")
        st.text(metadata.created_at.strftime("%Y-%m-%d %H:%M:%S"))
    
    with timestamp_cols[1]:
        st.caption("Processed At")
        st.text(metadata.processed_at.strftime("%Y-%m-%d %H:%M:%S"))


def render_retrieved_chunk(chunk: RetrievedChunk, show_full_text: bool = True) -> None:
    """
    Render a RetrievedChunk object (convenience wrapper).
    
    Args:
        chunk: RetrievedChunk object containing content, metadata, and score
        show_full_text: Whether to show full text by default (default: True)
    """
    render_chunk_inspector(
        chunk_content=chunk.content,
        metadata=chunk.metadata,
        relevance_score=chunk.score,
        show_full_text=show_full_text
    )


def render_chunk_comparison(
    chunks: list[RetrievedChunk],
    max_chunks: int = 3
) -> None:
    """
    Render a side-by-side comparison of multiple chunks.
    
    Args:
        chunks: List of RetrievedChunk objects to compare
        max_chunks: Maximum number of chunks to display (default: 3)
    """
    if not chunks:
        st.info("No chunks to compare")
        return
    
    display_chunks = chunks[:max_chunks]
    
    st.markdown(f"### 🔍 Chunk Comparison ({len(display_chunks)} chunks)")
    
    # Create columns for side-by-side comparison
    cols = st.columns(len(display_chunks))
    
    for idx, (col, chunk) in enumerate(zip(cols, display_chunks)):
        with col:
            st.markdown(f"#### Chunk {idx + 1}")
            st.caption(f"Score: {chunk.score:.3f}")
            
            # Show preview
            preview = chunk.content[:200]
            if len(chunk.content) > 200:
                preview += "..."
            
            st.text_area(
                "Content",
                preview,
                height=150,
                disabled=True,
                key=f"compare_chunk_{idx}_{chunk.metadata.chunk_id}"
            )
            
            # Key metadata
            st.caption(f"📄 {chunk.metadata.source_file}")
            st.caption(f"🧩 {chunk.metadata.chunk_index + 1}/{chunk.metadata.total_chunks}")
            st.caption(f"📊 {chunk.metadata.token_count} tokens")
            
            # View details button
            if st.button("View Details", key=f"view_details_{idx}_{chunk.metadata.chunk_id}"):
                st.session_state[f"selected_chunk_{idx}"] = chunk
    
    # Show detailed view for selected chunk
    for idx in range(len(display_chunks)):
        if f"selected_chunk_{idx}" in st.session_state:
            st.divider()
            render_retrieved_chunk(st.session_state[f"selected_chunk_{idx}"])
            if st.button("Close Details", key=f"close_details_{idx}"):
                del st.session_state[f"selected_chunk_{idx}"]
                st.rerun()
