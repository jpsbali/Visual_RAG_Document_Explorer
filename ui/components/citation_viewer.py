"""
Citation display component.

Shows source citations with expandable details.
"""

import streamlit as st
from typing import List, Optional, Callable
from config.models import Citation, RetrievedChunk


def get_relevance_color(score: float) -> str:
    """
    Get color code based on relevance score.
    
    Args:
        score: Relevance score (0.0 to 1.0)
    
    Returns:
        str: Color name for Streamlit
    """
    if score >= 0.8:
        return "green"
    elif score >= 0.5:
        return "orange"
    else:
        return "red"


def get_relevance_emoji(score: float) -> str:
    """
    Get emoji based on relevance score.
    
    Args:
        score: Relevance score (0.0 to 1.0)
    
    Returns:
        str: Emoji representing relevance level
    """
    if score >= 0.8:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    else:
        return "🔴"


def render_citation(
    citation: Citation,
    index: int,
    relevance_score: Optional[float] = None,
    on_view_chunk: Optional[Callable[[str], None]] = None
) -> None:
    """
    Render a single citation with expandable details.
    
    Args:
        citation: Citation object containing:
            - source_file: str
            - page_number: Optional[int]
            - chunk_id: str
            - relevant_text: str
            - claims_supported: List[str]
        index: Citation number (for display)
        relevance_score: Optional relevance score (0.0 to 1.0)
        on_view_chunk: Optional callback to view full chunk in explorer
    """
    # Create citation header
    header_parts = [f"**[{index}]** 📄 {citation.source_file}"]
    
    if citation.page_number is not None:
        header_parts.append(f"(Page {citation.page_number})")
    
    if relevance_score is not None:
        emoji = get_relevance_emoji(relevance_score)
        header_parts.append(f"{emoji} {relevance_score:.2f}")
    
    header = " ".join(header_parts)
    
    # Create expandable section
    with st.expander(header, expanded=False):
        # Relevance score with color coding
        if relevance_score is not None:
            color = get_relevance_color(relevance_score)
            st.markdown(f"**Relevance:** :{color}[{relevance_score:.3f}]")
        
        # Chunk ID
        st.caption(f"Chunk ID: `{citation.chunk_id}`")
        
        # Text excerpt
        st.markdown("**Excerpt:**")
        excerpt = citation.relevant_text
        if len(excerpt) > 200:
            excerpt = excerpt[:200] + "..."
        st.info(excerpt)
        
        # Full text toggle
        if len(citation.relevant_text) > 200:
            if st.checkbox("Show full text", key=f"full_text_{citation.chunk_id}"):
                st.text_area(
                    "Full Text",
                    citation.relevant_text,
                    height=200,
                    disabled=True,
                    key=f"full_text_area_{citation.chunk_id}"
                )
        
        # Claims supported
        if citation.claims_supported:
            st.markdown("**Claims Supported:**")
            for claim in citation.claims_supported:
                st.markdown(f"- {claim}")
        
        # View in explorer button
        if on_view_chunk:
            if st.button(
                "🔍 View in Explorer",
                key=f"view_chunk_{citation.chunk_id}",
                use_container_width=True
            ):
                on_view_chunk(citation.chunk_id)


def render_citation_list(
    citations: List[Citation],
    relevance_scores: Optional[List[float]] = None,
    on_view_chunk: Optional[Callable[[str], None]] = None,
    max_display: Optional[int] = None
) -> None:
    """
    Render a list of citations.
    
    Args:
        citations: List of Citation objects
        relevance_scores: Optional list of relevance scores (same length as citations)
        on_view_chunk: Optional callback to view full chunk in explorer
        max_display: Optional maximum number of citations to display
    """
    if not citations:
        st.info("No citations available")
        return
    
    # Limit display if specified
    display_citations = citations[:max_display] if max_display else citations
    
    st.markdown(f"**Sources ({len(citations)} total):**")
    
    for idx, citation in enumerate(display_citations, start=1):
        score = relevance_scores[idx - 1] if relevance_scores and idx - 1 < len(relevance_scores) else None
        render_citation(citation, idx, score, on_view_chunk)
    
    # Show message if citations were truncated
    if max_display and len(citations) > max_display:
        st.caption(f"Showing {max_display} of {len(citations)} citations")


def render_retrieved_chunks_as_citations(
    chunks: List[RetrievedChunk],
    on_view_chunk: Optional[Callable[[str], None]] = None,
    max_display: Optional[int] = None
) -> None:
    """
    Render retrieved chunks as citations (for when Citation objects aren't available).
    
    Args:
        chunks: List of RetrievedChunk objects
        on_view_chunk: Optional callback to view full chunk in explorer
        max_display: Optional maximum number of chunks to display
    """
    if not chunks:
        st.info("No sources available")
        return
    
    # Convert chunks to citations
    citations = []
    scores = []
    
    for chunk in chunks:
        citation = Citation(
            source_file=chunk.metadata.source_file,
            page_number=chunk.metadata.page_number,
            chunk_id=chunk.metadata.chunk_id,
            relevant_text=chunk.content,
            claims_supported=[]
        )
        citations.append(citation)
        scores.append(chunk.score)
    
    render_citation_list(citations, scores, on_view_chunk, max_display)


def render_citation_summary(citations: List[Citation]) -> None:
    """
    Render a summary of citations by source file.
    
    Args:
        citations: List of Citation objects
    """
    if not citations:
        return
    
    # Group by source file
    sources = {}
    for citation in citations:
        source = citation.source_file
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    
    # Display summary
    st.markdown("**Citation Summary:**")
    
    cols = st.columns(min(len(sources), 3))
    
    for idx, (source, count) in enumerate(sources.items()):
        with cols[idx % len(cols)]:
            st.metric(
                label=source,
                value=f"{count} citation{'s' if count != 1 else ''}"
            )
