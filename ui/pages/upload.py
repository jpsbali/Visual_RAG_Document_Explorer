"""
Document Upload Page for Streamlit Application.

Handles document ingestion and processing through the complete pipeline:
1. Document loading
2. Chunking
3. NER extraction
4. Summarization
5. Deduplication
6. Embedding generation
7. Vector DB indexing
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import time
import hashlib
import asyncio
import uuid

from core.document_processing.loaders import DocumentLoaderFactory
from core.document_processing.chunker import AdaptiveChunker
from core.document_processing.ner_router import NERRouter
from core.document_processing.summarizer import DocumentSummarizer
from core.document_processing.deduplication import DeduplicationService
from core.embeddings.embedding_router import get_embedding_service
from core.vectordb.router import get_vector_store
from core.llm.llm_router import get_llm_provider
from config.settings import Settings
from config.models import ChunkMetadata, NEREntities
from ui.components import render_sidebar
from langchain_core.documents import Document


def initialize_session_state():
    """Initialize session state variables."""
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []
    if "processing_settings" not in st.session_state:
        st.session_state.processing_settings = {
            "chunk_size_min": 200,
            "chunk_size_max": 1000,
            "chunk_overlap": 50,
            "ner_mode": "ensemble",
            "embedding_model": "voyage",
            "enable_summarization": True
        }


def render_configuration_sidebar(settings: Settings):
    """Render configuration options in the sidebar."""
    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Processing Configuration")
        
        with st.expander("Chunking Settings", expanded=False):
            chunk_size_min = st.slider(
                "Minimum Chunk Size",
                min_value=100,
                max_value=500,
                value=st.session_state.processing_settings["chunk_size_min"],
                step=50,
                help="Minimum number of characters per chunk"
            )
            
            chunk_size_max = st.slider(
                "Maximum Chunk Size",
                min_value=500,
                max_value=2000,
                value=st.session_state.processing_settings["chunk_size_max"],
                step=100,
                help="Maximum number of characters per chunk"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=200,
                value=st.session_state.processing_settings["chunk_overlap"],
                step=10,
                help="Number of overlapping characters between chunks"
            )
            
            st.session_state.processing_settings["chunk_size_min"] = chunk_size_min
            st.session_state.processing_settings["chunk_size_max"] = chunk_size_max
            st.session_state.processing_settings["chunk_overlap"] = chunk_overlap
        
        with st.expander("NER & Embedding Settings", expanded=False):
            ner_mode = st.selectbox(
                "NER Mode",
                options=["gliner", "spacy", "ensemble"],
                index=["gliner", "spacy", "ensemble"].index(
                    st.session_state.processing_settings["ner_mode"]
                ),
                help="Named Entity Recognition extraction mode"
            )
            
            embedding_model = st.selectbox(
                "Embedding Model",
                options=["voyage", "bge-m3"],
                index=["voyage", "bge-m3"].index(
                    st.session_state.processing_settings["embedding_model"]
                ),
                help="Embedding model for vector generation"
            )
            
            enable_summarization = st.checkbox(
                "Enable Summarization",
                value=st.session_state.processing_settings["enable_summarization"],
                help="Generate document summaries"
            )
            
            st.session_state.processing_settings["ner_mode"] = ner_mode
            st.session_state.processing_settings["embedding_model"] = embedding_model
            st.session_state.processing_settings["enable_summarization"] = enable_summarization


def validate_file(uploaded_file) -> tuple[bool, str]:
    """
    Validate uploaded file.
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Check file size (warn if > 10MB)
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 10:
        return False, f"⚠️ File size ({file_size_mb:.1f}MB) exceeds 10MB limit"
    
    # Check file extension
    file_ext = Path(uploaded_file.name).suffix.lower()
    supported_formats = [".pdf", ".docx", ".txt", ".html", ".htm", ".json"]
    if file_ext not in supported_formats:
        return False, f"❌ Unsupported file format: {file_ext}"
    
    return True, "✅ File validation passed"


def render_entity_preview(entities: NEREntities):
    """Render extracted entities in an expandable section."""
    st.subheader("🏷️ Extracted Entities")
    
    # Count entities
    entity_counts = {
        "Organizations": len(entities.organizations),
        "People": len(entities.people),
        "Dates": len(entities.dates),
        "Locations": len(entities.locations),
        "Topics": len(entities.topics)
    }
    
    # Display counts in columns
    cols = st.columns(5)
    for col, (entity_type, count) in zip(cols, entity_counts.items()):
        with col:
            st.metric(entity_type, count)
    
    # Display entity lists in expandable sections
    if entities.organizations:
        with st.expander(f"Organizations ({len(entities.organizations)})", expanded=False):
            for org in entities.organizations[:20]:  # Limit to first 20
                confidence = entities.confidence_scores.get(org, 0.0)
                st.write(f"• {org}" + (f" ({confidence:.2%})" if confidence > 0 else ""))
    
    if entities.people:
        with st.expander(f"People ({len(entities.people)})", expanded=False):
            for person in entities.people[:20]:
                confidence = entities.confidence_scores.get(person, 0.0)
                st.write(f"• {person}" + (f" ({confidence:.2%})" if confidence > 0 else ""))
    
    if entities.dates:
        with st.expander(f"Dates ({len(entities.dates)})", expanded=False):
            for date in entities.dates[:20]:
                st.write(f"• {date}")
    
    if entities.locations:
        with st.expander(f"Locations ({len(entities.locations)})", expanded=False):
            for location in entities.locations[:20]:
                confidence = entities.confidence_scores.get(location, 0.0)
                st.write(f"• {location}" + (f" ({confidence:.2%})" if confidence > 0 else ""))
    
    if entities.topics:
        with st.expander(f"Topics ({len(entities.topics)})", expanded=False):
            for topic in entities.topics[:20]:
                st.write(f"• {topic}")


def render_summary_display(summary: str):
    """Render document summary."""
    st.subheader("📝 Document Summary")
    
    # Calculate summary stats
    word_count = len(summary.split())
    char_count = len(summary)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(summary)
    with col2:
        st.metric("Words", word_count)
        st.metric("Characters", char_count)


def render_deduplication_feedback(is_duplicate: bool, similarity: float, existing_doc: Optional[Dict] = None):
    """Render deduplication check results."""
    if not is_duplicate:
        st.success("✅ **New Document** - No duplicates found")
        return True
    
    if similarity >= 0.99:
        st.error(f"❌ **Exact Duplicate** - Similarity: {similarity:.2%}")
        if existing_doc:
            st.write(f"Existing document: {existing_doc.get('filename', 'Unknown')}")
        return False
    else:
        st.warning(f"⚠️ **Near Duplicate** - Similarity: {similarity:.2%}")
        if existing_doc:
            st.write(f"Similar to: {existing_doc.get('filename', 'Unknown')}")
        
        # Ask user to confirm
        confirm = st.checkbox(
            "Process anyway?",
            key=f"confirm_duplicate_{similarity}",
            help="Check this box to process the document despite similarity"
        )
        return confirm


async def process_document(
    uploaded_file,
    settings: Settings,
    processing_settings: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Process a single uploaded document through the complete pipeline.
    
    Returns:
        Dictionary with processing results or None if failed
    """
    start_time = time.time()
    
    # Create uploads directory if it doesn't exist
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file temporarily
    temp_path = uploads_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Progress tracking
    progress_bar = st.progress(0, text="Starting processing...")
    status_container = st.container()
    
    try:
        # Update settings with user configuration
        settings.chunk_size_min = processing_settings["chunk_size_min"]
        settings.chunk_size_max = processing_settings["chunk_size_max"]
        settings.chunk_overlap = processing_settings["chunk_overlap"]
        settings.ner_mode = processing_settings["ner_mode"]
        settings.default_embedding_model = processing_settings["embedding_model"]
        
        # 1. Load document
        with status_container:
            st.write("📄 **Step 1/7:** Loading document...")
        progress_bar.progress(10, text="Loading document...")
        
        documents = DocumentLoaderFactory.load_document(str(temp_path))
        if not documents:
            raise ValueError("Failed to load document")
        
        document = documents[0]  # Get first document
        
        # 2. Chunk document
        with status_container:
            st.write("✂️ **Step 2/7:** Chunking document...")
        progress_bar.progress(25, text="Chunking document...")
        
        chunker = AdaptiveChunker(settings)
        chunks_with_metadata = chunker.chunk_documents([document])
        
        # Convert to Document objects for compatibility with rest of pipeline
        chunks = []
        for chunk_text, chunk_metadata in chunks_with_metadata:
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": chunk_metadata.chunk_id,
                    "source_file": chunk_metadata.source_file,
                    "file_type": chunk_metadata.file_type,
                    "chunk_index": chunk_metadata.chunk_index,
                    "total_chunks": chunk_metadata.total_chunks,
                }
            )
            chunks.append(chunk_doc)
        
        st.write(f"   ✓ Created {len(chunks)} chunks")
        
        # 3. Extract entities
        with status_container:
            st.write("🏷️ **Step 3/7:** Extracting entities...")
        progress_bar.progress(40, text="Extracting entities...")
        
        ner_router = NERRouter(settings)
        all_entities = NEREntities()
        
        for chunk in chunks:
            entities = ner_router.extract(chunk.page_content)
            # Aggregate entities
            all_entities.organizations.extend(entities.organizations)
            all_entities.people.extend(entities.people)
            all_entities.dates.extend(entities.dates)
            all_entities.locations.extend(entities.locations)
            all_entities.topics.extend(entities.topics)
        
        # Deduplicate entity lists
        all_entities.organizations = list(set(all_entities.organizations))
        all_entities.people = list(set(all_entities.people))
        all_entities.dates = list(set(all_entities.dates))
        all_entities.locations = list(set(all_entities.locations))
        all_entities.topics = list(set(all_entities.topics))
        
        total_entities = (
            len(all_entities.organizations) +
            len(all_entities.people) +
            len(all_entities.dates) +
            len(all_entities.locations) +
            len(all_entities.topics)
        )
        st.write(f"   ✓ Extracted {total_entities} unique entities")
        
        # Display entity preview
        render_entity_preview(all_entities)
        
        # 4. Generate summary (if enabled)
        summary = ""
        if processing_settings["enable_summarization"]:
            with status_container:
                st.write("📝 **Step 4/7:** Generating summary...")
            progress_bar.progress(55, text="Generating summary...")
            
            llm_provider = get_llm_provider(settings)
            summarizer = DocumentSummarizer(llm_provider)
            summary = await summarizer.summarize(document.page_content, uploaded_file.name)
            st.write(f"   ✓ Generated summary ({len(summary.split())} words)")
            
            # Display summary
            render_summary_display(summary)
        else:
            progress_bar.progress(55, text="Skipping summarization...")
            st.write("   ⊘ Summarization disabled")
        
        # 5. Check for duplicates
        with status_container:
            st.write("🔍 **Step 5/7:** Checking for duplicates...")
        progress_bar.progress(70, text="Checking for duplicates...")
        
        deduplicator = DeduplicationService(settings)
        content_hash = deduplicator.compute_content_hash(document.page_content)
        is_exact_duplicate = deduplicator.check_exact_duplicate(content_hash)
        
        if is_exact_duplicate:
            st.write("   ⚠️ Exact duplicate detected")
            should_continue = render_deduplication_feedback(True, 1.0)
            if not should_continue:
                return None
        else:
            # Check semantic similarity
            doc_embedding = await get_embedding_service(settings).embed_query(document.page_content[:1000])
            is_duplicate, similarity = await deduplicator.check_semantic_duplicate(
                document.page_content[:1000],
                doc_embedding
            )
            
            if is_duplicate or similarity > 0.85:
                st.write(f"   ⚠️ Near duplicate detected (similarity: {similarity:.2%})")
                should_continue = render_deduplication_feedback(True, similarity)
                if not should_continue:
                    return None
            else:
                st.write("   ✓ No duplicates found")
                render_deduplication_feedback(False, similarity)
            
            # Add to deduplication cache
            deduplicator.add_content_hash(content_hash)
            deduplicator.add_embedding(content_hash, doc_embedding)
        
        # 6. Generate embeddings
        with status_container:
            st.write("🧮 **Step 6/7:** Generating embeddings...")
        progress_bar.progress(85, text="Generating embeddings...")
        
        embedding_service = get_embedding_service(settings)
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = await embedding_service.embed_documents(chunk_texts)
        
        st.write(f"   ✓ Generated {len(embeddings)} embeddings")
        
        # 7. Index in vector DB
        with status_container:
            st.write("💾 **Step 7/7:** Indexing in vector database...")
        progress_bar.progress(95, text="Indexing in vector database...")
        
        vector_store = get_vector_store(settings)
        
        # Prepare metadata for each chunk
        chunk_ids = []
        metadatas = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.md5(
                f"{uploaded_file.name}_{i}_{datetime.now().isoformat()}".encode()
            ).hexdigest()
            chunk_ids.append(chunk_id)
            
            # Create metadata dict
            metadata = {
                "chunk_id": chunk_id,
                "source_file": uploaded_file.name,
                "file_type": Path(uploaded_file.name).suffix.lower().replace(".", ""),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content_hash": content_hash,
                "created_at": datetime.now().isoformat()
            }
            metadatas.append(metadata)
        
        # Upsert to vector store
        await vector_store.upsert(
            collection=settings.qdrant_collection if settings.default_vector_db == "qdrant" else settings.milvus_collection,
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas
        )
        
        st.write(f"   ✓ Indexed {len(chunks)} chunks")
        
        # Complete
        progress_bar.progress(100, text="✅ Processing complete!")
        
        processing_time = time.time() - start_time
        
        return {
            "filename": uploaded_file.name,
            "file_type": Path(uploaded_file.name).suffix.lower().replace(".", ""),
            "chunks": len(chunks),
            "entities": total_entities,
            "summary": summary,
            "processing_time": processing_time,
            "timestamp": datetime.now(),
            "status": "success"
        }
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None
    
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


def render_processing_results(result: Dict[str, Any]):
    """Render processing results after successful upload."""
    st.success(f"✅ Successfully processed **{result['filename']}**")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Chunks Created", result["chunks"])
    
    with col2:
        st.metric("Entities Extracted", result["entities"])
    
    with col3:
        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
    
    with col4:
        st.metric("Status", result["status"].upper())
    
    # Link to explorer
    st.info("📍 View this document in the **Explorer** page to search and analyze chunks")


def render_upload_history():
    """Render upload history section."""
    if not st.session_state.upload_history:
        st.info("No documents uploaded yet in this session")
        return
    
    st.subheader("📜 Upload History")
    
    for i, upload in enumerate(reversed(st.session_state.upload_history[-10:])):  # Show last 10
        with st.expander(
            f"{upload['filename']} - {upload['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
            expanded=False
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Type:** {upload['file_type'].upper()}")
                st.write(f"**Status:** {upload['status']}")
            
            with col2:
                st.write(f"**Chunks:** {upload['chunks']}")
                st.write(f"**Entities:** {upload['entities']}")
            
            with col3:
                st.write(f"**Time:** {upload['processing_time']:.2f}s")
            
            if upload.get("summary"):
                st.write("**Summary:**")
                st.caption(upload["summary"][:200] + "..." if len(upload["summary"]) > 200 else upload["summary"])


def render_upload_page():
    """Main upload page render function."""
    # Initialize session state
    initialize_session_state()
    
    # Load settings
    settings = Settings()
    
    # Render sidebar with configuration
    system_status = {
        "vector_db_connected": True,
        "vector_db_type": settings.default_vector_db,
        "llm_provider": settings.default_llm_provider,
        "llm_model": settings.default_model
    }
    
    stats = {
        "documents_indexed": len(st.session_state.upload_history),
        "total_chunks": sum(u["chunks"] for u in st.session_state.upload_history),
        "total_entities": sum(u["entities"] for u in st.session_state.upload_history)
    }
    
    render_sidebar(current_page="upload", system_status=system_status, stats=stats)
    render_configuration_sidebar(settings)
    
    # Main content
    st.title("📤 Document Upload")
    st.markdown("Upload and process documents through the complete RAG pipeline")
    
    st.divider()
    
    # File uploader
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "txt", "html", "json"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, HTML, JSON"
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} file(s) selected")
        
        # Validate files
        valid_files = []
        for uploaded_file in uploaded_files:
            is_valid, message = validate_file(uploaded_file)
            if is_valid:
                valid_files.append(uploaded_file)
                st.success(f"{uploaded_file.name}: {message}")
            else:
                st.error(f"{uploaded_file.name}: {message}")
        
        # Process button
        if valid_files and st.button("🚀 Process Documents", type="primary"):
            for uploaded_file in valid_files:
                st.divider()
                st.subheader(f"Processing: {uploaded_file.name}")
                
                with st.container():
                    # Process document (async wrapper)
                    result = asyncio.run(
                        process_document(
                            uploaded_file,
                            settings,
                            st.session_state.processing_settings
                        )
                    )
                    
                    if result:
                        # Display results
                        render_processing_results(result)
                        
                        # Add to history
                        st.session_state.upload_history.append(result)
    
    st.divider()
    
    # Upload history
    render_upload_history()
    
    # Clear history button
    if st.session_state.upload_history:
        if st.button("🗑️ Clear Upload History"):
            st.session_state.upload_history = []
            st.rerun()


# Main entry point
if __name__ == "__main__":
    render_upload_page()
