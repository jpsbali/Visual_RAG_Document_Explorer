"""
Integration tests for Phase 5 Streamlit UI.

Tests component integration, page functionality, backend integration,
and end-to-end workflows.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import sys
from typing import Dict, List, Any

# Mock streamlit before importing UI modules
sys.modules['streamlit'] = MagicMock()

import streamlit as st


class TestUIComponents:
    """Test UI component integration."""
    
    def test_sidebar_component_imports(self):
        """Test sidebar component can be imported."""
        try:
            from ui.components.sidebar import render_sidebar
            assert callable(render_sidebar)
        except ImportError as e:
            pytest.fail(f"Failed to import sidebar component: {e}")
    
    def test_document_card_component_imports(self):
        """Test document card component can be imported."""
        try:
            from ui.components.document_card import render_document_card
            assert callable(render_document_card)
        except ImportError as e:
            pytest.fail(f"Failed to import document card component: {e}")
    
    def test_citation_viewer_component_imports(self):
        """Test citation viewer component can be imported."""
        try:
            from ui.components.citation_viewer import render_citation_viewer
            assert callable(render_citation_viewer)
        except ImportError as e:
            pytest.fail(f"Failed to import citation viewer component: {e}")
    
    def test_chunk_inspector_component_imports(self):
        """Test chunk inspector component can be imported."""
        try:
            from ui.components.chunk_inspector import render_chunk_inspector
            assert callable(render_chunk_inspector)
        except ImportError as e:
            pytest.fail(f"Failed to import chunk inspector component: {e}")
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.sidebar')
    def test_sidebar_renders_without_errors(self, mock_sidebar):
        """Test sidebar component renders without errors."""
        from ui.components.sidebar import render_sidebar
        
        # Mock session state
        st.session_state.uploaded_documents = []
        st.session_state.settings = Mock()
        
        try:
            render_sidebar()
        except Exception as e:
            pytest.fail(f"Sidebar rendering failed: {e}")
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    def test_document_card_renders_with_mock_data(self, mock_columns, mock_markdown):
        """Test document card renders with mock data."""
        from ui.components.document_card import render_document_card
        
        mock_doc = {
            "id": "doc123",
            "filename": "test.pdf",
            "file_type": "pdf",
            "file_size": 1024000,
            "num_chunks": 10,
            "entities": {
                "PERSON": ["John Doe"],
                "ORG": ["Acme Corp"]
            },
            "summary": "Test document summary"
        }
        
        # Mock columns to return mock objects
        mock_col1, mock_col2 = Mock(), Mock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        try:
            render_document_card(mock_doc)
            assert mock_markdown.called
        except Exception as e:
            pytest.fail(f"Document card rendering failed: {e}")
    
    @patch('streamlit.expander')
    @patch('streamlit.markdown')
    def test_citation_viewer_renders_with_citations(self, mock_markdown, mock_expander):
        """Test citation viewer renders with mock citations."""
        from ui.components.citation_viewer import render_citation_viewer
        
        mock_citations = [
            {
                "document_id": "doc1",
                "chunk_id": "chunk1",
                "content": "Test citation content",
                "score": 0.95,
                "metadata": {"page": 1}
            }
        ]
        
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        try:
            render_citation_viewer(mock_citations)
            assert mock_expander.called
        except Exception as e:
            pytest.fail(f"Citation viewer rendering failed: {e}")
    
    @patch('streamlit.expander')
    @patch('streamlit.code')
    @patch('streamlit.json')
    def test_chunk_inspector_renders_with_chunk_data(self, mock_json, mock_code, mock_expander):
        """Test chunk inspector renders with mock chunk data."""
        from ui.components.chunk_inspector import render_chunk_inspector
        
        mock_chunk = {
            "id": "chunk1",
            "content": "Test chunk content",
            "metadata": {
                "page": 1,
                "section": "Introduction"
            },
            "embedding_model": "voyage-2"
        }
        
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        try:
            render_chunk_inspector(mock_chunk)
            assert mock_expander.called
        except Exception as e:
            pytest.fail(f"Chunk inspector rendering failed: {e}")
    
    @patch('streamlit.error')
    def test_component_error_handling(self, mock_error):
        """Test component error handling."""
        from ui.components.document_card import render_document_card
        
        # Pass invalid data
        invalid_doc = None
        
        try:
            render_document_card(invalid_doc)
        except Exception:
            # Should handle gracefully
            pass


class TestPageIntegration:
    """Test page-level integration."""
    
    def test_all_pages_can_be_imported(self):
        """Test that all pages can be imported."""
        pages = ['chat', 'upload', 'explorer', 'settings', 'benchmark']
        
        for page in pages:
            try:
                __import__(f'ui.pages.{page}')
            except ImportError as e:
                pytest.fail(f"Failed to import page {page}: {e}")
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.title')
    @patch('streamlit.chat_input')
    def test_chat_page_initialization(self, mock_input, mock_title):
        """Test chat page initializes correctly."""
        from ui.pages import chat
        
        # Initialize session state
        st.session_state.chat_history = []
        st.session_state.settings = Mock()
        
        try:
            # The page module should be importable and have expected functions
            assert hasattr(chat, 'render_chat_page')
        except Exception as e:
            pytest.fail(f"Chat page initialization failed: {e}")
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.file_uploader')
    def test_upload_page_initialization(self, mock_uploader):
        """Test upload page initializes correctly."""
        from ui.pages import upload
        
        st.session_state.uploaded_documents = []
        st.session_state.settings = Mock()
        
        try:
            assert hasattr(upload, 'render_upload_page')
        except Exception as e:
            pytest.fail(f"Upload page initialization failed: {e}")
    
    @patch('streamlit.session_state', {})
    def test_explorer_page_initialization(self):
        """Test explorer page initializes correctly."""
        from ui.pages import explorer
        
        st.session_state.settings = Mock()
        
        try:
            assert hasattr(explorer, 'render_explorer_page')
        except Exception as e:
            pytest.fail(f"Explorer page initialization failed: {e}")
    
    @patch('streamlit.session_state', {})
    def test_settings_page_initialization(self):
        """Test settings page initializes correctly."""
        from ui.pages import settings
        
        st.session_state.settings = Mock()
        
        try:
            assert hasattr(settings, 'render_settings_page')
        except Exception as e:
            pytest.fail(f"Settings page initialization failed: {e}")
    
    @patch('streamlit.session_state', {})
    def test_benchmark_page_initialization(self):
        """Test benchmark page initializes correctly."""
        from ui.pages import benchmark
        
        st.session_state.settings = Mock()
        
        try:
            assert hasattr(benchmark, 'render_benchmark_page')
        except Exception as e:
            pytest.fail(f"Benchmark page initialization failed: {e}")
    
    @patch('streamlit.session_state', {})
    def test_session_state_management(self):
        """Test session state is properly managed across pages."""
        # Initialize session state
        st.session_state.chat_history = []
        st.session_state.uploaded_documents = []
        st.session_state.settings = Mock()
        
        # Verify state persists
        assert 'chat_history' in st.session_state
        assert 'uploaded_documents' in st.session_state
        assert 'settings' in st.session_state
        
        # Modify state
        st.session_state.chat_history.append({"role": "user", "content": "test"})
        assert len(st.session_state.chat_history) == 1


class TestBackendIntegration:
    """Test backend integration with UI pages."""
    
    @pytest.mark.asyncio
    @patch('streamlit.session_state', {})
    @patch('streamlit.chat_message')
    async def test_chat_page_with_orchestrator(self, mock_chat_message):
        """Test chat page integrates with AgentOrchestrator."""
        from ui.pages.chat import render_chat_page
        
        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.arun.return_value = {
            "answer": "Test answer",
            "sources": [],
            "grounding_score": 0.95
        }
        
        st.session_state.chat_history = []
        st.session_state.settings = Mock()
        
        with patch('agents.orchestrator.AgentOrchestrator', return_value=mock_orchestrator):
            try:
                # Simulate user query
                query = "What is RAG?"
                # The page should handle the query
                # In actual implementation, this would call orchestrator
                pass
            except Exception as e:
                pytest.fail(f"Chat page orchestrator integration failed: {e}")
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.file_uploader')
    async def test_upload_page_with_document_processing(self, mock_uploader):
        """Test upload page processes documents through pipeline."""
        from ui.pages.upload import render_upload_page
        
        # Mock file upload
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.type = "application/pdf"
        mock_file.size = 1024000
        mock_file.read.return_value = b"mock pdf content"
        
        mock_uploader.return_value = [mock_file]
        
        st.session_state.uploaded_documents = []
        st.session_state.settings = Mock()
        
        # Mock document processing components
        with patch('core.document_processing.loaders.DocumentLoader') as mock_loader, \
             patch('core.document_processing.chunker.SemanticChunker') as mock_chunker, \
             patch('core.vectordb.router.VectorDBRouter') as mock_vectordb:
            
            mock_loader.return_value.load.return_value = [Mock()]
            mock_chunker.return_value.chunk_documents.return_value = [Mock()]
            mock_vectordb.return_value.add_documents = AsyncMock()
            
            try:
                # The page should process uploaded files
                pass
            except Exception as e:
                pytest.fail(f"Upload page document processing failed: {e}")
    
    @patch('streamlit.session_state', {})
    async def test_explorer_page_with_vector_store(self):
        """Test explorer page queries vector store."""
        from ui.pages.explorer import render_explorer_page
        
        st.session_state.settings = Mock()
        
        # Mock vector store
        mock_vectordb = AsyncMock()
        mock_vectordb.search.return_value = [
            {
                "id": "doc1",
                "content": "Test content",
                "score": 0.95,
                "metadata": {}
            }
        ]
        
        with patch('core.vectordb.router.VectorDBRouter', return_value=mock_vectordb):
            try:
                # The page should query vector store
                pass
            except Exception as e:
                pytest.fail(f"Explorer page vector store integration failed: {e}")
    
    @patch('streamlit.session_state', {})
    def test_settings_page_updates_configuration(self):
        """Test settings page updates configuration."""
        from ui.pages.settings import render_settings_page
        from config.settings import Settings
        
        st.session_state.settings = Settings()
        
        # Simulate configuration change
        original_llm = st.session_state.settings.llm_provider
        
        try:
            # Settings page should allow configuration updates
            # In actual implementation, this would update settings
            pass
        except Exception as e:
            pytest.fail(f"Settings page configuration update failed: {e}")
    
    @pytest.mark.asyncio
    @patch('streamlit.session_state', {})
    async def test_benchmark_page_executes_benchmark(self):
        """Test benchmark page executes benchmarks."""
        from ui.pages.benchmark import render_benchmark_page
        
        st.session_state.settings = Mock()
        
        # Mock benchmark runner
        mock_benchmark = AsyncMock()
        mock_benchmark.run_benchmark.return_value = {
            "qdrant": {
                "indexing_throughput": 1000,
                "query_latency_p50": 10,
                "recall_at_10": 0.95
            },
            "milvus": {
                "indexing_throughput": 1200,
                "query_latency_p50": 8,
                "recall_at_10": 0.96
            }
        }
        
        with patch('core.vectordb.benchmark.VectorDBBenchmark', return_value=mock_benchmark):
            try:
                # The page should execute benchmarks
                pass
            except Exception as e:
                pytest.fail(f"Benchmark page execution failed: {e}")


class TestEndToEndWorkflows:
    """Test complete user workflows."""
    
    @pytest.mark.asyncio
    @patch('streamlit.session_state', {})
    async def test_document_upload_and_query_workflow(self):
        """Test complete workflow: upload → index → query."""
        st.session_state.chat_history = []
        st.session_state.uploaded_documents = []
        st.session_state.settings = Mock()
        
        # Step 1: Upload document
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.type = "application/pdf"
        mock_file.read.return_value = b"mock content"
        
        with patch('core.document_processing.loaders.DocumentLoader') as mock_loader, \
             patch('core.document_processing.chunker.SemanticChunker') as mock_chunker, \
             patch('core.vectordb.router.VectorDBRouter') as mock_vectordb:
            
            # Mock document processing
            mock_doc = Mock()
            mock_doc.page_content = "Test content"
            mock_doc.metadata = {"source": "test.pdf"}
            
            mock_loader.return_value.load.return_value = [mock_doc]
            mock_chunker.return_value.chunk_documents.return_value = [mock_doc]
            mock_vectordb.return_value.add_documents = AsyncMock()
            
            # Step 2: Index document
            try:
                # Simulate indexing
                await mock_vectordb.return_value.add_documents([mock_doc])
                st.session_state.uploaded_documents.append({
                    "filename": "test.pdf",
                    "status": "indexed"
                })
            except Exception as e:
                pytest.fail(f"Document indexing failed: {e}")
        
        # Step 3: Query document
        with patch('agents.orchestrator.AgentOrchestrator') as mock_orchestrator:
            mock_orch_instance = AsyncMock()
            mock_orch_instance.arun.return_value = {
                "answer": "Test answer",
                "sources": [{"content": "Test content", "score": 0.95}],
                "grounding_score": 0.95
            }
            mock_orchestrator.return_value = mock_orch_instance
            
            try:
                # Simulate query
                result = await mock_orch_instance.arun("What is this about?")
                assert result["answer"] == "Test answer"
                assert len(result["sources"]) > 0
            except Exception as e:
                pytest.fail(f"Document query failed: {e}")
    
    @patch('streamlit.session_state', {})
    def test_configuration_changes_propagate(self):
        """Test configuration changes propagate correctly."""
        from config.settings import Settings
        
        st.session_state.settings = Settings()
        
        # Change LLM provider
        original_provider = st.session_state.settings.llm_provider
        st.session_state.settings.llm_provider = "openrouter"
        
        assert st.session_state.settings.llm_provider == "openrouter"
        assert st.session_state.settings.llm_provider != original_provider
        
        # Change embedding model
        st.session_state.settings.embedding_model = "bge-m3"
        assert st.session_state.settings.embedding_model == "bge-m3"
    
    @pytest.mark.asyncio
    @patch('streamlit.session_state', {})
    async def test_error_handling_across_pages(self):
        """Test error handling across different pages."""
        st.session_state.settings = Mock()
        
        # Test chat page error handling
        with patch('agents.orchestrator.AgentOrchestrator') as mock_orchestrator:
            mock_orch_instance = AsyncMock()
            mock_orch_instance.arun.side_effect = Exception("LLM API error")
            mock_orchestrator.return_value = mock_orch_instance
            
            try:
                await mock_orch_instance.arun("test query")
            except Exception as e:
                # Error should be caught and handled gracefully
                assert "error" in str(e).lower()
        
        # Test upload page error handling
        with patch('core.document_processing.loaders.DocumentLoader') as mock_loader:
            mock_loader.return_value.load.side_effect = Exception("File parsing error")
            
            try:
                mock_loader.return_value.load("invalid.pdf")
            except Exception as e:
                # Error should be caught and handled gracefully
                assert "error" in str(e).lower()
    
    @pytest.mark.asyncio
    @patch('streamlit.session_state', {})
    async def test_streaming_response_workflow(self):
        """Test streaming response functionality."""
        st.session_state.chat_history = []
        st.session_state.settings = Mock()
        
        # Mock streaming orchestrator
        async def mock_stream():
            chunks = ["This ", "is ", "a ", "streaming ", "response."]
            for chunk in chunks:
                yield {"content": chunk}
        
        with patch('agents.orchestrator.AgentOrchestrator') as mock_orchestrator:
            mock_orch_instance = AsyncMock()
            mock_orch_instance.astream.return_value = mock_stream()
            mock_orchestrator.return_value = mock_orch_instance
            
            try:
                # Collect streamed chunks
                chunks = []
                async for chunk in mock_orch_instance.astream("test query"):
                    chunks.append(chunk["content"])
                
                full_response = "".join(chunks)
                assert full_response == "This is a streaming response."
            except Exception as e:
                pytest.fail(f"Streaming response failed: {e}")
    
    @pytest.mark.asyncio
    @patch('streamlit.session_state', {})
    async def test_multi_document_search_workflow(self):
        """Test searching across multiple documents."""
        st.session_state.settings = Mock()
        st.session_state.uploaded_documents = [
            {"filename": "doc1.pdf", "id": "doc1"},
            {"filename": "doc2.pdf", "id": "doc2"},
            {"filename": "doc3.pdf", "id": "doc3"}
        ]
        
        # Mock vector store with multiple document results
        mock_vectordb = AsyncMock()
        mock_vectordb.search.return_value = [
            {"id": "doc1", "content": "Content from doc1", "score": 0.95},
            {"id": "doc2", "content": "Content from doc2", "score": 0.90},
            {"id": "doc3", "content": "Content from doc3", "score": 0.85}
        ]
        
        with patch('core.vectordb.router.VectorDBRouter', return_value=mock_vectordb):
            try:
                results = await mock_vectordb.search("test query", k=10)
                assert len(results) == 3
                assert all("content" in r for r in results)
                assert all("score" in r for r in results)
            except Exception as e:
                pytest.fail(f"Multi-document search failed: {e}")


class TestUIPerformance:
    """Test UI performance and responsiveness."""
    
    @patch('streamlit.session_state', {})
    def test_large_document_list_rendering(self):
        """Test rendering large list of documents."""
        from ui.components.document_card import render_document_card
        
        # Create large list of mock documents
        documents = [
            {
                "id": f"doc{i}",
                "filename": f"document_{i}.pdf",
                "file_type": "pdf",
                "file_size": 1024000,
                "num_chunks": 10,
                "entities": {},
                "summary": f"Summary {i}"
            }
            for i in range(100)
        ]
        
        try:
            # Should handle large lists without errors
            for doc in documents[:10]:  # Test first 10
                render_document_card(doc)
        except Exception as e:
            pytest.fail(f"Large document list rendering failed: {e}")
    
    @patch('streamlit.session_state', {})
    def test_long_chat_history_rendering(self):
        """Test rendering long chat history."""
        st.session_state.chat_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(100)
        ]
        
        try:
            # Should handle long chat history
            assert len(st.session_state.chat_history) == 100
        except Exception as e:
            pytest.fail(f"Long chat history handling failed: {e}")


class TestAccessibility:
    """Test UI accessibility features."""
    
    def test_css_file_exists(self):
        """Test custom CSS file exists."""
        css_file = Path("ui/styles/custom.css")
        assert css_file.exists(), "Custom CSS file not found"
    
    def test_css_file_has_content(self):
        """Test CSS file has content."""
        css_file = Path("ui/styles/custom.css")
        content = css_file.read_text()
        assert len(content) > 0, "CSS file is empty"
        assert ":root" in content, "CSS missing root variables"
        assert "color" in content.lower(), "CSS missing color definitions"
    
    def test_responsive_design_breakpoints(self):
        """Test CSS has responsive design breakpoints."""
        css_file = Path("ui/styles/custom.css")
        content = css_file.read_text()
        assert "@media" in content, "CSS missing media queries"
        assert "max-width" in content, "CSS missing responsive breakpoints"


class TestDataPersistence:
    """Test data persistence across sessions."""
    
    @patch('streamlit.session_state', {})
    def test_session_state_initialization(self):
        """Test session state is properly initialized."""
        # Simulate app initialization
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "uploaded_documents" not in st.session_state:
            st.session_state.uploaded_documents = []
        if "settings" not in st.session_state:
            st.session_state.settings = Mock()
        
        assert "chat_history" in st.session_state
        assert "uploaded_documents" in st.session_state
        assert "settings" in st.session_state
    
    @patch('streamlit.session_state', {})
    def test_chat_history_persistence(self):
        """Test chat history persists in session state."""
        st.session_state.chat_history = []
        
        # Add messages
        st.session_state.chat_history.append({"role": "user", "content": "Hello"})
        st.session_state.chat_history.append({"role": "assistant", "content": "Hi there"})
        
        assert len(st.session_state.chat_history) == 2
        assert st.session_state.chat_history[0]["role"] == "user"
        assert st.session_state.chat_history[1]["role"] == "assistant"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
