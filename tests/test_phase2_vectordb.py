"""
Tests for Phase 2 vector database implementations.

Tests both Qdrant and Milvus stores with the common interface.
"""

import pytest
import asyncio
from datetime import datetime
from config.settings import Settings
from config.models import ChunkMetadata, NEREntities
from core.vectordb.qdrant_store import QdrantStore
from core.vectordb.milvus_store import MilvusStore
from core.vectordb.router import get_vector_store, VectorDBRouter


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        qdrant_url="http://localhost:6333",
        milvus_url="http://localhost:19530",
        default_vector_db="qdrant"
    )


@pytest.fixture
def sample_metadata():
    """Create sample chunk metadata."""
    return ChunkMetadata(
        chunk_id="test_chunk_1",
        source_file="test.pdf",
        file_type="pdf",
        page_number=1,
        chunk_index=0,
        total_chunks=10,
        chunk_type="content",
        chunk_method="recursive",
        chunk_size=512,
        token_count=100,
        char_count=500,
        content_hash="abc123",
        content_preview="This is a test chunk",
        entities=NEREntities(
            people=["John Doe"],
            organizations=["Acme Corp"],
            dates=["2024-01-01"],
            locations=["New York"],
            topics=["technology"]
        ),
        keywords=["test", "sample"],
        created_at=datetime.now(),
        processed_at=datetime.now()
    )


@pytest.fixture
def test_data(sample_metadata):
    """Create test data for vector store operations."""
    return {
        "ids": ["chunk_1", "chunk_2", "chunk_3"],
        "embeddings": [
            [0.1] * 1024,
            [0.2] * 1024,
            [0.3] * 1024
        ],
        "documents": [
            "This is the first test document",
            "This is the second test document",
            "This is the third test document"
        ],
        "metadatas": [
            sample_metadata.model_dump(),
            sample_metadata.model_dump(),
            sample_metadata.model_dump()
        ]
    }


class TestVectorStoreBase:
    """Base test class for vector store implementations."""
    
    @pytest.mark.asyncio
    async def test_create_collection(self, store, test_collection):
        """Test collection creation."""
        await store.create_collection(
            name=test_collection,
            dimension=1024,
            distance_metric="cosine"
        )
        
        collections = await store.list_collections()
        assert test_collection in collections
        
    @pytest.mark.asyncio
    async def test_upsert_and_count(self, store, test_collection, test_data):
        """Test document upsert and count."""
        await store.create_collection(test_collection, dimension=1024)
        
        count = await store.upsert(
            collection=test_collection,
            ids=test_data["ids"],
            embeddings=test_data["embeddings"],
            documents=test_data["documents"],
            metadatas=test_data["metadatas"]
        )
        
        assert count == 3
        
        total = await store.count(test_collection)
        assert total == 3
        
    @pytest.mark.asyncio
    async def test_search(self, store, test_collection, test_data):
        """Test vector search."""
        await store.create_collection(test_collection, dimension=1024)
        await store.upsert(
            collection=test_collection,
            ids=test_data["ids"],
            embeddings=test_data["embeddings"],
            documents=test_data["documents"],
            metadatas=test_data["metadatas"]
        )
        
        # Search with query vector
        query_embedding = [0.15] * 1024
        results = await store.search(
            collection=test_collection,
            query_embedding=query_embedding,
            top_k=2
        )
        
        assert len(results) <= 2
        assert all(hasattr(r, "content") for r in results)
        assert all(hasattr(r, "metadata") for r in results)
        assert all(hasattr(r, "score") for r in results)
        
    @pytest.mark.asyncio
    async def test_search_with_filters(self, store, test_collection, test_data):
        """Test vector search with metadata filters."""
        await store.create_collection(test_collection, dimension=1024)
        await store.upsert(
            collection=test_collection,
            ids=test_data["ids"],
            embeddings=test_data["embeddings"],
            documents=test_data["documents"],
            metadatas=test_data["metadatas"]
        )
        
        # Search with filters
        query_embedding = [0.15] * 1024
        results = await store.search(
            collection=test_collection,
            query_embedding=query_embedding,
            top_k=10,
            filters={"organizations": ["Acme Corp"]}
        )
        
        assert len(results) > 0
        
    @pytest.mark.asyncio
    async def test_get(self, store, test_collection, test_data):
        """Test document retrieval by ID."""
        await store.create_collection(test_collection, dimension=1024)
        await store.upsert(
            collection=test_collection,
            ids=test_data["ids"],
            embeddings=test_data["embeddings"],
            documents=test_data["documents"],
            metadatas=test_data["metadatas"]
        )
        
        docs = await store.get(
            collection=test_collection,
            ids=["chunk_1", "chunk_2"]
        )
        
        assert len(docs) == 2
        
    @pytest.mark.asyncio
    async def test_delete(self, store, test_collection, test_data):
        """Test document deletion."""
        await store.create_collection(test_collection, dimension=1024)
        await store.upsert(
            collection=test_collection,
            ids=test_data["ids"],
            embeddings=test_data["embeddings"],
            documents=test_data["documents"],
            metadatas=test_data["metadatas"]
        )
        
        deleted = await store.delete(
            collection=test_collection,
            ids=["chunk_1"]
        )
        
        assert deleted == 1
        
        total = await store.count(test_collection)
        assert total == 2
        
    @pytest.mark.asyncio
    async def test_health_check(self, store):
        """Test health check."""
        is_healthy = await store.health_check()
        assert is_healthy is True
        
    @pytest.mark.asyncio
    async def test_delete_collection(self, store, test_collection):
        """Test collection deletion."""
        await store.create_collection(test_collection, dimension=1024)
        await store.delete_collection(test_collection)
        
        collections = await store.list_collections()
        assert test_collection not in collections


@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration"),
    reason="Requires Qdrant running"
)
class TestQdrantStore(TestVectorStoreBase):
    """Tests for Qdrant store implementation."""
    
    @pytest.fixture
    def store(self, settings):
        """Create Qdrant store instance."""
        return QdrantStore(settings)
        
    @pytest.fixture
    def test_collection(self):
        """Generate unique test collection name."""
        return f"test_qdrant_{int(datetime.now().timestamp())}"


@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration"),
    reason="Requires Milvus running"
)
class TestMilvusStore(TestVectorStoreBase):
    """Tests for Milvus store implementation."""
    
    @pytest.fixture
    def store(self, settings):
        """Create Milvus store instance."""
        return MilvusStore(settings)
        
    @pytest.fixture
    def test_collection(self):
        """Generate unique test collection name."""
        return f"test_milvus_{int(datetime.now().timestamp())}"


class TestVectorDBRouter:
    """Tests for vector database router."""
    
    def test_get_vector_store_qdrant(self, settings):
        """Test getting Qdrant store."""
        settings.default_vector_db = "qdrant"
        store = get_vector_store(settings)
        assert isinstance(store, QdrantStore)
        
    def test_get_vector_store_milvus(self, settings):
        """Test getting Milvus store."""
        settings.default_vector_db = "milvus"
        store = get_vector_store(settings)
        assert isinstance(store, MilvusStore)
        
    def test_get_vector_store_invalid(self, settings):
        """Test getting invalid store."""
        settings.default_vector_db = "invalid"
        with pytest.raises(ValueError):
            get_vector_store(settings)
            
    def test_router_initialization(self, settings):
        """Test router initialization."""
        router = VectorDBRouter(settings)
        assert router.active_store == settings.default_vector_db
        assert "qdrant" in router.stores
        assert "milvus" in router.stores
        
    def test_router_get_store(self, settings):
        """Test getting store from router."""
        router = VectorDBRouter(settings)
        
        qdrant = router.get_store("qdrant")
        assert isinstance(qdrant, QdrantStore)
        
        milvus = router.get_store("milvus")
        assert isinstance(milvus, MilvusStore)
        
    def test_router_switch_backend(self, settings):
        """Test switching backend."""
        router = VectorDBRouter(settings)
        
        router.switch_backend("milvus")
        assert router.get_active_backend() == "milvus"
        
        router.switch_backend("qdrant")
        assert router.get_active_backend() == "qdrant"
        
    def test_router_get_available_backends(self, settings):
        """Test getting available backends."""
        router = VectorDBRouter(settings)
        backends = router.get_available_backends()
        
        assert "qdrant" in backends
        assert "milvus" in backends


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services"
    )
