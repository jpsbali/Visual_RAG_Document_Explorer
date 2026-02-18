"""
Tests for Phase 2 benchmarking utilities.
"""

import pytest
from config.settings import Settings
from core.vectordb.benchmark import VectorDBBenchmark
from unittest.mock import Mock, AsyncMock, patch


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        qdrant_url="http://localhost:6333",
        milvus_url="http://localhost:19530"
    )


@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration"),
    reason="Requires Qdrant and Milvus running"
)
@pytest.mark.asyncio
class TestVectorDBBenchmark:
    """Tests for vector database benchmarking."""
    
    async def test_initialization(self, settings):
        """Test benchmark initialization."""
        benchmark = VectorDBBenchmark(settings)
        assert benchmark.qdrant is not None
        assert benchmark.milvus is not None
        
    async def test_generate_test_data(self, settings):
        """Test generating test data."""
        benchmark = VectorDBBenchmark(settings)
        
        test_data = benchmark._generate_test_data(
            num_documents=100,
            dimension=1024
        )
        
        assert len(test_data["ids"]) == 100
        assert len(test_data["embeddings"]) == 100
        assert len(test_data["documents"]) == 100
        assert len(test_data["metadatas"]) == 100
        assert len(test_data["embeddings"][0]) == 1024
        
    async def test_generate_test_queries(self, settings):
        """Test generating test queries."""
        benchmark = VectorDBBenchmark(settings)
        
        test_queries = benchmark._generate_test_queries(
            num_queries=50,
            dimension=1024
        )
        
        assert len(test_queries) == 50
        assert len(test_queries[0]) == 1024
        
    async def test_measure_memory(self, settings):
        """Test measuring memory usage."""
        benchmark = VectorDBBenchmark(settings)
        
        memory_mb = benchmark._measure_memory()
        
        assert memory_mb > 0
        assert isinstance(memory_mb, float)
        
    @pytest.mark.slow
    async def test_run_benchmark_small(self, settings):
        """Test running a small benchmark."""
        benchmark = VectorDBBenchmark(settings)
        
        # Run with small dataset
        results = await benchmark.run_benchmark(
            num_documents=10,
            dimension=128,
            num_queries=5
        )
        
        assert "qdrant" in results
        assert "milvus" in results
        
        # Check Qdrant results
        qdrant_result = results["qdrant"]
        assert qdrant_result.db_name == "qdrant"
        assert qdrant_result.num_documents == 10
        assert qdrant_result.throughput_ops_per_sec > 0
        assert qdrant_result.latency_p50_ms > 0
        
        # Check Milvus results
        milvus_result = results["milvus"]
        assert milvus_result.db_name == "milvus"
        assert milvus_result.num_documents == 10
        assert milvus_result.throughput_ops_per_sec > 0
        assert milvus_result.latency_p50_ms > 0


class TestBenchmarkMocked:
    """Tests for benchmark with mocked stores."""
    
    @pytest.mark.asyncio
    async def test_benchmark_indexing(self, settings):
        """Test benchmarking indexing performance."""
        benchmark = VectorDBBenchmark(settings)
        
        # Mock store
        mock_store = Mock()
        mock_store.upsert = AsyncMock(return_value=10)
        
        test_data = benchmark._generate_test_data(10, 128)
        
        result = await benchmark._benchmark_indexing(
            store=mock_store,
            collection="test",
            test_data=test_data
        )
        
        assert "elapsed" in result
        assert "throughput" in result
        assert result["throughput"] > 0
        
    @pytest.mark.asyncio
    async def test_benchmark_search(self, settings):
        """Test benchmarking search performance."""
        benchmark = VectorDBBenchmark(settings)
        
        # Mock store
        mock_store = Mock()
        mock_store.search = AsyncMock(return_value=[])
        
        test_queries = benchmark._generate_test_queries(5, 128)
        
        result = await benchmark._benchmark_search(
            store=mock_store,
            collection="test",
            test_queries=test_queries
        )
        
        assert "p50" in result
        assert "p95" in result
        assert "p99" in result
        assert len(result["latencies"]) == 5


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services"
    )
