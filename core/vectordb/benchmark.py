"""
Vector database benchmarking utilities.

Compares Qdrant vs Milvus performance on indexing, search, and memory usage.
"""

from config.models import BenchmarkResult
from config.settings import Settings
from .base import VectorStoreBase
from .qdrant_store import QdrantStore
from .milvus_store import MilvusStore
import time
import psutil
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class VectorDBBenchmark:
    """
    Benchmark vector database performance.
    
    Measures:
    - Indexing throughput (docs/sec)
    - Query latency (p50, p95, p99)
    - Recall@k
    - Memory usage
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize benchmark.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.qdrant = QdrantStore(settings)
        self.milvus = MilvusStore(settings)
        logger.info("Initialized VectorDBBenchmark")
        
    async def run_benchmark(
        self,
        num_documents: int = 1000,
        dimension: int = 1024,
        num_queries: int = 100
    ) -> dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmark on both databases.
        
        Args:
            num_documents: Number of documents to index
            dimension: Vector dimension
            num_queries: Number of search queries to run
            
        Returns:
            Dict mapping db_name to BenchmarkResult
        """
        logger.info(
            f"Starting benchmark: {num_documents} docs, "
            f"{dimension}D, {num_queries} queries"
        )
        
        # Generate test data
        test_data = self._generate_test_data(num_documents, dimension)
        test_queries = self._generate_test_queries(num_queries, dimension)
        
        results = {}
        
        # Benchmark Qdrant
        logger.info("Benchmarking Qdrant...")
        results["qdrant"] = await self._benchmark_store(
            store=self.qdrant,
            db_name="qdrant",
            test_data=test_data,
            test_queries=test_queries
        )
        
        # Benchmark Milvus
        logger.info("Benchmarking Milvus...")
        results["milvus"] = await self._benchmark_store(
            store=self.milvus,
            db_name="milvus",
            test_data=test_data,
            test_queries=test_queries
        )
        
        # Log comparison
        self._log_comparison(results)
        
        return results
        
    async def _benchmark_store(
        self,
        store: VectorStoreBase,
        db_name: str,
        test_data: dict,
        test_queries: list[list[float]]
    ) -> BenchmarkResult:
        """
        Benchmark a single vector store.
        
        Args:
            store: Vector store instance
            db_name: Database name
            test_data: Test data dict
            test_queries: Test query vectors
            
        Returns:
            BenchmarkResult object
        """
        collection_name = f"benchmark_{db_name}_{int(time.time())}"
        
        try:
            # Create collection
            await store.create_collection(
                name=collection_name,
                dimension=len(test_data["embeddings"][0])
            )
            
            # Benchmark indexing
            index_result = await self._benchmark_indexing(
                store=store,
                collection=collection_name,
                test_data=test_data
            )
            
            # Benchmark search
            search_result = await self._benchmark_search(
                store=store,
                collection=collection_name,
                test_queries=test_queries
            )
            
            # Measure memory
            memory_mb = self._measure_memory()
            
            # Cleanup
            await store.delete_collection(collection_name)
            
            return BenchmarkResult(
                db_name=db_name,
                operation="combined",
                num_documents=len(test_data["ids"]),
                latency_p50_ms=search_result["p50"],
                latency_p95_ms=search_result["p95"],
                latency_p99_ms=search_result["p99"],
                throughput_ops_per_sec=index_result["throughput"],
                memory_usage_mb=memory_mb,
                recall_at_k=search_result.get("recall_at_k"),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Benchmark failed for {db_name}: {e}")
            raise
        
    async def _benchmark_indexing(
        self,
        store: VectorStoreBase,
        collection: str,
        test_data: dict
    ) -> dict:
        """
        Benchmark indexing performance.
        
        Args:
            store: Vector store instance
            collection: Collection name
            test_data: Test data dict
            
        Returns:
            Dict with indexing metrics
        """
        start_time = time.time()
        
        await store.upsert(
            collection=collection,
            ids=test_data["ids"],
            embeddings=test_data["embeddings"],
            documents=test_data["documents"],
            metadatas=test_data["metadatas"]
        )
        
        elapsed = time.time() - start_time
        throughput = len(test_data["ids"]) / elapsed
        
        logger.info(
            f"Indexed {len(test_data['ids'])} docs in {elapsed:.2f}s "
            f"({throughput:.1f} docs/sec)"
        )
        
        return {
            "elapsed": elapsed,
            "throughput": throughput
        }
        
    async def _benchmark_search(
        self,
        store: VectorStoreBase,
        collection: str,
        test_queries: list[list[float]]
    ) -> dict:
        """
        Benchmark search performance.
        
        Args:
            store: Vector store instance
            collection: Collection name
            test_queries: Test query vectors
            
        Returns:
            Dict with search metrics
        """
        latencies = []
        
        for query_embedding in test_queries:
            start_time = time.time()
            
            await store.search(
                collection=collection,
                query_embedding=query_embedding,
                top_k=10
            )
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
            
        # Calculate percentiles
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        
        logger.info(
            f"Search latency - p50: {p50:.2f}ms, "
            f"p95: {p95:.2f}ms, p99: {p99:.2f}ms"
        )
        
        return {
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "latencies": latencies
        }
        
    def _generate_test_data(
        self, 
        num_documents: int, 
        dimension: int
    ) -> dict:
        """
        Generate synthetic test data.
        
        Args:
            num_documents: Number of documents
            dimension: Vector dimension
            
        Returns:
            Dict with test data
        """
        return {
            "ids": [f"doc_{i}" for i in range(num_documents)],
            "embeddings": [
                np.random.rand(dimension).tolist() 
                for _ in range(num_documents)
            ],
            "documents": [
                f"Test document {i} with some content" 
                for i in range(num_documents)
            ],
            "metadatas": [
                {
                    "chunk_id": f"doc_{i}",
                    "source_file": "test.pdf",
                    "file_type": "pdf",
                    "chunk_index": i,
                    "total_chunks": num_documents,
                    "chunk_type": "content",
                    "chunk_method": "test",
                    "chunk_size": 512,
                    "token_count": 100,
                    "char_count": 500,
                    "content_hash": f"hash_{i}",
                    "content_preview": f"Preview {i}",
                    "entities": {
                        "people": [],
                        "organizations": [],
                        "dates": [],
                        "locations": [],
                        "topics": [],
                        "custom": {},
                        "extractor": "ensemble",
                        "confidence_scores": {}
                    },
                    "keywords": [],
                    "created_at": datetime.now().isoformat(),
                    "processed_at": datetime.now().isoformat()
                }
                for i in range(num_documents)
            ]
        }
        
    def _generate_test_queries(
        self, 
        num_queries: int, 
        dimension: int
    ) -> list[list[float]]:
        """
        Generate synthetic query vectors.
        
        Args:
            num_queries: Number of queries
            dimension: Vector dimension
            
        Returns:
            List of query vectors
        """
        return [
            np.random.rand(dimension).tolist() 
            for _ in range(num_queries)
        ]
        
    def _measure_memory(self) -> float:
        """
        Measure current process memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
        
    def _log_comparison(self, results: dict[str, BenchmarkResult]) -> None:
        """
        Log comparison between benchmark results.
        
        Args:
            results: Dict of benchmark results
        """
        if "qdrant" in results and "milvus" in results:
            qdrant = results["qdrant"]
            milvus = results["milvus"]
            
            logger.info("=" * 60)
            logger.info("BENCHMARK COMPARISON")
            logger.info("=" * 60)
            logger.info(f"Indexing Throughput:")
            logger.info(f"  Qdrant: {qdrant.throughput_ops_per_sec:.1f} docs/sec")
            logger.info(f"  Milvus: {milvus.throughput_ops_per_sec:.1f} docs/sec")
            logger.info(f"Search Latency (p50):")
            logger.info(f"  Qdrant: {qdrant.latency_p50_ms:.2f}ms")
            logger.info(f"  Milvus: {milvus.latency_p50_ms:.2f}ms")
            logger.info(f"Search Latency (p95):")
            logger.info(f"  Qdrant: {qdrant.latency_p95_ms:.2f}ms")
            logger.info(f"  Milvus: {milvus.latency_p95_ms:.2f}ms")
            logger.info(f"Memory Usage:")
            logger.info(f"  Qdrant: {qdrant.memory_usage_mb:.1f}MB")
            logger.info(f"  Milvus: {milvus.memory_usage_mb:.1f}MB")
            logger.info("=" * 60)
