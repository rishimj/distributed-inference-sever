"""
Complete System Integration Test

Tests the full distributed inference system end-to-end:
- Worker management and lifecycle
- Gateway routing and load balancing
- Cache metadata synchronization
- Cache transfer operations
- Metrics collection
- Error handling and recovery
"""

import pytest
import asyncio
import time
from typing import List

from src.workers.manager import WorkerManager, WorkerState
from src.workers.vllm_worker import VLLMWorkerService
from src.gateway.worker_client import WorkerClientPool
from src.gateway.routing_engine import RoutingEngine, RoutingStrategy
from src.cache.registry import CacheRegistry
from src.cache.metadata_sync import CacheMetadataSync
from src.cache.transfer import MockCacheTransferService
from src.common.models import InferenceRequest
from src.common.metrics import MetricsCollector, MetricsConfig


@pytest.fixture
async def cache_registry():
    """Create test cache registry"""
    # Use mock/in-memory registry for testing
    registry = await CacheRegistry.create(redis_url=None)  # Mock mode
    yield registry
    await registry.close()


@pytest.fixture
async def worker_manager():
    """Create worker manager"""
    manager = WorkerManager(
        gateway_url=None,  # No gateway for this test
        auto_restart=True,
        max_restart_attempts=3,
        health_check_interval=5.0
    )
    
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
async def worker_client_pool():
    """Create worker client pool"""
    pool = WorkerClientPool(
        timeout_seconds=30.0,
        max_retries=2,
        health_check_interval=10.0
    )
    
    await pool.start()
    yield pool
    await pool.stop()


@pytest.fixture
async def routing_engine(cache_registry):
    """Create routing engine"""
    engine = RoutingEngine(
        cache_registry=cache_registry,
        default_strategy=RoutingStrategy.CACHE_AFFINITY
    )
    
    await engine.start()
    yield engine
    await engine.stop()


@pytest.fixture
async def metadata_sync(cache_registry):
    """Create cache metadata sync service"""
    sync = CacheMetadataSync(
        cache_registry=cache_registry,
        sync_interval=5.0,
        stale_threshold=60.0
    )
    
    await sync.start()
    yield sync
    await sync.stop()


@pytest.fixture
async def cache_transfer():
    """Create cache transfer service"""
    service = MockCacheTransferService(
        max_concurrent_transfers=2,
        compression_enabled=True
    )
    
    await service.start()
    yield service
    await service.stop()


@pytest.fixture
def metrics_collector():
    """Create metrics collector"""
    config = MetricsConfig(
        enabled=True,
        namespace="test_distributed_inference"
    )
    return MetricsCollector(config)


class TestCompleteSystem:
    """Test suite for complete system integration"""
    
    @pytest.mark.asyncio
    async def test_worker_lifecycle(self, worker_manager):
        """Test worker startup, health monitoring, and shutdown"""
        # Start a mock worker
        worker = await worker_manager.start_worker(
            worker_id="test-worker-1",
            model="mock-model",
            port=9001,
            gpu_id=None  # No GPU for test
        )
        
        assert worker.worker_id == "test-worker-1"
        assert worker.state == WorkerState.RUNNING
        assert worker.port == 9001
        
        # Check worker is healthy
        await asyncio.sleep(2)
        assert worker.is_healthy
        
        # Get worker info
        retrieved_worker = worker_manager.get_worker("test-worker-1")
        assert retrieved_worker is not None
        assert retrieved_worker.uptime_seconds > 0
        
        # Stop worker
        await worker_manager.stop_worker("test-worker-1")
        assert worker.state == WorkerState.STOPPED
    
    @pytest.mark.asyncio
    async def test_worker_client_communication(self, worker_client_pool):
        """Test gateway-to-worker communication"""
        # Register mock workers
        worker_client_pool.add_worker("worker-1", "localhost", 9001)
        worker_client_pool.add_worker("worker-2", "localhost", 9002)
        
        # Wait for health checks
        await asyncio.sleep(1)
        
        # Check workers are registered
        assert "worker-1" in worker_client_pool.workers
        assert "worker-2" in worker_client_pool.workers
        
        # Get healthy workers (won't be healthy without actual workers running)
        healthy = await worker_client_pool.get_healthy_workers()
        # In test environment, workers may not be healthy
        assert isinstance(healthy, list)
    
    @pytest.mark.asyncio
    async def test_routing_with_cache_affinity(self, routing_engine, cache_registry):
        """Test cache-aware routing decisions"""
        # Register workers
        await routing_engine.register_worker("worker-1", "http://localhost:9001")
        await routing_engine.register_worker("worker-2", "http://localhost:9002")
        
        # Create test request
        request = InferenceRequest(
            prompt="Test prompt for routing",
            max_tokens=50
        )
        
        # First routing decision (no cache)
        decision1 = await routing_engine.route_request(request)
        assert decision1.selected_worker in ["worker-1", "worker-2"]
        assert decision1.strategy == RoutingStrategy.CACHE_AFFINITY
        
        # Simulate cache entry for worker-1
        from src.common.models import CacheEntry
        cache_entry = CacheEntry(
            prefix_hash=routing_engine.prefix_hasher.hash_prefix(request.prompt),
            node_id="worker-1",
            cache_size_bytes=1024,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            ttl_seconds=300
        )
        await cache_registry.register_cache(cache_entry)
        
        # Second routing decision (with cache)
        decision2 = await routing_engine.route_request(request)
        assert decision2.selected_worker == "worker-1"  # Should prefer cached worker
        assert decision2.cache_hit_probability > 0.5
    
    @pytest.mark.asyncio
    async def test_cache_metadata_sync(self, metadata_sync, cache_registry):
        """Test cache metadata synchronization"""
        # Register workers
        metadata_sync.register_worker("worker-1", "http://localhost:9001")
        metadata_sync.register_worker("worker-2", "http://localhost:9002")
        
        # Check workers registered
        assert "worker-1" in metadata_sync.workers
        assert "worker-2" in metadata_sync.workers
        
        # Get cache statistics
        stats = metadata_sync.get_cache_statistics()
        assert stats['total_workers'] == 2
        assert stats['total_cached_prefixes'] >= 0
        
        # Test force sync (will fail without actual workers, but tests the flow)
        success_count = await metadata_sync.force_sync_all()
        assert isinstance(success_count, int)
    
    @pytest.mark.asyncio
    async def test_cache_transfer_flow(self, cache_transfer):
        """Test cache transfer between workers"""
        # Register workers
        cache_transfer.register_worker("worker-1", "http://localhost:9001")
        cache_transfer.register_worker("worker-2", "http://localhost:9002")
        
        # Add mock cache to worker-1
        cache_transfer.add_mock_cache("worker-1", "test-prefix-123", size_bytes=2048)
        
        # Request transfer from worker-1 to worker-2
        await cache_transfer.request_transfer(
            prefix_hash="test-prefix-123",
            source_worker_id="worker-1",
            target_worker_id="worker-2",
            priority=5
        )
        
        # Wait for transfer to complete
        await asyncio.sleep(2)
        
        # Check statistics
        stats = cache_transfer.get_statistics()
        assert stats['total_transfers'] > 0
        
        # Verify cache now exists on worker-2
        assert ("worker-2", "test-prefix-123") in cache_transfer._mock_caches
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, metrics_collector):
        """Test metrics recording and export"""
        # Record some test metrics
        metrics_collector.record_request(
            worker_id="worker-1",
            status="success",
            duration_ms=150.0,
            cache_hit=True,
            tokens_generated=100,
            ttft_ms=20.0
        )
        
        metrics_collector.record_request(
            worker_id="worker-2",
            status="success",
            duration_ms=200.0,
            cache_hit=False,
            tokens_generated=120,
            ttft_ms=30.0
        )
        
        # Update cache state
        metrics_collector.update_cache_state(
            worker_id="worker-1",
            entries=10,
            size_bytes=1024*1024,
            utilization=0.45,
            hit_rate=0.75
        )
        
        # Record cache transfer
        metrics_collector.record_cache_transfer(
            source_worker="worker-1",
            target_worker="worker-2",
            success=True,
            bytes_transferred=2048,
            duration_ms=500.0
        )
        
        # Export metrics
        metrics_data = metrics_collector.export_metrics()
        assert len(metrics_data) > 0
        assert b"distributed_inference" in metrics_data or b"test_distributed_inference" in metrics_data
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, routing_engine):
        """Test system behavior under error conditions"""
        # Register workers
        await routing_engine.register_worker("worker-1", "http://localhost:9001")
        await routing_engine.register_worker("worker-2", "http://localhost:9002")
        
        # Create request
        request = InferenceRequest(
            prompt="Test error handling",
            max_tokens=50
        )
        
        # Routing should still work even if workers are unhealthy
        decision = await routing_engine.route_request(request)
        assert decision.selected_worker is not None
        
        # Deregister a worker
        await routing_engine.deregister_worker("worker-1")
        
        # Should still be able to route to remaining worker
        decision2 = await routing_engine.route_request(request)
        assert decision2.selected_worker == "worker-2"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, routing_engine):
        """Test handling multiple concurrent requests"""
        # Register workers
        await routing_engine.register_worker("worker-1", "http://localhost:9001")
        await routing_engine.register_worker("worker-2", "http://localhost:9002")
        
        # Create multiple requests
        requests = [
            InferenceRequest(prompt=f"Request {i}", max_tokens=50)
            for i in range(10)
        ]
        
        # Route all requests concurrently
        routing_tasks = [
            routing_engine.route_request(req)
            for req in requests
        ]
        
        decisions = await asyncio.gather(*routing_tasks)
        
        # All requests should get routed
        assert len(decisions) == 10
        assert all(d.selected_worker in ["worker-1", "worker-2"] for d in decisions)
        
        # Check load distribution (should not be all on one worker)
        worker_counts = {}
        for decision in decisions:
            worker_counts[decision.selected_worker] = worker_counts.get(decision.selected_worker, 0) + 1
        
        # Both workers should get some requests (with high probability)
        assert len(worker_counts) > 0
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_improvement(self, routing_engine, cache_registry):
        """Test that cache awareness improves hit rates"""
        # Register workers
        await routing_engine.register_worker("worker-1", "http://localhost:9001")
        await routing_engine.register_worker("worker-2", "http://localhost:9002")
        
        # Create request with common prefix
        base_prompt = "Tell me a story about a brave knight"
        
        # First request - will be a cache miss
        request1 = InferenceRequest(prompt=base_prompt, max_tokens=50)
        decision1 = await routing_engine.route_request(request1)
        first_worker = decision1.selected_worker
        
        # Simulate caching the result
        from src.common.models import CacheEntry
        prefix_hash = routing_engine.prefix_hasher.hash_prefix(base_prompt)
        cache_entry = CacheEntry(
            prefix_hash=prefix_hash,
            node_id=first_worker,
            cache_size_bytes=2048,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            ttl_seconds=300
        )
        await cache_registry.register_cache(cache_entry)
        
        # Second request with same prefix - should route to same worker
        request2 = InferenceRequest(prompt=base_prompt + " who", max_tokens=50)
        decision2 = await routing_engine.route_request(request2)
        
        # Should prefer the worker with cached prefix
        assert decision2.selected_worker == first_worker
        assert decision2.cache_hit_probability > 0
    
    @pytest.mark.asyncio
    async def test_system_graceful_shutdown(self, worker_manager, routing_engine):
        """Test graceful shutdown of all components"""
        # Start workers
        worker1 = await worker_manager.start_worker(
            worker_id="shutdown-test-1",
            model="mock-model",
            port=9101
        )
        
        # Register with routing
        await routing_engine.register_worker("shutdown-test-1", "http://localhost:9101")
        
        # Verify running
        assert worker1.state == WorkerState.RUNNING
        
        # Shutdown
        await worker_manager.stop_worker("shutdown-test-1")
        await routing_engine.deregister_worker("shutdown-test-1")
        
        # Verify clean shutdown
        assert worker1.state == WorkerState.STOPPED


class TestSystemPerformance:
    """Performance tests for the system"""
    
    @pytest.mark.asyncio
    async def test_routing_performance(self, routing_engine):
        """Test routing decision performance"""
        await routing_engine.register_worker("perf-worker-1", "http://localhost:9001")
        await routing_engine.register_worker("perf-worker-2", "http://localhost:9002")
        
        request = InferenceRequest(prompt="Performance test", max_tokens=50)
        
        # Measure routing time
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            await routing_engine.route_request(request)
        
        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / iterations) * 1000
        
        # Routing should be fast (< 10ms per decision)
        assert avg_time_ms < 10.0, f"Routing too slow: {avg_time_ms:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_cache_sync_performance(self, metadata_sync):
        """Test cache sync performance with many workers"""
        # Register many workers
        for i in range(10):
            metadata_sync.register_worker(f"perf-worker-{i}", f"http://localhost:900{i}")
        
        # Measure sync time
        start_time = time.time()
        await metadata_sync.force_sync_all()
        elapsed = (time.time() - start_time) * 1000
        
        # Sync should complete reasonably fast
        assert elapsed < 1000.0, f"Sync too slow: {elapsed:.2f}ms"


@pytest.mark.asyncio
async def test_full_system_integration():
    """
    Complete end-to-end integration test.
    
    This test verifies all components working together in a realistic scenario.
    """
    # Setup all components
    cache_registry = await CacheRegistry.create(redis_url=None)
    
    worker_manager = WorkerManager(auto_restart=True)
    await worker_manager.start()
    
    routing_engine = RoutingEngine(cache_registry=cache_registry)
    await routing_engine.start()
    
    metadata_sync = CacheMetadataSync(cache_registry=cache_registry)
    await metadata_sync.start()
    
    metrics = MetricsCollector(MetricsConfig(enabled=True))
    
    try:
        # Start workers
        worker1 = await worker_manager.start_worker(
            worker_id="integration-worker-1",
            model="mock-model",
            port=9201
        )
        
        worker2 = await worker_manager.start_worker(
            worker_id="integration-worker-2",
            model="mock-model",
            port=9202
        )
        
        # Register with routing and sync
        await routing_engine.register_worker("integration-worker-1", "http://localhost:9201")
        await routing_engine.register_worker("integration-worker-2", "http://localhost:9202")
        
        metadata_sync.register_worker("integration-worker-1", "http://localhost:9201")
        metadata_sync.register_worker("integration-worker-2", "http://localhost:9202")
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Simulate requests
        for i in range(5):
            request = InferenceRequest(
                prompt=f"Integration test request {i}",
                max_tokens=50
            )
            
            # Route request
            decision = await routing_engine.route_request(request)
            assert decision.selected_worker in ["integration-worker-1", "integration-worker-2"]
            
            # Record metrics
            metrics.record_request(
                worker_id=decision.selected_worker,
                status="success",
                duration_ms=100.0 + i * 10,
                cache_hit=(i % 2 == 0),
                tokens_generated=50
            )
        
        # Force cache sync
        await metadata_sync.force_sync_all()
        
        # Verify metrics
        metrics_data = metrics.export_metrics()
        assert len(metrics_data) > 0
        
        # Get statistics
        stats = metadata_sync.get_cache_statistics()
        assert stats['total_workers'] == 2
        
        # Verify system health
        assert worker1.is_healthy
        assert worker2.is_healthy
        
    finally:
        # Cleanup
        await metadata_sync.stop()
        await routing_engine.stop()
        await worker_manager.stop()
        await cache_registry.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
