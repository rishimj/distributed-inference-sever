"""
Simplified tests for gateway-to-worker communication client.
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, patch

from src.gateway.worker_client import WorkerClientPool, MockWorkerClientPool, WorkerConnection, WorkerStatus
from src.common.models import InferenceRequest


class TestWorkerConnection:
    """Test WorkerConnection data class"""
    
    def test_initial_state(self):
        """Test initial connection state"""
        conn = WorkerConnection("node-1", "http://localhost:8000")
        
        assert conn.node_id == "node-1"
        assert conn.base_url == "http://localhost:8000"
        assert conn.status == WorkerStatus.UNKNOWN
        assert conn.session is None
        assert conn.consecutive_failures == 0
        assert conn.total_requests == 0
        assert conn.successful_requests == 0
        assert conn.success_rate == 1.0  # No requests yet
        assert not conn.is_healthy  # Unknown status
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        conn = WorkerConnection("node-1", "http://localhost:8000")
        
        # No requests
        assert conn.success_rate == 1.0
        
        # Some successful requests
        conn.total_requests = 10
        conn.successful_requests = 8
        assert conn.success_rate == 0.8
        
        # All failed
        conn.successful_requests = 0
        assert conn.success_rate == 0.0
    
    def test_is_healthy_logic(self):
        """Test health determination logic"""
        conn = WorkerConnection("node-1", "http://localhost:8000")
        
        # Unknown status
        assert not conn.is_healthy
        
        # Healthy status but high failure rate
        conn.status = WorkerStatus.HEALTHY
        conn.consecutive_failures = 5
        assert not conn.is_healthy
        
        # Healthy status but low success rate
        conn.consecutive_failures = 0
        conn.total_requests = 10
        conn.successful_requests = 7  # 70% success rate
        assert not conn.is_healthy
        
        # All conditions met
        conn.successful_requests = 9  # 90% success rate
        assert conn.is_healthy


class TestWorkerClientPool:
    """Test WorkerClientPool functionality"""
    
    def test_init_parameters(self):
        """Test initialization parameters"""
        pool = WorkerClientPool(
            timeout_seconds=10.0,
            max_retries=3,
            health_check_interval=60.0,
            connection_pool_size=20
        )
        
        assert pool.timeout.total == 10.0
        assert pool.max_retries == 3
        assert pool.health_check_interval == 60.0
        assert pool.connection_pool_size == 20
        assert len(pool.workers) == 0
        assert len(pool.healthy_workers) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test start and stop lifecycle"""
        pool = WorkerClientPool()
        
        # Initially not running
        assert not pool._running
        assert pool._health_check_task is None
        assert pool._cleanup_task is None
        
        # Start pool
        await pool.start()
        assert pool._running
        assert pool._health_check_task is not None
        assert pool._cleanup_task is not None
        
        # Stop pool
        await pool.stop()
        assert not pool._running
        assert len(pool.workers) == 0
        assert len(pool.healthy_workers) == 0
    
    @pytest.mark.asyncio
    async def test_add_remove_workers(self):
        """Test adding and removing workers"""
        pool = WorkerClientPool()
        await pool.start()
        
        try:
            # Add workers
            pool.add_worker("node-1", "localhost", 8001)
            pool.add_worker("node-2", "localhost", 8002)
            
            assert len(pool.workers) == 2
            assert "node-1" in pool.workers
            assert "node-2" in pool.workers
            
            worker1 = pool.workers["node-1"]
            assert worker1.base_url == "http://localhost:8001"
            
            # Remove worker
            pool.remove_worker("node-1")
            assert len(pool.workers) == 1
            assert "node-1" not in pool.workers
            assert "node-2" in pool.workers
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_get_healthy_workers(self):
        """Test getting healthy workers list"""
        pool = WorkerClientPool()
        await pool.start()
        
        try:
            # No workers initially
            healthy = await pool.get_healthy_workers()
            assert len(healthy) == 0
            
            # Add workers but they're not healthy yet
            pool.add_worker("node-1", "localhost", 8001)
            pool.add_worker("node-2", "localhost", 8002)
            
            healthy = await pool.get_healthy_workers()
            assert len(healthy) == 0
            
            # Manually mark as healthy
            pool.healthy_workers.add("node-1")
            healthy = await pool.get_healthy_workers()
            assert healthy == ["node-1"]
        finally:
            await pool.stop()


class TestMockWorkerClientPool:
    """Test MockWorkerClientPool for testing scenarios"""
    
    @pytest.mark.asyncio
    async def test_mock_inference_request(self):
        """Test mock inference request"""
        pool = MockWorkerClientPool()
        await pool.start()
        
        try:
            # Add a worker
            pool.add_worker("node-1", "localhost", 8001)
            
            # Set up mock response
            pool.set_mock_response("node-1", "Hello, world!", 150.0)
            
            request = InferenceRequest(
                request_id="test-123",
                prompt="Say hello",
                max_tokens=50,
                temperature=0.7
            )
            
            response = await pool.send_inference_request("node-1", request)
            
            assert response.request_id == "test-123"
            assert response.generated_text == "Hello, world!"
            assert response.tokens_generated > 0
            assert response.processing_time_ms > 0
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_mock_streaming_request(self):
        """Test mock streaming request"""
        pool = MockWorkerClientPool()
        await pool.start()
        
        try:
            pool.add_worker("node-1", "localhost", 8001)
            pool.set_mock_response("node-1", "Hello world test", 100.0)
            
            request = InferenceRequest(
                request_id="stream-123",
                prompt="Say hello world test",
                max_tokens=50,
                temperature=0.7
            )
            
            chunks = []
            async for chunk in pool.send_streaming_request("node-1", request):
                chunks.append(chunk)
            
            # Should have 3 chunks (one per word)
            assert len(chunks) == 3
            
            # Check first chunk
            assert "choices" in chunks[0]
            assert chunks[0]["choices"][0]["delta"]["content"] == "Hello "
            assert chunks[0]["choices"][0]["finish_reason"] is None
            
            # Check last chunk
            assert chunks[-1]["choices"][0]["delta"]["content"] == "test"
            assert chunks[-1]["choices"][0]["finish_reason"] == "length"
            assert "usage" in chunks[-1]
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_mock_failures(self):
        """Test mock failure simulation"""
        pool = MockWorkerClientPool()
        await pool.start()
        
        try:
            pool.add_worker("node-1", "localhost", 8001)
            
            # Set node to fail next 2 requests
            pool.set_mock_failure("node-1", 2)
            
            request = InferenceRequest(
                request_id="fail-test",
                prompt="This should fail",
                max_tokens=10
            )
            
            # First request should fail
            with pytest.raises(aiohttp.ClientError):
                await pool.send_inference_request("node-1", request)
            
            # Second request should also fail
            with pytest.raises(aiohttp.ClientError):
                await pool.send_inference_request("node-1", request)
            
            # Third request should succeed
            response = await pool.send_inference_request("node-1", request)
            assert response.request_id == "fail-test"
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_mock_health_checks(self):
        """Test mock health check behavior"""
        pool = MockWorkerClientPool()
        await pool.start()
        
        try:
            pool.add_worker("node-1", "localhost", 8001)
            
            # Initially healthy
            await pool._check_worker_health("node-1", pool.workers["node-1"])
            assert pool.workers["node-1"].status == WorkerStatus.HEALTHY
            assert "node-1" in pool.healthy_workers
            
            # Set to fail
            pool.set_mock_failure("node-1", 5)
            await pool._check_worker_health("node-1", pool.workers["node-1"])
            assert pool.workers["node-1"].status == WorkerStatus.UNHEALTHY
            assert "node-1" not in pool.healthy_workers
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_error_scenarios(self):
        """Test various error scenarios"""
        pool = MockWorkerClientPool()
        await pool.start()
        
        try:
            # Test request to non-existent worker
            request = InferenceRequest(
                request_id="error-test",
                prompt="Test prompt",
                max_tokens=10
            )
            
            with pytest.raises(ValueError, match="Worker non-existent not found"):
                await pool.send_inference_request("non-existent", request)
            
        finally:
            await pool.stop()


class TestWorkerClientIntegration:
    """Integration tests for worker client functionality"""
    
    def test_update_worker_metrics(self):
        """Test worker metrics updates"""
        worker = WorkerConnection("test-node", "http://localhost:8000")
        pool = WorkerClientPool()
        
        # Test successful request
        pool._update_worker_metrics(worker, True, 100.0)
        assert worker.total_requests == 1
        assert worker.successful_requests == 1
        assert worker.consecutive_failures == 0
        assert worker.avg_response_time_ms == 100.0
        
        # Test another successful request with different timing
        pool._update_worker_metrics(worker, True, 200.0)
        assert worker.total_requests == 2
        assert worker.successful_requests == 2
        # Response time should be moving average
        expected_avg = 0.9 * 100.0 + 0.1 * 200.0
        assert abs(worker.avg_response_time_ms - expected_avg) < 0.01
        
        # Test failed request
        pool._update_worker_metrics(worker, False, 500.0)
        assert worker.total_requests == 3
        assert worker.successful_requests == 2
        assert worker.consecutive_failures == 1
        assert worker.success_rate == 2/3


@pytest.mark.asyncio
async def test_background_tasks():
    """Test background health check and cleanup tasks"""
    pool = WorkerClientPool(health_check_interval=0.1)  # Very fast for testing
    
    # Add a worker
    pool.add_worker("test-node", "localhost", 8000)
    
    # Start pool
    await pool.start()
    
    try:
        # Let health checks run
        await asyncio.sleep(0.2)
        
        # Background tasks should be running
        assert pool._health_check_task is not None
        assert not pool._health_check_task.done()
        assert pool._cleanup_task is not None
        assert not pool._cleanup_task.done()
        
    finally:
        await pool.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])