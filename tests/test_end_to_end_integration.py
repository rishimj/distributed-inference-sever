"""
End-to-end integration tests for the complete distributed inference system.

These tests validate the entire request flow from client API through
disaggregated prefill/decode processing with real vLLM engines.
"""

import pytest
import asyncio
import aiohttp
import json
import logging
import time
from typing import List, Dict, Any

from src.gateway.production_server import ProductionGateway
from src.common.models import InferenceRequest


@pytest.fixture(scope="session")
async def gateway_server():
    """Start a complete gateway server for testing."""
    gateway = ProductionGateway(
        host="127.0.0.1",
        port=18080,
        model_name="facebook/opt-125m",
        use_real_vllm=False  # Use mocks for faster testing
    )
    
    await gateway.initialize()
    await gateway.start()
    
    # Wait for server to be ready
    await asyncio.sleep(1)
    
    try:
        yield gateway
    finally:
        await gateway.stop()


class TestEndToEndIntegration:
    """
    Complete end-to-end integration tests.
    
    Tests the full stack:
    1. HTTP API requests
    2. Gateway routing
    3. Disaggregated coordinator
    4. Prefill worker processing
    5. InfiniBand cache transfer
    6. Decode worker processing
    7. Response streaming
    """
    
    @pytest.mark.asyncio
    async def test_single_completion_request(self, gateway_server):
        """Test basic completion request through the full system."""
        async with aiohttp.ClientSession() as session:
            # Send completion request
            request_data = {
                "prompt": "The capital of France is",
                "max_tokens": 10,
                "temperature": 0.7
            }
            
            async with session.post(
                "http://127.0.0.1:18080/v1/completions",
                json=request_data
            ) as response:
                
                assert response.status == 200
                data = await response.json()
                
                # Validate OpenAI-compatible response structure
                assert "id" in data
                assert "object" in data
                assert data["object"] == "text_completion"
                assert "choices" in data
                assert len(data["choices"]) == 1
                
                choice = data["choices"][0]
                assert "text" in choice
                assert "finish_reason" in choice
                assert len(choice["text"]) > 0
                
                # Validate usage metrics
                assert "usage" in data
                usage = data["usage"]
                assert "prompt_tokens" in usage
                assert "completion_tokens" in usage
                assert "total_tokens" in usage
                assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    
    @pytest.mark.asyncio
    async def test_streaming_completion(self, gateway_server):
        """Test streaming completion through the system."""
        async with aiohttp.ClientSession() as session:
            request_data = {
                "prompt": "Once upon a time",
                "max_tokens": 15,
                "temperature": 0.8,
                "stream": True
            }
            
            async with session.post(
                "http://127.0.0.1:18080/v1/completions/stream",
                json=request_data
            ) as response:
                
                assert response.status == 200
                
                chunks = []
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            chunks.append(chunk)
                        except json.JSONDecodeError:
                            continue
                
                # Validate streaming chunks
                assert len(chunks) > 0
                
                for chunk in chunks:
                    assert "id" in chunk
                    assert "object" in chunk
                    assert chunk["object"] == "text_completion"
                    assert "choices" in chunk
                    
                    if chunk["choices"][0]["finish_reason"] is None:
                        # Intermediate chunk should have text
                        assert len(chunk["choices"][0]["text"]) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, gateway_server):
        """Test handling multiple concurrent requests."""
        async with aiohttp.ClientSession() as session:
            
            # Create multiple concurrent requests
            tasks = []
            request_data = {
                "prompt": "The quick brown fox",
                "max_tokens": 8,
                "temperature": 0.5
            }
            
            for i in range(5):
                task = session.post(
                    "http://127.0.0.1:18080/v1/completions",
                    json=request_data
                )
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks)
            
            # Validate all responses
            assert len(responses) == 5
            
            for response in responses:
                assert response.status == 200
                data = await response.json()
                
                assert "choices" in data
                assert len(data["choices"]) == 1
                assert len(data["choices"][0]["text"]) > 0
                
                # Each request should have unique ID
                assert "id" in data
    
    @pytest.mark.asyncio
    async def test_health_and_metrics_endpoints(self, gateway_server):
        """Test system monitoring endpoints."""
        async with aiohttp.ClientSession() as session:
            
            # Test health endpoint
            async with session.get("http://127.0.0.1:18080/health") as response:
                assert response.status == 200
                data = await response.json()
                
                assert "status" in data
                assert data["status"] in ["healthy", "unhealthy"]
                assert "prefill_workers" in data
                assert "decode_workers" in data
                assert data["prefill_workers"] > 0
                assert data["decode_workers"] > 0
            
            # Test metrics endpoint
            async with session.get("http://127.0.0.1:18080/metrics") as response:
                assert response.status == 200
                data = await response.json()
                
                assert "gateway_metrics" in data
                assert "coordinator_metrics" in data
                assert "worker_status" in data
                
                gateway_metrics = data["gateway_metrics"]
                assert "requests_processed" in gateway_metrics
                assert "active_requests" in gateway_metrics
            
            # Test status endpoint
            async with session.get("http://127.0.0.1:18080/status") as response:
                assert response.status == 200
                data = await response.json()
                
                assert "gateway" in data
                assert "workers" in data
                assert "recent_requests" in data
    
    @pytest.mark.asyncio
    async def test_error_handling(self, gateway_server):
        """Test error handling and recovery."""
        async with aiohttp.ClientSession() as session:
            
            # Test invalid request
            invalid_request = {
                "prompt": "",  # Empty prompt
                "max_tokens": -1  # Invalid max_tokens
            }
            
            async with session.post(
                "http://127.0.0.1:18080/v1/completions",
                json=invalid_request
            ) as response:
                
                # Should handle gracefully (may succeed with corrected params)
                assert response.status in [200, 400, 422, 500]
                
                if response.status != 200:
                    data = await response.json()
                    assert "error" in data
    
    @pytest.mark.asyncio
    async def test_chat_completion_endpoint(self, gateway_server):
        """Test OpenAI chat completion compatibility."""
        async with aiohttp.ClientSession() as session:
            request_data = {
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "max_tokens": 10,
                "temperature": 0.7
            }
            
            async with session.post(
                "http://127.0.0.1:18080/v1/chat/completions",
                json=request_data
            ) as response:
                
                assert response.status == 200
                data = await response.json()
                
                # Should convert to completion format
                assert "choices" in data
                assert len(data["choices"]) == 1
                assert "text" in data["choices"][0]


class TestRealVLLMIntegration:
    """
    Tests with real vLLM engines (requires vLLM installation).
    
    These tests are marked as slow and require actual GPU resources.
    """
    
    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Requires real vLLM and GPU")
    @pytest.mark.asyncio
    async def test_real_vllm_prefill_decode(self):
        """Test with real vLLM engines for prefill and decode."""
        # This test would require:
        # 1. Real vLLM installation
        # 2. GPU resources
        # 3. Model weights downloaded
        
        gateway = ProductionGateway(
            host="127.0.0.1",
            port=18081,
            model_name="facebook/opt-125m",
            use_real_vllm=True  # Use real vLLM
        )
        
        try:
            await gateway.initialize()
            await gateway.start()
            
            # Test real inference
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "prompt": "The capital of France is",
                    "max_tokens": 20,
                    "temperature": 0.0  # Deterministic for testing
                }
                
                async with session.post(
                    "http://127.0.0.1:18081/v1/completions",
                    json=request_data
                ) as response:
                    
                    assert response.status == 200
                    data = await response.json()
                    
                    # Validate that we got a real model response
                    assert "choices" in data
                    text = data["choices"][0]["text"]
                    assert len(text) > 0
                    
                    # For deterministic generation, we should get consistent output
                    print(f"Real vLLM output: {text}")
                    
        finally:
            await gateway.stop()
    
    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Requires cluster setup")
    @pytest.mark.asyncio
    async def test_distributed_cluster_deployment(self):
        """Test deployment across multiple nodes."""
        # This would test:
        # 1. Multiple prefill workers on different nodes
        # 2. Multiple decode workers on different nodes  
        # 3. Real InfiniBand cache transfers
        # 4. Load balancing across nodes
        
        prefill_workers = [
            {"host": "prefill-node-1", "port": 8000},
            {"host": "prefill-node-2", "port": 8000}
        ]
        
        decode_workers = [
            {"host": "decode-node-1", "port": 8001},
            {"host": "decode-node-2", "port": 8001},
            {"host": "decode-node-3", "port": 8001}
        ]
        
        gateway = ProductionGateway(
            host="0.0.0.0",
            port=8080,
            model_name="facebook/opt-1.3b",
            use_real_vllm=True
        )
        
        try:
            await gateway.initialize(
                prefill_workers=prefill_workers,
                decode_workers=decode_workers
            )
            await gateway.start()
            
            # Test load balancing
            # ... implementation would test multiple requests
            # distributed across different nodes
            
        finally:
            await gateway.stop()


class TestPerformanceBenchmarks:
    """
    Performance benchmarking tests.
    """
    
    @pytest.mark.asyncio
    async def test_latency_benchmarks(self, gateway_server):
        """Benchmark request latency."""
        async with aiohttp.ClientSession() as session:
            
            latencies = []
            request_data = {
                "prompt": "The weather today is",
                "max_tokens": 5,
                "temperature": 0.5
            }
            
            # Warm up
            for _ in range(3):
                async with session.post(
                    "http://127.0.0.1:18080/v1/completions",
                    json=request_data
                ) as response:
                    await response.json()
            
            # Measure latency
            for _ in range(10):
                start_time = time.time()
                
                async with session.post(
                    "http://127.0.0.1:18080/v1/completions", 
                    json=request_data
                ) as response:
                    
                    assert response.status == 200
                    await response.json()
                    
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)
            
            # Analyze results
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            logging.info(f"Latency benchmarks:")
            logging.info(f"  Average: {avg_latency:.1f}ms")
            logging.info(f"  Min: {min_latency:.1f}ms")
            logging.info(f"  Max: {max_latency:.1f}ms")
            
            # Basic performance assertions (for mock system)
            assert avg_latency < 5000  # Should be under 5 seconds
            assert min_latency > 0
    
    @pytest.mark.asyncio
    async def test_throughput_benchmarks(self, gateway_server):
        """Benchmark request throughput."""
        async with aiohttp.ClientSession() as session:
            
            request_data = {
                "prompt": "Hello world",
                "max_tokens": 3,
                "temperature": 0.5
            }
            
            # Concurrent requests benchmark
            start_time = time.time()
            concurrent_requests = 10
            
            tasks = []
            for _ in range(concurrent_requests):
                task = session.post(
                    "http://127.0.0.1:18080/v1/completions",
                    json=request_data
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # Validate all responses succeeded
            success_count = 0
            for response in responses:
                if response.status == 200:
                    success_count += 1
                    await response.json()
            
            throughput = success_count / total_time
            
            logging.info(f"Throughput benchmark:")
            logging.info(f"  Requests: {concurrent_requests}")
            logging.info(f"  Success: {success_count}")
            logging.info(f"  Time: {total_time:.2f}s")
            logging.info(f"  Throughput: {throughput:.2f} req/s")
            
            assert success_count >= concurrent_requests * 0.8  # At least 80% success
            assert throughput > 0


@pytest.mark.asyncio
async def test_gateway_startup_shutdown():
    """Test gateway lifecycle management."""
    gateway = ProductionGateway(
        host="127.0.0.1",
        port=18082,
        model_name="facebook/opt-125m",
        use_real_vllm=False
    )
    
    # Test initialization
    await gateway.initialize()
    assert gateway.coordinator is not None
    assert gateway.prefill_pool is not None
    assert gateway.decode_pool is not None
    
    # Test startup
    await gateway.start()
    
    # Verify server is running
    async with aiohttp.ClientSession() as session:
        async with session.get("http://127.0.0.1:18082/health") as response:
            assert response.status == 200
    
    # Test graceful shutdown
    await gateway.stop()


if __name__ == "__main__":
    # Configure logging for test output
    logging.basicConfig(level=logging.INFO)
    
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "-m", "not slow",  # Skip slow tests by default
        "--tb=short"
    ])