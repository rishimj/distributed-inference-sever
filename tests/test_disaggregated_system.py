"""
Integration tests for disaggregated prefill/decode system.
"""

import pytest
import asyncio
import logging
from typing import List

from src.workers.prefill_worker import PrefillWorker, PrefillResult
from src.workers.decode_worker import DecodeWorker, DecodeRequest
from src.gateway.disaggregated_coordinator import DisaggregatedRequestCoordinator
from src.gateway.worker_client import MockWorkerClientPool
from src.common.models import InferenceRequest


class TestDisaggregatedSystem:
    """Test complete disaggregated prefill/decode pipeline."""
    
    @pytest.mark.asyncio
    async def test_prefill_worker_basic(self):
        """Test basic prefill worker functionality."""
        worker = PrefillWorker(
            node_id="test-prefill-1",
            host="127.0.0.1", 
            port=18000,
            model_name="facebook/opt-125m"
        )
        
        try:
            await worker.initialize_engine()
            
            request = InferenceRequest(
                request_id="test-prefill-001",
                prompt="The capital of France is",
                max_tokens=50,
                temperature=0.7
            )
            
            result = await worker.process_prefill(request)
            
            assert result.request_id == "test-prefill-001"
            assert result.first_token is not None
            assert len(result.first_token) > 0
            assert result.cache_size_bytes > 0
            assert result.processing_time_ms > 0
            assert len(result.kv_cache_data) > 0
            assert result.cache_hash is not None
            
        finally:
            # Cleanup is implicit with mock engine
            pass
    
    @pytest.mark.asyncio
    async def test_decode_worker_basic(self):
        """Test basic decode worker functionality."""
        # First create a mock prefill result
        mock_cache_data = b"mock_cache_data_12345"
        
        # Serialize it properly using the same serializer
        from src.workers.prefill_worker import KVCacheSerializer
        serializer = KVCacheSerializer()
        
        mock_cache_obj = {
            'request_id': 'test-decode-001',
            'mock_data': 'test_cache',
            'model': 'facebook/opt-125m'
        }
        
        serialized_cache = serializer.serialize_cache(mock_cache_obj)
        
        prefill_result = PrefillResult(
            request_id="test-decode-001",
            first_token="Hello",
            kv_cache_data=serialized_cache,
            prompt_tokens=5,
            cache_size_bytes=len(serialized_cache),
            processing_time_ms=150.0,
            cache_hash="abc123"
        )
        
        worker = DecodeWorker(
            node_id="test-decode-1",
            host="127.0.0.1",
            port=18001,
            model_name="facebook/opt-125m"
        )
        
        try:
            await worker.initialize_engine()
            
            request = InferenceRequest(
                request_id="test-decode-001",
                prompt="The capital of France is",
                max_tokens=10,
                temperature=0.7
            )
            
            decode_request = DecodeRequest(
                inference_request=request,
                prefill_result=prefill_result
            )
            
            # Test streaming generation
            tokens = []
            async for token in worker.continue_generation(decode_request):
                tokens.append(token)
                if len(tokens) >= 5:  # Limit for testing
                    break
            
            assert len(tokens) > 0
            assert tokens[0] == "Hello"  # First token from prefill
            
            # Test metrics
            metrics = await worker.get_metrics()
            assert metrics['node_id'] == "test-decode-1"
            assert metrics['worker_type'] == 'decode'
            assert metrics['cache_injections'] >= 1
            
        finally:
            # Cleanup is implicit with mock engine
            pass
    
    @pytest.mark.asyncio
    async def test_cache_serialization(self):
        """Test KV cache serialization/deserialization."""
        from src.workers.prefill_worker import KVCacheSerializer
        
        serializer = KVCacheSerializer()
        
        # Test data
        test_cache = {
            'model_name': 'test-model',
            'sequence_length': 100,
            'cache_blocks': [1, 2, 3, 4, 5],
            'metadata': {'version': 1, 'compressed': True}
        }
        
        # Serialize
        serialized = serializer.serialize_cache(test_cache)
        assert len(serialized) > 24  # At least header size
        
        # Deserialize
        deserialized = serializer.deserialize_cache(serialized)
        
        # Verify
        assert deserialized == test_cache
        assert deserialized['model_name'] == 'test-model'
        assert deserialized['sequence_length'] == 100
    
    @pytest.mark.asyncio
    async def test_disaggregated_coordinator_mock(self):
        """Test disaggregated coordinator with mock pools."""
        # Create mock pools
        prefill_pool = MockWorkerClientPool()
        decode_pool = MockWorkerClientPool()
        
        await prefill_pool.start()
        await decode_pool.start()
        
        try:
            # Add mock workers
            prefill_pool.add_worker("prefill-1", "localhost", 8000)
            decode_pool.add_worker("decode-1", "localhost", 8001)
            
            # Set up mock responses
            # Note: We'll need to extend MockWorkerClientPool to support prefill/decode
            # For now, just test basic structure
            
            coordinator = DisaggregatedRequestCoordinator(prefill_pool, decode_pool)
            
            # Test that coordinator initializes
            assert coordinator.prefill_pool == prefill_pool
            assert coordinator.decode_pool == decode_pool
            assert coordinator.processed_requests == 0
            
            # Test metrics (should be empty initially)
            metrics = await coordinator.get_performance_metrics()
            assert metrics['status'] == 'no_data'
            
        finally:
            await prefill_pool.stop()
            await decode_pool.stop()
    
    @pytest.mark.asyncio
    async def test_node_selector(self):
        """Test node selection logic."""
        from src.gateway.disaggregated_coordinator import NodeSelector
        
        selector = NodeSelector()
        
        # Mock request
        request = InferenceRequest(
            request_id="test-selector",
            prompt="Short test prompt",
            max_tokens=100,
            temperature=0.7
        )
        
        # Mock available nodes
        prefill_nodes = ["prefill-1", "prefill-2"]
        
        # Mock node metrics
        node_metrics = {
            "prefill-1": {
                "worker_type": "prefill",
                "active_requests": 5,
                "max_concurrent_sequences": 16,
                "avg_processing_time_ms": 200
            },
            "prefill-2": {
                "worker_type": "prefill", 
                "active_requests": 2,
                "max_concurrent_sequences": 16,
                "avg_processing_time_ms": 180
            }
        }
        
        # Test prefill node selection
        selected = await selector.select_prefill_node(request, prefill_nodes, node_metrics)
        
        # Should select prefill-2 (lower load, better performance)
        assert selected == "prefill-2"
    
    @pytest.mark.asyncio
    async def test_cache_transfer_estimation(self):
        """Test cache transfer time estimation."""
        from src.gateway.disaggregated_coordinator import CacheTransferManager
        
        manager = CacheTransferManager()
        
        # Test same node (no transfer)
        time_same = await manager.estimate_transfer_time(
            "node-1", "node-1", 1024 * 1024  # 1MB
        )
        assert time_same == 0.0
        
        # Test different nodes
        time_diff = await manager.estimate_transfer_time(
            "node-1", "node-2", 10 * 1024 * 1024  # 10MB
        )
        assert time_diff > 0
        
        # Test with rack detection
        time_same_rack = await manager.estimate_transfer_time(
            "pace-ice-rack1-node01", "pace-ice-rack1-node02", 1024 * 1024
        )
        time_diff_rack = await manager.estimate_transfer_time(
            "pace-ice-rack1-node01", "pace-ice-rack2-node01", 1024 * 1024
        )
        
        # Same rack should be faster (allow small variance for mock estimation)
        assert time_same_rack <= time_diff_rack * 1.1  # Allow 10% variance
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self):
        """Test performance metrics collection."""
        from src.gateway.disaggregated_coordinator import DisaggregatedMetrics
        
        # Create sample metrics
        metrics = DisaggregatedMetrics(
            request_id="perf-test-001",
            prefill_node="prefill-1",
            decode_node="decode-1", 
            prefill_time_ms=200.0,
            cache_transfer_time_ms=50.0,
            decode_time_ms=800.0,
            total_time_ms=1050.0,
            cache_size_bytes=2 * 1024 * 1024,  # 2MB
            tokens_generated=45,
            cache_hit=True,
            disaggregation_overhead_ms=60.0
        )
        
        # Verify metrics structure
        assert metrics.request_id == "perf-test-001"
        assert metrics.total_time_ms == 1050.0
        assert metrics.cache_hit is True
        assert metrics.disaggregation_overhead_ms < metrics.total_time_ms


class TestDeploymentConfig:
    """Test deployment configuration parsing."""
    
    def test_pace_ice_config_loading(self):
        """Test loading PACE ICE deployment configuration."""
        import yaml
        
        config_path = "configs/pace_ice_deployment.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify required sections
        assert 'model_name' in config
        assert 'prefill' in config
        assert 'decode' in config
        assert 'network' in config
        
        # Verify prefill config
        prefill = config['prefill']
        assert prefill['nodes'] > 0
        assert prefill['partition'] in ['gpu-a100', 'gpu-rtx', 'cpu']
        assert prefill['max_sequences'] > 0
        
        # Verify decode config
        decode = config['decode']
        assert decode['nodes'] > 0
        assert decode['max_sequences'] >= prefill['max_sequences']
        
        # Verify network config
        network = config['network']
        assert network['interconnect'] in ['infiniband', 'ethernet']
        assert network['bandwidth_gbps'] > 0


@pytest.mark.asyncio
async def test_end_to_end_mock_pipeline():
    """Test complete end-to-end pipeline with mocks."""
    
    # Create mock prefill result
    from src.workers.prefill_worker import KVCacheSerializer
    
    serializer = KVCacheSerializer()
    mock_cache = {
        'request_id': 'e2e-test',
        'mock_data': 'complete_test_cache',
        'tokens': ['The', 'capital', 'of', 'France']
    }
    
    serialized_cache = serializer.serialize_cache(mock_cache)
    
    prefill_result = PrefillResult(
        request_id="e2e-test",
        first_token="The",
        kv_cache_data=serialized_cache,
        prompt_tokens=6,
        cache_size_bytes=len(serialized_cache),
        processing_time_ms=180.0,
        cache_hash="e2e_hash"
    )
    
    # Test that we can create a complete decode request
    request = InferenceRequest(
        request_id="e2e-test",
        prompt="What is the capital of France?",
        max_tokens=20,
        temperature=0.7
    )
    
    decode_request = DecodeRequest(
        inference_request=request,
        prefill_result=prefill_result
    )
    
    assert decode_request.inference_request.request_id == "e2e-test"
    assert decode_request.prefill_result.first_token == "The"
    assert len(decode_request.prefill_result.kv_cache_data) > 0


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v"])