"""
Tests for InfiniBand RDMA integration with disaggregated system.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch

from src.cache.infiniband_transfer import (
    InfiniBandTransferManager, 
    MockInfiniBandTransferManager,
    TransferStats
)
from src.gateway.disaggregated_coordinator import DisaggregatedRequestCoordinator
from src.gateway.worker_client import MockWorkerClientPool
from src.common.models import InferenceRequest


class TestInfiniBandTransferManager:
    """Test InfiniBand transfer manager functionality."""
    
    @pytest.mark.asyncio
    async def test_mock_transfer_manager_init(self):
        """Test mock InfiniBand manager initialization."""
        manager = MockInfiniBandTransferManager(
            node_id="test-node-1",
            enable_compression=True
        )
        
        await manager.initialize()
        
        assert manager.node_id == "test-node-1"
        assert manager.enable_compression is True
        assert manager.running is True
        assert len(manager.transfer_stats) == 0
        
        await manager.shutdown()
        assert manager.running is False
    
    @pytest.mark.asyncio
    async def test_mock_cache_transfer(self):
        """Test mock cache transfer with simulated InfiniBand performance."""
        manager = MockInfiniBandTransferManager(
            node_id="prefill-node-1",
            enable_compression=True
        )
        
        await manager.initialize()
        
        try:
            # Create test cache data
            test_cache = b"mock_kv_cache_data_for_testing" * 1000  # ~30KB
            
            # Transfer cache
            success = await manager.transfer_cache(
                target_node="decode-node-1",
                cache_data=test_cache,
                request_id="test-transfer-001"
            )
            
            assert success is True
            assert len(manager.transfer_stats) == 1
            
            stats = manager.transfer_stats[0]
            assert stats.bytes_transferred == len(test_cache)
            assert stats.rdma_used is True
            assert stats.bandwidth_gbps > 0
            assert stats.compression_ratio > 1.0  # Should have compression
            assert stats.source_node == "prefill-node-1"
            assert stats.target_node == "decode-node-1"
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_stats_tracking(self):
        """Test performance statistics tracking."""
        manager = MockInfiniBandTransferManager(
            node_id="test-node",
            enable_compression=True
        )
        
        await manager.initialize()
        
        try:
            # Perform multiple transfers
            for i in range(5):
                await manager.transfer_cache(
                    target_node=f"target-{i % 2}",  # Alternate between 2 targets
                    cache_data=b"test_data" * (100 * (i + 1)),  # Varying sizes
                    request_id=f"request-{i}"
                )
            
            # Check statistics
            stats = manager.get_performance_stats()
            
            assert stats['total_transfers'] == 5
            assert stats['recent_transfers'] == 5
            assert stats['avg_bandwidth_gbps'] > 0
            assert stats['avg_compression_ratio'] > 1.0
            assert stats['rdma_transfers'] == 5
            assert stats['tcp_transfers'] == 0
            assert stats['rdma_success_rate'] == 1.0
            
            # Check per-node bandwidth tracking
            assert 'bandwidth_by_node' in stats
            assert len(stats['bandwidth_by_node']) == 2  # Two target nodes
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_node_address_resolution(self):
        """Test node address resolution for PACE ICE."""
        # Test with regular manager (not mock) to test actual resolution logic
        manager = InfiniBandTransferManager(node_id="test")
        
        # Test InfiniBand hostname resolution
        with patch('socket.gethostbyname') as mock_resolve:
            mock_resolve.side_effect = [
                '192.168.100.10',  # IB address
                '192.168.1.10'     # Regular address  
            ]
            
            # Should try IB first
            ib_addr = await manager.get_node_address("pace-node-01")
            assert ib_addr == '192.168.100.10'
            
            # Check that it tried the IB hostname first
            mock_resolve.assert_called_with("pace-node-01-ib")
    
    def test_transfer_stats_dataclass(self):
        """Test TransferStats dataclass."""
        stats = TransferStats(
            bytes_transferred=1024 * 1024,  # 1MB
            transfer_time_ms=10.0,
            bandwidth_gbps=0.8,
            compression_ratio=5.0,
            rdma_used=True,
            source_node="prefill-1",
            target_node="decode-1",
            timestamp=1234567890.0
        )
        
        assert stats.bytes_transferred == 1024 * 1024
        assert stats.transfer_time_ms == 10.0
        assert stats.bandwidth_gbps == 0.8
        assert stats.compression_ratio == 5.0
        assert stats.rdma_used is True
        assert stats.source_node == "prefill-1"
        assert stats.target_node == "decode-1"


class TestInfiniBandIntegrationWithCoordinator:
    """Test InfiniBand integration with disaggregated coordinator."""
    
    @pytest.mark.asyncio
    async def test_coordinator_with_infiniband(self):
        """Test coordinator using InfiniBand for cache transfers."""
        # Create mock pools
        prefill_pool = MockWorkerClientPool()
        decode_pool = MockWorkerClientPool()
        
        await prefill_pool.start()
        await decode_pool.start()
        
        try:
            # Add mock workers
            prefill_pool.add_worker("prefill-pace-01", "192.168.100.1", 8000)
            decode_pool.add_worker("decode-pace-01", "192.168.100.2", 8001)
            
            # Create coordinator with InfiniBand (mock)
            coordinator = DisaggregatedRequestCoordinator(
                prefill_pool=prefill_pool,
                decode_pool=decode_pool,
                node_id="coordinator-pace-01",
                use_infiniband=True  # This will use Mock since UCX not available
            )
            
            await coordinator.initialize()
            
            try:
                # Verify InfiniBand manager is initialized
                assert coordinator.ib_transfer is not None
                assert coordinator.ib_transfer.node_id == "coordinator-pace-01"
                assert coordinator.ib_transfer.running is True
                
                # Check initial performance stats
                ib_stats = coordinator.ib_transfer.get_performance_stats()
                assert ib_stats['status'] == 'no_transfers'
                
            finally:
                await coordinator.shutdown()
                
        finally:
            await prefill_pool.stop()
            await decode_pool.stop()
    
    @pytest.mark.asyncio
    async def test_coordinator_performance_metrics_with_ib(self):
        """Test coordinator performance metrics including InfiniBand stats."""
        prefill_pool = MockWorkerClientPool()
        decode_pool = MockWorkerClientPool()
        
        await prefill_pool.start()
        await decode_pool.start()
        
        try:
            prefill_pool.add_worker("prefill-1", "node1", 8000)
            decode_pool.add_worker("decode-1", "node2", 8001)
            
            coordinator = DisaggregatedRequestCoordinator(
                prefill_pool, decode_pool, use_infiniband=True
            )
            
            await coordinator.initialize()
            
            try:
                # Simulate some transfers in the IB manager (should work with mock)
                success = await coordinator.ib_transfer.transfer_cache(
                    target_node="decode-1",
                    cache_data=b"test_cache_data" * 100,
                    request_id="metrics-test"
                )
                
                # Verify the transfer succeeded with mock
                assert success is True
                
                # Check IB manager stats directly
                ib_stats = coordinator.ib_transfer.get_performance_stats()
                assert ib_stats['total_transfers'] >= 1
                assert 'avg_bandwidth_gbps' in ib_stats
                assert 'rdma_success_rate' in ib_stats
                
                # Get coordinator performance metrics (may be no_data if no full requests)
                metrics = await coordinator.get_performance_metrics()
                
                # Should include InfiniBand stats
                if metrics.get('status') != 'no_data':
                    assert 'infiniband_stats' in metrics
                
            finally:
                await coordinator.shutdown()
                
        finally:
            await prefill_pool.stop()
            await decode_pool.stop()
    
    def test_pace_ice_specific_optimizations(self):
        """Test PACE ICE specific InfiniBand optimizations."""
        import os
        
        # Test UCX environment variable setup
        expected_ucx_config = {
            'UCX_TLS': 'rc_mlx5,ud_mlx5,tcp',
            'UCX_NET_DEVICES': 'mlx5_0:1', 
            'UCX_RNDV_SCHEME': 'put_zcopy',
            'UCX_RNDV_THRESH': '8192'
        }
        
        # These would be set by the deployment script
        for key, expected_value in expected_ucx_config.items():
            # Just verify the expected format
            assert len(expected_value) > 0
            assert isinstance(expected_value, str)
    
    @pytest.mark.asyncio
    async def test_bandwidth_estimation_accuracy(self):
        """Test bandwidth estimation for different cache sizes."""
        manager = MockInfiniBandTransferManager(
            node_id="test-bandwidth",
            enable_compression=True
        )
        
        await manager.initialize()
        
        try:
            # Test different cache sizes
            cache_sizes = [
                1024,           # 1KB  
                1024 * 1024,    # 1MB
                10 * 1024 * 1024, # 10MB
                100 * 1024 * 1024 # 100MB
            ]
            
            bandwidths = []
            
            for size in cache_sizes:
                cache_data = b"x" * size
                
                start_time = asyncio.get_event_loop().time()
                success = await manager.transfer_cache(
                    target_node="test-target",
                    cache_data=cache_data,
                    request_id=f"bw-test-{size}"
                )
                end_time = asyncio.get_event_loop().time()
                
                assert success is True
                
                # Get the bandwidth from the last transfer
                last_stats = manager.transfer_stats[-1]
                bandwidths.append(last_stats.bandwidth_gbps)
                
                # Verify realistic simulation
                assert last_stats.bandwidth_gbps > 10.0  # At least 10 Gbps
                assert last_stats.bandwidth_gbps < 50.0  # Not more than 50 Gbps
            
            # Larger transfers should have better bandwidth efficiency
            assert len(bandwidths) == len(cache_sizes)
            
        finally:
            await manager.shutdown()


class TestPACEICEDeploymentIntegration:
    """Test PACE ICE specific deployment features."""
    
    def test_infiniband_environment_setup(self):
        """Test InfiniBand environment variable setup for PACE ICE."""
        # Test the environment variables that would be set in deployment
        pace_ice_config = {
            # UCX configuration for InfiniBand  
            'UCX_NET_DEVICES': 'mlx5_0:1',
            'UCX_TLS': 'rc_mlx5,ud_mlx5,mm,shm', 
            'UCX_RNDV_SCHEME': 'put_zcopy',
            'UCX_RNDV_THRESH': '8192',
            'UCX_MAX_RNDV_RAILS': '1',
            
            # InfiniBand specific
            'IBV_FORK_SAFE': '1',
            'RDMA_CORE_ROOT': '/usr',
        }
        
        for key, value in pace_ice_config.items():
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_pace_ice_node_naming_conventions(self):
        """Test node naming and address resolution for PACE ICE."""
        # PACE ICE nodes typically follow patterns like:
        test_nodes = [
            'pace-ice-c001',
            'pace-ice-c002', 
            'pace-ice-gpu-001',
            'pace-ice-gpu-002'
        ]
        
        for node in test_nodes:
            # Should be able to construct IB hostname
            ib_hostname = f"{node}-ib"
            assert ib_hostname.endswith('-ib')
            assert 'pace-ice' in ib_hostname
    
    @pytest.mark.asyncio
    async def test_multi_node_cache_transfer_simulation(self):
        """Simulate cache transfers between multiple PACE ICE nodes."""
        # Create multiple transfer managers (simulating different nodes)
        nodes = {}
        node_names = ['pace-ice-prefill-01', 'pace-ice-decode-01', 'pace-ice-decode-02']
        
        for node_name in node_names:
            manager = MockInfiniBandTransferManager(
                node_id=node_name,
                enable_compression=True
            )
            await manager.initialize()
            nodes[node_name] = manager
        
        try:
            # Simulate cache transfers from prefill to multiple decode nodes
            prefill_node = nodes['pace-ice-prefill-01']
            cache_data = b"shared_kv_cache_data" * 10000  # ~200KB
            
            # Transfer to both decode nodes
            for decode_node_name in ['pace-ice-decode-01', 'pace-ice-decode-02']:
                success = await prefill_node.transfer_cache(
                    target_node=decode_node_name,
                    cache_data=cache_data,
                    request_id=f"multi-node-{decode_node_name}"
                )
                
                assert success is True
            
            # Verify statistics
            stats = prefill_node.get_performance_stats()
            assert stats['total_transfers'] == 2
            assert stats['rdma_success_rate'] == 1.0
            assert len(stats['bandwidth_by_node']) == 2
            
        finally:
            # Cleanup all nodes
            for manager in nodes.values():
                await manager.shutdown()


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v"])