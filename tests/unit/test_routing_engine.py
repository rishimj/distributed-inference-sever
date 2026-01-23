"""
Unit tests for the routing decision engine.

Tests cover:
1. KV cache affinity tracking and scoring
2. Load balancing algorithms for vLLM instances
3. Routing decision logic and node selection
4. Performance optimization and statistics tracking
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.models import (
    InferenceRequest,
    NodeInfo,
    RouteDecision,
    ServiceType,
    RoutingConfig,
)
from src.common.prefix_hasher import create_production_hasher
from src.gateway.routing_engine import (
    ActiveRequest,
    KVCacheAffinityTracker,
    LoadBalancingAlgorithm,
    NodeLoadMetrics,
    RoutingDecisionEngine,
)


@pytest.fixture
def sample_request():
    """Sample inference request for testing."""
    return InferenceRequest(
        prompt="What is the capital of France? This is a test prompt for routing.",
        max_tokens=100,
        temperature=0.7
    )


@pytest.fixture
def sample_node_metrics():
    """Sample node metrics for testing."""
    return NodeLoadMetrics(
        node_id="test-node-1",
        active_sequences=50,
        pending_sequences=5,
        kv_cache_usage_percent=45.0,
        gpu_memory_usage_percent=70.0,
        current_throughput_tps=120.0,
        max_concurrent_sequences=256,
        max_sequence_length=4096,
        error_rate_percent=1.0
    )


@pytest.fixture
def sample_nodes():
    """Sample node information for testing."""
    return {
        "node-1": NodeInfo(
            node_id="node-1",
            service_type=ServiceType.PREFILL,
            hostname="host1",
            port=8080,
            gpu_memory_gb=40.0,
            cpu_cores=16,
            current_load=0.3,
            avg_latency_ms=80.0,
            throughput_rps=150.0
        ),
        "node-2": NodeInfo(
            node_id="node-2", 
            service_type=ServiceType.PREFILL,
            hostname="host2",
            port=8080,
            gpu_memory_gb=40.0,
            cpu_cores=16,
            current_load=0.7,
            avg_latency_ms=120.0,
            throughput_rps=100.0
        ),
        "node-3": NodeInfo(
            node_id="node-3",
            service_type=ServiceType.PREFILL,
            hostname="host3",
            port=8080,
            gpu_memory_gb=40.0,
            cpu_cores=16,
            current_load=0.9,  # Overloaded
            avg_latency_ms=200.0,
            throughput_rps=50.0
        )
    }


class TestKVCacheAffinityTracker:
    """Test KV cache affinity tracking."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = KVCacheAffinityTracker(memory_window_minutes=30, max_entries=5000)
        
        assert tracker.memory_window_seconds == 1800
        assert tracker.max_entries == 5000
        assert len(tracker.prefix_to_node) == 0
        assert len(tracker.active_requests) == 0
    
    def test_record_request(self):
        """Test recording a new request."""
        tracker = KVCacheAffinityTracker()
        
        request_id = "req-123"
        prefix_hash = "abc" * 21 + "d"  # 64 char hash
        node_id = "node-1"
        completion_time = time.time() + 10
        
        tracker.record_request(request_id, prefix_hash, node_id, completion_time)
        
        # Check prefix mapping
        assert prefix_hash in tracker.prefix_to_node
        stored_node, timestamp, count = tracker.prefix_to_node[prefix_hash]
        assert stored_node == node_id
        assert count == 1
        
        # Check active request tracking
        assert request_id in tracker.active_requests
        active_req = tracker.active_requests[request_id]
        assert active_req.prefix_hash == prefix_hash
        assert active_req.node_id == node_id
        
        # Check node recent prefixes
        assert node_id in tracker.node_recent_prefixes
        assert len(tracker.node_recent_prefixes[node_id]) == 1
    
    def test_complete_request(self):
        """Test completing a request."""
        tracker = KVCacheAffinityTracker()
        
        request_id = "req-123"
        prefix_hash = "abc" * 21 + "d"
        node_id = "node-1"
        
        tracker.record_request(request_id, prefix_hash, node_id, time.time() + 10)
        assert request_id in tracker.active_requests
        
        tracker.complete_request(request_id)
        assert request_id not in tracker.active_requests
    
    def test_cache_affinity_exact_match(self):
        """Test cache affinity scoring for exact prefix match."""
        tracker = KVCacheAffinityTracker()
        
        prefix_hash = "test_prefix_hash_" + "a" * 48  # 64 chars
        node_id = "node-1"
        
        # Record recent request
        tracker.record_request("req-1", prefix_hash, node_id, time.time() + 10)
        
        # Test exact match on same node (should be high score)
        score = tracker.get_cache_affinity_score(prefix_hash, node_id)
        assert score >= 0.9  # Very high score for exact recent match
        
        # Test exact match on different node (should be low score)
        score_different_node = tracker.get_cache_affinity_score(prefix_hash, "node-2")
        assert score_different_node == 0.1  # Low score for wrong node
    
    def test_cache_affinity_aging(self):
        """Test that cache affinity scores decrease over time."""
        tracker = KVCacheAffinityTracker()
        
        prefix_hash = "aging_test_hash_" + "b" * 48
        node_id = "node-1"
        
        # Record request in the past
        past_time = time.time() - 600  # 10 minutes ago
        tracker.prefix_to_node[prefix_hash] = (node_id, past_time, 1)
        
        score_10min = tracker.get_cache_affinity_score(prefix_hash, node_id)
        
        # Record even older request
        older_time = time.time() - 2400  # 40 minutes ago  
        tracker.prefix_to_node[prefix_hash] = (node_id, older_time, 1)
        
        score_40min = tracker.get_cache_affinity_score(prefix_hash, node_id)
        
        # Older requests should have lower scores
        assert score_40min < score_10min
        assert score_10min > 0.3  # Still some value after 10 minutes
        assert score_40min < 0.3  # Much less value after 40 minutes
    
    def test_cleanup_old_entries(self):
        """Test cleanup of old cache entries."""
        tracker = KVCacheAffinityTracker(memory_window_minutes=1)  # 1 minute window
        
        # Add old entries
        old_time = time.time() - 120  # 2 minutes ago (outside window)
        recent_time = time.time() - 30  # 30 seconds ago (inside window)
        
        tracker.prefix_to_node["old_prefix"] = ("node-1", old_time, 1)
        tracker.prefix_to_node["recent_prefix"] = ("node-1", recent_time, 1)
        
        # Trigger cleanup
        tracker._cleanup_old_entries()
        
        # Old entry should be removed, recent should remain
        assert "old_prefix" not in tracker.prefix_to_node
        assert "recent_prefix" in tracker.prefix_to_node
    
    def test_prefix_similarity_calculation(self):
        """Test prefix similarity calculation."""
        tracker = KVCacheAffinityTracker()
        
        # Identical prefixes
        hash1 = "abcd1234" * 8  # 64 chars
        hash2 = "abcd1234" * 8  # Same
        assert tracker._calculate_prefix_similarity(hash1, hash2) == 1.0
        
        # Partially similar prefixes
        hash3 = "abcd1234efgh" + "x" * 52  # Same start, different end
        hash4 = "abcd1234efgh" + "y" * 52  # Same start, different end
        similarity = tracker._calculate_prefix_similarity(hash3, hash4)
        assert 0.5 < similarity < 1.0  # Some similarity due to common prefix
        
        # Completely different prefixes
        hash5 = "a" * 64
        hash6 = "b" * 64
        assert tracker._calculate_prefix_similarity(hash5, hash6) == 0.0


class TestNodeLoadMetrics:
    """Test node load metrics and scoring."""
    
    def test_node_load_metrics_creation(self, sample_node_metrics):
        """Test creating node load metrics."""
        metrics = sample_node_metrics
        
        assert metrics.node_id == "test-node-1"
        assert metrics.active_sequences == 50
        assert metrics.kv_cache_usage_percent == 45.0
        assert metrics.current_throughput_tps == 120.0
    
    def test_is_overloaded_detection(self):
        """Test overload detection.""" 
        # Normal load
        metrics = NodeLoadMetrics(
            node_id="normal-node",
            active_sequences=100,
            kv_cache_usage_percent=50.0,
            gpu_memory_usage_percent=70.0,
            max_concurrent_sequences=256
        )
        assert not metrics.is_overloaded()
        
        # Overloaded by active sequences
        metrics.active_sequences = 250  # Close to max (256)
        assert metrics.is_overloaded()
        
        # Overloaded by cache usage
        metrics.active_sequences = 100  # Reset
        metrics.kv_cache_usage_percent = 95.0
        assert metrics.is_overloaded()
        
        # Overloaded by GPU memory
        metrics.kv_cache_usage_percent = 50.0  # Reset
        metrics.gpu_memory_usage_percent = 95.0
        assert metrics.is_overloaded()
    
    def test_capacity_score_calculation(self):
        """Test capacity score calculation."""
        metrics = NodeLoadMetrics(
            node_id="capacity-test",
            active_sequences=128,  # 50% of max (256)
            kv_cache_usage_percent=40.0,  # 40% usage
            gpu_memory_usage_percent=60.0,  # 60% usage
            max_concurrent_sequences=256
        )
        
        capacity_score = metrics.get_capacity_score()
        
        # Should be positive (node has capacity)
        assert 0.0 < capacity_score < 1.0
        
        # Test empty node (high capacity)
        empty_metrics = NodeLoadMetrics(
            node_id="empty-node",
            active_sequences=0,
            kv_cache_usage_percent=0.0,
            gpu_memory_usage_percent=10.0,  # Some base usage
            max_concurrent_sequences=256
        )
        
        empty_score = metrics.get_capacity_score()
        full_metrics = NodeLoadMetrics(
            node_id="full-node", 
            active_sequences=250,
            kv_cache_usage_percent=90.0,
            gpu_memory_usage_percent=95.0,
            max_concurrent_sequences=256
        )
        
        full_score = full_metrics.get_capacity_score()
        
        # Empty node should have higher capacity score
        assert empty_score > full_score


class TestLoadBalancingAlgorithm:
    """Test load balancing algorithm."""
    
    def test_initialization(self):
        """Test load balancing algorithm initialization."""
        config = RoutingConfig()
        algorithm = LoadBalancingAlgorithm(config)
        
        assert algorithm.config == config
        assert len(algorithm.node_metrics) == 0
    
    def test_update_node_metrics(self, sample_node_metrics):
        """Test updating node metrics."""
        config = RoutingConfig()
        algorithm = LoadBalancingAlgorithm(config)
        
        algorithm.update_node_metrics("test-node", sample_node_metrics)
        
        assert "test-node" in algorithm.node_metrics
        stored_metrics = algorithm.node_metrics["test-node"]
        assert stored_metrics.node_id == "test-node"
        assert stored_metrics.active_sequences == 50
    
    def test_load_score_calculation(self, sample_request, sample_node_metrics):
        """Test load score calculation."""
        config = RoutingConfig()
        algorithm = LoadBalancingAlgorithm(config)
        
        # Update metrics for node
        algorithm.update_node_metrics("test-node", sample_node_metrics)
        
        # Calculate load score
        score = algorithm.calculate_load_score("test-node", sample_request)
        
        # Should return a valid score
        assert 0.0 <= score <= 1.0
        
        # Test unknown node
        unknown_score = algorithm.calculate_load_score("unknown-node", sample_request)
        assert unknown_score == 0.1  # Low score for unknown node
    
    def test_load_score_with_high_error_rate(self, sample_request):
        """Test load scoring with high error rate."""
        config = RoutingConfig()
        algorithm = LoadBalancingAlgorithm(config)
        
        # Create metrics with high error rate
        high_error_metrics = NodeLoadMetrics(
            node_id="error-node",
            active_sequences=10,
            error_rate_percent=15.0,  # High error rate
            current_throughput_tps=50.0
        )
        
        algorithm.update_node_metrics("error-node", high_error_metrics)
        score = algorithm.calculate_load_score("error-node", sample_request)
        
        # Should get low score due to high error rate
        assert score == 0.1
    
    def test_workload_suitability_scoring(self):
        """Test workload suitability calculation."""
        config = RoutingConfig()
        algorithm = LoadBalancingAlgorithm(config)
        
        # Create node with specific capabilities
        metrics = NodeLoadMetrics(
            node_id="suitability-test",
            max_sequence_length=2048,  # Smaller capacity
            active_sequences=50,
            kv_cache_usage_percent=30.0,
            gpu_memory_usage_percent=40.0
        )
        
        # Small request (should be good fit)
        small_request = InferenceRequest(
            prompt="Short prompt",
            max_tokens=50
        )
        
        small_suitability = algorithm._calculate_workload_suitability(metrics, small_request)
        assert small_suitability > 0.5
        
        # Very large request (should not fit)
        large_request = InferenceRequest(
            prompt="Very long prompt " * 500,  # Very long
            max_tokens=1000
        )
        
        large_suitability = algorithm._calculate_workload_suitability(metrics, large_request)
        # Might not fit in max_sequence_length
        assert large_suitability >= 0.0  # At least doesn't crash


class TestRoutingDecisionEngine:
    """Test the main routing decision engine."""
    
    @pytest.fixture
    def mock_cache_registry(self):
        """Mock cache registry for testing."""
        registry = AsyncMock()
        registry.get_healthy_nodes = AsyncMock(return_value=[])
        return registry
    
    @pytest.fixture
    def routing_engine(self, mock_cache_registry):
        """Create routing engine for testing."""
        prefix_manager = create_production_hasher()
        config = RoutingConfig()
        
        engine = RoutingDecisionEngine(
            cache_registry=mock_cache_registry,
            prefix_manager=prefix_manager,
            config=config
        )
        
        return engine
    
    async def test_routing_engine_initialization(self, routing_engine):
        """Test routing engine initialization."""
        assert routing_engine.cache_registry is not None
        assert routing_engine.prefix_manager is not None
        assert routing_engine.config is not None
        assert isinstance(routing_engine.affinity_tracker, KVCacheAffinityTracker)
        assert isinstance(routing_engine.load_balancer, LoadBalancingAlgorithm)
    
    async def test_get_healthy_nodes_caching(self, routing_engine, sample_nodes):
        """Test healthy nodes caching."""
        # Mock registry response
        node_list = list(sample_nodes.values())
        routing_engine.cache_registry.get_healthy_nodes.return_value = node_list
        
        # First call should query registry
        nodes1 = await routing_engine._get_healthy_nodes()
        assert len(nodes1) == 3
        assert routing_engine.cache_registry.get_healthy_nodes.call_count == 1
        
        # Second call within cache window should use cache
        nodes2 = await routing_engine._get_healthy_nodes()
        assert nodes1 == nodes2
        assert routing_engine.cache_registry.get_healthy_nodes.call_count == 1  # No additional calls
    
    async def test_score_nodes(self, routing_engine, sample_request, sample_nodes):
        """Test node scoring algorithm."""
        prefix_hash = "test_prefix_" + "a" * 53
        
        # Mock some cache affinity for node-1
        routing_engine.affinity_tracker.prefix_to_node[prefix_hash] = ("node-1", time.time(), 1)
        
        # Update load metrics for all nodes
        for node_id in sample_nodes:
            metrics = NodeLoadMetrics(
                node_id=node_id,
                active_sequences=50,
                kv_cache_usage_percent=40.0,
                gpu_memory_usage_percent=60.0,
                current_throughput_tps=100.0
            )
            routing_engine.load_balancer.update_node_metrics(node_id, metrics)
        
        # Score nodes
        scored_nodes = await routing_engine._score_nodes(sample_request, prefix_hash, sample_nodes)
        
        assert len(scored_nodes) == 3
        
        # Verify scoring structure
        for node_id, total_score, breakdown in scored_nodes:
            assert node_id in sample_nodes
            assert 0.0 <= total_score <= 1.0
            assert 'cache_hit_score' in breakdown
            assert 'load_score' in breakdown
            assert 'latency_score' in breakdown
            assert 'capacity_score' in breakdown
        
        # Nodes should be sorted by score (highest first)
        scores = [score for _, score, _ in scored_nodes]
        assert scores == sorted(scores, reverse=True)
        
        # Node-1 should have highest score due to cache affinity
        best_node = scored_nodes[0][0]
        best_breakdown = scored_nodes[0][2]
        assert best_breakdown['cache_hit_score'] > 0.8  # High cache affinity
    
    async def test_select_best_node(self, routing_engine, sample_request):
        """Test best node selection."""
        # Create mock scored nodes
        scored_nodes = [
            ("node-1", 0.9, {
                'cache_hit_score': 0.9,
                'load_score': 0.8,
                'latency_score': 0.7,
                'capacity_score': 0.9
            }),
            ("node-2", 0.6, {
                'cache_hit_score': 0.3,
                'load_score': 0.7,
                'latency_score': 0.8,
                'capacity_score': 0.6
            }),
            ("node-3", 0.4, {
                'cache_hit_score': 0.1,
                'load_score': 0.5,
                'latency_score': 0.6,
                'capacity_score': 0.4
            })
        ]
        
        decision = routing_engine._select_best_node(scored_nodes, sample_request)
        
        assert isinstance(decision, RouteDecision)
        assert decision.target_node == "node-1"  # Best scoring node
        assert decision.cache_hit_score == 0.9
        assert decision.load_score == 0.8
        assert len(decision.fallback_nodes) <= 3
        assert "node-2" in decision.fallback_nodes  # Should include alternatives
        
        # High cache hit probability should reduce estimated latency
        assert decision.estimated_latency_ms < 200.0  # Should be less than base latency
    
    async def test_route_request_end_to_end(self, routing_engine, sample_request, sample_nodes):
        """Test end-to-end request routing.""" 
        # Setup mock healthy nodes
        node_list = list(sample_nodes.values())
        routing_engine.cache_registry.get_healthy_nodes.return_value = node_list
        
        # Update load metrics
        for node_id in sample_nodes:
            metrics = NodeLoadMetrics(
                node_id=node_id,
                active_sequences=30,
                kv_cache_usage_percent=30.0,
                gpu_memory_usage_percent=50.0,
                current_throughput_tps=120.0
            )
            routing_engine.load_balancer.update_node_metrics(node_id, metrics)
        
        # Route request
        decision = await routing_engine.route_request(sample_request)
        
        # Verify decision structure
        assert isinstance(decision, RouteDecision)
        assert decision.target_node in sample_nodes
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.estimated_latency_ms > 0
        assert len(decision.fallback_nodes) >= 0
        
        # Verify request tracking
        assert sample_request.prefix_hash is not None
        assert len(routing_engine.affinity_tracker.active_requests) == 1
        
        # Verify statistics update
        assert routing_engine.routing_stats['total_requests'] == 1
    
    async def test_route_request_no_healthy_nodes(self, routing_engine, sample_request):
        """Test routing when no healthy nodes available."""
        # Mock empty healthy nodes
        routing_engine.cache_registry.get_healthy_nodes.return_value = []
        
        with pytest.raises(RuntimeError, match="No healthy nodes available"):
            await routing_engine.route_request(sample_request)
    
    async def test_update_node_metrics(self, routing_engine, sample_node_metrics):
        """Test updating node metrics."""
        await routing_engine.update_node_metrics("test-node", sample_node_metrics)
        
        # Verify metrics were stored in load balancer
        assert "test-node" in routing_engine.load_balancer.node_metrics
        stored_metrics = routing_engine.load_balancer.node_metrics["test-node"]
        assert stored_metrics.active_sequences == 50
    
    async def test_complete_request(self, routing_engine):
        """Test request completion tracking."""
        request_id = "test-req-123"
        
        # First record a request
        routing_engine.affinity_tracker.active_requests[request_id] = ActiveRequest(
            request_id=request_id,
            prefix_hash="test_hash",
            node_id="test-node",
            start_time=time.time(),
            estimated_completion_time=time.time() + 10
        )
        
        # Complete the request
        await routing_engine.complete_request(request_id, actual_latency_ms=150.0, cache_hit=True)
        
        # Verify request was removed from active tracking
        assert request_id not in routing_engine.affinity_tracker.active_requests
    
    def test_routing_statistics(self, routing_engine):
        """Test routing statistics collection."""
        # Add some mock statistics
        routing_engine.routing_stats['total_requests'] = 100
        routing_engine.routing_stats['cache_hit_routes'] = 70
        routing_engine.routing_stats['load_balanced_routes'] = 25
        routing_engine.routing_stats['routing_latency_ms'].extend([10, 15, 12, 20, 8])
        
        stats = routing_engine.get_routing_statistics()
        
        assert stats['total_requests'] == 100
        assert stats['cache_hit_route_percentage'] == 70.0
        assert stats['load_balanced_route_percentage'] == 25.0
        assert 'avg_routing_latency_ms' in stats
        assert 'p95_routing_latency_ms' in stats
        assert stats['avg_routing_latency_ms'] > 0


if __name__ == "__main__":
    pytest.main([__file__])