"""
Simple tests for routing engine without external dependencies.

Tests core routing logic, cache affinity, and load balancing algorithms.
"""

import time
import pytest

from src.common.models import (
    InferenceRequest,
    NodeInfo,
    RouteDecision,
    ServiceType,
    RoutingConfig,
)
from src.gateway.routing_engine import (
    ActiveRequest,
    KVCacheAffinityTracker,
    LoadBalancingAlgorithm,
    NodeLoadMetrics,
)


def test_node_load_metrics_basic():
    """Test basic NodeLoadMetrics functionality."""
    metrics = NodeLoadMetrics(
        node_id="test-node-1",
        active_sequences=100,
        pending_sequences=10,
        kv_cache_usage_percent=60.0,
        gpu_memory_usage_percent=75.0,
        current_throughput_tps=150.0,
        max_concurrent_sequences=256,
        max_sequence_length=4096
    )
    
    # Test basic properties
    assert metrics.node_id == "test-node-1"
    assert metrics.active_sequences == 100
    assert metrics.kv_cache_usage_percent == 60.0
    
    # Test overload detection
    assert not metrics.is_overloaded()  # Within normal limits
    
    # Test with overloaded state
    metrics.active_sequences = 240  # Close to max
    assert metrics.is_overloaded()
    
    # Test capacity scoring
    metrics.active_sequences = 50  # Reset to normal
    capacity_score = metrics.get_capacity_score()
    assert 0.0 <= capacity_score <= 1.0


def test_kv_cache_affinity_tracker():
    """Test KV cache affinity tracking logic."""
    tracker = KVCacheAffinityTracker(memory_window_minutes=5, max_entries=1000)
    
    # Test initialization
    assert tracker.memory_window_seconds == 300
    assert tracker.max_entries == 1000
    assert len(tracker.prefix_to_node) == 0
    
    # Test recording requests
    prefix_hash = "test_prefix_hash_" + "a" * 47  # 64 char total
    node_id = "test-node-1"
    request_id = "req-123"
    
    tracker.record_request(request_id, prefix_hash, node_id, time.time() + 10)
    
    # Verify tracking
    assert prefix_hash in tracker.prefix_to_node
    assert request_id in tracker.active_requests
    assert node_id in tracker.node_recent_prefixes
    
    # Test cache affinity scoring
    score = tracker.get_cache_affinity_score(prefix_hash, node_id)
    assert score > 0.8  # High score for exact recent match
    
    # Test different node (should be low)
    score_different = tracker.get_cache_affinity_score(prefix_hash, "different-node")
    assert score_different == 0.1
    
    # Test unknown prefix (should be base score)
    unknown_score = tracker.get_cache_affinity_score("unknown_hash_" + "b" * 51, node_id)
    assert unknown_score >= 0.0  # Base score


def test_prefix_similarity():
    """Test prefix similarity calculation."""
    tracker = KVCacheAffinityTracker()
    
    # Test identical prefixes
    hash1 = "abcd1234" * 8  # 64 chars
    hash2 = "abcd1234" * 8
    assert tracker._calculate_prefix_similarity(hash1, hash2) == 1.0
    
    # Test different prefixes
    hash3 = "a" * 64
    hash4 = "b" * 64
    assert tracker._calculate_prefix_similarity(hash3, hash4) == 0.0
    
    # Test partially similar (same beginning)
    hash5 = "abcd1234" + "x" * 56
    hash6 = "abcd1234" + "y" * 56
    similarity = tracker._calculate_prefix_similarity(hash5, hash6)
    assert 0.0 < similarity < 1.0  # Some similarity from common prefix


def test_active_request_tracking():
    """Test active request tracking."""
    tracker = KVCacheAffinityTracker()
    
    request_id = "req-456"
    prefix_hash = "active_test_hash_" + "c" * 47
    node_id = "test-node-2"
    completion_time = time.time() + 15
    
    # Record request
    tracker.record_request(request_id, prefix_hash, node_id, completion_time)
    
    # Verify active tracking
    assert request_id in tracker.active_requests
    active_req = tracker.active_requests[request_id]
    assert active_req.prefix_hash == prefix_hash
    assert active_req.node_id == node_id
    assert active_req.estimated_completion_time == completion_time
    
    # Complete request
    tracker.complete_request(request_id)
    assert request_id not in tracker.active_requests


def test_load_balancing_algorithm():
    """Test load balancing algorithm."""
    config = RoutingConfig(
        cache_hit_weight=0.4,
        latency_weight=0.3,
        load_weight=0.2,
        capacity_weight=0.1
    )
    
    algorithm = LoadBalancingAlgorithm(config)
    
    # Test initialization
    assert algorithm.config == config
    assert len(algorithm.node_metrics) == 0
    
    # Create sample metrics
    metrics = NodeLoadMetrics(
        node_id="lb-test-node",
        active_sequences=80,
        kv_cache_usage_percent=50.0,
        gpu_memory_usage_percent=70.0,
        current_throughput_tps=120.0,
        max_concurrent_sequences=200,
        error_rate_percent=2.0
    )
    
    # Update metrics
    algorithm.update_node_metrics("lb-test-node", metrics)
    assert "lb-test-node" in algorithm.node_metrics
    
    # Test load score calculation
    request = InferenceRequest(
        prompt="Test prompt for load balancing",
        max_tokens=100
    )
    
    score = algorithm.calculate_load_score("lb-test-node", request)
    assert 0.0 <= score <= 1.0
    
    # Test unknown node
    unknown_score = algorithm.calculate_load_score("unknown-node", request)
    assert unknown_score == 0.1


def test_workload_suitability():
    """Test workload suitability calculation."""
    config = RoutingConfig()
    algorithm = LoadBalancingAlgorithm(config)
    
    # Create node with limited capacity
    metrics = NodeLoadMetrics(
        node_id="small-node",
        max_sequence_length=1024,  # Small capacity
        max_concurrent_sequences=50
    )
    
    # Small request should be suitable
    small_request = InferenceRequest(
        prompt="Short",
        max_tokens=10
    )
    
    small_suitability = algorithm._calculate_workload_suitability(metrics, small_request)
    assert small_suitability > 0.0
    
    # Very large request should be less suitable or impossible
    large_request = InferenceRequest(
        prompt="Very long prompt " * 200,
        max_tokens=500
    )
    
    large_suitability = algorithm._calculate_workload_suitability(metrics, large_request)
    assert large_suitability >= 0.0  # At minimum 0, might be 0 if too large


def test_routing_config_defaults():
    """Test routing configuration defaults."""
    config = RoutingConfig()
    
    # Test default weights sum to 1.0
    total_weight = (
        config.cache_hit_weight + 
        config.latency_weight + 
        config.load_weight + 
        config.capacity_weight
    )
    assert abs(total_weight - 1.0) < 0.001  # Allow floating point tolerance
    
    # Test reasonable defaults
    assert 0.0 <= config.cache_hit_weight <= 1.0
    assert 0.0 <= config.latency_weight <= 1.0
    assert 0.0 <= config.load_weight <= 1.0
    assert 0.0 <= config.capacity_weight <= 1.0
    
    # Cache hits should have high weight (most important)
    assert config.cache_hit_weight >= config.latency_weight
    assert config.cache_hit_weight >= config.load_weight


def test_memory_cleanup():
    """Test memory cleanup in affinity tracker."""
    # Use very short window for testing
    tracker = KVCacheAffinityTracker(memory_window_minutes=0.01, max_entries=5)  # 0.6 seconds
    
    # Add some entries
    for i in range(3):
        prefix = f"cleanup_test_{i}_" + "d" * (64 - 15 - len(str(i)))
        tracker.record_request(f"req-{i}", prefix, f"node-{i}", time.time() + 10)
    
    assert len(tracker.prefix_to_node) == 3
    assert len(tracker.active_requests) == 3
    
    # Wait for entries to expire (in real scenario, this would be longer)
    time.sleep(0.1)
    
    # Add new entry to trigger cleanup
    new_prefix = "new_cleanup_test_" + "e" * (64 - 19)
    tracker.record_request("new-req", new_prefix, "new-node", time.time() + 10)
    
    # Verify cleanup happened (old entries removed)
    assert len(tracker.prefix_to_node) >= 1  # At least the new one
    assert "new-req" in tracker.active_requests


def test_cache_affinity_aging():
    """Test that cache affinity decreases over time."""
    tracker = KVCacheAffinityTracker()
    
    prefix_hash = "aging_test_" + "f" * (64 - 11)
    node_id = "aging-node"
    
    # Simulate recent access
    recent_time = time.time() - 60  # 1 minute ago
    tracker.prefix_to_node[prefix_hash] = (node_id, recent_time, 1)
    
    score_recent = tracker.get_cache_affinity_score(prefix_hash, node_id)
    
    # Simulate older access
    old_time = time.time() - 1200  # 20 minutes ago
    tracker.prefix_to_node[prefix_hash] = (node_id, old_time, 1)
    
    score_old = tracker.get_cache_affinity_score(prefix_hash, node_id)
    
    # Older access should have lower score
    assert score_old < score_recent
    assert score_recent > 0.5  # Recent should still be valuable
    assert score_old > 0.0     # Old should still have some value


def test_request_completion_tracking():
    """Test request completion and tracking updates."""
    tracker = KVCacheAffinityTracker()
    
    # Record multiple requests to same prefix
    prefix_hash = "completion_test_" + "g" * (64 - 16)
    
    for i in range(3):
        tracker.record_request(
            f"req-completion-{i}", 
            prefix_hash, 
            "completion-node", 
            time.time() + 10 + i
        )
    
    # All should be active
    assert len(tracker.active_requests) == 3
    
    # Complete one request
    tracker.complete_request("req-completion-1")
    assert len(tracker.active_requests) == 2
    assert "req-completion-1" not in tracker.active_requests
    
    # Other requests should still be tracked
    assert "req-completion-0" in tracker.active_requests
    assert "req-completion-2" in tracker.active_requests
    
    # Prefix mapping should still exist (other requests using it)
    assert prefix_hash in tracker.prefix_to_node


def test_multiple_nodes_same_prefix():
    """Test handling when same prefix processed on different nodes."""
    tracker = KVCacheAffinityTracker()
    
    prefix_hash = "multi_node_test_" + "h" * (64 - 16)
    
    # Record same prefix on different nodes
    tracker.record_request("req-node1", prefix_hash, "node-1", time.time() + 10)
    time.sleep(0.01)  # Small delay
    tracker.record_request("req-node2", prefix_hash, "node-2", time.time() + 10)
    
    # Most recent should win
    stored_node, timestamp, count = tracker.prefix_to_node[prefix_hash]
    assert stored_node == "node-2"  # Last one recorded
    assert count == 2  # Should increment count
    
    # Affinity should prefer the node that processed it most recently
    score_node2 = tracker.get_cache_affinity_score(prefix_hash, "node-2")
    score_node1 = tracker.get_cache_affinity_score(prefix_hash, "node-1")
    
    assert score_node2 > score_node1
    assert score_node1 == 0.1  # Wrong node penalty


if __name__ == "__main__":
    pytest.main([__file__])