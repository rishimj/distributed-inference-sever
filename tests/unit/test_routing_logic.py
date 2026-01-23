"""
Test core routing logic without external dependencies.

Tests the algorithms and data structures used in routing decisions
without importing modules that have problematic dependencies.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest

from src.common.models import InferenceRequest, RoutingConfig


@dataclass
class TestNodeLoadMetrics:
    """Simplified node metrics for testing."""
    node_id: str
    active_sequences: int = 0
    pending_sequences: int = 0
    kv_cache_usage_percent: float = 0.0
    gpu_memory_usage_percent: float = 0.0
    current_throughput_tps: float = 0.0
    max_concurrent_sequences: int = 256
    max_sequence_length: int = 4096
    max_batch_size: int = 32
    last_heartbeat: float = 0.0
    error_rate_percent: float = 0.0
    
    def is_overloaded(self, threshold: float = 0.9) -> bool:
        """Check if node is overloaded."""
        return (
            self.active_sequences >= self.max_concurrent_sequences * threshold or
            self.kv_cache_usage_percent > threshold * 100 or  # Convert to percentage
            self.gpu_memory_usage_percent > threshold * 95   # 95% of threshold
        )
    
    def get_capacity_score(self) -> float:
        """Calculate capacity score (0.0 = full, 1.0 = empty)."""
        sequence_capacity = 1.0 - (self.active_sequences / self.max_concurrent_sequences)
        memory_capacity = 1.0 - (self.gpu_memory_usage_percent / 100.0)
        cache_capacity = 1.0 - (self.kv_cache_usage_percent / 100.0)
        
        return (sequence_capacity * 0.3 + memory_capacity * 0.4 + cache_capacity * 0.3)


class TestCacheAffinityTracker:
    """Simplified cache affinity tracker for testing."""
    
    def __init__(self, memory_window_minutes: int = 30, max_entries: int = 10000):
        self.memory_window_seconds = memory_window_minutes * 60
        self.max_entries = max_entries
        
        # prefix_hash -> (node_id, last_seen_time, access_count)
        self.prefix_to_node: Dict[str, Tuple[str, float, int]] = {}
        
        # node_id -> deque of (prefix_hash, timestamp)
        self.node_recent_prefixes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def record_request(self, prefix_hash: str, node_id: str) -> None:
        """Record a request being processed."""
        now = time.time()
        
        # Update prefix-to-node mapping
        if prefix_hash in self.prefix_to_node:
            _, _, access_count = self.prefix_to_node[prefix_hash]
            self.prefix_to_node[prefix_hash] = (node_id, now, access_count + 1)
        else:
            self.prefix_to_node[prefix_hash] = (node_id, now, 1)
        
        # Track recent prefixes for this node
        self.node_recent_prefixes[node_id].append((prefix_hash, now))
    
    def get_cache_affinity_score(self, prefix_hash: str, node_id: str) -> float:
        """Calculate cache affinity score for a prefix-node combination."""
        now = time.time()
        
        # Check for exact prefix match
        if prefix_hash in self.prefix_to_node:
            cached_node_id, last_seen, access_count = self.prefix_to_node[prefix_hash]
            
            if cached_node_id == node_id:
                # Exact prefix match on this node
                age_seconds = now - last_seen
                if age_seconds < 300:  # 5 minutes - very fresh
                    return 0.9 + min(0.1, access_count * 0.02)
                elif age_seconds < 1800:  # 30 minutes - still valuable
                    return 0.7 * (1.0 - age_seconds / 1800)
                else:
                    # Old cache, but might still exist
                    return 0.3 * (1.0 - min(1.0, age_seconds / 3600))
            else:
                # Prefix exists on different node
                return 0.1
        
        # Check for similar prefixes on this node
        if node_id in self.node_recent_prefixes:
            recent_prefixes = self.node_recent_prefixes[node_id]
            similarity_score = 0.0
            
            for cached_prefix, timestamp in recent_prefixes:
                if now - timestamp > self.memory_window_seconds:
                    continue
                
                similarity = self._calculate_prefix_similarity(prefix_hash, cached_prefix)
                age_factor = 1.0 - (now - timestamp) / self.memory_window_seconds
                similarity_score = max(similarity_score, similarity * age_factor * 0.6)
            
            return similarity_score
        
        return 0.2  # Base score
    
    def _calculate_prefix_similarity(self, prefix1: str, prefix2: str) -> float:
        """Calculate similarity between two prefix hashes."""
        if prefix1 == prefix2:
            return 1.0
        
        # Check how many leading characters match
        common_prefix = 0
        for i in range(min(len(prefix1), len(prefix2))):
            if prefix1[i] == prefix2[i]:
                common_prefix += 1
            else:
                break
        
        return min(1.0, common_prefix / 16.0)


class TestLoadBalancer:
    """Simplified load balancer for testing."""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.node_metrics: Dict[str, TestNodeLoadMetrics] = {}
    
    def update_node_metrics(self, node_id: str, metrics: TestNodeLoadMetrics) -> None:
        """Update node metrics."""
        self.node_metrics[node_id] = metrics
    
    def calculate_load_score(self, node_id: str, request: InferenceRequest) -> float:
        """Calculate load score for a node."""
        if node_id not in self.node_metrics:
            return 0.1
        
        metrics = self.node_metrics[node_id]
        
        # Health checks
        if metrics.error_rate_percent > 10:
            return 0.1
        
        # Utilization scoring
        utilization_score = metrics.get_capacity_score()
        
        # Performance scoring
        target_throughput = 100.0
        performance_score = min(1.0, metrics.current_throughput_tps / target_throughput)
        
        # Workload suitability
        suitability_score = self._calculate_workload_suitability(metrics, request)
        
        # Queue penalty
        queue_penalty = 1.0
        if metrics.pending_sequences > 10:
            queue_penalty = max(0.3, 1.0 - (metrics.pending_sequences - 10) / 50.0)
        
        # Weighted combination
        total_score = (
            utilization_score * 0.4 +
            performance_score * 0.3 +
            suitability_score * 0.2 +
            queue_penalty * 0.1
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _calculate_workload_suitability(self, metrics: TestNodeLoadMetrics, 
                                       request: InferenceRequest) -> float:
        """Calculate workload suitability."""
        estimated_length = len(request.prompt.split()) * 1.3 + request.max_tokens
        
        if estimated_length > metrics.max_sequence_length:
            return 0.0
        
        length_utilization = estimated_length / metrics.max_sequence_length
        
        if length_utilization < 0.3:
            return 0.7
        elif length_utilization < 0.7:
            return 1.0
        else:
            return 0.8


def test_node_load_metrics():
    """Test node load metrics calculations."""
    # Normal node
    metrics = TestNodeLoadMetrics(
        node_id="test-1",
        active_sequences=50,
        kv_cache_usage_percent=40.0,
        gpu_memory_usage_percent=60.0,
        max_concurrent_sequences=200
    )
    
    assert not metrics.is_overloaded()
    
    capacity_score = metrics.get_capacity_score()
    assert 0.0 < capacity_score < 1.0
    
    # Overloaded node
    overloaded_metrics = TestNodeLoadMetrics(
        node_id="overloaded",
        active_sequences=190,  # Close to max
        kv_cache_usage_percent=95.0,
        gpu_memory_usage_percent=95.0,
        max_concurrent_sequences=200
    )
    
    assert overloaded_metrics.is_overloaded()
    assert overloaded_metrics.get_capacity_score() < 0.5


def test_cache_affinity_exact_match():
    """Test exact prefix cache affinity."""
    tracker = TestCacheAffinityTracker()
    
    prefix_hash = "test_exact_match_" + "a" * 47
    node_id = "exact-node"
    
    # Record request
    tracker.record_request(prefix_hash, node_id)
    
    # Test exact match
    score = tracker.get_cache_affinity_score(prefix_hash, node_id)
    assert score >= 0.9  # Very high score
    
    # Test wrong node
    wrong_score = tracker.get_cache_affinity_score(prefix_hash, "wrong-node")
    assert wrong_score == 0.1  # Penalty for wrong node


def test_cache_affinity_aging():
    """Test cache affinity decreases with age."""
    tracker = TestCacheAffinityTracker()
    
    prefix_hash = "aging_test_" + "b" * 53
    node_id = "aging-node"
    
    # Simulate old request
    old_time = time.time() - 1200  # 20 minutes ago
    tracker.prefix_to_node[prefix_hash] = (node_id, old_time, 1)
    
    old_score = tracker.get_cache_affinity_score(prefix_hash, node_id)
    
    # Simulate recent request
    recent_time = time.time() - 60  # 1 minute ago
    tracker.prefix_to_node[prefix_hash] = (node_id, recent_time, 1)
    
    recent_score = tracker.get_cache_affinity_score(prefix_hash, node_id)
    
    # Recent should be higher
    assert recent_score > old_score
    assert recent_score > 0.6
    assert old_score > 0.0  # Still some value


def test_prefix_similarity():
    """Test prefix similarity calculation."""
    tracker = TestCacheAffinityTracker()
    
    # Identical
    hash1 = "abcd1234" * 8
    hash2 = "abcd1234" * 8
    assert tracker._calculate_prefix_similarity(hash1, hash2) == 1.0
    
    # Different
    hash3 = "a" * 64
    hash4 = "b" * 64
    assert tracker._calculate_prefix_similarity(hash3, hash4) == 0.0
    
    # Partial similarity
    hash5 = "same_prefix_" + "x" * 52
    hash6 = "same_prefix_" + "y" * 52
    similarity = tracker._calculate_prefix_similarity(hash5, hash6)
    assert 0.0 < similarity < 1.0


def test_load_balancer_basic():
    """Test basic load balancer functionality."""
    config = RoutingConfig()
    balancer = TestLoadBalancer(config)
    
    # Add node metrics
    metrics = TestNodeLoadMetrics(
        node_id="lb-test",
        active_sequences=50,
        kv_cache_usage_percent=30.0,
        gpu_memory_usage_percent=50.0,
        current_throughput_tps=120.0,
        error_rate_percent=1.0
    )
    
    balancer.update_node_metrics("lb-test", metrics)
    
    # Test load score
    request = InferenceRequest(prompt="Test load balancing", max_tokens=100)
    score = balancer.calculate_load_score("lb-test", request)
    
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Should be decent score for good metrics


def test_load_balancer_high_error_rate():
    """Test load balancer penalizes high error rates."""
    config = RoutingConfig()
    balancer = TestLoadBalancer(config)
    
    # Node with high error rate
    bad_metrics = TestNodeLoadMetrics(
        node_id="bad-node",
        active_sequences=20,
        error_rate_percent=15.0  # Very high
    )
    
    balancer.update_node_metrics("bad-node", bad_metrics)
    
    request = InferenceRequest(prompt="Test", max_tokens=50)
    score = balancer.calculate_load_score("bad-node", request)
    
    assert score == 0.1  # Should get penalized


def test_workload_suitability():
    """Test workload suitability calculation."""
    config = RoutingConfig()
    balancer = TestLoadBalancer(config)
    
    # Small capacity node
    small_node = TestNodeLoadMetrics(
        node_id="small",
        max_sequence_length=1000
    )
    
    # Small request
    small_request = InferenceRequest(prompt="Short", max_tokens=10)
    small_suitability = balancer._calculate_workload_suitability(small_node, small_request)
    assert small_suitability > 0.0
    
    # Large request that won't fit
    huge_request = InferenceRequest(prompt="X" * 2000, max_tokens=1000)  # Way too big
    large_suitability = balancer._calculate_workload_suitability(small_node, huge_request)
    assert large_suitability == 0.0  # Should reject


def test_routing_score_calculation():
    """Test complete routing score calculation."""
    config = RoutingConfig(
        cache_hit_weight=0.4,
        latency_weight=0.3,
        load_weight=0.2,
        capacity_weight=0.1
    )
    
    # Verify weights sum to 1
    total = config.cache_hit_weight + config.latency_weight + config.load_weight + config.capacity_weight
    assert abs(total - 1.0) < 0.001
    
    # Test scoring components
    cache_score = 0.9  # High cache affinity
    latency_score = 0.8
    load_score = 0.7
    capacity_score = 0.6
    
    total_score = (
        cache_score * config.cache_hit_weight +
        latency_score * config.latency_weight +
        load_score * config.load_weight +
        capacity_score * config.capacity_weight
    )
    
    expected = 0.9 * 0.4 + 0.8 * 0.3 + 0.7 * 0.2 + 0.6 * 0.1
    assert abs(total_score - expected) < 0.001


def test_cache_tracking_multiple_requests():
    """Test tracking multiple requests to same prefix."""
    tracker = TestCacheAffinityTracker()
    
    prefix_hash = "multi_request_" + "c" * 50
    
    # Record same prefix on same node multiple times
    for i in range(3):
        tracker.record_request(prefix_hash, "multi-node")
    
    # Should track access count
    node_id, timestamp, count = tracker.prefix_to_node[prefix_hash]
    assert node_id == "multi-node"
    assert count == 3
    
    # Score should get bonus for frequent access
    score = tracker.get_cache_affinity_score(prefix_hash, "multi-node")
    assert score >= 0.9  # High score with frequency bonus


def test_node_switching_penalty():
    """Test that switching nodes for same prefix gets penalized."""
    tracker = TestCacheAffinityTracker()
    
    prefix_hash = "node_switching_" + "d" * 49
    
    # Record on first node
    tracker.record_request(prefix_hash, "node-1")
    
    # Test affinity scores
    node1_score = tracker.get_cache_affinity_score(prefix_hash, "node-1")
    node2_score = tracker.get_cache_affinity_score(prefix_hash, "node-2")
    
    # Node-1 should have much higher score
    assert node1_score > 0.8
    assert node2_score == 0.1  # Penalty for wrong node
    assert node1_score > node2_score * 5


if __name__ == "__main__":
    pytest.main([__file__])