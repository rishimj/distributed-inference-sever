"""
Simple tests for cache registry service without Redis dependencies.

Tests core logic and data structures without external dependencies.
"""

import json
import time

import pytest

from src.common.models import (
    CacheEntry,
    CacheKey,
    CacheLevel,
    NodeInfo,
    ServiceType,
)


def test_cache_key_serialization():
    """Test that cache keys can be serialized properly for Redis storage."""
    cache_key = CacheKey(
        prefix_hash="a" * 64,
        model_name="test-model",
        sequence_length=512
    )
    
    # Test string conversion
    key_str = cache_key.to_string()
    assert "test-model" in key_str
    assert "a" * 64 in key_str
    assert "512" in key_str
    
    # Test round-trip conversion
    restored_key = CacheKey.from_string(key_str)
    assert restored_key == cache_key


def test_cache_entry_serialization():
    """Test that cache entries can be serialized for Redis storage."""
    cache_key = CacheKey(
        prefix_hash="b" * 64,
        model_name="test-model",
        sequence_length=256
    )
    
    entry = CacheEntry(
        cache_key=cache_key,
        kv_data=b"test_data",
        cache_level=CacheLevel.L1_GPU,
        node_id="test-node",
        size_bytes=1024
    )
    
    # Simulate Redis serialization
    entry_data = {
        'cache_key': cache_key.dict(),
        'cache_level': entry.cache_level.value,
        'node_id': entry.node_id,
        'size_bytes': entry.size_bytes,
        'created_at': entry.created_at,
        'last_accessed': entry.last_accessed,
        'access_count': entry.access_count,
        'hit_rate': entry.hit_rate,
        'replica_nodes': entry.replica_nodes,
        'is_primary': entry.is_primary
    }
    
    # Should be JSON serializable
    serialized = json.dumps(entry_data)
    deserialized = json.loads(serialized)
    
    assert deserialized['node_id'] == "test-node"
    assert deserialized['size_bytes'] == 1024
    assert deserialized['cache_level'] == "l1_gpu"


def test_node_info_serialization():
    """Test that node info can be serialized for Redis storage."""
    node = NodeInfo(
        node_id="test-node",
        service_type=ServiceType.PREFILL,
        hostname="test-host",
        port=8080,
        gpu_memory_gb=40.0,
        cpu_cores=16,
        is_healthy=True,
        current_load=0.5
    )
    
    # Simulate Redis serialization
    node_data = {
        'node_id': node.node_id,
        'service_type': node.service_type.value,
        'hostname': node.hostname,
        'port': node.port,
        'gpu_memory_gb': node.gpu_memory_gb,
        'cpu_cores': node.cpu_cores,
        'memory_gb': node.memory_gb,
        'is_healthy': node.is_healthy,
        'current_load': node.current_load,
        'last_heartbeat': node.last_heartbeat,
        'avg_latency_ms': node.avg_latency_ms,
        'throughput_rps': node.throughput_rps
    }
    
    # Should be JSON serializable
    serialized = json.dumps(node_data)
    deserialized = json.loads(serialized)
    
    assert deserialized['node_id'] == "test-node"
    assert deserialized['service_type'] == "prefill"
    assert deserialized['hostname'] == "test-host"


def test_redis_key_generation():
    """Test Redis key generation logic."""
    namespace = "test_cache"
    
    def make_redis_key(key_type: str, identifier: str) -> str:
        """Simulate the _make_redis_key method."""
        return f"{namespace}:{key_type}:{identifier}"
    
    # Test different key types
    entry_key = make_redis_key("entry", "model:hash123:256")
    assert entry_key == "test_cache:entry:model:hash123:256"
    
    node_key = make_redis_key("node", "node-123")
    assert node_key == "test_cache:node:node-123"
    
    index_key = make_redis_key("node_index", "node-123")
    assert index_key == "test_cache:node_index:node-123"


def test_cache_statistics_calculation():
    """Test cache statistics calculation logic."""
    stats = {
        'cache_hits': 85,
        'cache_misses': 15,
        'entries_added': 100,
        'entries_removed': 20,
        'queries_served': 100
    }
    
    # Calculate derived statistics
    hit_rate = stats['cache_hits'] / max(1, stats['queries_served'])
    miss_rate = stats['cache_misses'] / max(1, stats['queries_served'])
    
    assert hit_rate == 0.85
    assert miss_rate == 0.15
    assert hit_rate + miss_rate == 1.0


def test_access_pattern_tracking():
    """Test cache entry access pattern tracking."""
    cache_key = CacheKey(
        prefix_hash="c" * 64,
        model_name="test",
        sequence_length=128
    )
    
    entry = CacheEntry(
        cache_key=cache_key,
        kv_data=b"data",
        cache_level=CacheLevel.L2_CPU,
        node_id="node-1",
        size_bytes=512
    )
    
    # Initial state
    initial_count = entry.access_count
    initial_time = entry.last_accessed
    
    # Simulate access
    time.sleep(0.001)  # Ensure time difference
    entry.update_access()
    
    # Should update statistics
    assert entry.access_count == initial_count + 1
    assert entry.last_accessed > initial_time
    
    # Test age calculation
    age = entry.age_seconds()
    assert age >= 0
    assert age < 1  # Should be very recent
    
    # Test recency score
    recency = entry.recency_score()
    assert 0.0 <= recency <= 1.0


def test_node_health_checking():
    """Test node health checking logic."""
    node = NodeInfo(
        node_id="test-node",
        service_type=ServiceType.DECODE,
        hostname="test-host",
        port=8080,
        current_load=0.95,  # High load
        last_heartbeat=time.time() - 30  # 30 seconds ago
    )
    
    # Test overload detection
    assert node.is_overloaded() is True
    assert node.is_overloaded(threshold=0.98) is False
    
    # Test heartbeat age
    age = node.heartbeat_age_seconds()
    assert 25 <= age <= 35  # Should be around 30 seconds


def test_cache_entry_filtering():
    """Test filtering cache entries by various criteria."""
    entries = []
    
    # Create test entries
    for i in range(5):
        cache_key = CacheKey(
            prefix_hash=("d" * 63) + str(i),
            model_name=f"model-{i % 2}",  # Two different models
            sequence_length=256 + i * 64
        )
        entry = CacheEntry(
            cache_key=cache_key,
            kv_data=b"data",
            cache_level=CacheLevel.L1_GPU if i % 2 == 0 else CacheLevel.L2_CPU,
            node_id=f"node-{i % 3}",  # Three different nodes
            size_bytes=1024 * (i + 1)
        )
        entries.append(entry)
    
    # Filter by model name
    model_0_entries = [e for e in entries if e.cache_key.model_name == "model-0"]
    assert len(model_0_entries) == 3  # entries 0, 2, 4
    
    # Filter by cache level
    l1_entries = [e for e in entries if e.cache_level == CacheLevel.L1_GPU]
    assert len(l1_entries) == 3  # entries 0, 2, 4
    
    # Filter by node
    node_0_entries = [e for e in entries if e.node_id == "node-0"]
    assert len(node_0_entries) == 2  # entries 0, 3
    
    # Calculate total size per node
    node_sizes = {}
    for entry in entries:
        if entry.node_id not in node_sizes:
            node_sizes[entry.node_id] = 0
        node_sizes[entry.node_id] += entry.size_bytes
    
    assert len(node_sizes) == 3
    assert all(size > 0 for size in node_sizes.values())


if __name__ == "__main__":
    pytest.main([__file__])