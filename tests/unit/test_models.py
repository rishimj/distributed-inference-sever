"""
Unit tests for core data models.

These tests verify:
1. Model validation works correctly
2. Serialization/deserialization is working  
3. Business logic methods behave as expected
4. Edge cases and error conditions are handled
"""

import time
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.common.models import (
    CacheEntry,
    CacheKey,
    CacheLevel,
    CacheRegistry,
    InferenceRequest,
    InferenceResponse,
    NodeInfo,
    RouteDecision,
    ServiceType,
)


class TestInferenceRequest:
    """Test the InferenceRequest model."""
    
    def test_valid_request_creation(self):
        """Test creating a valid inference request."""
        request = InferenceRequest(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.8
        )
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 100
        assert request.temperature == 0.8
        assert request.request_id is not None
        assert isinstance(request.timestamp, float)
    
    def test_request_validation_prompt_too_long(self):
        """Test that overly long prompts are rejected."""
        long_prompt = "x" * 100001  # Exceeds 100K limit
        
        with pytest.raises(ValidationError) as exc_info:
            InferenceRequest(prompt=long_prompt)
        
        assert "Prompt too long" in str(exc_info.value)
    
    def test_request_validation_empty_prompt(self):
        """Test that empty prompts are rejected."""
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="")
    
    def test_request_validation_invalid_temperature(self):
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="test", temperature=3.0)  # > 2.0
        
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="test", temperature=-0.1)  # < 0.0
    
    def test_compute_prefix_hash(self):
        """Test prefix hash computation."""
        request = InferenceRequest(prompt="Hello world, this is a test prompt")
        
        hash1 = request.compute_prefix_hash(prefix_length=10)
        hash2 = request.compute_prefix_hash(prefix_length=10)
        
        # Same prefix should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 char hex string
        
        # Different prefix lengths should produce different hashes
        hash3 = request.compute_prefix_hash(prefix_length=5)
        assert hash1 != hash3
    
    def test_compute_prefix_hash_short_prompt(self):
        """Test prefix hash with prompt shorter than prefix_length."""
        request = InferenceRequest(prompt="short")
        hash_val = request.compute_prefix_hash(prefix_length=100)
        
        # Should not error, just use entire prompt
        assert len(hash_val) == 64


class TestCacheKey:
    """Test the CacheKey model."""
    
    def test_cache_key_creation(self):
        """Test creating a cache key."""
        key = CacheKey(
            prefix_hash="a" * 64,  # Valid SHA-256 length
            model_name="llama-7b",
            sequence_length=512
        )
        
        assert key.prefix_hash == "a" * 64
        assert key.model_name == "llama-7b"
        assert key.sequence_length == 512
    
    def test_cache_key_validation_invalid_hash_length(self):
        """Test that invalid hash lengths are rejected."""
        with pytest.raises(ValidationError):
            CacheKey(
                prefix_hash="short",  # Too short
                sequence_length=512
            )
        
        with pytest.raises(ValidationError):
            CacheKey(
                prefix_hash="a" * 70,  # Too long
                sequence_length=512
            )
    
    def test_cache_key_string_conversion(self):
        """Test conversion to/from string representation."""
        original = CacheKey(
            prefix_hash="b" * 64,
            model_name="gpt-3.5",
            sequence_length=1024
        )
        
        key_str = original.to_string()
        restored = CacheKey.from_string(key_str)
        
        assert restored == original
        assert "gpt-3.5" in key_str
        assert "1024" in key_str
    
    def test_cache_key_from_string_invalid_format(self):
        """Test parsing invalid string format."""
        with pytest.raises(ValueError):
            CacheKey.from_string("invalid:format")


class TestCacheEntry:
    """Test the CacheEntry model."""
    
    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        cache_key = CacheKey(
            prefix_hash="c" * 64,
            model_name="test-model",
            sequence_length=256
        )
        
        entry = CacheEntry(
            cache_key=cache_key,
            kv_data=b"test_kv_data",
            cache_level=CacheLevel.L1_GPU,
            node_id="node-1",
            size_bytes=1024
        )
        
        assert entry.cache_key == cache_key
        assert entry.kv_data == b"test_kv_data"
        assert entry.cache_level == CacheLevel.L1_GPU
        assert entry.node_id == "node-1"
        assert entry.size_bytes == 1024
        assert entry.access_count == 0
        assert entry.is_primary is True
    
    def test_update_access(self):
        """Test updating access statistics."""
        cache_key = CacheKey(
            prefix_hash="d" * 64,
            sequence_length=128
        )
        entry = CacheEntry(
            cache_key=cache_key,
            kv_data=b"test",
            cache_level=CacheLevel.L2_CPU,
            node_id="node-2",
            size_bytes=512
        )
        
        initial_access_time = entry.last_accessed
        initial_count = entry.access_count
        
        # Wait a tiny bit to ensure timestamp changes
        time.sleep(0.001)
        entry.update_access()
        
        assert entry.last_accessed > initial_access_time
        assert entry.access_count == initial_count + 1
    
    def test_age_seconds(self):
        """Test age calculation."""
        cache_key = CacheKey(
            prefix_hash="e" * 64,
            sequence_length=64
        )
        entry = CacheEntry(
            cache_key=cache_key,
            kv_data=b"test",
            cache_level=CacheLevel.L3_NETWORK,
            node_id="node-3",
            size_bytes=256
        )
        
        age = entry.age_seconds()
        assert age >= 0
        assert age < 1  # Should be very recent
    
    def test_recency_score(self):
        """Test recency score calculation."""
        cache_key = CacheKey(
            prefix_hash="f" * 64,
            sequence_length=32
        )
        entry = CacheEntry(
            cache_key=cache_key,
            kv_data=b"test",
            cache_level=CacheLevel.L1_GPU,
            node_id="node-4", 
            size_bytes=128
        )
        
        # Recent access should have high score
        score1 = entry.recency_score()
        assert 0.8 <= score1 <= 1.0
        
        # Simulate old access
        with patch('time.time', return_value=time.time() + 3600):  # 1 hour later
            score2 = entry.recency_score()
            assert score2 < score1
            assert 0.4 <= score2 <= 0.6  # Should be around 0.5 due to 1-hour half-life


class TestNodeInfo:
    """Test the NodeInfo model."""
    
    def test_node_info_creation(self):
        """Test creating node info."""
        node = NodeInfo(
            service_type=ServiceType.PREFILL,
            hostname="test-host",
            port=8080,
            gpu_memory_gb=40.0,
            cpu_cores=16
        )
        
        assert node.service_type == ServiceType.PREFILL
        assert node.hostname == "test-host"
        assert node.port == 8080
        assert node.gpu_memory_gb == 40.0
        assert node.cpu_cores == 16
        assert node.is_healthy is True
        assert node.current_load == 0.0
    
    def test_is_overloaded(self):
        """Test overload detection."""
        node = NodeInfo(
            service_type=ServiceType.DECODE,
            hostname="test",
            port=8080,
            current_load=0.95
        )
        
        assert node.is_overloaded() is True
        assert node.is_overloaded(threshold=0.98) is False
    
    def test_heartbeat_age(self):
        """Test heartbeat age calculation."""
        node = NodeInfo(
            service_type=ServiceType.GATEWAY,
            hostname="test",
            port=8080
        )
        
        age = node.heartbeat_age_seconds()
        assert age >= 0
        assert age < 1


class TestRouteDecision:
    """Test the RouteDecision model."""
    
    def test_route_decision_creation(self):
        """Test creating a route decision."""
        decision = RouteDecision(
            target_node="node-1",
            confidence=0.85,
            cache_hit_score=0.9,
            load_score=0.8,
            latency_score=0.7,
            capacity_score=0.6
        )
        
        assert decision.target_node == "node-1"
        assert decision.confidence == 0.85
        assert decision.cache_hit_score == 0.9
    
    def test_total_score_calculation(self):
        """Test total score calculation with weights."""
        decision = RouteDecision(
            target_node="node-1",
            confidence=1.0,
            cache_hit_score=1.0,  # 40% weight
            latency_score=1.0,    # 30% weight  
            load_score=1.0,       # 20% weight
            capacity_score=1.0    # 10% weight
        )
        
        # Perfect scores should give 1.0 (with floating point tolerance)
        assert abs(decision.total_score() - 1.0) < 0.001
        
        # Test with different scores
        decision2 = RouteDecision(
            target_node="node-2",
            confidence=0.7,
            cache_hit_score=0.8,
            latency_score=0.6,
            load_score=0.4, 
            capacity_score=0.2
        )
        
        expected = 0.8 * 0.4 + 0.6 * 0.3 + 0.4 * 0.2 + 0.2 * 0.1
        assert abs(decision2.total_score() - expected) < 0.001


class TestCacheRegistry:
    """Test the CacheRegistry model."""
    
    def test_cache_registry_creation(self):
        """Test creating an empty cache registry."""
        registry = CacheRegistry()
        
        assert len(registry.entries) == 0
        assert len(registry.nodes) == 0
        assert registry.total_cache_size_bytes == 0
    
    def test_add_remove_entry(self):
        """Test adding and removing cache entries."""
        registry = CacheRegistry()
        
        cache_key = CacheKey(
            prefix_hash="g" * 64,
            sequence_length=512
        )
        entry = CacheEntry(
            cache_key=cache_key,
            kv_data=b"test_data",
            cache_level=CacheLevel.L1_GPU,
            node_id="node-1",
            size_bytes=1024
        )
        
        # Add entry
        initial_time = registry.last_updated
        registry.add_entry(entry)
        
        assert len(registry.entries) == 1
        assert registry.total_cache_size_bytes == 1024
        assert registry.last_updated >= initial_time
        
        # Remove entry
        removed = registry.remove_entry(cache_key)
        
        assert removed == entry
        assert len(registry.entries) == 0
        assert registry.total_cache_size_bytes == 0
    
    def test_find_entries_by_prefix(self):
        """Test finding entries by prefix hash."""
        registry = CacheRegistry()
        prefix_hash = "h" * 64
        
        # Add entries with same prefix but different sequence lengths
        for seq_len in [128, 256, 512]:
            cache_key = CacheKey(
                prefix_hash=prefix_hash,
                sequence_length=seq_len
            )
            entry = CacheEntry(
                cache_key=cache_key,
                kv_data=b"data",
                cache_level=CacheLevel.L2_CPU,
                node_id="node-1",
                size_bytes=100
            )
            registry.add_entry(entry)
        
        # Add entry with different prefix
        other_key = CacheKey(
            prefix_hash="i" * 64,
            sequence_length=256
        )
        other_entry = CacheEntry(
            cache_key=other_key,
            kv_data=b"other",
            cache_level=CacheLevel.L2_CPU,
            node_id="node-1", 
            size_bytes=50
        )
        registry.add_entry(other_entry)
        
        # Find by prefix
        matches = registry.find_entries_by_prefix(prefix_hash)
        
        assert len(matches) == 3
        assert all(e.cache_key.prefix_hash == prefix_hash for e in matches)
    
    def test_get_node_cache_size(self):
        """Test calculating cache size per node."""
        registry = CacheRegistry()
        
        # Add entries for different nodes
        for i, node_id in enumerate(["node-1", "node-2", "node-1"]):
            cache_key = CacheKey(
                prefix_hash=("j" * 63) + str(i),
                sequence_length=128
            )
            entry = CacheEntry(
                cache_key=cache_key,
                kv_data=b"data",
                cache_level=CacheLevel.L1_GPU,
                node_id=node_id,
                size_bytes=1000
            )
            registry.add_entry(entry)
        
        assert registry.get_node_cache_size("node-1") == 2000  # 2 entries
        assert registry.get_node_cache_size("node-2") == 1000  # 1 entry
        assert registry.get_node_cache_size("node-3") == 0     # No entries
    
    def test_get_cache_utilization(self):
        """Test cache utilization calculation by level.""" 
        registry = CacheRegistry()
        
        # Add entries at different cache levels
        levels_and_sizes = [
            (CacheLevel.L1_GPU, 1000),
            (CacheLevel.L1_GPU, 2000), 
            (CacheLevel.L2_CPU, 5000),
            (CacheLevel.L3_NETWORK, 10000)
        ]
        
        for i, (level, size) in enumerate(levels_and_sizes):
            cache_key = CacheKey(
                prefix_hash=("k" * 63) + str(i),
                sequence_length=64
            )
            entry = CacheEntry(
                cache_key=cache_key,
                kv_data=b"data",
                cache_level=level,
                node_id="node-1",
                size_bytes=size
            )
            registry.add_entry(entry)
        
        utilization = registry.get_cache_utilization()
        
        assert utilization[CacheLevel.L1_GPU] == 3000    # 1000 + 2000
        assert utilization[CacheLevel.L2_CPU] == 5000    # 5000
        assert utilization[CacheLevel.L3_NETWORK] == 10000  # 10000
        assert utilization[CacheLevel.L4_COLD] == 0      # No entries


if __name__ == "__main__":
    pytest.main([__file__])