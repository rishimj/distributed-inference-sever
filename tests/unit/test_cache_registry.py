"""
Unit tests for cache registry service.

These tests verify:
1. Cache entry registration and lookup
2. Node management and heartbeats
3. Query interfaces for routing decisions
4. Statistics tracking and reporting
5. Background cleanup and maintenance

Note: These tests use a mock Redis for isolation
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.models import (
    CacheEntry,
    CacheKey,
    CacheLevel,
    NodeInfo,
    ServiceType,
)


class MockRedis:
    """Mock Redis client for testing."""
    
    def __init__(self):
        self.data = {}
        self.sets = {}  # For Redis sets
        
    async def ping(self):
        return True
    
    async def close(self):
        pass
    
    async def setex(self, key, ttl, value):
        self.data[key] = value
        return True
    
    async def get(self, key):
        return self.data.get(key)
    
    async def delete(self, key):
        self.data.pop(key, None)
        return True
    
    async def scan(self, cursor, match=None):
        keys = []
        if match:
            # Simple pattern matching for tests
            pattern = match.replace("*", "")
            keys = [k for k in self.data.keys() if pattern in k]
        else:
            keys = list(self.data.keys())
        
        return 0, keys  # cursor=0 means no more results
    
    async def smembers(self, key):
        return self.sets.get(key, set())
    
    async def sadd(self, key, value):
        if key not in self.sets:
            self.sets[key] = set()
        self.sets[key].add(value)
        return 1
    
    async def srem(self, key, value):
        if key in self.sets:
            self.sets[key].discard(value)
            return 1
        return 0
    
    async def expire(self, key, seconds):
        return True
    
    async def info(self, section=None):
        return {'used_memory': 1024000}  # 1MB


@pytest.fixture
def mock_redis():
    """Fixture providing a mock Redis client."""
    return MockRedis()


@pytest.fixture
def cache_registry(mock_redis):
    """Fixture providing a cache registry service with mock Redis."""
    with patch('src.cache.registry.aioredis.from_url', return_value=mock_redis):
        # Import here to avoid the import error
        from src.cache.registry import CacheRegistryService
        service = CacheRegistryService(
            redis_url="redis://localhost:6379",
            namespace="test_cache",
            ttl_seconds=3600
        )
        service.redis = mock_redis
        return service


@pytest.fixture
def sample_cache_entry():
    """Fixture providing a sample cache entry."""
    cache_key = CacheKey(
        prefix_hash="a" * 64,
        model_name="test-model",
        sequence_length=512
    )
    
    return CacheEntry(
        cache_key=cache_key,
        kv_data=b"test_kv_data",
        cache_level=CacheLevel.L1_GPU,
        node_id="test-node-1",
        size_bytes=1024
    )


@pytest.fixture
def sample_node_info():
    """Fixture providing sample node information."""
    return NodeInfo(
        node_id="test-node-1",
        service_type=ServiceType.PREFILL,
        hostname="test-host",
        port=8080,
        gpu_memory_gb=40.0,
        cpu_cores=16,
        memory_gb=128.0,
        is_healthy=True,
        current_load=0.5
    )


class TestCacheRegistryService:
    """Test the cache registry service."""
    
    async def test_start_and_stop(self, cache_registry, mock_redis):
        """Test service start and stop.""" 
        await cache_registry.start()
        assert cache_registry.redis is mock_redis
        
        await cache_registry.stop()
    
    async def test_register_cache_entry(self, cache_registry, sample_cache_entry):
        """Test cache entry registration."""
        await cache_registry.start()
        
        result = await cache_registry.register_cache_entry(sample_cache_entry)
        assert result is True
        
        # Check that entry is stored in Redis
        key = cache_registry._make_redis_key("entry", sample_cache_entry.cache_key.to_string())
        stored_data = await cache_registry.redis.get(key)
        assert stored_data is not None
        
        entry_data = json.loads(stored_data)
        assert entry_data['node_id'] == "test-node-1"
        assert entry_data['size_bytes'] == 1024
        
        # Check local cache
        assert sample_cache_entry.cache_key.to_string() in cache_registry._local_cache
        
        # Check statistics
        assert cache_registry._stats['entries_added'] == 1
    
    async def test_get_cache_entry_from_local_cache(self, cache_registry, sample_cache_entry):
        """Test cache entry retrieval from local cache."""
        await cache_registry.start()
        
        # Register entry first
        await cache_registry.register_cache_entry(sample_cache_entry)
        
        # Get entry (should hit local cache)
        retrieved_entry = await cache_registry.get_cache_entry(sample_cache_entry.cache_key)
        
        assert retrieved_entry is not None
        assert retrieved_entry.node_id == "test-node-1"
        assert retrieved_entry.size_bytes == 1024
        assert cache_registry._stats['cache_hits'] == 1
    
    async def test_get_cache_entry_from_redis(self, cache_registry, sample_cache_entry):
        """Test cache entry retrieval from Redis when not in local cache."""
        await cache_registry.start()
        
        # Register entry first
        await cache_registry.register_cache_entry(sample_cache_entry)
        
        # Clear local cache to force Redis lookup
        cache_registry._local_cache.clear()
        cache_registry._local_cache_ttl.clear()
        
        # Get entry (should hit Redis)
        retrieved_entry = await cache_registry.get_cache_entry(sample_cache_entry.cache_key)
        
        assert retrieved_entry is not None
        assert retrieved_entry.node_id == "test-node-1"
        assert retrieved_entry.size_bytes == 1024
        
        # Should now be in local cache
        assert sample_cache_entry.cache_key.to_string() in cache_registry._local_cache
    
    async def test_get_cache_entry_miss(self, cache_registry):
        """Test cache entry retrieval when entry doesn't exist."""
        await cache_registry.start()
        
        cache_key = CacheKey(
            prefix_hash="b" * 64,
            model_name="nonexistent",
            sequence_length=256
        )
        
        retrieved_entry = await cache_registry.get_cache_entry(cache_key)
        
        assert retrieved_entry is None
        assert cache_registry._stats['cache_misses'] == 1
    
    async def test_update_cache_access(self, cache_registry, sample_cache_entry):
        """Test updating cache access statistics."""
        await cache_registry.start()
        
        # Register entry first
        await cache_registry.register_cache_entry(sample_cache_entry)
        
        # Update access
        result = await cache_registry.update_cache_access(sample_cache_entry.cache_key)
        assert result is True
        
        # Get updated entry
        updated_entry = await cache_registry.get_cache_entry(sample_cache_entry.cache_key)
        assert updated_entry.access_count > 0
        assert updated_entry.last_accessed > sample_cache_entry.last_accessed
    
    async def test_unregister_cache_entry(self, cache_registry, sample_cache_entry):
        """Test cache entry unregistration."""
        await cache_registry.start()
        
        # Register entry first
        await cache_registry.register_cache_entry(sample_cache_entry)
        
        # Unregister entry
        result = await cache_registry.unregister_cache_entry(sample_cache_entry.cache_key)
        assert result is True
        
        # Entry should no longer exist
        retrieved_entry = await cache_registry.get_cache_entry(sample_cache_entry.cache_key)
        assert retrieved_entry is None
        
        # Check statistics
        assert cache_registry._stats['entries_removed'] == 1
    
    async def test_find_entries_by_prefix(self, cache_registry):
        """Test finding entries by prefix hash."""
        await cache_registry.start()
        
        prefix_hash = "c" * 64
        
        # Create multiple entries with same prefix
        for i in range(3):
            cache_key = CacheKey(
                prefix_hash=prefix_hash,
                model_name=f"model-{i}",
                sequence_length=256 + i * 128
            )
            entry = CacheEntry(
                cache_key=cache_key,
                kv_data=b"data",
                cache_level=CacheLevel.L2_CPU,
                node_id=f"node-{i}",
                size_bytes=512
            )
            await cache_registry.register_cache_entry(entry)
        
        # Create entry with different prefix
        other_key = CacheKey(
            prefix_hash="d" * 64,
            model_name="other-model",
            sequence_length=128
        )
        other_entry = CacheEntry(
            cache_key=other_key,
            kv_data=b"other",
            cache_level=CacheLevel.L1_GPU,
            node_id="other-node",
            size_bytes=256
        )
        await cache_registry.register_cache_entry(other_entry)
        
        # Find by prefix
        matching_entries = await cache_registry.find_entries_by_prefix(prefix_hash)
        
        assert len(matching_entries) == 3
        assert all(e.cache_key.prefix_hash == prefix_hash for e in matching_entries)
    
    async def test_register_node(self, cache_registry, sample_node_info):
        """Test node registration."""
        await cache_registry.start()
        
        result = await cache_registry.register_node(sample_node_info)
        assert result is True
        
        # Check Redis storage
        key = cache_registry._make_redis_key("node", sample_node_info.node_id)
        stored_data = await cache_registry.redis.get(key)
        assert stored_data is not None
        
        node_data = json.loads(stored_data)
        assert node_data['node_id'] == "test-node-1"
        assert node_data['hostname'] == "test-host"
        
        # Check local tracking
        assert sample_node_info.node_id in cache_registry._nodes
    
    async def test_update_node_heartbeat(self, cache_registry, sample_node_info):
        """Test node heartbeat updates."""
        await cache_registry.start()
        
        # Register node first
        await cache_registry.register_node(sample_node_info)
        
        # Update heartbeat with new load
        result = await cache_registry.update_node_heartbeat("test-node-1", load=0.8)
        assert result is True
        
        # Check that load was updated
        key = cache_registry._make_redis_key("node", "test-node-1")
        stored_data = await cache_registry.redis.get(key)
        node_data = json.loads(stored_data)
        
        assert node_data['current_load'] == 0.8
        assert node_data['last_heartbeat'] > sample_node_info.last_heartbeat
    
    async def test_get_healthy_nodes(self, cache_registry):
        """Test getting healthy nodes."""
        await cache_registry.start()
        
        # Register multiple nodes
        nodes = [
            NodeInfo(
                node_id=f"node-{i}",
                service_type=ServiceType.PREFILL if i % 2 == 0 else ServiceType.DECODE,
                hostname=f"host-{i}",
                port=8080 + i,
                is_healthy=i < 3,  # First 3 nodes are healthy
                last_heartbeat=time.time()
            )
            for i in range(5)
        ]
        
        for node in nodes:
            await cache_registry.register_node(node)
        
        # Get all healthy nodes
        healthy_nodes = await cache_registry.get_healthy_nodes()
        assert len(healthy_nodes) == 3
        
        # Get healthy prefill nodes only
        prefill_nodes = await cache_registry.get_healthy_nodes(service_type="prefill")
        assert len(prefill_nodes) == 2  # nodes 0 and 2
        assert all(node.service_type == ServiceType.PREFILL for node in prefill_nodes)
    
    async def test_get_cache_entries_by_node(self, cache_registry):
        """Test getting cache entries for a specific node."""
        await cache_registry.start()
        
        node_id = "test-node-1"
        
        # Create entries for the node
        for i in range(3):
            cache_key = CacheKey(
                prefix_hash=("e" * 63) + str(i),
                model_name="test",
                sequence_length=256
            )
            entry = CacheEntry(
                cache_key=cache_key,
                kv_data=b"data",
                cache_level=CacheLevel.L1_GPU,
                node_id=node_id,
                size_bytes=512
            )
            await cache_registry.register_cache_entry(entry)
        
        # Create entry for different node
        other_key = CacheKey(
            prefix_hash="f" * 64,
            model_name="test",
            sequence_length=128
        )
        other_entry = CacheEntry(
            cache_key=other_key,
            kv_data=b"other",
            cache_level=CacheLevel.L2_CPU,
            node_id="other-node",
            size_bytes=256
        )
        await cache_registry.register_cache_entry(other_entry)
        
        # Get entries for specific node
        node_entries = await cache_registry.get_cache_entries_by_node(node_id)
        
        assert len(node_entries) == 3
        assert all(entry.node_id == node_id for entry in node_entries)
    
    async def test_unregister_node(self, cache_registry, sample_node_info, sample_cache_entry):
        """Test node unregistration with cleanup."""
        await cache_registry.start()
        
        # Register node and entry
        await cache_registry.register_node(sample_node_info)
        await cache_registry.register_cache_entry(sample_cache_entry)
        
        # Unregister node
        result = await cache_registry.unregister_node("test-node-1")
        assert result is True
        
        # Node should no longer exist
        key = cache_registry._make_redis_key("node", "test-node-1")
        stored_data = await cache_registry.redis.get(key)
        assert stored_data is None
        
        # Cache entries for the node should be cleaned up
        retrieved_entry = await cache_registry.get_cache_entry(sample_cache_entry.cache_key)
        assert retrieved_entry is None
    
    async def test_get_cache_statistics(self, cache_registry, sample_cache_entry):
        """Test cache statistics retrieval."""
        await cache_registry.start()
        
        # Add some entries and activity
        await cache_registry.register_cache_entry(sample_cache_entry)
        await cache_registry.get_cache_entry(sample_cache_entry.cache_key)  # Hit
        
        # Try to get non-existent entry
        nonexistent_key = CacheKey(
            prefix_hash="g" * 64,
            model_name="nonexistent",
            sequence_length=128
        )
        await cache_registry.get_cache_entry(nonexistent_key)  # Miss
        
        stats = await cache_registry.get_cache_statistics()
        
        assert stats['entries_added'] == 1
        assert stats['cache_hits'] >= 1
        assert stats['cache_misses'] >= 1
        assert stats['queries_served'] >= 2
        assert stats['total_entries'] >= 1
        assert 'local_cache_hit_rate' in stats
    
    def test_make_redis_key(self, cache_registry):
        """Test Redis key generation."""
        key = cache_registry._make_redis_key("entry", "test-key")
        assert key == "test_cache:entry:test-key"
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_entries(self, cache_registry):
        """Test local cache cleanup of expired entries."""
        await cache_registry.start()
        
        # Add entry to local cache with past expiry
        cache_key = "test-key"
        cache_registry._local_cache[cache_key] = "test-value"
        cache_registry._local_cache_ttl[cache_key] = time.time() - 100  # Expired
        
        # Run cleanup manually (normally runs in background)
        await cache_registry._cleanup_expired_entries()
        
        # Expired entry should be removed
        assert cache_key not in cache_registry._local_cache
        assert cache_key not in cache_registry._local_cache_ttl


class TestCacheRegistryServiceErrors:
    """Test error handling in cache registry service."""
    
    async def test_start_with_redis_failure(self):
        """Test service start with Redis connection failure."""
        with patch('src.cache.registry.aioredis.from_url') as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping.side_effect = Exception("Connection failed")
            mock_from_url.return_value = mock_redis
            
            from src.cache.registry import CacheRegistryService
            service = CacheRegistryService()
            
            with pytest.raises(Exception, match="Connection failed"):
                await service.start()
    
    async def test_register_cache_entry_failure(self, cache_registry, sample_cache_entry):
        """Test cache entry registration with Redis failure."""
        await cache_registry.start()
        
        # Mock Redis failure
        cache_registry.redis.setex = AsyncMock(side_effect=Exception("Redis error"))
        
        result = await cache_registry.register_cache_entry(sample_cache_entry)
        assert result is False
    
    async def test_get_cache_entry_failure(self, cache_registry):
        """Test cache entry retrieval with Redis failure."""
        await cache_registry.start()
        
        cache_key = CacheKey(
            prefix_hash="h" * 64,
            model_name="test",
            sequence_length=256
        )
        
        # Mock Redis failure
        cache_registry.redis.get = AsyncMock(side_effect=Exception("Redis error"))
        
        result = await cache_registry.get_cache_entry(cache_key)
        assert result is None
        assert cache_registry._stats['cache_misses'] == 1


if __name__ == "__main__":
    pytest.main([__file__])