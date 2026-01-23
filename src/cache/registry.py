"""
Cache registry service for distributed KV cache coordination.

This module provides:
- Global registry of cache entries across all nodes
- Cache location tracking and metadata management
- Query interface for cache-aware routing decisions
- Cache replication and consistency coordination

Design decisions:
- Redis as backing store for high availability and performance
- Async operations to avoid blocking main execution
- TTL-based cleanup for automatic garbage collection
- Eventual consistency model with configurable refresh intervals
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set
from uuid import uuid4

import aioredis
import structlog

from ..common.models import CacheEntry, CacheKey, CacheLevel, CacheRegistry, NodeInfo


logger = structlog.get_logger()


class CacheRegistryService:
    """
    Distributed cache registry service.
    
    This service coordinates cache entries across multiple nodes and provides
    a global view of cache availability for routing decisions.
    
    Design choice: Centralized registry with Redis backing
    Tradeoff: Single source of truth vs potential bottleneck
    Scaling approach: Redis cluster for horizontal scaling
    """
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 namespace: str = "cache_registry",
                 ttl_seconds: int = 3600):
        """
        Initialize cache registry service.
        
        Args:
            redis_url: Redis connection URL
            namespace: Redis key namespace prefix
            ttl_seconds: Default TTL for cache entries
        """
        self.redis_url = redis_url
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        self.redis: Optional[aioredis.Redis] = None
        
        # In-memory cache for frequently accessed data
        self._local_cache: Dict[str, CacheEntry] = {}
        self._local_cache_ttl: Dict[str, float] = {}
        self._local_ttl_seconds = 30  # Local cache TTL
        
        # Node tracking
        self._nodes: Dict[str, NodeInfo] = {}
        self._node_last_heartbeat: Dict[str, float] = {}
        
        # Statistics tracking
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'entries_added': 0,
            'entries_removed': 0,
            'queries_served': 0
        }
    
    async def start(self) -> None:
        """Start the cache registry service."""
        try:
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Cache registry service started", redis_url=self.redis_url)
            
            # Start background tasks
            asyncio.create_task(self._cleanup_expired_entries())
            asyncio.create_task(self._update_local_cache())
            
        except Exception as e:
            logger.error("Failed to start cache registry service", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the cache registry service."""
        if self.redis:
            await self.redis.close()
            logger.info("Cache registry service stopped")
    
    # Cache Entry Management
    
    async def register_cache_entry(self, entry: CacheEntry) -> bool:
        """
        Register a new cache entry in the registry.
        
        Args:
            entry: Cache entry to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            key = self._make_redis_key("entry", entry.cache_key.to_string())
            
            # Serialize entry to JSON
            entry_data = {
                'cache_key': entry.cache_key.dict(),
                'kv_data_size': len(entry.kv_data),  # Don't store actual data
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
            
            # Store in Redis with TTL
            await self.redis.setex(
                key, 
                self.ttl_seconds, 
                json.dumps(entry_data)
            )
            
            # Update local cache
            self._local_cache[entry.cache_key.to_string()] = entry
            self._local_cache_ttl[entry.cache_key.to_string()] = time.time() + self._local_ttl_seconds
            
            # Update statistics
            self._stats['entries_added'] += 1
            
            # Add to node index
            await self._add_to_node_index(entry.node_id, entry.cache_key)
            
            logger.info(
                "Cache entry registered",
                cache_key=entry.cache_key.to_string(),
                node_id=entry.node_id,
                size_bytes=entry.size_bytes
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to register cache entry",
                cache_key=entry.cache_key.to_string(),
                error=str(e)
            )
            return False
    
    async def unregister_cache_entry(self, cache_key: CacheKey) -> bool:
        """
        Unregister a cache entry from the registry.
        
        Args:
            cache_key: Cache key to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            key = self._make_redis_key("entry", cache_key.to_string())
            
            # Get entry info before deletion for cleanup
            entry_data = await self.redis.get(key)
            if entry_data:
                entry_info = json.loads(entry_data)
                node_id = entry_info['node_id']
                
                # Remove from node index
                await self._remove_from_node_index(node_id, cache_key)
            
            # Remove from Redis
            await self.redis.delete(key)
            
            # Remove from local cache
            self._local_cache.pop(cache_key.to_string(), None)
            self._local_cache_ttl.pop(cache_key.to_string(), None)
            
            # Update statistics
            self._stats['entries_removed'] += 1
            
            logger.info("Cache entry unregistered", cache_key=cache_key.to_string())
            return True
            
        except Exception as e:
            logger.error(
                "Failed to unregister cache entry",
                cache_key=cache_key.to_string(),
                error=str(e)
            )
            return False
    
    async def get_cache_entry(self, cache_key: CacheKey) -> Optional[CacheEntry]:
        """
        Get cache entry information by key.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cache entry if found, None otherwise
        """
        self._stats['queries_served'] += 1
        
        # Check local cache first
        cache_key_str = cache_key.to_string()
        if (cache_key_str in self._local_cache and 
            time.time() < self._local_cache_ttl.get(cache_key_str, 0)):
            
            self._stats['cache_hits'] += 1
            return self._local_cache[cache_key_str]
        
        # Query Redis
        try:
            key = self._make_redis_key("entry", cache_key_str)
            entry_data = await self.redis.get(key)
            
            if entry_data:
                entry_info = json.loads(entry_data)
                
                # Reconstruct cache entry (without actual KV data)
                entry = CacheEntry(
                    cache_key=CacheKey(**entry_info['cache_key']),
                    kv_data=b"",  # Not stored in registry
                    cache_level=CacheLevel(entry_info['cache_level']),
                    node_id=entry_info['node_id'],
                    size_bytes=entry_info['size_bytes'],
                    created_at=entry_info['created_at'],
                    last_accessed=entry_info['last_accessed'],
                    access_count=entry_info['access_count'],
                    hit_rate=entry_info['hit_rate'],
                    replica_nodes=entry_info['replica_nodes'],
                    is_primary=entry_info['is_primary']
                )
                
                # Update local cache
                self._local_cache[cache_key_str] = entry
                self._local_cache_ttl[cache_key_str] = time.time() + self._local_ttl_seconds
                
                self._stats['cache_hits'] += 1
                return entry
            else:
                self._stats['cache_misses'] += 1
                return None
                
        except Exception as e:
            logger.error(
                "Failed to get cache entry",
                cache_key=cache_key_str,
                error=str(e)
            )
            self._stats['cache_misses'] += 1
            return None
    
    async def update_cache_access(self, cache_key: CacheKey) -> bool:
        """
        Update access statistics for a cache entry.
        
        Args:
            cache_key: Cache key that was accessed
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            key = self._make_redis_key("entry", cache_key.to_string())
            
            # Get current entry
            entry_data = await self.redis.get(key)
            if not entry_data:
                return False
            
            entry_info = json.loads(entry_data)
            
            # Update access statistics
            entry_info['last_accessed'] = time.time()
            entry_info['access_count'] += 1
            
            # Recalculate hit rate (simplified)
            total_accesses = entry_info['access_count']
            entry_info['hit_rate'] = min(1.0, total_accesses / (total_accesses + 1))
            
            # Store updated entry
            await self.redis.setex(
                key,
                self.ttl_seconds,
                json.dumps(entry_info)
            )
            
            # Update local cache if present
            cache_key_str = cache_key.to_string()
            if cache_key_str in self._local_cache:
                local_entry = self._local_cache[cache_key_str]
                local_entry.last_accessed = entry_info['last_accessed']
                local_entry.access_count = entry_info['access_count']
                local_entry.hit_rate = entry_info['hit_rate']
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to update cache access",
                cache_key=cache_key.to_string(),
                error=str(e)
            )
            return False
    
    # Query Interface
    
    async def find_entries_by_prefix(self, prefix_hash: str) -> List[CacheEntry]:
        """
        Find cache entries matching a prefix hash.
        
        Args:
            prefix_hash: Prefix hash to search for
            
        Returns:
            List of matching cache entries
        """
        try:
            # Use Redis SCAN to find matching keys
            pattern = self._make_redis_key("entry", f"*:{prefix_hash}:*")
            cursor = 0
            entries = []
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern)
                
                for key in keys:
                    entry_data = await self.redis.get(key)
                    if entry_data:
                        entry_info = json.loads(entry_data)
                        
                        # Reconstruct cache entry
                        entry = CacheEntry(
                            cache_key=CacheKey(**entry_info['cache_key']),
                            kv_data=b"",
                            cache_level=CacheLevel(entry_info['cache_level']),
                            node_id=entry_info['node_id'],
                            size_bytes=entry_info['size_bytes'],
                            created_at=entry_info['created_at'],
                            last_accessed=entry_info['last_accessed'],
                            access_count=entry_info['access_count'],
                            hit_rate=entry_info['hit_rate'],
                            replica_nodes=entry_info['replica_nodes'],
                            is_primary=entry_info['is_primary']
                        )
                        entries.append(entry)
                
                if cursor == 0:
                    break
            
            return entries
            
        except Exception as e:
            logger.error(
                "Failed to find entries by prefix",
                prefix_hash=prefix_hash,
                error=str(e)
            )
            return []
    
    async def get_cache_entries_by_node(self, node_id: str) -> List[CacheEntry]:
        """
        Get all cache entries stored on a specific node.
        
        Args:
            node_id: Node ID to query
            
        Returns:
            List of cache entries on the node
        """
        try:
            # Get node index
            node_index_key = self._make_redis_key("node_index", node_id)
            cache_keys = await self.redis.smembers(node_index_key)
            
            entries = []
            for cache_key_str in cache_keys:
                cache_key = CacheKey.from_string(cache_key_str.decode('utf-8'))
                entry = await self.get_cache_entry(cache_key)
                if entry:
                    entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(
                "Failed to get cache entries by node",
                node_id=node_id,
                error=str(e)
            )
            return []
    
    async def get_cache_statistics(self) -> Dict:
        """
        Get cache registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        try:
            # Get Redis info
            redis_info = await self.redis.info("memory")
            
            # Count total entries
            pattern = self._make_redis_key("entry", "*")
            cursor = 0
            total_entries = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern)
                total_entries += len(keys)
                if cursor == 0:
                    break
            
            # Combine with local statistics
            stats = self._stats.copy()
            stats.update({
                'total_entries': total_entries,
                'redis_memory_usage': redis_info.get('used_memory', 0),
                'local_cache_size': len(self._local_cache),
                'registered_nodes': len(self._nodes),
                'local_cache_hit_rate': (
                    self._stats['cache_hits'] / max(1, self._stats['queries_served'])
                )
            })
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get cache statistics", error=str(e))
            return self._stats.copy()
    
    # Node Management
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """
        Register a node in the cache registry.
        
        Args:
            node_info: Node information to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            key = self._make_redis_key("node", node_info.node_id)
            
            node_data = {
                'node_id': node_info.node_id,
                'service_type': node_info.service_type.value,
                'hostname': node_info.hostname,
                'port': node_info.port,
                'gpu_memory_gb': node_info.gpu_memory_gb,
                'cpu_cores': node_info.cpu_cores,
                'memory_gb': node_info.memory_gb,
                'is_healthy': node_info.is_healthy,
                'current_load': node_info.current_load,
                'last_heartbeat': node_info.last_heartbeat,
                'avg_latency_ms': node_info.avg_latency_ms,
                'throughput_rps': node_info.throughput_rps
            }
            
            await self.redis.setex(
                key,
                self.ttl_seconds,
                json.dumps(node_data)
            )
            
            # Update local node tracking
            self._nodes[node_info.node_id] = node_info
            self._node_last_heartbeat[node_info.node_id] = time.time()
            
            logger.info("Node registered", node_id=node_info.node_id)
            return True
            
        except Exception as e:
            logger.error(
                "Failed to register node",
                node_id=node_info.node_id,
                error=str(e)
            )
            return False
    
    async def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the cache registry.
        
        Args:
            node_id: Node ID to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            # Remove node info
            node_key = self._make_redis_key("node", node_id)
            await self.redis.delete(node_key)
            
            # Remove node index
            node_index_key = self._make_redis_key("node_index", node_id)
            await self.redis.delete(node_index_key)
            
            # Clean up cache entries for this node
            entries = await self.get_cache_entries_by_node(node_id)
            for entry in entries:
                await self.unregister_cache_entry(entry.cache_key)
            
            # Update local tracking
            self._nodes.pop(node_id, None)
            self._node_last_heartbeat.pop(node_id, None)
            
            logger.info("Node unregistered", node_id=node_id)
            return True
            
        except Exception as e:
            logger.error(
                "Failed to unregister node",
                node_id=node_id,
                error=str(e)
            )
            return False
    
    async def update_node_heartbeat(self, node_id: str, load: float = None) -> bool:
        """
        Update node heartbeat and optionally current load.
        
        Args:
            node_id: Node ID to update
            load: Optional current load value
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            key = self._make_redis_key("node", node_id)
            
            # Get current node data
            node_data = await self.redis.get(key)
            if not node_data:
                return False
            
            node_info = json.loads(node_data)
            
            # Update heartbeat and load
            node_info['last_heartbeat'] = time.time()
            if load is not None:
                node_info['current_load'] = load
            
            # Store updated node info
            await self.redis.setex(
                key,
                self.ttl_seconds,
                json.dumps(node_info)
            )
            
            # Update local tracking
            self._node_last_heartbeat[node_id] = time.time()
            if node_id in self._nodes and load is not None:
                self._nodes[node_id].current_load = load
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to update node heartbeat",
                node_id=node_id,
                error=str(e)
            )
            return False
    
    async def get_healthy_nodes(self, service_type: str = None) -> List[NodeInfo]:
        """
        Get list of healthy nodes, optionally filtered by service type.
        
        Args:
            service_type: Optional service type filter
            
        Returns:
            List of healthy node information
        """
        try:
            # Scan for all node keys
            pattern = self._make_redis_key("node", "*")
            cursor = 0
            healthy_nodes = []
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern)
                
                for key in keys:
                    node_data = await self.redis.get(key)
                    if node_data:
                        node_info = json.loads(node_data)
                        
                        # Check if node is healthy
                        if (node_info['is_healthy'] and 
                            time.time() - node_info['last_heartbeat'] < 60):  # 1 minute threshold
                            
                            # Filter by service type if specified
                            if service_type is None or node_info['service_type'] == service_type:
                                # Reconstruct NodeInfo
                                from ..common.models import ServiceType
                                node = NodeInfo(
                                    node_id=node_info['node_id'],
                                    service_type=ServiceType(node_info['service_type']),
                                    hostname=node_info['hostname'],
                                    port=node_info['port'],
                                    gpu_memory_gb=node_info['gpu_memory_gb'],
                                    cpu_cores=node_info['cpu_cores'],
                                    memory_gb=node_info['memory_gb'],
                                    is_healthy=node_info['is_healthy'],
                                    current_load=node_info['current_load'],
                                    last_heartbeat=node_info['last_heartbeat'],
                                    avg_latency_ms=node_info['avg_latency_ms'],
                                    throughput_rps=node_info['throughput_rps']
                                )
                                healthy_nodes.append(node)
                
                if cursor == 0:
                    break
            
            return healthy_nodes
            
        except Exception as e:
            logger.error(
                "Failed to get healthy nodes",
                service_type=service_type,
                error=str(e)
            )
            return []
    
    # Private Methods
    
    def _make_redis_key(self, key_type: str, identifier: str) -> str:
        """Create a namespaced Redis key."""
        return f"{self.namespace}:{key_type}:{identifier}"
    
    async def _add_to_node_index(self, node_id: str, cache_key: CacheKey) -> None:
        """Add cache key to node index for efficient node-based queries."""
        try:
            node_index_key = self._make_redis_key("node_index", node_id)
            await self.redis.sadd(node_index_key, cache_key.to_string())
            await self.redis.expire(node_index_key, self.ttl_seconds)
        except Exception as e:
            logger.warning("Failed to add to node index", error=str(e))
    
    async def _remove_from_node_index(self, node_id: str, cache_key: CacheKey) -> None:
        """Remove cache key from node index."""
        try:
            node_index_key = self._make_redis_key("node_index", node_id)
            await self.redis.srem(node_index_key, cache_key.to_string())
        except Exception as e:
            logger.warning("Failed to remove from node index", error=str(e))
    
    async def _cleanup_expired_entries(self) -> None:
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up local cache
                now = time.time()
                expired_keys = [
                    key for key, expiry in self._local_cache_ttl.items()
                    if now > expiry
                ]
                
                for key in expired_keys:
                    self._local_cache.pop(key, None)
                    self._local_cache_ttl.pop(key, None)
                
                logger.debug("Cleaned up expired local cache entries", count=len(expired_keys))
                
            except Exception as e:
                logger.error("Error in cleanup task", error=str(e))
    
    async def _update_local_cache(self) -> None:
        """Background task to refresh local cache with hot entries.""" 
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # This is a placeholder for more sophisticated local cache management
                # In practice, you might want to:
                # 1. Identify hot cache keys based on access patterns
                # 2. Pre-load frequently accessed entries
                # 3. Implement LRU eviction for local cache
                
                logger.debug("Local cache refresh cycle completed")
                
            except Exception as e:
                logger.error("Error in local cache update task", error=str(e))