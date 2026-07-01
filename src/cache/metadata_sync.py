"""
Cache Metadata Synchronization Service

Synchronizes KV cache metadata between vLLM workers and the central cache registry.
Enables cache-aware routing by keeping track of which prefixes are cached on which nodes.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

import aiohttp
import structlog

from ..common.models import CacheEntry
from .registry import CacheRegistry

logger = structlog.get_logger()


@dataclass
class WorkerCacheState:
    """Represents the cache state of a single worker"""
    worker_id: str
    node_url: str
    
    # Prefix tracking
    cached_prefixes: Dict[str, float]  # prefix_hash -> last_access_time
    cache_capacity_bytes: int = 0
    cache_used_bytes: int = 0
    
    # Statistics
    cache_hit_rate: float = 0.0
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Sync metadata
    last_sync_time: float = 0.0
    sync_failures: int = 0
    
    @property
    def cache_utilization(self) -> float:
        """Calculate cache utilization percentage"""
        if self.cache_capacity_bytes == 0:
            return 0.0
        return self.cache_used_bytes / self.cache_capacity_bytes
    
    @property
    def cache_count(self) -> int:
        """Number of cached prefixes"""
        return len(self.cached_prefixes)


class CacheMetadataSync:
    """
    Synchronizes cache metadata between workers and central registry.
    
    Features:
    - Periodic polling of worker cache states
    - Push updates to central cache registry
    - Track cache hit rates and utilization
    - Detect cache evictions and updates
    - Enable cache-aware routing decisions
    """
    
    def __init__(self,
                 cache_registry: CacheRegistry,
                 sync_interval: float = 10.0,
                 stale_threshold: float = 300.0):
        """
        Initialize cache metadata sync service.
        
        Args:
            cache_registry: Central cache registry instance
            sync_interval: Seconds between sync cycles
            stale_threshold: Seconds before cache entry considered stale
        """
        self.cache_registry = cache_registry
        self.sync_interval = sync_interval
        self.stale_threshold = stale_threshold
        
        # Worker tracking
        self.workers: Dict[str, WorkerCacheState] = {}
        
        # Global statistics
        self.total_syncs = 0
        self.failed_syncs = 0
        self.last_global_sync = 0.0
        
        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            "CacheMetadataSync initialized",
            sync_interval=sync_interval,
            stale_threshold=stale_threshold
        )
    
    async def start(self) -> None:
        """Start the sync service"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("CacheMetadataSync started")
    
    async def stop(self) -> None:
        """Stop the sync service"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("CacheMetadataSync stopped")
    
    def register_worker(self, worker_id: str, node_url: str) -> None:
        """Register a worker for cache sync"""
        if worker_id in self.workers:
            logger.warning("Worker already registered", worker_id=worker_id)
            return
        
        self.workers[worker_id] = WorkerCacheState(
            worker_id=worker_id,
            node_url=node_url,
            cached_prefixes={}
        )
        
        logger.info("Worker registered for cache sync", worker_id=worker_id, node_url=node_url)
    
    def deregister_worker(self, worker_id: str) -> None:
        """Deregister a worker from cache sync"""
        if worker_id not in self.workers:
            return
        
        # Clean up worker's cache entries from registry
        asyncio.create_task(self._cleanup_worker_cache_entries(worker_id))
        
        del self.workers[worker_id]
        
        logger.info("Worker deregistered from cache sync", worker_id=worker_id)
    
    async def force_sync_worker(self, worker_id: str) -> bool:
        """Force immediate sync for a specific worker"""
        if worker_id not in self.workers:
            logger.warning("Worker not found for force sync", worker_id=worker_id)
            return False
        
        worker_state = self.workers[worker_id]
        return await self._sync_worker(worker_state)
    
    async def force_sync_all(self) -> int:
        """Force immediate sync for all workers. Returns number of successful syncs."""
        success_count = 0
        
        sync_tasks = []
        for worker_state in self.workers.values():
            sync_tasks.append(self._sync_worker(worker_state))
        
        results = await asyncio.gather(*sync_tasks, return_exceptions=True)
        
        for result in results:
            if result is True:
                success_count += 1
        
        return success_count
    
    def get_worker_cache_state(self, worker_id: str) -> Optional[WorkerCacheState]:
        """Get cache state for a worker"""
        return self.workers.get(worker_id)
    
    def get_all_worker_states(self) -> List[WorkerCacheState]:
        """Get cache states for all workers"""
        return list(self.workers.values())
    
    def get_prefix_locations(self, prefix_hash: str) -> List[str]:
        """Get list of worker IDs that have a specific prefix cached"""
        locations = []
        
        current_time = time.time()
        
        for worker_id, worker_state in self.workers.items():
            if prefix_hash in worker_state.cached_prefixes:
                last_access = worker_state.cached_prefixes[prefix_hash]
                
                # Only include if not stale
                if current_time - last_access < self.stale_threshold:
                    locations.append(worker_id)
        
        return locations
    
    def get_cache_statistics(self) -> Dict:
        """Get global cache statistics"""
        total_cached_prefixes = sum(w.cache_count for w in self.workers.values())
        total_requests = sum(w.total_requests for w in self.workers.values())
        total_hits = sum(w.cache_hits for w in self.workers.values())
        
        avg_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        avg_utilization = sum(w.cache_utilization for w in self.workers.values()) / len(self.workers) if self.workers else 0.0
        
        return {
            'total_workers': len(self.workers),
            'total_cached_prefixes': total_cached_prefixes,
            'total_requests': total_requests,
            'total_cache_hits': total_hits,
            'global_hit_rate': avg_hit_rate,
            'average_cache_utilization': avg_utilization,
            'total_syncs': self.total_syncs,
            'failed_syncs': self.failed_syncs,
            'last_sync_time': self.last_global_sync
        }
    
    async def _sync_loop(self) -> None:
        """Background task for periodic cache sync"""
        while self._running:
            try:
                await self._sync_all_workers()
                self.last_global_sync = time.time()
                await asyncio.sleep(self.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Sync loop error", error=str(e))
                await asyncio.sleep(5)
    
    async def _sync_all_workers(self) -> None:
        """Sync cache metadata for all workers"""
        if not self.workers:
            return
        
        sync_tasks = []
        for worker_state in self.workers.values():
            sync_tasks.append(self._sync_worker(worker_state))
        
        results = await asyncio.gather(*sync_tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        failure_count = len(results) - success_count
        
        self.total_syncs += success_count
        self.failed_syncs += failure_count
        
        logger.debug(
            "Cache sync completed",
            workers=len(self.workers),
            success=success_count,
            failures=failure_count
        )
    
    async def _sync_worker(self, worker_state: WorkerCacheState) -> bool:
        """Sync cache metadata for a single worker"""
        try:
            # Fetch cache state from worker
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{worker_state.node_url}/prefixes",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    
                    if response.status != 200:
                        logger.warning(
                            "Failed to fetch worker prefixes",
                            worker_id=worker_state.worker_id,
                            status=response.status
                        )
                        worker_state.sync_failures += 1
                        return False
                    
                    data = await response.json()
                    
                    # Extract prefix data
                    new_prefixes = data.get('processed_prefixes', {})
                    
                    # Update worker state
                    worker_state.cached_prefixes = new_prefixes
                    worker_state.last_sync_time = time.time()
                    worker_state.sync_failures = 0
                    
                    # Fetch health data for statistics
                    async with session.get(
                        f"{worker_state.node_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as health_response:
                        
                        if health_response.status == 200:
                            health_data = await health_response.json()
                            
                            # Update statistics
                            worker_state.total_requests = health_data.get('total_requests', 0)
                            worker_state.cache_used_bytes = health_data.get('engine_stats', {}).get('gpu_cache_usage_sys', 0) * 1e9
                            worker_state.cache_capacity_bytes = 16 * 1e9  # Assume 16GB GPU
                    
                    # Update central registry
                    await self._update_registry_from_worker(worker_state, new_prefixes)
                    
                    return True
        
        except Exception as e:
            logger.warning(
                "Worker sync failed",
                worker_id=worker_state.worker_id,
                error=str(e)
            )
            worker_state.sync_failures += 1
            return False
    
    async def _update_registry_from_worker(self,
                                          worker_state: WorkerCacheState,
                                          prefixes: Dict[str, float]) -> None:
        """Update cache registry with worker's prefix data"""
        current_time = time.time()
        
        for prefix_hash, last_access_time in prefixes.items():
            try:
                # Create or update cache entry
                cache_entry = CacheEntry(
                    prefix_hash=prefix_hash,
                    node_id=worker_state.worker_id,
                    cache_size_bytes=0,  # Unknown, estimated
                    created_at=last_access_time,
                    last_accessed=last_access_time,
                    access_count=1,
                    ttl_seconds=int(self.stale_threshold)
                )
                
                # Register in cache registry
                await self.cache_registry.register_cache(cache_entry)
            
            except Exception as e:
                logger.warning(
                    "Failed to update registry for prefix",
                    prefix_hash=prefix_hash[:16],
                    worker_id=worker_state.worker_id,
                    error=str(e)
                )
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up stale cache entries"""
        while self._running:
            try:
                await self._cleanup_stale_entries()
                await asyncio.sleep(60)  # Run every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(10)
    
    async def _cleanup_stale_entries(self) -> None:
        """Remove stale cache entries from tracking"""
        current_time = time.time()
        
        for worker_state in self.workers.values():
            stale_prefixes = []
            
            for prefix_hash, last_access_time in worker_state.cached_prefixes.items():
                if current_time - last_access_time > self.stale_threshold:
                    stale_prefixes.append(prefix_hash)
            
            # Remove stale entries
            for prefix_hash in stale_prefixes:
                del worker_state.cached_prefixes[prefix_hash]
                
                # Also remove from registry
                try:
                    await self.cache_registry.invalidate_cache(
                        prefix_hash,
                        worker_state.worker_id
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to invalidate stale cache entry",
                        prefix_hash=prefix_hash[:16],
                        error=str(e)
                    )
            
            if stale_prefixes:
                logger.debug(
                    "Cleaned up stale cache entries",
                    worker_id=worker_state.worker_id,
                    count=len(stale_prefixes)
                )
    
    async def _cleanup_worker_cache_entries(self, worker_id: str) -> None:
        """Clean up all cache entries for a deregistered worker"""
        try:
            # Get all prefixes for this worker from registry
            # and invalidate them
            
            # This would require a method in cache registry to bulk invalidate
            # For now, we'll just log
            logger.info(
                "Cleaning up cache entries for deregistered worker",
                worker_id=worker_id
            )
        
        except Exception as e:
            logger.error(
                "Failed to cleanup worker cache entries",
                worker_id=worker_id,
                error=str(e)
            )


async def main():
    """Example usage"""
    from .registry import CacheRegistry
    
    # Initialize cache registry
    cache_registry = await CacheRegistry.create(
        redis_url="redis://localhost:6379"
    )
    
    # Initialize metadata sync
    sync_service = CacheMetadataSync(
        cache_registry=cache_registry,
        sync_interval=10.0
    )
    
    await sync_service.start()
    
    try:
        # Register workers
        sync_service.register_worker("worker-1", "http://localhost:8001")
        sync_service.register_worker("worker-2", "http://localhost:8002")
        
        # Run sync loop
        while True:
            stats = sync_service.get_cache_statistics()
            logger.info("Cache statistics", **stats)
            await asyncio.sleep(30)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    
    finally:
        await sync_service.stop()
        await cache_registry.close()


if __name__ == "__main__":
    import structlog
    
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer()
        ]
    )
    
    asyncio.run(main())
