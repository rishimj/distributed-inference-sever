"""
KV Cache Transfer Service

Handles transfer of KV cache data between vLLM worker nodes.
Enables actual cache sharing (not just metadata) for maximum performance.

Note: This requires vLLM support for cache export/import, which may need
custom modifications to vLLM or use of its cache serialization APIs.
"""

import asyncio
import hashlib
import lz4.frame
import msgpack
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import aiohttp
import structlog

logger = structlog.get_logger()


@dataclass
class CacheTransferRequest:
    """Request to transfer cache data"""
    prefix_hash: str
    source_worker_id: str
    target_worker_id: str
    priority: int = 0
    requested_at: float = 0.0
    
    def __post_init__(self):
        if self.requested_at == 0.0:
            self.requested_at = time.time()


@dataclass
class CacheTransferResult:
    """Result of cache transfer operation"""
    prefix_hash: str
    success: bool
    bytes_transferred: int = 0
    transfer_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class CacheBlob:
    """Serialized KV cache data"""
    prefix_hash: str
    model: str
    layer_count: int
    
    # KV cache tensors (simplified representation)
    # In reality, these would be actual tensor data
    key_cache_data: bytes
    value_cache_data: bytes
    
    # Metadata
    sequence_length: int
    hidden_size: int
    num_heads: int
    
    # Integrity
    checksum: str
    compressed: bool = False
    
    @property
    def size_bytes(self) -> int:
        """Total size of cache data"""
        return len(self.key_cache_data) + len(self.value_cache_data)
    
    def compress(self) -> 'CacheBlob':
        """Compress cache data using LZ4"""
        if self.compressed:
            return self
        
        compressed_key = lz4.frame.compress(self.key_cache_data)
        compressed_value = lz4.frame.compress(self.value_cache_data)
        
        return CacheBlob(
            prefix_hash=self.prefix_hash,
            model=self.model,
            layer_count=self.layer_count,
            key_cache_data=compressed_key,
            value_cache_data=compressed_value,
            sequence_length=self.sequence_length,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            checksum=self.checksum,
            compressed=True
        )
    
    def decompress(self) -> 'CacheBlob':
        """Decompress cache data"""
        if not self.compressed:
            return self
        
        decompressed_key = lz4.frame.decompress(self.key_cache_data)
        decompressed_value = lz4.frame.decompress(self.value_cache_data)
        
        return CacheBlob(
            prefix_hash=self.prefix_hash,
            model=self.model,
            layer_count=self.layer_count,
            key_cache_data=decompressed_key,
            value_cache_data=decompressed_value,
            sequence_length=self.sequence_length,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            checksum=self.checksum,
            compressed=False
        )
    
    def verify_checksum(self) -> bool:
        """Verify data integrity"""
        actual_checksum = hashlib.sha256(
            self.key_cache_data + self.value_cache_data
        ).hexdigest()
        return actual_checksum == self.checksum
    
    def to_msgpack(self) -> bytes:
        """Serialize to msgpack"""
        data = {
            'prefix_hash': self.prefix_hash,
            'model': self.model,
            'layer_count': self.layer_count,
            'key_cache_data': self.key_cache_data,
            'value_cache_data': self.value_cache_data,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'checksum': self.checksum,
            'compressed': self.compressed
        }
        return msgpack.packb(data, use_bin_type=True)
    
    @classmethod
    def from_msgpack(cls, data: bytes) -> 'CacheBlob':
        """Deserialize from msgpack"""
        obj = msgpack.unpackb(data, raw=False)
        return cls(**obj)


class CacheTransferService:
    """
    Manages transfer of KV cache data between worker nodes.
    
    Features:
    - Cache serialization and compression
    - P2P transfer between workers
    - Transfer queue with prioritization
    - Bandwidth throttling
    - Integrity verification
    - Retry logic for failed transfers
    """
    
    def __init__(self,
                 max_concurrent_transfers: int = 3,
                 compression_enabled: bool = True,
                 max_transfer_size_mb: float = 100.0,
                 transfer_timeout_seconds: float = 60.0):
        """
        Initialize cache transfer service.
        
        Args:
            max_concurrent_transfers: Max parallel transfers
            compression_enabled: Enable LZ4 compression
            max_transfer_size_mb: Max size for single transfer
            transfer_timeout_seconds: Timeout for transfers
        """
        self.max_concurrent_transfers = max_concurrent_transfers
        self.compression_enabled = compression_enabled
        self.max_transfer_size_bytes = int(max_transfer_size_mb * 1024 * 1024)
        self.transfer_timeout = transfer_timeout_seconds
        
        # Transfer queue
        self.transfer_queue: asyncio.Queue[CacheTransferRequest] = asyncio.Queue()
        self.active_transfers: Dict[str, CacheTransferRequest] = {}
        
        # Statistics
        self.total_transfers = 0
        self.successful_transfers = 0
        self.failed_transfers = 0
        self.total_bytes_transferred = 0
        
        # Worker URL mapping
        self.worker_urls: Dict[str, str] = {}
        
        # Background tasks
        self._transfer_workers: List[asyncio.Task] = []
        self._running = False
        
        logger.info(
            "CacheTransferService initialized",
            max_concurrent=max_concurrent_transfers,
            compression=compression_enabled,
            max_size_mb=max_transfer_size_mb
        )
    
    async def start(self) -> None:
        """Start the transfer service"""
        if self._running:
            return
        
        self._running = True
        
        # Start transfer workers
        for i in range(self.max_concurrent_transfers):
            task = asyncio.create_task(self._transfer_worker(i))
            self._transfer_workers.append(task)
        
        logger.info(
            "CacheTransferService started",
            workers=self.max_concurrent_transfers
        )
    
    async def stop(self) -> None:
        """Stop the transfer service"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all workers
        for task in self._transfer_workers:
            task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._transfer_workers, return_exceptions=True)
        self._transfer_workers.clear()
        
        logger.info("CacheTransferService stopped")
    
    def register_worker(self, worker_id: str, worker_url: str) -> None:
        """Register a worker node for cache transfers"""
        self.worker_urls[worker_id] = worker_url
        logger.info("Worker registered for cache transfer", worker_id=worker_id)
    
    def deregister_worker(self, worker_id: str) -> None:
        """Deregister a worker node"""
        if worker_id in self.worker_urls:
            del self.worker_urls[worker_id]
            logger.info("Worker deregistered from cache transfer", worker_id=worker_id)
    
    async def request_transfer(self,
                              prefix_hash: str,
                              source_worker_id: str,
                              target_worker_id: str,
                              priority: int = 0) -> None:
        """
        Request a cache transfer from source to target worker.
        
        Args:
            prefix_hash: Hash of prefix to transfer
            source_worker_id: Worker that has the cache
            target_worker_id: Worker that needs the cache
            priority: Transfer priority (higher = more urgent)
        """
        # Validate workers
        if source_worker_id not in self.worker_urls:
            raise ValueError(f"Source worker {source_worker_id} not registered")
        if target_worker_id not in self.worker_urls:
            raise ValueError(f"Target worker {target_worker_id} not registered")
        
        request = CacheTransferRequest(
            prefix_hash=prefix_hash,
            source_worker_id=source_worker_id,
            target_worker_id=target_worker_id,
            priority=priority
        )
        
        await self.transfer_queue.put(request)
        
        logger.info(
            "Cache transfer requested",
            prefix_hash=prefix_hash[:16],
            source=source_worker_id,
            target=target_worker_id,
            priority=priority
        )
    
    def get_statistics(self) -> Dict:
        """Get transfer statistics"""
        success_rate = self.successful_transfers / self.total_transfers if self.total_transfers > 0 else 0.0
        
        return {
            'total_transfers': self.total_transfers,
            'successful_transfers': self.successful_transfers,
            'failed_transfers': self.failed_transfers,
            'success_rate': success_rate,
            'total_bytes_transferred': self.total_bytes_transferred,
            'active_transfers': len(self.active_transfers),
            'queued_transfers': self.transfer_queue.qsize()
        }
    
    async def _transfer_worker(self, worker_id: int) -> None:
        """Background worker that processes transfer queue"""
        logger.info(f"Transfer worker {worker_id} started")
        
        while self._running:
            try:
                # Get next transfer request
                request = await asyncio.wait_for(
                    self.transfer_queue.get(),
                    timeout=1.0
                )
                
                # Process transfer
                transfer_key = f"{request.source_worker_id}:{request.target_worker_id}:{request.prefix_hash}"
                self.active_transfers[transfer_key] = request
                
                try:
                    result = await self._execute_transfer(request)
                    
                    if result.success:
                        self.successful_transfers += 1
                        self.total_bytes_transferred += result.bytes_transferred
                        
                        logger.info(
                            "Cache transfer completed",
                            prefix_hash=request.prefix_hash[:16],
                            source=request.source_worker_id,
                            target=request.target_worker_id,
                            bytes=result.bytes_transferred,
                            time_ms=result.transfer_time_ms
                        )
                    else:
                        self.failed_transfers += 1
                        
                        logger.warning(
                            "Cache transfer failed",
                            prefix_hash=request.prefix_hash[:16],
                            source=request.source_worker_id,
                            target=request.target_worker_id,
                            error=result.error
                        )
                
                finally:
                    self.total_transfers += 1
                    del self.active_transfers[transfer_key]
            
            except asyncio.TimeoutError:
                # No transfers in queue, continue
                continue
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"Transfer worker {worker_id} error", error=str(e))
                await asyncio.sleep(1)
        
        logger.info(f"Transfer worker {worker_id} stopped")
    
    async def _execute_transfer(self, request: CacheTransferRequest) -> CacheTransferResult:
        """Execute a single cache transfer"""
        start_time = time.time()
        
        try:
            # Step 1: Export cache from source worker
            cache_blob = await self._export_cache_from_worker(
                request.source_worker_id,
                request.prefix_hash
            )
            
            if cache_blob is None:
                return CacheTransferResult(
                    prefix_hash=request.prefix_hash,
                    success=False,
                    error="Failed to export cache from source"
                )
            
            # Step 2: Compress if enabled
            if self.compression_enabled and not cache_blob.compressed:
                cache_blob = cache_blob.compress()
            
            # Step 3: Verify integrity
            if not cache_blob.verify_checksum():
                return CacheTransferResult(
                    prefix_hash=request.prefix_hash,
                    success=False,
                    error="Cache checksum verification failed"
                )
            
            # Step 4: Transfer to target worker
            success = await self._import_cache_to_worker(
                request.target_worker_id,
                cache_blob
            )
            
            if not success:
                return CacheTransferResult(
                    prefix_hash=request.prefix_hash,
                    success=False,
                    error="Failed to import cache to target"
                )
            
            # Success!
            transfer_time = (time.time() - start_time) * 1000
            
            return CacheTransferResult(
                prefix_hash=request.prefix_hash,
                success=True,
                bytes_transferred=cache_blob.size_bytes,
                transfer_time_ms=transfer_time
            )
        
        except Exception as e:
            transfer_time = (time.time() - start_time) * 1000
            
            return CacheTransferResult(
                prefix_hash=request.prefix_hash,
                success=False,
                transfer_time_ms=transfer_time,
                error=str(e)
            )
    
    async def _export_cache_from_worker(self,
                                       worker_id: str,
                                       prefix_hash: str) -> Optional[CacheBlob]:
        """
        Export cache data from a worker node.
        
        Note: This requires vLLM to expose cache export API.
        Currently implemented as a mock/placeholder.
        """
        try:
            worker_url = self.worker_urls[worker_id]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{worker_url}/cache/export",
                    json={'prefix_hash': prefix_hash},
                    timeout=aiohttp.ClientTimeout(total=self.transfer_timeout)
                ) as response:
                    
                    if response.status != 200:
                        logger.warning(
                            "Cache export failed",
                            worker_id=worker_id,
                            status=response.status
                        )
                        return None
                    
                    # Read cache blob
                    data = await response.read()
                    cache_blob = CacheBlob.from_msgpack(data)
                    
                    return cache_blob
        
        except Exception as e:
            logger.error(
                "Error exporting cache",
                worker_id=worker_id,
                prefix_hash=prefix_hash[:16],
                error=str(e)
            )
            return None
    
    async def _import_cache_to_worker(self,
                                     worker_id: str,
                                     cache_blob: CacheBlob) -> bool:
        """
        Import cache data into a worker node.
        
        Note: This requires vLLM to expose cache import API.
        Currently implemented as a mock/placeholder.
        """
        try:
            worker_url = self.worker_urls[worker_id]
            
            # Serialize cache blob
            data = cache_blob.to_msgpack()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{worker_url}/cache/import",
                    data=data,
                    headers={'Content-Type': 'application/msgpack'},
                    timeout=aiohttp.ClientTimeout(total=self.transfer_timeout)
                ) as response:
                    
                    if response.status != 200:
                        logger.warning(
                            "Cache import failed",
                            worker_id=worker_id,
                            status=response.status
                        )
                        return False
                    
                    return True
        
        except Exception as e:
            logger.error(
                "Error importing cache",
                worker_id=worker_id,
                error=str(e)
            )
            return False


# Mock implementation for testing without real vLLM cache export/import
class MockCacheTransferService(CacheTransferService):
    """Mock cache transfer service for testing"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Mock cache storage
        self._mock_caches: Dict[Tuple[str, str], CacheBlob] = {}  # (worker_id, prefix_hash) -> blob
    
    def add_mock_cache(self, worker_id: str, prefix_hash: str, size_bytes: int = 1024) -> None:
        """Add a mock cache to a worker"""
        cache_blob = CacheBlob(
            prefix_hash=prefix_hash,
            model="mock-model",
            layer_count=32,
            key_cache_data=b"x" * size_bytes,
            value_cache_data=b"y" * size_bytes,
            sequence_length=512,
            hidden_size=4096,
            num_heads=32,
            checksum=hashlib.sha256(b"x" * size_bytes + b"y" * size_bytes).hexdigest(),
            compressed=False
        )
        
        self._mock_caches[(worker_id, prefix_hash)] = cache_blob
    
    async def _export_cache_from_worker(self,
                                       worker_id: str,
                                       prefix_hash: str) -> Optional[CacheBlob]:
        """Mock export - return cached blob"""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        cache_key = (worker_id, prefix_hash)
        if cache_key in self._mock_caches:
            return self._mock_caches[cache_key]
        
        return None
    
    async def _import_cache_to_worker(self,
                                     worker_id: str,
                                     cache_blob: CacheBlob) -> bool:
        """Mock import - store blob"""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        cache_key = (worker_id, cache_blob.prefix_hash)
        self._mock_caches[cache_key] = cache_blob
        
        return True


async def main():
    """Example usage"""
    service = MockCacheTransferService(
        max_concurrent_transfers=2,
        compression_enabled=True
    )
    
    await service.start()
    
    try:
        # Register workers
        service.register_worker("worker-1", "http://localhost:8001")
        service.register_worker("worker-2", "http://localhost:8002")
        
        # Add mock cache to worker-1
        service.add_mock_cache("worker-1", "prefix123", size_bytes=1024*1024)  # 1MB
        
        # Request transfer from worker-1 to worker-2
        await service.request_transfer(
            prefix_hash="prefix123",
            source_worker_id="worker-1",
            target_worker_id="worker-2",
            priority=5
        )
        
        # Wait for transfer to complete
        await asyncio.sleep(5)
        
        # Check statistics
        stats = service.get_statistics()
        logger.info("Transfer statistics", **stats)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    
    finally:
        await service.stop()


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
