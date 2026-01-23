"""
Gateway-to-Worker HTTP Client

Handles communication between the gateway routing engine and vLLM worker nodes.
Provides connection pooling, retry logic, and error handling for optimal performance.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, AsyncGenerator, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json

from ..common.models import InferenceRequest, InferenceResponse, NodeInfo

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class WorkerConnection:
    """Represents a connection to a worker node"""
    node_id: str
    base_url: str
    session: Optional[aiohttp.ClientSession] = None
    status: WorkerStatus = WorkerStatus.UNKNOWN
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    avg_response_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def is_healthy(self) -> bool:
        return (
            self.status == WorkerStatus.HEALTHY and 
            self.consecutive_failures < 3 and
            self.success_rate > 0.8
        )


class WorkerClientPool:
    """
    Manages HTTP connections to multiple vLLM worker nodes.
    
    Features:
    - Connection pooling with automatic cleanup
    - Health monitoring with circuit breaker pattern
    - Request retry logic with backoff
    - Load balancing across healthy workers
    - Performance metrics tracking
    """
    
    def __init__(self, 
                 timeout_seconds: float = 30.0,
                 max_retries: int = 2,
                 health_check_interval: float = 30.0,
                 connection_pool_size: int = 10):
        
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        self.connection_pool_size = connection_pool_size
        
        # Worker connections
        self.workers: Dict[str, WorkerConnection] = {}
        self.healthy_workers: Set[str] = set()
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"Initialized WorkerClientPool with timeout={timeout_seconds}s, max_retries={max_retries}")
    
    async def start(self) -> None:
        """Start the client pool and background tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Start background health checks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("WorkerClientPool started")
    
    async def stop(self) -> None:
        """Stop the client pool and cleanup connections"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for worker in self.workers.values():
            if worker.session:
                await worker.session.close()
        
        self.workers.clear()
        self.healthy_workers.clear()
        
        logger.info("WorkerClientPool stopped")
    
    def add_worker(self, node_id: str, host: str, port: int) -> None:
        """Add a worker node to the pool"""
        base_url = f"http://{host}:{port}"
        
        if node_id in self.workers:
            logger.warning(f"Worker {node_id} already exists, updating URL to {base_url}")
        
        self.workers[node_id] = WorkerConnection(
            node_id=node_id,
            base_url=base_url
        )
        
        logger.info(f"Added worker {node_id} at {base_url}")
    
    def remove_worker(self, node_id: str) -> None:
        """Remove a worker node from the pool"""
        if node_id in self.workers:
            worker = self.workers[node_id]
            if worker.session:
                asyncio.create_task(worker.session.close())
            
            del self.workers[node_id]
            self.healthy_workers.discard(node_id)
            
            logger.info(f"Removed worker {node_id}")
    
    async def get_healthy_workers(self) -> List[str]:
        """Get list of healthy worker node IDs"""
        return list(self.healthy_workers)
    
    async def get_worker_info(self, node_id: str) -> Optional[NodeInfo]:
        """Get detailed information about a worker node"""
        if node_id not in self.workers:
            return None
        
        worker = self.workers[node_id]
        
        try:
            session = await self._get_session(worker)
            async with session.get(f"{worker.base_url}/info") as response:
                if response.status == 200:
                    data = await response.json()
                    return NodeInfo(
                        node_id=node_id,
                        host=worker.base_url.split("//")[1].split(":")[0],
                        port=int(worker.base_url.split(":")[-1]),
                        status="healthy" if worker.is_healthy else "unhealthy",
                        gpu_memory_used=data.get("gpu_memory_used", 0),
                        gpu_memory_total=data.get("gpu_memory_total", 1),
                        active_requests=data.get("active_requests", 0),
                        total_requests_served=data.get("total_requests", 0),
                        average_response_time=worker.avg_response_time_ms
                    )
        except Exception as e:
            logger.warning(f"Failed to get worker info for {node_id}: {e}")
        
        return None
    
    async def send_inference_request(self, 
                                   node_id: str, 
                                   request: InferenceRequest) -> InferenceResponse:
        """
        Send inference request to a specific worker node.
        Includes retry logic and error handling.
        """
        if node_id not in self.workers:
            raise ValueError(f"Worker {node_id} not found")
        
        worker = self.workers[node_id]
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                session = await self._get_session(worker)
                request_data = request.model_dump()
                
                async with session.post(
                    f"{worker.base_url}/generate",
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        response_time = (time.time() - start_time) * 1000
                        
                        # Update worker metrics
                        self._update_worker_metrics(worker, True, response_time)
                        
                        return InferenceResponse(**response_data)
                    
                    elif response.status == 503:  # Service unavailable
                        # Worker is overloaded, mark as unhealthy temporarily
                        worker.consecutive_failures += 1
                        if worker.consecutive_failures >= 3:
                            worker.status = WorkerStatus.UNHEALTHY
                            self.healthy_workers.discard(node_id)
                        
                        raise aiohttp.ClientResponseError(
                            request.method, request.url,
                            status=response.status,
                            message="Worker overloaded"
                        )
                    
                    else:
                        error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request.method, request.url,
                            status=response.status,
                            message=error_text
                        )
            
            except Exception as e:
                last_exception = e
                response_time = (time.time() - start_time) * 1000
                self._update_worker_metrics(worker, False, response_time)
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = 0.5 * (2 ** attempt)
                    await asyncio.sleep(delay)
                    logger.warning(f"Request to {node_id} failed (attempt {attempt + 1}): {e}. Retrying in {delay}s")
                else:
                    logger.error(f"Request to {node_id} failed after {self.max_retries + 1} attempts: {e}")
        
        # All retries failed
        worker.consecutive_failures += 1
        if worker.consecutive_failures >= 3:
            worker.status = WorkerStatus.UNHEALTHY
            self.healthy_workers.discard(node_id)
        
        raise last_exception
    
    async def send_streaming_request(self, 
                                   node_id: str, 
                                   request: InferenceRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send streaming inference request to a worker node.
        Yields response chunks as they arrive.
        """
        if node_id not in self.workers:
            raise ValueError(f"Worker {node_id} not found")
        
        worker = self.workers[node_id]
        session = await self._get_session(worker)
        
        try:
            start_time = time.time()
            request_data = request.model_dump()
            request_data['stream'] = True  # Force streaming
            
            async with session.post(
                f"{worker.base_url}/generate_stream",
                json=request_data,
                timeout=self.timeout
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request.method, request.url,
                        status=response.status,
                        message=error_text
                    )
                
                # Process streaming response
                first_chunk = True
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            yield chunk
                            
                            if first_chunk:
                                first_chunk = False
                                # Start tracking response time from first chunk
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse streaming chunk: {data_str}")
                
                # Update metrics on successful completion
                response_time = (time.time() - start_time) * 1000
                self._update_worker_metrics(worker, True, response_time)
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_worker_metrics(worker, False, response_time)
            raise
    
    async def _get_session(self, worker: WorkerConnection) -> aiohttp.ClientSession:
        """Get or create HTTP session for worker"""
        if worker.session is None or worker.session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.connection_pool_size,
                keepalive_timeout=30
            )
            worker.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout
            )
        
        return worker.session
    
    def _update_worker_metrics(self, worker: WorkerConnection, success: bool, response_time_ms: float) -> None:
        """Update worker performance metrics"""
        worker.total_requests += 1
        
        if success:
            worker.successful_requests += 1
            worker.consecutive_failures = 0
            
            # Update rolling average response time
            if worker.avg_response_time_ms == 0:
                worker.avg_response_time_ms = response_time_ms
            else:
                # Exponential moving average with alpha=0.1
                worker.avg_response_time_ms = (
                    0.9 * worker.avg_response_time_ms + 0.1 * response_time_ms
                )
        else:
            worker.consecutive_failures += 1
    
    async def _health_check_loop(self) -> None:
        """Background task to monitor worker health"""
        while self._running:
            try:
                await self._check_all_workers_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def _check_all_workers_health(self) -> None:
        """Check health of all registered workers"""
        if not self.workers:
            return
        
        health_tasks = []
        for node_id, worker in self.workers.items():
            health_tasks.append(self._check_worker_health(node_id, worker))
        
        await asyncio.gather(*health_tasks, return_exceptions=True)
    
    async def _check_worker_health(self, node_id: str, worker: WorkerConnection) -> None:
        """Check health of a single worker"""
        try:
            session = await self._get_session(worker)
            
            start_time = time.time()
            async with session.get(
                f"{worker.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5.0)  # Shorter timeout for health checks
            ) as response:
                
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "healthy":
                        if worker.status != WorkerStatus.HEALTHY:
                            logger.info(f"Worker {node_id} is now healthy")
                        
                        worker.status = WorkerStatus.HEALTHY
                        worker.consecutive_failures = 0
                        worker.last_health_check = time.time()
                        self.healthy_workers.add(node_id)
                    else:
                        worker.status = WorkerStatus.UNHEALTHY
                        self.healthy_workers.discard(node_id)
                else:
                    worker.status = WorkerStatus.UNHEALTHY
                    self.healthy_workers.discard(node_id)
        
        except Exception as e:
            logger.warning(f"Health check failed for worker {node_id}: {e}")
            worker.status = WorkerStatus.UNHEALTHY
            worker.consecutive_failures += 1
            self.healthy_workers.discard(node_id)
    
    async def _cleanup_loop(self) -> None:
        """Background task to cleanup stale connections"""
        while self._running:
            try:
                await self._cleanup_stale_sessions()
                await asyncio.sleep(300)  # Run every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    async def _cleanup_stale_sessions(self) -> None:
        """Cleanup stale HTTP sessions to prevent memory leaks"""
        current_time = time.time()
        
        for worker in self.workers.values():
            # Close sessions that haven't been health checked in 10 minutes
            if (worker.session and 
                not worker.session.closed and 
                current_time - worker.last_health_check > 600):
                
                logger.info(f"Closing stale session for worker {worker.node_id}")
                await worker.session.close()
                worker.session = None


class MockWorkerClientPool(WorkerClientPool):
    """
    Mock implementation for testing without real vLLM workers.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mock_responses: Dict[str, str] = {}
        self._mock_latencies: Dict[str, float] = {}
        self._mock_failures: Dict[str, int] = {}  # node_id -> failure_count
    
    def set_mock_response(self, node_id: str, response_text: str, latency_ms: float = 100.0):
        """Set mock response for a worker node"""
        self._mock_responses[node_id] = response_text
        self._mock_latencies[node_id] = latency_ms
    
    def set_mock_failure(self, node_id: str, failure_count: int = 1):
        """Set a worker to fail the next N requests"""
        self._mock_failures[node_id] = failure_count
    
    async def send_inference_request(self, 
                                   node_id: str, 
                                   request: InferenceRequest) -> InferenceResponse:
        """Mock implementation of inference request"""
        
        # Check if worker exists
        if node_id not in self.workers:
            raise ValueError(f"Worker {node_id} not found")
        
        # Check for mock failures
        if node_id in self._mock_failures and self._mock_failures[node_id] > 0:
            self._mock_failures[node_id] -= 1
            raise aiohttp.ClientError("Mock failure")
        
        # Simulate network latency
        latency = self._mock_latencies.get(node_id, 100.0) / 1000.0
        await asyncio.sleep(latency)
        
        # Generate mock response
        response_text = self._mock_responses.get(
            node_id, 
            f"Mock response from {node_id} for prompt: {request.prompt[:50]}..."
        )
        
        return InferenceResponse(
            request_id=request.request_id,
            generated_text=response_text,
            tokens_generated=len(response_text.split()),
            processing_time_ms=latency,
            cache_hit=False,
            processed_by=node_id
        )
    
    async def send_streaming_request(self, 
                                   node_id: str, 
                                   request: InferenceRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Mock implementation of streaming request"""
        
        # Check if worker exists
        if node_id not in self.workers:
            raise ValueError(f"Worker {node_id} not found")
        
        # Check for mock failures
        if node_id in self._mock_failures and self._mock_failures[node_id] > 0:
            self._mock_failures[node_id] -= 1
            raise aiohttp.ClientError("Mock streaming failure")
        
        response_text = self._mock_responses.get(
            node_id,
            f"Mock streaming response from {node_id} for prompt: {request.prompt[:30]}..."
        )
        
        # Simulate streaming by yielding words one at a time
        words = response_text.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.1)  # Simulate generation delay
            
            yield {
                "id": f"chunk_{i}",
                "choices": [{
                    "delta": {"content": word + " " if i < len(words) - 1 else word},
                    "finish_reason": None if i < len(words) - 1 else "length"
                }],
                "usage": None if i < len(words) - 1 else {
                    "prompt_tokens": len(request.prompt.split()),
                    "completion_tokens": len(words),
                    "total_tokens": len(request.prompt.split()) + len(words)
                }
            }
    
    async def _check_worker_health(self, node_id: str, worker: WorkerConnection) -> None:
        """Mock health check - always healthy unless explicitly failed"""
        if node_id in self._mock_failures and self._mock_failures[node_id] > 0:
            worker.status = WorkerStatus.UNHEALTHY
            self.healthy_workers.discard(node_id)
        else:
            worker.status = WorkerStatus.HEALTHY
            worker.last_health_check = time.time()
            self.healthy_workers.add(node_id)