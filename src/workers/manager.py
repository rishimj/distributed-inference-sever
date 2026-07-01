"""
Worker Manager - Process lifecycle management for vLLM workers.

Handles:
1. Starting/stopping worker processes
2. Health monitoring and auto-restart
3. Resource allocation and configuration
4. Worker registration with gateway
"""

import asyncio
import json
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from enum import Enum

import aiohttp
import structlog

logger = structlog.get_logger()


class WorkerState(Enum):
    """Worker process states"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class WorkerProcess:
    """Represents a managed worker process"""
    worker_id: str
    port: int
    model: str
    gpu_id: Optional[int] = None
    
    # Process management
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    state: WorkerState = WorkerState.STOPPED
    
    # Lifecycle tracking
    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    restart_count: int = 0
    consecutive_failures: int = 0
    
    # Configuration
    config: Dict = field(default_factory=dict)
    
    # Health status
    last_health_check: Optional[float] = None
    is_healthy: bool = False
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate worker uptime"""
        if self.start_time is None:
            return 0.0
        if self.state == WorkerState.RUNNING:
            return time.time() - self.start_time
        if self.stop_time:
            return self.stop_time - self.start_time
        return 0.0
    
    @property
    def base_url(self) -> str:
        """Get worker's base URL"""
        return f"http://localhost:{self.port}"


class WorkerManager:
    """
    Manages lifecycle of multiple vLLM worker processes.
    
    Features:
    - Start/stop workers with proper configuration
    - Health monitoring with auto-restart
    - GPU allocation and resource management
    - Integration with gateway for service discovery
    """
    
    def __init__(self,
                 gateway_url: Optional[str] = None,
                 worker_script_path: Optional[str] = None,
                 auto_restart: bool = True,
                 max_restart_attempts: int = 3,
                 health_check_interval: float = 30.0):
        """
        Initialize worker manager.
        
        Args:
            gateway_url: URL of gateway service for worker registration
            worker_script_path: Path to vLLM worker script
            auto_restart: Enable automatic restart on failure
            max_restart_attempts: Max consecutive restart attempts
            health_check_interval: Seconds between health checks
        """
        self.gateway_url = gateway_url
        self.auto_restart = auto_restart
        self.max_restart_attempts = max_restart_attempts
        self.health_check_interval = health_check_interval
        
        # Find worker script path
        if worker_script_path:
            self.worker_script_path = Path(worker_script_path)
        else:
            # Default to vllm_worker.py in same directory
            self.worker_script_path = Path(__file__).parent / "vllm_worker.py"
        
        # Worker tracking
        self.workers: Dict[str, WorkerProcess] = {}
        self.available_ports: Set[int] = set(range(8001, 8100))  # Port pool
        self.allocated_gpus: Dict[int, str] = {}  # gpu_id -> worker_id
        
        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            "WorkerManager initialized",
            worker_script=str(self.worker_script_path),
            auto_restart=auto_restart
        )
    
    async def start(self) -> None:
        """Start the worker manager"""
        if self._running:
            return
        
        self._running = True
        
        # Start health monitoring
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        logger.info("WorkerManager started")
    
    async def stop(self) -> None:
        """Stop the worker manager and all workers"""
        if not self._running:
            return
        
        self._running = False
        
        # Stop health monitoring
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop all workers
        stop_tasks = []
        for worker_id in list(self.workers.keys()):
            stop_tasks.append(self.stop_worker(worker_id))
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logger.info("WorkerManager stopped")
    
    async def start_worker(self,
                          worker_id: str,
                          model: str,
                          port: Optional[int] = None,
                          gpu_id: Optional[int] = None,
                          max_num_seqs: int = 256,
                          max_model_len: int = 4096,
                          gpu_memory_utilization: float = 0.8,
                          **kwargs) -> WorkerProcess:
        """
        Start a new vLLM worker process.
        
        Args:
            worker_id: Unique identifier for the worker
            model: Model name/path for vLLM
            port: Port to bind worker (auto-allocated if None)
            gpu_id: GPU device ID (auto-allocated if None)
            max_num_seqs: Max concurrent sequences
            max_model_len: Max sequence length
            gpu_memory_utilization: GPU memory fraction
            **kwargs: Additional vLLM configuration
        
        Returns:
            WorkerProcess instance
        """
        # Validate worker_id
        if worker_id in self.workers:
            raise ValueError(f"Worker {worker_id} already exists")
        
        # Allocate port
        if port is None:
            if not self.available_ports:
                raise RuntimeError("No available ports for worker")
            port = self.available_ports.pop()
        elif port in self.available_ports:
            self.available_ports.remove(port)
        
        # Allocate GPU
        if gpu_id is not None and gpu_id in self.allocated_gpus:
            raise ValueError(f"GPU {gpu_id} already allocated to worker {self.allocated_gpus[gpu_id]}")
        
        # Create worker process object
        worker = WorkerProcess(
            worker_id=worker_id,
            port=port,
            model=model,
            gpu_id=gpu_id,
            config={
                'max_num_seqs': max_num_seqs,
                'max_model_len': max_model_len,
                'gpu_memory_utilization': gpu_memory_utilization,
                **kwargs
            }
        )
        
        self.workers[worker_id] = worker
        
        if gpu_id is not None:
            self.allocated_gpus[gpu_id] = worker_id
        
        # Start the worker process
        try:
            await self._start_worker_process(worker)
            
            # Wait for worker to become healthy
            max_wait = 60  # 60 seconds
            start_wait = time.time()
            
            while time.time() - start_wait < max_wait:
                if await self._check_worker_health(worker):
                    worker.is_healthy = True
                    worker.consecutive_failures = 0
                    
                    # Register with gateway if configured
                    if self.gateway_url:
                        await self._register_worker_with_gateway(worker)
                    
                    logger.info(
                        "Worker started successfully",
                        worker_id=worker_id,
                        port=port,
                        model=model,
                        gpu_id=gpu_id
                    )
                    
                    return worker
                
                await asyncio.sleep(2)
            
            # Failed to start
            raise RuntimeError(f"Worker {worker_id} failed to become healthy within {max_wait}s")
        
        except Exception as e:
            logger.error("Failed to start worker", worker_id=worker_id, error=str(e))
            
            # Cleanup on failure
            await self.stop_worker(worker_id, force=True)
            raise
    
    async def stop_worker(self, worker_id: str, force: bool = False) -> None:
        """
        Stop a worker process.
        
        Args:
            worker_id: Worker to stop
            force: If True, kill immediately; otherwise graceful shutdown
        """
        if worker_id not in self.workers:
            logger.warning("Worker not found", worker_id=worker_id)
            return
        
        worker = self.workers[worker_id]
        
        if worker.state in (WorkerState.STOPPED, WorkerState.STOPPING):
            return
        
        worker.state = WorkerState.STOPPING
        
        logger.info("Stopping worker", worker_id=worker_id, force=force)
        
        try:
            # Deregister from gateway
            if self.gateway_url:
                await self._deregister_worker_from_gateway(worker)
            
            # Stop the process
            if worker.process:
                if force:
                    worker.process.kill()
                else:
                    worker.process.terminate()
                
                # Wait for process to exit
                try:
                    worker.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Worker didn't stop gracefully, killing", worker_id=worker_id)
                    worker.process.kill()
                    worker.process.wait(timeout=5)
            
            worker.state = WorkerState.STOPPED
            worker.stop_time = time.time()
            worker.is_healthy = False
            
            # Release resources
            self.available_ports.add(worker.port)
            if worker.gpu_id is not None and worker.gpu_id in self.allocated_gpus:
                del self.allocated_gpus[worker.gpu_id]
            
            logger.info("Worker stopped", worker_id=worker_id)
        
        except Exception as e:
            logger.error("Error stopping worker", worker_id=worker_id, error=str(e))
            worker.state = WorkerState.FAILED
    
    async def restart_worker(self, worker_id: str) -> None:
        """Restart a worker process"""
        if worker_id not in self.workers:
            raise ValueError(f"Worker {worker_id} not found")
        
        worker = self.workers[worker_id]
        
        logger.info("Restarting worker", worker_id=worker_id, restart_count=worker.restart_count)
        
        # Stop the worker
        await self.stop_worker(worker_id, force=False)
        
        # Increment restart counter
        worker.restart_count += 1
        
        # Wait a bit before restarting
        await asyncio.sleep(5)
        
        # Restart
        try:
            await self._start_worker_process(worker)
            worker.consecutive_failures = 0
        except Exception as e:
            logger.error("Failed to restart worker", worker_id=worker_id, error=str(e))
            worker.consecutive_failures += 1
            worker.state = WorkerState.FAILED
    
    def get_worker(self, worker_id: str) -> Optional[WorkerProcess]:
        """Get worker by ID"""
        return self.workers.get(worker_id)
    
    def list_workers(self) -> List[WorkerProcess]:
        """Get list of all workers"""
        return list(self.workers.values())
    
    def get_healthy_workers(self) -> List[WorkerProcess]:
        """Get list of healthy workers"""
        return [w for w in self.workers.values() if w.is_healthy and w.state == WorkerState.RUNNING]
    
    async def _start_worker_process(self, worker: WorkerProcess) -> None:
        """Start the actual worker subprocess"""
        worker.state = WorkerState.STARTING
        
        # Build command
        cmd = [
            "python",
            str(self.worker_script_path),
            "--model", worker.model,
            "--port", str(worker.port),
            "--host", "0.0.0.0",
            "--max-num-seqs", str(worker.config.get('max_num_seqs', 256)),
            "--max-model-len", str(worker.config.get('max_model_len', 4096)),
            "--gpu-memory-utilization", str(worker.config.get('gpu_memory_utilization', 0.8)),
        ]
        
        # Set up environment
        env = {}
        if worker.gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(worker.gpu_id)
        
        # Start process
        try:
            worker.process = subprocess.Popen(
                cmd,
                env=env or None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            worker.pid = worker.process.pid
            worker.state = WorkerState.RUNNING
            worker.start_time = time.time()
            
            logger.info(
                "Worker process started",
                worker_id=worker.worker_id,
                pid=worker.pid,
                port=worker.port,
                gpu_id=worker.gpu_id
            )
        
        except Exception as e:
            logger.error("Failed to start worker process", worker_id=worker.worker_id, error=str(e))
            worker.state = WorkerState.FAILED
            raise
    
    async def _check_worker_health(self, worker: WorkerProcess) -> bool:
        """Check if worker is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{worker.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        worker.last_health_check = time.time()
                        return data.get('status') == 'healthy'
            
            return False
        
        except Exception as e:
            logger.debug("Health check failed", worker_id=worker.worker_id, error=str(e))
            return False
    
    async def _health_monitor_loop(self) -> None:
        """Background task to monitor worker health"""
        while self._running:
            try:
                await self._check_all_workers_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e))
                await asyncio.sleep(10)
    
    async def _check_all_workers_health(self) -> None:
        """Check health of all running workers"""
        for worker in self.workers.values():
            if worker.state != WorkerState.RUNNING:
                continue
            
            try:
                is_healthy = await self._check_worker_health(worker)
                
                if is_healthy:
                    worker.is_healthy = True
                    worker.consecutive_failures = 0
                else:
                    worker.is_healthy = False
                    worker.consecutive_failures += 1
                    
                    logger.warning(
                        "Worker health check failed",
                        worker_id=worker.worker_id,
                        consecutive_failures=worker.consecutive_failures
                    )
                    
                    # Auto-restart if configured
                    if (self.auto_restart and 
                        worker.consecutive_failures >= 3 and
                        worker.restart_count < self.max_restart_attempts):
                        
                        logger.info(
                            "Auto-restarting unhealthy worker",
                            worker_id=worker.worker_id
                        )
                        
                        asyncio.create_task(self.restart_worker(worker.worker_id))
                    
                    elif worker.restart_count >= self.max_restart_attempts:
                        logger.error(
                            "Worker exceeded max restart attempts",
                            worker_id=worker.worker_id,
                            restart_count=worker.restart_count
                        )
                        worker.state = WorkerState.FAILED
            
            except Exception as e:
                logger.error(
                    "Error checking worker health",
                    worker_id=worker.worker_id,
                    error=str(e)
                )
    
    async def _register_worker_with_gateway(self, worker: WorkerProcess) -> None:
        """Register worker with gateway service"""
        if not self.gateway_url:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.gateway_url}/workers/register",
                    json={
                        'worker_id': worker.worker_id,
                        'host': 'localhost',
                        'port': worker.port,
                        'model': worker.model,
                        'gpu_id': worker.gpu_id,
                        'capabilities': worker.config
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(
                            "Worker registered with gateway",
                            worker_id=worker.worker_id,
                            gateway_url=self.gateway_url
                        )
                    else:
                        logger.warning(
                            "Failed to register worker with gateway",
                            worker_id=worker.worker_id,
                            status=response.status
                        )
        
        except Exception as e:
            logger.error(
                "Error registering worker with gateway",
                worker_id=worker.worker_id,
                error=str(e)
            )
    
    async def _deregister_worker_from_gateway(self, worker: WorkerProcess) -> None:
        """Deregister worker from gateway service"""
        if not self.gateway_url:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.gateway_url}/workers/deregister",
                    json={'worker_id': worker.worker_id},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(
                            "Worker deregistered from gateway",
                            worker_id=worker.worker_id
                        )
        
        except Exception as e:
            logger.error(
                "Error deregistering worker from gateway",
                worker_id=worker.worker_id,
                error=str(e)
            )


async def main():
    """Example usage"""
    manager = WorkerManager(
        gateway_url="http://localhost:8000",
        auto_restart=True
    )
    
    await manager.start()
    
    try:
        # Start workers
        await manager.start_worker(
            worker_id="worker-1",
            model="meta-llama/Llama-2-7b-hf",
            gpu_id=0
        )
        
        await manager.start_worker(
            worker_id="worker-2",
            model="meta-llama/Llama-2-7b-hf",
            gpu_id=1
        )
        
        # Keep running
        while True:
            healthy_workers = manager.get_healthy_workers()
            logger.info(f"Healthy workers: {len(healthy_workers)}")
            await asyncio.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    
    finally:
        await manager.stop()


if __name__ == "__main__":
    import sys
    import structlog
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer()
        ]
    )
    
    asyncio.run(main())
