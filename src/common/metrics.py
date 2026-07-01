"""
Monitoring and Metrics Collection

Provides Prometheus-compatible metrics for the distributed inference system.
Tracks performance, resource utilization, and system health.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import structlog
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    enabled: bool = True
    namespace: str = "distributed_inference"
    include_process_metrics: bool = True
    include_platform_metrics: bool = True
    
    # Histogram buckets for latency metrics (in milliseconds)
    latency_buckets: List[float] = field(default_factory=lambda: [
        10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000
    ])


class MetricsCollector:
    """
    Central metrics collector for the distributed inference system.
    
    Tracks:
    - Request throughput and latency
    - Cache hit rates and utilization
    - Worker health and resource usage
    - System-level statistics
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """Initialize metrics collector"""
        self.config = config or MetricsConfig()
        
        if not self.config.enabled:
            logger.info("Metrics collection disabled")
            return
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Initialize metric groups
        self._init_request_metrics()
        self._init_cache_metrics()
        self._init_worker_metrics()
        self._init_system_metrics()
        
        logger.info("MetricsCollector initialized", namespace=self.config.namespace)
    
    def _init_request_metrics(self) -> None:
        """Initialize request-related metrics"""
        ns = self.config.namespace
        
        # Request counts
        self.requests_total = Counter(
            f"{ns}_requests_total",
            "Total number of inference requests",
            ["status", "worker_id"],
            registry=self.registry
        )
        
        self.requests_in_flight = Gauge(
            f"{ns}_requests_in_flight",
            "Number of requests currently being processed",
            ["worker_id"],
            registry=self.registry
        )
        
        # Latency metrics
        self.request_duration_ms = Histogram(
            f"{ns}_request_duration_milliseconds",
            "Request processing duration in milliseconds",
            ["worker_id", "cache_hit"],
            buckets=self.config.latency_buckets,
            registry=self.registry
        )
        
        self.time_to_first_token_ms = Histogram(
            f"{ns}_ttft_milliseconds",
            "Time to first token in milliseconds",
            ["worker_id", "cache_hit"],
            buckets=self.config.latency_buckets,
            registry=self.registry
        )
        
        # Throughput
        self.tokens_generated_total = Counter(
            f"{ns}_tokens_generated_total",
            "Total number of tokens generated",
            ["worker_id"],
            registry=self.registry
        )
        
        self.tokens_per_second = Gauge(
            f"{ns}_tokens_per_second",
            "Current token generation rate",
            ["worker_id"],
            registry=self.registry
        )
    
    def _init_cache_metrics(self) -> None:
        """Initialize cache-related metrics"""
        ns = self.config.namespace
        
        # Cache operations
        self.cache_hits_total = Counter(
            f"{ns}_cache_hits_total",
            "Total number of cache hits",
            ["worker_id"],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            f"{ns}_cache_misses_total",
            "Total number of cache misses",
            ["worker_id"],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            f"{ns}_cache_hit_rate",
            "Cache hit rate (0.0 to 1.0)",
            ["worker_id"],
            registry=self.registry
        )
        
        # Cache storage
        self.cache_entries_total = Gauge(
            f"{ns}_cache_entries_total",
            "Number of cache entries stored",
            ["worker_id"],
            registry=self.registry
        )
        
        self.cache_size_bytes = Gauge(
            f"{ns}_cache_size_bytes",
            "Total size of cached data in bytes",
            ["worker_id"],
            registry=self.registry
        )
        
        self.cache_utilization = Gauge(
            f"{ns}_cache_utilization",
            "Cache utilization (0.0 to 1.0)",
            ["worker_id"],
            registry=self.registry
        )
        
        # Cache transfers
        self.cache_transfers_total = Counter(
            f"{ns}_cache_transfers_total",
            "Total number of cache transfer operations",
            ["source_worker", "target_worker", "status"],
            registry=self.registry
        )
        
        self.cache_transfer_bytes_total = Counter(
            f"{ns}_cache_transfer_bytes_total",
            "Total bytes transferred in cache operations",
            registry=self.registry
        )
        
        self.cache_transfer_duration_ms = Histogram(
            f"{ns}_cache_transfer_duration_milliseconds",
            "Cache transfer duration in milliseconds",
            buckets=[100, 500, 1000, 2000, 5000, 10000, 30000],
            registry=self.registry
        )
    
    def _init_worker_metrics(self) -> None:
        """Initialize worker-related metrics"""
        ns = self.config.namespace
        
        # Worker status
        self.workers_total = Gauge(
            f"{ns}_workers_total",
            "Total number of workers",
            ["status"],
            registry=self.registry
        )
        
        self.worker_uptime_seconds = Gauge(
            f"{ns}_worker_uptime_seconds",
            "Worker uptime in seconds",
            ["worker_id"],
            registry=self.registry
        )
        
        self.worker_restarts_total = Counter(
            f"{ns}_worker_restarts_total",
            "Total number of worker restarts",
            ["worker_id", "reason"],
            registry=self.registry
        )
        
        # Resource usage
        self.worker_gpu_memory_used_bytes = Gauge(
            f"{ns}_worker_gpu_memory_used_bytes",
            "GPU memory used by worker in bytes",
            ["worker_id", "gpu_id"],
            registry=self.registry
        )
        
        self.worker_gpu_memory_total_bytes = Gauge(
            f"{ns}_worker_gpu_memory_total_bytes",
            "Total GPU memory available to worker in bytes",
            ["worker_id", "gpu_id"],
            registry=self.registry
        )
        
        self.worker_gpu_utilization = Gauge(
            f"{ns}_worker_gpu_utilization",
            "GPU utilization (0.0 to 1.0)",
            ["worker_id", "gpu_id"],
            registry=self.registry
        )
        
        self.worker_cpu_utilization = Gauge(
            f"{ns}_worker_cpu_utilization",
            "CPU utilization (0.0 to 1.0)",
            ["worker_id"],
            registry=self.registry
        )
    
    def _init_system_metrics(self) -> None:
        """Initialize system-level metrics"""
        ns = self.config.namespace
        
        # Routing metrics
        self.routing_decisions_total = Counter(
            f"{ns}_routing_decisions_total",
            "Total number of routing decisions",
            ["strategy", "result"],
            registry=self.registry
        )
        
        self.routing_duration_ms = Histogram(
            f"{ns}_routing_duration_milliseconds",
            "Time taken for routing decisions in milliseconds",
            buckets=[1, 5, 10, 25, 50, 100],
            registry=self.registry
        )
        
        # System health
        self.system_errors_total = Counter(
            f"{ns}_system_errors_total",
            "Total number of system errors",
            ["component", "error_type"],
            registry=self.registry
        )
        
        self.system_health_score = Gauge(
            f"{ns}_system_health_score",
            "Overall system health score (0.0 to 1.0)",
            registry=self.registry
        )
    
    # Convenience methods for recording metrics
    
    def record_request(self, worker_id: str, status: str, duration_ms: float, 
                      cache_hit: bool = False, tokens_generated: int = 0,
                      ttft_ms: Optional[float] = None) -> None:
        """Record a completed request"""
        if not self.config.enabled:
            return
        
        self.requests_total.labels(status=status, worker_id=worker_id).inc()
        
        cache_hit_str = "true" if cache_hit else "false"
        self.request_duration_ms.labels(
            worker_id=worker_id,
            cache_hit=cache_hit_str
        ).observe(duration_ms)
        
        if ttft_ms is not None:
            self.time_to_first_token_ms.labels(
                worker_id=worker_id,
                cache_hit=cache_hit_str
            ).observe(ttft_ms)
        
        if tokens_generated > 0:
            self.tokens_generated_total.labels(worker_id=worker_id).inc(tokens_generated)
        
        if cache_hit:
            self.cache_hits_total.labels(worker_id=worker_id).inc()
        else:
            self.cache_misses_total.labels(worker_id=worker_id).inc()
    
    def record_cache_transfer(self, source_worker: str, target_worker: str,
                             success: bool, bytes_transferred: int = 0,
                             duration_ms: float = 0.0) -> None:
        """Record a cache transfer operation"""
        if not self.config.enabled:
            return
        
        status = "success" if success else "failure"
        self.cache_transfers_total.labels(
            source_worker=source_worker,
            target_worker=target_worker,
            status=status
        ).inc()
        
        if success and bytes_transferred > 0:
            self.cache_transfer_bytes_total.inc(bytes_transferred)
            self.cache_transfer_duration_ms.observe(duration_ms)
    
    def update_worker_status(self, worker_id: str, status: str, uptime_seconds: float = 0) -> None:
        """Update worker status metrics"""
        if not self.config.enabled:
            return
        
        if uptime_seconds > 0:
            self.worker_uptime_seconds.labels(worker_id=worker_id).set(uptime_seconds)
    
    def update_worker_resources(self, worker_id: str, gpu_id: int = 0,
                               gpu_memory_used: int = 0, gpu_memory_total: int = 1,
                               gpu_utilization: float = 0.0, cpu_utilization: float = 0.0) -> None:
        """Update worker resource metrics"""
        if not self.config.enabled:
            return
        
        self.worker_gpu_memory_used_bytes.labels(
            worker_id=worker_id,
            gpu_id=str(gpu_id)
        ).set(gpu_memory_used)
        
        self.worker_gpu_memory_total_bytes.labels(
            worker_id=worker_id,
            gpu_id=str(gpu_id)
        ).set(gpu_memory_total)
        
        self.worker_gpu_utilization.labels(
            worker_id=worker_id,
            gpu_id=str(gpu_id)
        ).set(gpu_utilization)
        
        self.worker_cpu_utilization.labels(worker_id=worker_id).set(cpu_utilization)
    
    def update_cache_state(self, worker_id: str, entries: int = 0,
                          size_bytes: int = 0, utilization: float = 0.0,
                          hit_rate: float = 0.0) -> None:
        """Update cache state metrics"""
        if not self.config.enabled:
            return
        
        self.cache_entries_total.labels(worker_id=worker_id).set(entries)
        self.cache_size_bytes.labels(worker_id=worker_id).set(size_bytes)
        self.cache_utilization.labels(worker_id=worker_id).set(utilization)
        self.cache_hit_rate.labels(worker_id=worker_id).set(hit_rate)
    
    def record_routing_decision(self, strategy: str, result: str, duration_ms: float) -> None:
        """Record a routing decision"""
        if not self.config.enabled:
            return
        
        self.routing_decisions_total.labels(strategy=strategy, result=result).inc()
        self.routing_duration_ms.observe(duration_ms)
    
    def record_error(self, component: str, error_type: str) -> None:
        """Record a system error"""
        if not self.config.enabled:
            return
        
        self.system_errors_total.labels(component=component, error_type=error_type).inc()
    
    def update_system_health(self, health_score: float) -> None:
        """Update overall system health score"""
        if not self.config.enabled:
            return
        
        self.system_health_score.set(health_score)
    
    def set_requests_in_flight(self, worker_id: str, count: int) -> None:
        """Set current number of in-flight requests"""
        if not self.config.enabled:
            return
        
        self.requests_in_flight.labels(worker_id=worker_id).set(count)
    
    def set_tokens_per_second(self, worker_id: str, tps: float) -> None:
        """Set current token generation rate"""
        if not self.config.enabled:
            return
        
        self.tokens_per_second.labels(worker_id=worker_id).set(tps)
    
    def set_worker_count(self, status: str, count: int) -> None:
        """Set worker count by status"""
        if not self.config.enabled:
            return
        
        self.workers_total.labels(status=status).set(count)
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        if not self.config.enabled:
            return b""
        
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get content type for metrics export"""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
_global_metrics: Optional[MetricsCollector] = None


def initialize_metrics(config: Optional[MetricsConfig] = None) -> MetricsCollector:
    """Initialize global metrics collector"""
    global _global_metrics
    
    if _global_metrics is not None:
        logger.warning("Metrics already initialized, returning existing instance")
        return _global_metrics
    
    _global_metrics = MetricsCollector(config)
    return _global_metrics


def get_metrics() -> Optional[MetricsCollector]:
    """Get global metrics collector instance"""
    return _global_metrics


# Context manager for timing operations
class Timer:
    """Context manager for timing operations and recording to metrics"""
    
    def __init__(self, operation: str = "operation"):
        self.operation = operation
        self.start_time: Optional[float] = None
        self.duration_ms: float = 0.0
    
    def __enter__(self) -> 'Timer':
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.duration_ms = (time.time() - self.start_time) * 1000
        return False
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return self.duration_ms


async def main():
    """Example usage"""
    # Initialize metrics
    config = MetricsConfig(
        enabled=True,
        namespace="distributed_inference_test"
    )
    
    metrics = initialize_metrics(config)
    
    # Simulate some metrics
    for i in range(10):
        with Timer() as timer:
            await asyncio.sleep(0.1)  # Simulate work
        
        metrics.record_request(
            worker_id="worker-1",
            status="success",
            duration_ms=timer.elapsed_ms,
            cache_hit=i % 2 == 0,
            tokens_generated=100,
            ttft_ms=10.0
        )
    
    # Export metrics
    metrics_output = metrics.export_metrics()
    print(metrics_output.decode('utf-8'))


if __name__ == "__main__":
    import asyncio
    
    asyncio.run(main())
