"""
Routing Decision Engine for vLLM-based Distributed Inference.

This module implements intelligent routing decisions based on:
1. KV cache likelihood (prefix matching probability)
2. Load balancing across vLLM instances
3. Node health and capacity
4. Request characteristics and priority

Key Design Decisions:
- Local memory for active requests (faster than Redis for recent requests)
- Weighted scoring algorithm combining cache hit probability and load
- vLLM-specific optimizations for PagedAttention and KV cache management
- Hierarchical routing: prefix-based → load-based → fallback
"""

import asyncio
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import structlog

try:
    from ..cache.registry import CacheRegistryService
except ImportError:
    # For testing without Redis dependencies
    class CacheRegistryService:
        pass
from ..common.models import (
    InferenceRequest, 
    NodeInfo, 
    RouteDecision, 
    ServiceType,
    RoutingConfig
)
from ..common.prefix_hasher import PrefixHashManager

logger = structlog.get_logger()


@dataclass
class ActiveRequest:
    """Track active requests for local memory optimization."""
    request_id: str
    prefix_hash: str
    node_id: str
    start_time: float
    estimated_completion_time: float
    tokens_processed: int = 0
    is_prefill_complete: bool = False


@dataclass
class NodeLoadMetrics:
    """Real-time load metrics for a vLLM node."""
    node_id: str
    
    # vLLM specific metrics
    active_sequences: int = 0           # Number of concurrent sequences
    pending_sequences: int = 0          # Queued sequences waiting for processing
    kv_cache_usage_percent: float = 0.0 # KV cache memory utilization
    gpu_memory_usage_percent: float = 0.0
    
    # Performance metrics
    current_throughput_tps: float = 0.0 # Tokens per second
    avg_prefill_latency_ms: float = 100.0
    avg_decode_latency_ms: float = 20.0
    
    # Capacity metrics
    max_concurrent_sequences: int = 256  # vLLM configuration dependent
    max_sequence_length: int = 4096
    max_batch_size: int = 32
    
    # Health metrics
    last_heartbeat: float = 0.0
    error_rate_percent: float = 0.0
    
    def is_overloaded(self, threshold: float = 0.9) -> bool:
        """Check if node is overloaded based on multiple factors."""
        return (
            self.active_sequences >= self.max_concurrent_sequences * threshold or
            self.kv_cache_usage_percent > threshold or
            self.gpu_memory_usage_percent > threshold * 0.95  # GPU memory is more critical
        )
    
    def get_capacity_score(self) -> float:
        """Calculate capacity score (0.0 = full, 1.0 = empty)."""
        sequence_capacity = 1.0 - (self.active_sequences / self.max_concurrent_sequences)
        memory_capacity = 1.0 - (self.gpu_memory_usage_percent / 100.0)
        cache_capacity = 1.0 - (self.kv_cache_usage_percent / 100.0)
        
        # Weighted average - memory and cache are more critical
        return (sequence_capacity * 0.3 + memory_capacity * 0.4 + cache_capacity * 0.3)


class KVCacheAffinityTracker:
    """
    Track KV cache affinity for intelligent routing.
    
    Key insight: KV cache is most useful when the current prompt shares a prefix
    with a recently processed prompt on the same vLLM instance.
    """
    
    def __init__(self, memory_window_minutes: int = 30, max_entries: int = 10000):
        """
        Initialize cache affinity tracker.
        
        Args:
            memory_window_minutes: How long to remember recent requests
            max_entries: Maximum entries to keep in memory (LRU eviction)
        """
        self.memory_window_seconds = memory_window_minutes * 60
        self.max_entries = max_entries
        
        # prefix_hash -> (node_id, last_seen_time, access_count)
        self.prefix_to_node: Dict[str, Tuple[str, float, int]] = {}
        
        # node_id -> deque of (prefix_hash, timestamp) for LRU tracking
        self.node_recent_prefixes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Track active requests for immediate routing decisions
        self.active_requests: Dict[str, ActiveRequest] = {}
        
    def record_request(self, request_id: str, prefix_hash: str, node_id: str, 
                      estimated_completion_time: float) -> None:
        """Record a new request being processed."""
        now = time.time()
        
        # Update prefix-to-node mapping
        if prefix_hash in self.prefix_to_node:
            _, _, access_count = self.prefix_to_node[prefix_hash]
            self.prefix_to_node[prefix_hash] = (node_id, now, access_count + 1)
        else:
            self.prefix_to_node[prefix_hash] = (node_id, now, 1)
        
        # Track recent prefixes for this node
        self.node_recent_prefixes[node_id].append((prefix_hash, now))
        
        # Track active request
        self.active_requests[request_id] = ActiveRequest(
            request_id=request_id,
            prefix_hash=prefix_hash,
            node_id=node_id,
            start_time=now,
            estimated_completion_time=estimated_completion_time
        )
        
        # Cleanup old entries
        self._cleanup_old_entries()
    
    def complete_request(self, request_id: str, actual_completion_time: float = None) -> None:
        """Mark a request as completed."""
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            if actual_completion_time:
                request.estimated_completion_time = actual_completion_time
            del self.active_requests[request_id]
    
    def get_cache_affinity_score(self, prefix_hash: str, node_id: str) -> float:
        """
        Calculate cache affinity score for a prefix-node combination.
        
        Returns:
            Float between 0.0 and 1.0:
            - 1.0 = Very high likelihood of cache hit (recent exact match)
            - 0.5-0.8 = Medium likelihood (similar prefix recently processed)
            - 0.0-0.3 = Low likelihood (no recent related activity)
        """
        now = time.time()
        
        # Check for exact prefix match
        if prefix_hash in self.prefix_to_node:
            cached_node_id, last_seen, access_count = self.prefix_to_node[prefix_hash]
            
            if cached_node_id == node_id:
                # Exact prefix match on this node
                age_seconds = now - last_seen
                if age_seconds < 300:  # 5 minutes - very fresh
                    return 0.9 + min(0.1, access_count * 0.02)  # Bonus for frequent access
                elif age_seconds < 1800:  # 30 minutes - still valuable
                    return 0.7 * (1.0 - age_seconds / 1800)
                else:
                    # Old cache, but might still exist in vLLM memory
                    return 0.3 * (1.0 - min(1.0, age_seconds / 3600))
            else:
                # Prefix exists on different node - negative affinity
                return 0.1
        
        # Check for similar prefixes on this node (prefix collision potential)
        if node_id in self.node_recent_prefixes:
            recent_prefixes = self.node_recent_prefixes[node_id]
            similarity_score = 0.0
            
            for cached_prefix, timestamp in recent_prefixes:
                if now - timestamp > self.memory_window_seconds:
                    continue
                
                # Simple similarity: check if prefixes share common beginning
                similarity = self._calculate_prefix_similarity(prefix_hash, cached_prefix)
                age_factor = 1.0 - (now - timestamp) / self.memory_window_seconds
                similarity_score = max(similarity_score, similarity * age_factor * 0.6)
            
            return similarity_score
        
        return 0.2  # Base score for unknown prefix on any node
    
    def _calculate_prefix_similarity(self, prefix1: str, prefix2: str) -> float:
        """Calculate similarity between two prefix hashes."""
        if prefix1 == prefix2:
            return 1.0
        
        # For SHA-256 hashes, we can't easily determine semantic similarity
        # Instead, we use the assumption that similar prefixes might have
        # been generated from similar text patterns
        
        # Simple heuristic: check how many leading characters match
        common_prefix = 0
        for i in range(min(len(prefix1), len(prefix2))):
            if prefix1[i] == prefix2[i]:
                common_prefix += 1
            else:
                break
        
        # Convert to similarity score (16 matching chars = high similarity)
        return min(1.0, common_prefix / 16.0)
    
    def _cleanup_old_entries(self) -> None:
        """Remove old entries to prevent memory bloat."""
        now = time.time()
        cutoff_time = now - self.memory_window_seconds
        
        # Clean up prefix mappings
        to_remove = []
        for prefix_hash, (node_id, timestamp, count) in self.prefix_to_node.items():
            if timestamp < cutoff_time:
                to_remove.append(prefix_hash)
        
        for prefix_hash in to_remove:
            del self.prefix_to_node[prefix_hash]
        
        # Clean up node recent prefixes
        for node_id, prefix_deque in self.node_recent_prefixes.items():
            while prefix_deque and prefix_deque[0][1] < cutoff_time:
                prefix_deque.popleft()
        
        # Clean up completed active requests
        completed_requests = []
        for request_id, request in self.active_requests.items():
            if request.estimated_completion_time < now:
                completed_requests.append(request_id)
        
        for request_id in completed_requests:
            del self.active_requests[request_id]
        
        # LRU eviction if too many entries
        if len(self.prefix_to_node) > self.max_entries:
            # Sort by last access time and remove oldest
            sorted_prefixes = sorted(
                self.prefix_to_node.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            
            entries_to_remove = len(sorted_prefixes) - self.max_entries + 1000  # Remove extra
            for i in range(entries_to_remove):
                prefix_hash = sorted_prefixes[i][0]
                del self.prefix_to_node[prefix_hash]


class LoadBalancingAlgorithm:
    """
    Advanced load balancing for vLLM instances.
    
    Combines multiple factors:
    1. Current load (active sequences, memory usage)
    2. Performance characteristics (throughput, latency)
    3. Workload suitability (sequence length, complexity)
    """
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.node_metrics: Dict[str, NodeLoadMetrics] = {}
        
    def update_node_metrics(self, node_id: str, metrics: NodeLoadMetrics) -> None:
        """Update real-time metrics for a node."""
        metrics.last_heartbeat = time.time()
        self.node_metrics[node_id] = metrics
    
    def calculate_load_score(self, node_id: str, request: InferenceRequest) -> float:
        """
        Calculate load-based routing score (0.0 = worst, 1.0 = best).
        
        Considers:
        - Current utilization (sequences, memory, cache)
        - Performance characteristics
        - Workload suitability
        """
        if node_id not in self.node_metrics:
            return 0.1  # Unknown node, low score
        
        metrics = self.node_metrics[node_id]
        now = time.time()
        
        # Health check
        if now - metrics.last_heartbeat > 30:  # 30 seconds
            return 0.0  # Stale metrics, avoid this node
        
        if metrics.error_rate_percent > 10:  # High error rate
            return 0.1
        
        # Utilization scoring
        utilization_score = metrics.get_capacity_score()
        
        # Performance scoring based on current throughput
        target_throughput = 100.0  # tokens/second target
        performance_score = min(1.0, metrics.current_throughput_tps / target_throughput)
        
        # Workload suitability scoring
        suitability_score = self._calculate_workload_suitability(metrics, request)
        
        # Queue depth penalty
        queue_penalty = 1.0
        if metrics.pending_sequences > 10:
            queue_penalty = max(0.3, 1.0 - (metrics.pending_sequences - 10) / 50.0)
        
        # Weighted combination
        total_score = (
            utilization_score * 0.4 +        # Most important: can it handle the load?
            performance_score * 0.3 +        # Current performance
            suitability_score * 0.2 +        # Workload match
            queue_penalty * 0.1              # Queue consideration
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _calculate_workload_suitability(self, metrics: NodeLoadMetrics, 
                                       request: InferenceRequest) -> float:
        """Calculate how suitable this node is for the specific request."""
        # Estimate total sequence length
        estimated_length = len(request.prompt.split()) * 1.3 + request.max_tokens  # Rough token estimate
        
        # Check if request fits in node's capabilities
        if estimated_length > metrics.max_sequence_length:
            return 0.0  # Can't handle this request
        
        # Score based on how well the request fits
        length_utilization = estimated_length / metrics.max_sequence_length
        
        if length_utilization < 0.3:
            # Small request - prefer nodes with smaller capacity for efficiency
            return 0.7
        elif length_utilization < 0.7:
            # Medium request - good fit
            return 1.0
        else:
            # Large request - still okay but may block other requests
            return 0.8


class RoutingDecisionEngine:
    """
    Main routing decision engine combining all factors.
    """
    
    def __init__(self, 
                 cache_registry: CacheRegistryService,
                 prefix_manager: PrefixHashManager,
                 config: RoutingConfig = None):
        """
        Initialize routing engine.
        
        Args:
            cache_registry: Global cache registry service
            prefix_manager: Prefix hashing manager
            config: Routing configuration
        """
        self.cache_registry = cache_registry
        self.prefix_manager = prefix_manager
        self.config = config or RoutingConfig()
        
        # Local state for fast routing decisions
        self.affinity_tracker = KVCacheAffinityTracker()
        self.load_balancer = LoadBalancingAlgorithm(self.config)
        
        # Node management
        self.available_nodes: Dict[str, NodeInfo] = {}
        self.healthy_nodes: Dict[str, NodeInfo] = {}
        
        # Performance tracking
        self.routing_stats = {
            'total_requests': 0,
            'cache_hit_routes': 0,
            'load_balanced_routes': 0,
            'fallback_routes': 0,
            'routing_latency_ms': deque(maxlen=1000)
        }
    
    async def route_request(self, request: InferenceRequest) -> RouteDecision:
        """
        Make routing decision for an inference request.
        
        Algorithm:
        1. Calculate prefix hash for cache lookup
        2. Score all healthy nodes based on cache affinity + load
        3. Select best node with fallback options
        4. Track routing decision for future optimization
        """
        start_time = time.time()
        
        try:
            # Generate prefix hash for cache lookup
            prefix_hash = self.prefix_manager.get_routing_hash(request.prompt)
            request.prefix_hash = prefix_hash
            
            # Get healthy nodes
            healthy_nodes = await self._get_healthy_nodes()
            
            if not healthy_nodes:
                raise RuntimeError("No healthy nodes available")
            
            # Score all nodes
            node_scores = await self._score_nodes(request, prefix_hash, healthy_nodes)
            
            # Select best node with fallback options
            decision = self._select_best_node(node_scores, request)
            
            # Record routing decision for tracking
            await self._record_routing_decision(request, decision)
            
            # Update statistics
            routing_latency = (time.time() - start_time) * 1000
            self.routing_stats['routing_latency_ms'].append(routing_latency)
            self.routing_stats['total_requests'] += 1
            
            logger.info(
                "routing_decision",
                request_id=request.request_id,
                target_node=decision.target_node,
                cache_hit_score=decision.cache_hit_score,
                load_score=decision.load_score,
                total_score=decision.total_score(),
                routing_latency_ms=routing_latency
            )
            
            return decision
            
        except Exception as e:
            logger.error(
                "routing_failed",
                request_id=request.request_id,
                error=str(e)
            )
            
            # Emergency fallback to any available node
            if healthy_nodes:
                fallback_node = list(healthy_nodes.values())[0]
                return RouteDecision(
                    target_node=fallback_node.node_id,
                    confidence=0.1,
                    cache_hit_score=0.0,
                    load_score=0.1,
                    estimated_latency_ms=5000.0,  # Conservative estimate
                    cache_hit_probability=0.0
                )
            
            raise RuntimeError("No nodes available and routing failed")
    
    async def _get_healthy_nodes(self) -> Dict[str, NodeInfo]:
        """Get currently healthy nodes from cache registry."""
        # Check local cache first (updated every few seconds)
        now = time.time()
        
        if (hasattr(self, '_last_node_refresh') and 
            now - self._last_node_refresh < 10):  # 10 second cache
            return self.healthy_nodes
        
        # Refresh from registry
        try:
            nodes = await self.cache_registry.get_healthy_nodes(service_type="prefill")
            self.healthy_nodes = {node.node_id: node for node in nodes}
            self._last_node_refresh = now
            
            return self.healthy_nodes
            
        except Exception as e:
            logger.warning("Failed to refresh healthy nodes", error=str(e))
            # Use cached nodes if refresh fails
            return self.healthy_nodes
    
    async def _score_nodes(self, request: InferenceRequest, prefix_hash: str, 
                          nodes: Dict[str, NodeInfo]) -> List[Tuple[str, float, Dict]]:
        """
        Score all nodes for routing decision.
        
        Returns:
            List of (node_id, total_score, score_breakdown) tuples
        """
        scored_nodes = []
        
        for node_id, node_info in nodes.items():
            # Cache affinity score (local memory lookup)
            cache_score = self.affinity_tracker.get_cache_affinity_score(prefix_hash, node_id)
            
            # Load balancing score
            load_score = self.load_balancer.calculate_load_score(node_id, request)
            
            # Latency score (based on node performance characteristics)
            latency_score = self._calculate_latency_score(node_info)
            
            # Capacity score (can it handle this request?)
            capacity_score = self._calculate_capacity_score(node_info, request)
            
            # Weighted total score
            total_score = (
                cache_score * self.config.cache_hit_weight +
                load_score * self.config.load_weight +
                latency_score * self.config.latency_weight +
                capacity_score * self.config.capacity_weight
            )
            
            score_breakdown = {
                'cache_hit_score': cache_score,
                'load_score': load_score,
                'latency_score': latency_score,
                'capacity_score': capacity_score
            }
            
            scored_nodes.append((node_id, total_score, score_breakdown))
        
        # Sort by score (highest first)
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return scored_nodes
    
    def _calculate_latency_score(self, node_info: NodeInfo) -> float:
        """Calculate latency-based score for a node."""
        target_latency = 100.0  # ms
        actual_latency = node_info.avg_latency_ms
        
        if actual_latency <= target_latency:
            return 1.0
        else:
            # Exponential decay for higher latencies
            return max(0.1, 0.5 ** ((actual_latency - target_latency) / target_latency))
    
    def _calculate_capacity_score(self, node_info: NodeInfo, request: InferenceRequest) -> float:
        """Calculate capacity-based score for a node."""
        # Simple capacity scoring based on current load
        if node_info.is_overloaded():
            return 0.1
        elif node_info.current_load > 0.8:
            return 0.5
        elif node_info.current_load > 0.6:
            return 0.8
        else:
            return 1.0
    
    def _select_best_node(self, scored_nodes: List[Tuple[str, float, Dict]], 
                         request: InferenceRequest) -> RouteDecision:
        """Select the best node and prepare routing decision."""
        if not scored_nodes:
            raise RuntimeError("No nodes available for selection")
        
        # Primary node (best score)
        best_node_id, best_score, best_breakdown = scored_nodes[0]
        
        # Fallback nodes (top 3 alternatives)
        fallback_nodes = [node_id for node_id, score, _ in scored_nodes[1:4]]
        
        # Estimate latency based on cache hit probability and node performance
        cache_hit_probability = best_breakdown['cache_hit_score']
        base_latency = 200.0  # Base processing latency
        
        if cache_hit_probability > 0.7:
            estimated_latency = base_latency * 0.3  # Significant speedup
        elif cache_hit_probability > 0.4:
            estimated_latency = base_latency * 0.6  # Moderate speedup
        else:
            estimated_latency = base_latency  # No cache benefit
        
        # Confidence based on score spread
        if len(scored_nodes) > 1:
            second_best_score = scored_nodes[1][1]
            confidence = min(1.0, (best_score - second_best_score) + 0.5)
        else:
            confidence = best_score
        
        return RouteDecision(
            target_node=best_node_id,
            confidence=confidence,
            cache_hit_score=best_breakdown['cache_hit_score'],
            load_score=best_breakdown['load_score'],
            latency_score=best_breakdown['latency_score'],
            capacity_score=best_breakdown['capacity_score'],
            estimated_latency_ms=estimated_latency,
            cache_hit_probability=cache_hit_probability,
            fallback_nodes=fallback_nodes
        )
    
    async def _record_routing_decision(self, request: InferenceRequest, 
                                      decision: RouteDecision) -> None:
        """Record routing decision for tracking and optimization."""
        # Update affinity tracker
        estimated_completion = time.time() + (decision.estimated_latency_ms / 1000.0)
        
        self.affinity_tracker.record_request(
            request_id=request.request_id,
            prefix_hash=request.prefix_hash,
            node_id=decision.target_node,
            estimated_completion_time=estimated_completion
        )
        
        # Update routing statistics
        if decision.cache_hit_probability > 0.7:
            self.routing_stats['cache_hit_routes'] += 1
        elif decision.load_score == max(decision.cache_hit_score, decision.load_score, 
                                       decision.latency_score, decision.capacity_score):
            self.routing_stats['load_balanced_routes'] += 1
        else:
            self.routing_stats['fallback_routes'] += 1
    
    async def update_node_metrics(self, node_id: str, metrics: NodeLoadMetrics) -> None:
        """Update real-time node metrics for routing decisions."""
        self.load_balancer.update_node_metrics(node_id, metrics)
    
    async def complete_request(self, request_id: str, 
                              actual_latency_ms: float = None,
                              cache_hit: bool = None) -> None:
        """Record request completion for routing optimization."""
        completion_time = time.time()
        
        # Update affinity tracker
        self.affinity_tracker.complete_request(request_id, completion_time)
        
        # TODO: Use actual performance data to improve routing algorithms
        # - Update latency models based on actual vs predicted latency
        # - Update cache hit models based on actual cache performance
        # - Adjust scoring weights based on observed performance
    
    def get_routing_statistics(self) -> Dict:
        """Get routing performance statistics."""
        stats = self.routing_stats.copy()
        
        if stats['routing_latency_ms']:
            stats['avg_routing_latency_ms'] = sum(stats['routing_latency_ms']) / len(stats['routing_latency_ms'])
            stats['p95_routing_latency_ms'] = sorted(stats['routing_latency_ms'])[int(len(stats['routing_latency_ms']) * 0.95)]
        else:
            stats['avg_routing_latency_ms'] = 0
            stats['p95_routing_latency_ms'] = 0
        
        if stats['total_requests'] > 0:
            stats['cache_hit_route_percentage'] = (stats['cache_hit_routes'] / stats['total_requests']) * 100
            stats['load_balanced_route_percentage'] = (stats['load_balanced_routes'] / stats['total_requests']) * 100
        
        return stats