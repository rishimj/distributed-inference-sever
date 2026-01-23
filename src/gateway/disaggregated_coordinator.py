"""
Disaggregated Request Coordinator

Orchestrates the prefill → decode handoff across specialized nodes.
Handles routing, cache transfer, and performance optimization for
disaggregated inference on distributed clusters like PACE ICE.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass
from enum import Enum

import aiohttp

from ..common.models import InferenceRequest
from .worker_client import WorkerClientPool
from ..workers.prefill_worker import PrefillResult
from ..workers.decode_worker import DecodeRequest, DecodeResult
from ..cache.infiniband_transfer import InfiniBandTransferManager, MockInfiniBandTransferManager


class NodeType(str, Enum):
    """Types of specialized nodes in disaggregated architecture."""
    PREFILL = "prefill"
    DECODE = "decode"
    HYBRID = "hybrid"  # Can handle both phases


@dataclass
class DisaggregatedMetrics:
    """Performance metrics for disaggregated processing."""
    request_id: str
    prefill_node: str
    decode_node: str
    prefill_time_ms: float
    cache_transfer_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    cache_size_bytes: int
    tokens_generated: int
    cache_hit: bool
    disaggregation_overhead_ms: float


class NodeSelector:
    """
    Intelligent node selection for prefill and decode phases.
    
    Considers:
    - Node specialization and current load
    - Cache locality and transfer costs
    - Network topology and bandwidth
    - Historical performance patterns
    """
    
    def __init__(self):
        self.node_performance_history: Dict[str, List[float]] = {}
        self.cache_affinity_map: Dict[str, str] = {}  # prefix_hash -> preferred_node
        
    async def select_prefill_node(self, 
                                 request: InferenceRequest,
                                 available_nodes: List[str],
                                 node_metrics: Dict[str, Dict]) -> str:
        """
        Select optimal prefill node based on:
        - Current load and compute capacity
        - Prefill specialization
        - Estimated processing time
        """
        best_node = None
        best_score = -1.0
        
        for node_id in available_nodes:
            metrics = node_metrics.get(node_id, {})
            
            # Skip if not a prefill-capable node
            worker_type = metrics.get('worker_type', 'unknown')
            if worker_type not in ['prefill', 'hybrid']:
                continue
            
            # Calculate selection score
            score = self._calculate_prefill_score(request, node_id, metrics)
            
            if score > best_score:
                best_score = score
                best_node = node_id
        
        if not best_node:
            raise Exception("No suitable prefill nodes available")
        
        logging.info(f"Selected prefill node {best_node} with score {best_score:.3f}")
        return best_node
    
    async def select_decode_node(self, 
                                request: InferenceRequest,
                                prefill_result: PrefillResult,
                                available_nodes: List[str],
                                node_metrics: Dict[str, Dict],
                                prefill_node: str) -> str:
        """
        Select optimal decode node considering:
        - Decode specialization and capacity
        - Cache transfer costs
        - Memory bandwidth requirements
        - Estimated generation length
        """
        best_node = None
        best_score = -1.0
        
        for node_id in available_nodes:
            metrics = node_metrics.get(node_id, {})
            
            # Skip if not decode-capable
            worker_type = metrics.get('worker_type', 'unknown')
            if worker_type not in ['decode', 'hybrid']:
                continue
            
            # Calculate selection score
            score = self._calculate_decode_score(
                request, prefill_result, node_id, metrics, prefill_node
            )
            
            if score > best_score:
                best_score = score
                best_node = node_id
        
        if not best_node:
            raise Exception("No suitable decode nodes available")
        
        logging.info(f"Selected decode node {best_node} with score {best_score:.3f}")
        return best_node
    
    def _calculate_prefill_score(self, 
                                request: InferenceRequest, 
                                node_id: str, 
                                metrics: Dict) -> float:
        """
        Calculate prefill node selection score.
        Higher score = better choice.
        """
        # Base score for prefill capability
        worker_type = metrics.get('worker_type', 'unknown')
        if worker_type == 'prefill':
            base_score = 1.0
        elif worker_type == 'hybrid':
            base_score = 0.7
        else:
            return 0.0
        
        # Load factor (lower load = higher score)
        active_requests = metrics.get('active_requests', 0)
        max_capacity = metrics.get('max_concurrent_sequences', 16)
        load_factor = 1.0 - (active_requests / max_capacity)
        load_score = max(0.1, load_factor)
        
        # Performance factor
        avg_processing_time = metrics.get('avg_processing_time_ms', 1000)
        performance_score = min(1.0, 500.0 / avg_processing_time)  # Target 500ms
        
        # Prompt length suitability
        prompt_length = len(request.prompt)
        if prompt_length < 1000:
            length_score = 1.0  # Good for prefill optimization
        elif prompt_length < 2000:
            length_score = 0.8
        else:
            length_score = 0.6  # Very long prompts are harder to batch
        
        # Weighted combination
        total_score = (
            base_score * 0.3 +
            load_score * 0.4 +
            performance_score * 0.2 +
            length_score * 0.1
        )
        
        return total_score
    
    def _calculate_decode_score(self, 
                               request: InferenceRequest,
                               prefill_result: PrefillResult,
                               node_id: str, 
                               metrics: Dict,
                               prefill_node: str) -> float:
        """
        Calculate decode node selection score.
        Considers cache transfer costs and decode specialization.
        """
        # Base score for decode capability
        worker_type = metrics.get('worker_type', 'unknown')
        if worker_type == 'decode':
            base_score = 1.0
        elif worker_type == 'hybrid':
            base_score = 0.7
        else:
            return 0.0
        
        # Load factor
        active_sequences = metrics.get('active_sequences', 0)
        max_capacity = metrics.get('max_concurrent_sequences', 64)
        load_factor = 1.0 - (active_sequences / max_capacity)
        load_score = max(0.1, load_factor)
        
        # Performance factor
        tokens_per_second = metrics.get('tokens_per_second', 10)
        performance_score = min(1.0, tokens_per_second / 50.0)  # Target 50 TPS
        
        # Cache locality bonus (same node = no transfer cost)
        if node_id == prefill_node:
            locality_score = 1.0  # No transfer needed
        else:
            # Penalize based on cache size (larger = more expensive to transfer)
            cache_size_mb = prefill_result.cache_size_bytes / (1024 * 1024)
            transfer_penalty = min(0.3, cache_size_mb / 100)  # Penalty for large caches
            locality_score = max(0.4, 1.0 - transfer_penalty)
        
        # Generation length suitability
        expected_tokens = request.max_tokens
        if expected_tokens > 500:
            length_score = 1.0  # Decode nodes excel at long sequences
        elif expected_tokens > 100:
            length_score = 0.8
        else:
            length_score = 0.6  # Short sequences might be better on hybrid
        
        # Weighted combination
        total_score = (
            base_score * 0.25 +
            load_score * 0.35 +
            performance_score * 0.15 +
            locality_score * 0.15 +
            length_score * 0.1
        )
        
        return total_score


class CacheTransferManager:
    """
    Manages KV cache transfers between prefill and decode nodes.
    Optimized for high-bandwidth cluster networks like InfiniBand.
    """
    
    def __init__(self):
        self.transfer_stats: Dict[str, List[float]] = {}
        
    async def estimate_transfer_time(self, 
                                   source_node: str,
                                   target_node: str,
                                   cache_size_bytes: int) -> float:
        """
        Estimate cache transfer time based on:
        - Network topology and bandwidth
        - Historical transfer performance
        - Cache size and compression ratio
        """
        if source_node == target_node:
            return 0.0  # No transfer needed
        
        # Get historical performance
        transfer_key = f"{source_node}->{target_node}"
        historical_times = self.transfer_stats.get(transfer_key, [])
        
        if historical_times:
            # Use median of recent transfers
            avg_mbps = sum(historical_times[-10:]) / len(historical_times[-10:])
        else:
            # Estimate based on network type
            if self._is_same_rack(source_node, target_node):
                avg_mbps = 10000  # 10 Gbps intra-rack
            elif self._is_same_cluster(source_node, target_node):
                avg_mbps = 25000  # 25 Gbps InfiniBand inter-rack
            else:
                avg_mbps = 1000   # 1 Gbps inter-cluster
        
        # Account for compression (typically 10x for KV cache)
        compressed_size = cache_size_bytes / 10
        
        # Estimate transfer time
        transfer_time_seconds = (compressed_size * 8) / (avg_mbps * 1_000_000)
        
        # Add overhead for serialization/deserialization
        overhead_ms = min(50, cache_size_bytes / (1024 * 1024) * 2)  # 2ms per MB
        
        return transfer_time_seconds * 1000 + overhead_ms
    
    def _is_same_rack(self, node1: str, node2: str) -> bool:
        """Check if nodes are on the same rack (faster interconnect)."""
        # Parse node names to determine rack
        # Example: node1 = "pace-ice-rack1-node05", node2 = "pace-ice-rack1-node08"
        if 'rack' in node1 and 'rack' in node2:
            rack1 = node1.split('rack')[1].split('-')[0]
            rack2 = node2.split('rack')[1].split('-')[0]
            return rack1 == rack2
        return False
    
    def _is_same_cluster(self, node1: str, node2: str) -> bool:
        """Check if nodes are in the same cluster."""
        # Simple heuristic based on node naming
        cluster1 = node1.split('-')[0] if '-' in node1 else node1
        cluster2 = node2.split('-')[0] if '-' in node2 else node2
        return cluster1 == cluster2
    
    def record_transfer_performance(self, 
                                  source_node: str,
                                  target_node: str,
                                  cache_size_bytes: int,
                                  transfer_time_ms: float):
        """Record actual transfer performance for future estimates."""
        if transfer_time_ms > 0 and cache_size_bytes > 0:
            # Calculate effective bandwidth in Mbps
            bandwidth_mbps = (cache_size_bytes * 8) / (transfer_time_ms / 1000) / 1_000_000
            
            transfer_key = f"{source_node}->{target_node}"
            if transfer_key not in self.transfer_stats:
                self.transfer_stats[transfer_key] = []
            
            self.transfer_stats[transfer_key].append(bandwidth_mbps)
            
            # Keep only recent measurements
            if len(self.transfer_stats[transfer_key]) > 50:
                self.transfer_stats[transfer_key] = self.transfer_stats[transfer_key][-50:]


class DisaggregatedRequestCoordinator:
    """
    Main coordinator for disaggregated prefill/decode processing.
    
    Orchestrates the complete flow:
    1. Route request to optimal prefill node
    2. Process prefill phase
    3. Select optimal decode node
    4. Transfer cache if needed
    5. Continue generation on decode node
    6. Track performance metrics
    """
    
    def __init__(self, 
                 prefill_pool: WorkerClientPool,
                 decode_pool: WorkerClientPool,
                 node_id: str = "coordinator-1",
                 use_infiniband: bool = True):
        
        self.prefill_pool = prefill_pool
        self.decode_pool = decode_pool
        
        # Optimization components
        self.node_selector = NodeSelector()
        self.cache_manager = CacheTransferManager()
        
        # InfiniBand cache transfer manager
        if use_infiniband:
            try:
                # Try to import UCX to see if real InfiniBand is available
                import ucx
                self.ib_transfer = InfiniBandTransferManager(
                    node_id=node_id,
                    enable_compression=True
                )
            except ImportError:
                # Fall back to mock if UCX not available
                self.ib_transfer = MockInfiniBandTransferManager(
                    node_id=node_id,
                    enable_compression=True
                )
        else:
            self.ib_transfer = MockInfiniBandTransferManager(
                node_id=node_id,
                enable_compression=True
            )
        
        # Metrics tracking
        self.processed_requests = 0
        self.total_latency_improvement = 0.0
        self.disaggregation_metrics: List[DisaggregatedMetrics] = []
        
        logging.info(f"Initialized DisaggregatedRequestCoordinator with InfiniBand: {use_infiniband}")
    
    async def initialize(self) -> None:
        """Initialize the coordinator and InfiniBand components."""
        await self.ib_transfer.initialize()
        logging.info("DisaggregatedRequestCoordinator initialized")
    
    async def shutdown(self) -> None:
        """Clean shutdown of coordinator."""
        await self.ib_transfer.shutdown()
        logging.info("DisaggregatedRequestCoordinator shut down")
    
    async def process_request(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """
        Process request with disaggregated prefill/decode pipeline.
        
        Yields tokens as they're generated for streaming response.
        """
        total_start_time = time.time()
        
        try:
            # 1. Get available nodes
            prefill_nodes = await self.prefill_pool.get_healthy_workers()
            decode_nodes = await self.decode_pool.get_healthy_workers()
            
            if not prefill_nodes:
                raise Exception("No healthy prefill nodes available")
            if not decode_nodes:
                raise Exception("No healthy decode nodes available")
            
            # 2. Get node metrics for selection
            prefill_metrics = {}
            for node_id in prefill_nodes:
                info = await self.prefill_pool.get_worker_info(node_id)
                if info:
                    prefill_metrics[node_id] = {
                        'worker_type': 'prefill',
                        'active_requests': info.active_requests,
                        'avg_processing_time_ms': info.average_response_time,
                        'max_concurrent_sequences': 16  # Default for prefill
                    }
            
            decode_metrics = {}
            for node_id in decode_nodes:
                info = await self.decode_pool.get_worker_info(node_id)
                if info:
                    decode_metrics[node_id] = {
                        'worker_type': 'decode', 
                        'active_sequences': info.active_requests,
                        'tokens_per_second': 20.0,  # Default estimate
                        'max_concurrent_sequences': 64  # Default for decode
                    }
            
            # 3. Select optimal prefill node
            prefill_node = await self.node_selector.select_prefill_node(
                request, prefill_nodes, prefill_metrics
            )
            
            # 4. Process prefill phase
            prefill_start_time = time.time()
            prefill_result = await self.prefill_pool.send_prefill_request(
                prefill_node, request
            )
            prefill_time = (time.time() - prefill_start_time) * 1000
            
            # 5. Select optimal decode node
            decode_node = await self.node_selector.select_decode_node(
                request, prefill_result, decode_nodes, decode_metrics, prefill_node
            )
            
            # 6. Transfer cache if needed
            cache_transfer_time = 0.0
            if prefill_node != decode_node:
                transfer_start = time.time()
                
                logging.info(f"Transferring {prefill_result.cache_size_bytes} bytes cache "
                           f"from {prefill_node} to {decode_node} via InfiniBand")
                
                # Use InfiniBand RDMA for high-speed cache transfer
                transfer_success = await self.ib_transfer.transfer_cache(
                    target_node=decode_node,
                    cache_data=prefill_result.kv_cache_data,
                    request_id=request.request_id
                )
                
                if not transfer_success:
                    raise Exception(f"Cache transfer failed from {prefill_node} to {decode_node}")
                
                cache_transfer_time = (time.time() - transfer_start) * 1000
                
                # Record performance in both managers
                self.cache_manager.record_transfer_performance(
                    prefill_node, decode_node,
                    prefill_result.cache_size_bytes,
                    cache_transfer_time
                )
                
                logging.info(f"InfiniBand cache transfer completed in {cache_transfer_time:.1f}ms")
            else:
                logging.info("Cache transfer not needed - same node for prefill and decode")
            
            # 7. Continue generation on decode node
            decode_start_time = time.time()
            
            decode_request = DecodeRequest(
                inference_request=request,
                prefill_result=prefill_result
            )
            
            token_count = 0
            async for token in self.decode_pool.send_decode_stream(decode_node, decode_request):
                yield token
                token_count += 1
            
            decode_time = (time.time() - decode_start_time) * 1000
            total_time = (time.time() - total_start_time) * 1000
            
            # 8. Calculate disaggregation overhead
            disaggregation_overhead = cache_transfer_time + 10  # 10ms coordination overhead
            
            # 9. Record performance metrics
            metrics = DisaggregatedMetrics(
                request_id=request.request_id,
                prefill_node=prefill_node,
                decode_node=decode_node,
                prefill_time_ms=prefill_time,
                cache_transfer_time_ms=cache_transfer_time,
                decode_time_ms=decode_time,
                total_time_ms=total_time,
                cache_size_bytes=prefill_result.cache_size_bytes,
                tokens_generated=token_count,
                cache_hit=True,
                disaggregation_overhead_ms=disaggregation_overhead
            )
            
            self.disaggregation_metrics.append(metrics)
            self.processed_requests += 1
            
            # Estimate improvement vs monolithic
            estimated_monolithic_time = prefill_time + decode_time + 200  # Assume 200ms extra
            improvement = max(0, estimated_monolithic_time - total_time)
            self.total_latency_improvement += improvement
            
            logging.info(f"Disaggregated request {request.request_id} completed: "
                        f"prefill={prefill_time:.1f}ms, transfer={cache_transfer_time:.1f}ms, "
                        f"decode={decode_time:.1f}ms, total={total_time:.1f}ms, "
                        f"improvement={improvement:.1f}ms")
            
        except Exception as e:
            logging.error(f"Disaggregated processing failed for {request.request_id}: {e}")
            raise
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for disaggregated system."""
        if not self.disaggregation_metrics:
            return {'status': 'no_data'}
        
        recent_metrics = self.disaggregation_metrics[-100:]  # Last 100 requests
        
        avg_prefill_time = sum(m.prefill_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_transfer_time = sum(m.cache_transfer_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_decode_time = sum(m.decode_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_total_time = sum(m.total_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_tokens = sum(m.tokens_generated for m in recent_metrics) / len(recent_metrics)
        
        cache_hit_rate = sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics)
        
        tokens_per_second = (
            sum(m.tokens_generated for m in recent_metrics) / 
            (sum(m.total_time_ms for m in recent_metrics) / 1000)
        )
        
        # Calculate efficiency metrics
        disaggregation_overhead_ratio = (
            sum(m.disaggregation_overhead_ms for m in recent_metrics) / 
            sum(m.total_time_ms for m in recent_metrics)
        )
        
        avg_improvement = self.total_latency_improvement / self.processed_requests
        
        # Get InfiniBand transfer statistics
        ib_stats = self.ib_transfer.get_performance_stats()
        
        return {
            'processed_requests': self.processed_requests,
            'avg_prefill_time_ms': avg_prefill_time,
            'avg_cache_transfer_time_ms': avg_transfer_time,
            'avg_decode_time_ms': avg_decode_time,
            'avg_total_time_ms': avg_total_time,
            'avg_tokens_generated': avg_tokens,
            'tokens_per_second': tokens_per_second,
            'cache_hit_rate': cache_hit_rate,
            'disaggregation_overhead_ratio': disaggregation_overhead_ratio,
            'avg_latency_improvement_ms': avg_improvement,
            'infiniband_stats': ib_stats,
            'transfer_efficiency': {
                node_pair: stats[-5:] for node_pair, stats in 
                self.cache_manager.transfer_stats.items()
            }
        }