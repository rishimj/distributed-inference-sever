"""
Core data structures and models for the distributed inference server.

This module defines the fundamental data types used across all services.
Design decisions:
- Using Pydantic for validation and serialization performance
- Immutable data structures where possible for thread safety
- Rich type hints for better IDE support and runtime validation
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class ServiceType(str, Enum):
    """Service types in the distributed system."""
    GATEWAY = "gateway"
    PREFILL = "prefill" 
    DECODE = "decode"
    CACHE = "cache"


class CacheLevel(str, Enum):
    """Cache storage tiers with different performance characteristics."""
    L1_GPU = "l1_gpu"          # Fastest: GPU memory, ~1-10ms access
    L2_CPU = "l2_cpu"          # Fast: CPU memory, ~10-50ms access  
    L3_NETWORK = "l3_network"  # Medium: Network storage, ~50-200ms access
    L4_COLD = "l4_cold"        # Slow: Cold storage, ~200ms+ access


class RequestStatus(str, Enum):
    """Request processing status."""
    PENDING = "pending"
    ROUTING = "routing"
    PREFILLING = "prefilling" 
    DECODING = "decoding"
    COMPLETED = "completed"
    FAILED = "failed"


# Core Request Models

class InferenceRequest(BaseModel):
    """
    Incoming inference request with all necessary information.
    
    Design choice: Include both raw and processed fields for efficiency.
    Tradeoff: Slight memory overhead vs avoiding recomputation.
    """
    
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    
    # Processing metadata
    timestamp: float = Field(default_factory=time.time)
    priority: int = Field(default=5, ge=1, le=10)  # 1=highest, 10=lowest
    
    # Cache-related fields
    prefix_hash: Optional[str] = None
    estimated_tokens: Optional[int] = None
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt_length(cls, v: str) -> str:
        if len(v) > 100000:  # ~100K chars limit
            raise ValueError('Prompt too long')
        return v
    
    def compute_prefix_hash(self, prefix_length: int = 512) -> str:
        """
        Compute hash of prompt prefix for cache lookup.
        
        Args:
            prefix_length: Number of characters to include in prefix
            
        Returns:
            SHA-256 hash of the prefix
            
        Design choice: Use SHA-256 for collision resistance.
        Tradeoff: Slower than MD5 but much safer for production use.
        """
        prefix = self.prompt[:prefix_length]
        return hashlib.sha256(prefix.encode('utf-8')).hexdigest()


class InferenceResponse(BaseModel):
    """Response from inference processing."""
    
    request_id: str
    generated_text: str
    tokens_generated: int
    processing_time_ms: float
    cache_hit: bool
    
    # Performance metrics
    ttft_ms: Optional[float] = None  # Time to first token
    tps: Optional[float] = None      # Tokens per second
    
    # Node information
    processed_by: Optional[str] = None
    cache_node: Optional[str] = None


# Cache System Models

class CacheKey(BaseModel):
    """
    Unique identifier for cached KV data.
    
    Design choice: Include model info in key for multi-model support.
    Tradeoff: Longer keys vs better cache isolation.
    """
    
    prefix_hash: str = Field(min_length=64, max_length=64)  # SHA-256 hash
    model_name: str = Field(default="default")
    sequence_length: int = Field(ge=1)
    
    def to_string(self) -> str:
        """Convert to string representation for storage keys."""
        return f"{self.model_name}:{self.prefix_hash}:{self.sequence_length}"
    
    @classmethod
    def from_string(cls, key_str: str) -> 'CacheKey':
        """Parse from string representation."""
        parts = key_str.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid cache key format: {key_str}")
        
        return cls(
            model_name=parts[0],
            prefix_hash=parts[1], 
            sequence_length=int(parts[2])
        )


class CacheEntry(BaseModel):
    """
    Cached KV data with metadata.
    
    Design choice: Include access patterns for intelligent eviction.
    Tradeoff: Extra metadata overhead vs better cache management.
    """
    
    cache_key: CacheKey
    kv_data: bytes  # Serialized KV cache tensors
    
    # Storage metadata
    cache_level: CacheLevel
    node_id: str
    size_bytes: int
    
    # Access patterns for eviction decisions
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = Field(default=0)
    hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Replication info
    replica_nodes: List[str] = Field(default_factory=list)
    is_primary: bool = Field(default=True)
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at
    
    def recency_score(self) -> float:
        """
        Calculate recency score for eviction (0.0 to 1.0).
        Higher score = more recently accessed.
        """
        age = time.time() - self.last_accessed
        # Exponential decay with half-life of 1 hour
        return 0.5 ** (age / 3600)


# Node and Service Models

class NodeInfo(BaseModel):
    """Information about a node in the cluster."""
    
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    service_type: ServiceType
    hostname: str
    port: int
    
    # Capabilities
    gpu_memory_gb: float = Field(default=0.0, ge=0.0)
    cpu_cores: int = Field(default=1, ge=1)
    memory_gb: float = Field(default=8.0, ge=1.0)
    
    # Current status
    is_healthy: bool = Field(default=True)
    current_load: float = Field(default=0.0, ge=0.0, le=1.0)
    last_heartbeat: float = Field(default_factory=time.time)
    
    # Performance characteristics
    avg_latency_ms: float = Field(default=100.0, ge=0.0)
    throughput_rps: float = Field(default=10.0, ge=0.0)
    
    def is_overloaded(self, threshold: float = 0.9) -> bool:
        """Check if node is overloaded."""
        return self.current_load > threshold
    
    def heartbeat_age_seconds(self) -> float:
        """Get age of last heartbeat in seconds.""" 
        return time.time() - self.last_heartbeat


class RouteDecision(BaseModel):
    """
    Routing decision with scoring information.
    
    Design choice: Include detailed scoring for debugging and optimization.
    Tradeoff: Extra computation vs better observability.
    """
    
    target_node: str
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Scoring components
    cache_hit_score: float = Field(default=0.0, ge=0.0, le=1.0)
    load_score: float = Field(default=0.0, ge=0.0, le=1.0) 
    latency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    capacity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Predictions
    estimated_latency_ms: float = Field(default=100.0, ge=0.0)
    cache_hit_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Fallback options
    fallback_nodes: List[str] = Field(default_factory=list)
    
    def total_score(self) -> float:
        """Calculate weighted total routing score."""
        return (
            self.cache_hit_score * 0.4 +      # Cache hits are most important
            self.latency_score * 0.3 +        # Then latency
            self.load_score * 0.2 +           # Then current load
            self.capacity_score * 0.1         # Finally capacity
        )


# Cache Registry Models

class CacheRegistry(BaseModel):
    """
    Registry of all cache entries in the system.
    
    Design choice: Centralized registry for global cache awareness.
    Tradeoff: Single point of contention vs cache coordination efficiency.
    """
    
    entries: Dict[str, CacheEntry] = Field(default_factory=dict)
    nodes: Dict[str, NodeInfo] = Field(default_factory=dict)
    
    # Registry metadata
    last_updated: float = Field(default_factory=time.time)
    total_cache_size_bytes: int = Field(default=0)
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def add_entry(self, entry: CacheEntry) -> None:
        """Add cache entry to registry."""
        key = entry.cache_key.to_string()
        self.entries[key] = entry
        self.total_cache_size_bytes += entry.size_bytes
        self.last_updated = time.time()
    
    def remove_entry(self, cache_key: CacheKey) -> Optional[CacheEntry]:
        """Remove cache entry from registry.""" 
        key = cache_key.to_string()
        entry = self.entries.pop(key, None)
        if entry:
            self.total_cache_size_bytes -= entry.size_bytes
            self.last_updated = time.time()
        return entry
    
    def find_entries_by_prefix(self, prefix_hash: str) -> List[CacheEntry]:
        """Find all cache entries matching a prefix hash."""
        return [
            entry for entry in self.entries.values()
            if entry.cache_key.prefix_hash == prefix_hash
        ]
    
    def get_node_cache_size(self, node_id: str) -> int:
        """Get total cache size for a specific node."""
        return sum(
            entry.size_bytes for entry in self.entries.values()
            if entry.node_id == node_id
        )
    
    def get_cache_utilization(self) -> Dict[CacheLevel, float]:
        """Get utilization by cache level."""
        utilization = {}
        for level in CacheLevel:
            entries = [e for e in self.entries.values() if e.cache_level == level]
            total_size = sum(e.size_bytes for e in entries)
            utilization[level] = total_size
        return utilization


# Error Models

class InferenceError(BaseModel):
    """Error information for failed inference requests."""
    
    request_id: str
    error_type: str
    error_message: str
    timestamp: float = Field(default_factory=time.time)
    node_id: Optional[str] = None
    retry_count: int = Field(default=0)
    is_retryable: bool = Field(default=True)


# Configuration Models

@dataclass(frozen=True)
class CacheConfig:
    """
    Cache configuration with performance tuning parameters.
    
    Design choice: Frozen dataclass for immutable configuration.
    Tradeoff: Cannot modify after creation vs thread safety.
    """
    
    # Cache size limits (bytes)
    l1_gpu_limit: int = 40 * 1024**3      # 40GB GPU memory
    l2_cpu_limit: int = 100 * 1024**3     # 100GB CPU memory  
    l3_network_limit: int = 1000 * 1024**3  # 1TB network storage
    
    # Eviction thresholds
    l1_eviction_threshold: float = 0.9
    l2_eviction_threshold: float = 0.8
    l3_eviction_threshold: float = 0.95
    
    # Replication settings
    min_replicas: int = 1
    max_replicas: int = 3
    replication_threshold: float = 0.7  # Hit rate to trigger replication
    
    # Performance tuning
    compression_enabled: bool = True
    compression_level: int = 1  # LZ4 compression level
    transfer_chunk_size: int = 64 * 1024  # 64KB chunks for transfer
    
    # TTL settings (seconds)
    l1_ttl: int = 3600      # 1 hour
    l2_ttl: int = 86400     # 24 hours
    l3_ttl: int = 604800    # 1 week
    l4_ttl: int = 2592000   # 30 days


@dataclass(frozen=True)  
class RoutingConfig:
    """
    Routing configuration parameters.
    
    Design choice: Separate config for easy A/B testing of routing strategies.
    """
    
    # Scoring weights
    cache_hit_weight: float = 0.4
    latency_weight: float = 0.3
    load_weight: float = 0.2
    capacity_weight: float = 0.1
    
    # Thresholds
    overload_threshold: float = 0.9
    healthy_threshold: float = 0.7
    min_confidence: float = 0.5
    
    # Timeouts
    routing_timeout_ms: int = 50
    health_check_interval_ms: int = 5000
    node_timeout_ms: int = 30000
    
    # Fallback behavior
    enable_fallback: bool = True
    max_fallback_attempts: int = 3
    fallback_penalty: float = 0.1