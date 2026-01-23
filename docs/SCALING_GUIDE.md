# Scaling Guide

## Overview
This document outlines strategies for scaling the distributed inference server across different dimensions and traffic patterns.

## Scaling Dimensions

### 1. Horizontal Scaling

#### Gateway Service Scaling
```yaml
# Target: Handle increased request volume
Strategy: Load balancer + multiple gateway instances
Bottlenecks: 
  - Cache registry query latency
  - Prefix computation overhead
Solutions:
  - Local cache registry cache (TTL: 30s)
  - Batch prefix computations
  - Async prefix hashing pipeline
```

#### Prefill Cluster Scaling
```yaml
# Target: Handle larger prompt processing load
Strategy: Queue-based auto-scaling
Metrics:
  - Queue depth > 100 requests: Scale up
  - Average queue time > 5s: Scale up  
  - GPU utilization < 30%: Scale down
Hardware: GPU-optimized instances (A100, H100)
```

#### Decode Cluster Scaling
```yaml
# Target: Handle more concurrent generations
Strategy: Session-based scaling
Metrics:
  - Active sessions > node_capacity * 0.8: Scale up
  - P95 latency > 200ms: Scale up
  - Active sessions < node_capacity * 0.3: Scale down
Hardware: Balanced CPU+GPU instances
```

### 2. Vertical Scaling

#### Memory Optimization
```python
# Cache tier sizing guidelines
L1_GPU_Memory = min(total_gpu_memory * 0.6, 40GB)  # Reserve for computation
L2_CPU_Memory = total_cpu_memory * 0.8  # OS overhead
L3_Network_Cache = cluster_memory * cache_replication_factor
```

#### Compute Optimization
```python
# Resource allocation per service type
Gateway_CPU = 4-8_cores  # Prefix computation intensive
Prefill_GPU = 1-4_high_end_gpus  # Parallel attention
Decode_GPU = 1_gpu + 8-16_cpu_cores  # Sequential generation
Cache_Manager_CPU = 2-4_cores + fast_storage
```

### 3. Geographic Scaling

#### Multi-Region Deployment
```
Region A: [Gateway] → [Local Cache] → [Prefill/Decode Cluster]
          ↓ (cache replication)
Region B: [Gateway] → [Local Cache] → [Prefill/Decode Cluster]
```

**Cache Replication Strategy**:
- **Hot Prefixes**: Replicated to all regions
- **Warm Prefixes**: Replicated to 2-3 nearest regions  
- **Cold Prefixes**: Single region with on-demand transfer

## Traffic Pattern Adaptations

### 1. Bursty Traffic
```yaml
Characteristics: Sudden 10x-100x traffic spikes
Challenges: 
  - Cache cold start problems
  - Resource allocation delays
Solutions:
  - Pre-warmed instance pools
  - Aggressive cache preloading
  - Circuit breakers for overload protection
```

### 2. High Concurrency
```yaml
Characteristics: Many simultaneous requests
Challenges:
  - Memory pressure from concurrent sessions
  - Network bandwidth saturation
Solutions:
  - Request batching and queueing
  - Streaming responses to reduce memory
  - Network compression and multiplexing
```

### 3. Long Context Requests
```yaml
Characteristics: Very long prompts (>32K tokens)
Challenges:
  - Large KV cache sizes
  - Network transfer bottlenecks
Solutions:
  - Dedicated long-context nodes
  - Cache compression (50-80% size reduction)
  - Progressive cache loading
```

## Performance Optimization by Scale

### Small Scale (1-10 nodes)
```yaml
Focus: Simplicity and cost efficiency
Architecture:
  - Single gateway instance
  - Combined prefill/decode nodes
  - Redis-only cache (no L4)
Optimizations:
  - In-memory cache only
  - Simple round-robin routing
  - Minimal monitoring overhead
```

### Medium Scale (10-100 nodes)
```yaml
Focus: Performance and reliability
Architecture:
  - Multiple gateway instances with LB
  - Separate prefill/decode clusters
  - Multi-tier cache with network storage
Optimizations:
  - Cache-aware routing
  - Auto-scaling policies
  - Comprehensive monitoring
```

### Large Scale (100+ nodes)
```yaml
Focus: Maximum efficiency and automation
Architecture:
  - Multi-region deployments
  - Specialized node types
  - Advanced cache strategies
Optimizations:
  - ML-based predictive scaling
  - Cache placement optimization
  - Advanced failure recovery
```

## Resource Planning Guidelines

### Capacity Planning Formulas

#### Gateway Capacity
```python
def gateway_capacity(target_rps: int, avg_prefix_length: int) -> Dict:
    cpu_cores = max(4, target_rps * avg_prefix_length / 10000)
    memory_gb = max(8, target_rps * 0.01)  # Request buffering
    return {"cpu_cores": cpu_cores, "memory_gb": memory_gb}
```

#### Prefill Capacity
```python
def prefill_capacity(target_tps: int, avg_prompt_tokens: int) -> Dict:
    # Tokens per second processing capacity
    gpu_memory_gb = max(40, avg_prompt_tokens * 0.002)  # 2MB per 1K tokens
    num_gpus = max(1, target_tps * avg_prompt_tokens / 50000)
    return {"num_gpus": num_gpus, "gpu_memory_gb": gpu_memory_gb}
```

#### Cache Sizing
```python
def cache_sizing(daily_requests: int, cache_hit_target: float) -> Dict:
    # Estimate unique prefixes and cache requirements
    unique_prefixes = daily_requests * (1 - cache_hit_target)
    avg_cache_size_mb = 50  # Average KV cache size
    
    l1_size_gb = min(100, unique_prefixes * 0.1 * avg_cache_size_mb / 1024)
    l2_size_gb = min(1000, unique_prefixes * 0.3 * avg_cache_size_mb / 1024)
    l3_size_gb = unique_prefixes * avg_cache_size_mb / 1024
    
    return {
        "l1_gpu_cache_gb": l1_size_gb,
        "l2_cpu_cache_gb": l2_size_gb, 
        "l3_network_cache_gb": l3_size_gb
    }
```

## Cost Optimization Strategies

### 1. Instance Mix Optimization
```yaml
High Performance Tier:
  - 20% A100/H100 instances for prefill
  - Low latency, high cost
Medium Performance Tier:
  - 60% L4/RTX instances for decode
  - Balanced cost/performance
Cost Optimized Tier:
  - 20% CPU instances for cache/gateway
  - Maximum cost efficiency
```

### 2. Dynamic Resource Allocation
```yaml
Peak Hours (9 AM - 6 PM):
  - Full resource allocation
  - All cache tiers active
Off-Peak Hours (6 PM - 9 AM):
  - Scale down decode clusters
  - Reduce cache replication
  - Consolidate to fewer nodes
```

### 3. Spot Instance Strategy
```yaml
Workload Types:
  - Gateway: On-demand (critical path)
  - Prefill: Mixed spot/on-demand (batch-like)
  - Decode: On-demand (user-facing)
  - Cache: Spot instances with persistence
```

## Monitoring at Scale

### Key Performance Indicators
```yaml
Service Level Objectives:
  - P99 latency < 500ms (cached requests)
  - P99 latency < 2000ms (cache miss)
  - Availability > 99.9%
  - Cache hit rate > 80%

Resource Utilization Targets:
  - GPU utilization: 70-90%
  - Memory utilization: 60-80%
  - Network utilization: 50-70%
  - Cache hit rate: 80-95%
```

### Scaling Alerts
```yaml
Scale Up Triggers:
  - P95 latency > SLO for 5 minutes
  - Resource utilization > 85% for 10 minutes
  - Queue depth > threshold for 3 minutes
  - Cache miss rate > 40% for 15 minutes

Scale Down Triggers:
  - Resource utilization < 30% for 30 minutes
  - No scaling activity for 1 hour
  - Off-peak hours with low traffic
```

### Predictive Scaling
```yaml
ML Models for Prediction:
  - Time series forecasting for traffic patterns
  - Request pattern analysis for cache preloading
  - User behavior modeling for capacity planning

Data Sources:
  - Historical traffic patterns
  - Request characteristics (length, complexity)
  - System performance metrics
  - External factors (time, events, seasonality)
```