# Distributed LLM Inference System: Detailed Technical Design

## Executive Summary

This document describes a production-grade distributed inference system for Large Language Models (LLMs) built on vLLM, incorporating KV-cache-aware routing, tiered storage with LMCache, and high-performance event-driven architecture. The system is designed for deployment on Georgia Tech's PACE ICE infrastructure with InfiniBand interconnects.

**Key Performance Targets (Conservative Estimates):**
- 20-40% reduction in Time-To-First-Token (TTFT) through KV cache reuse
- 2-3x improvement in cache hit scenarios vs cache miss
- Support for 100+ concurrent requests across multiple nodes
- Sub-10ms routing decision latency

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Component Deep Dive](#2-component-deep-dive)
3. [KV-Cache-Aware Routing](#3-kv-cache-aware-routing)
4. [Event-Driven Architecture with ZeroMQ](#4-event-driven-architecture-with-zeromq)
5. [Tiered Storage with LMCache](#5-tiered-storage-with-lmcache)
6. [Communication Protocols](#6-communication-protocols)
7. [Performance Analysis](#7-performance-analysis)
8. [PACE ICE Deployment](#8-pace-ice-deployment)
9. [Failure Handling](#9-failure-handling)
10. [Monitoring and Observability](#10-monitoring-and-observability)
11. [Design Tradeoffs](#11-design-tradeoffs)
12. [Future Considerations](#12-future-considerations)

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Requests                                │
│                          (HTTP/HTTPS REST API)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Load Balancer (L4/L7)                            │
│                         (HAProxy / NGINX / K8s Ingress)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│     Gateway Node 1    │ │     Gateway Node 2    │ │     Gateway Node N    │
│  ┌─────────────────┐  │ │  ┌─────────────────┐  │ │  ┌─────────────────┐  │
│  │  KV-Cache Index │  │ │  │  KV-Cache Index │  │ │  │  KV-Cache Index │  │
│  │  (In-Memory)    │  │ │  │  (In-Memory)    │  │ │  │  (In-Memory)    │  │
│  └────────┬────────┘  │ │  └────────┬────────┘  │ │  └────────┬────────┘  │
│           │           │ │           │           │ │           │           │
│  ┌────────▼────────┐  │ │  ┌────────▼────────┐  │ │  ┌────────▼────────┐  │
│  │  Routing Engine │  │ │  │  Routing Engine │  │ │  │  Routing Engine │  │
│  └─────────────────┘  │ │  └─────────────────┘  │ │  └─────────────────┘  │
└───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
            │                         │                         │
            │    ZeroMQ Subscribe     │                         │
            │◄────────────────────────┼─────────────────────────┤
            │                         │                         │
            │    HTTP POST (Inference Requests)                 │
            ├─────────────────────────┼─────────────────────────┤
            │                         │                         │
            ▼                         ▼                         ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│   vLLM Worker Node 1  │ │   vLLM Worker Node 2  │ │   vLLM Worker Node M  │
│  ┌─────────────────┐  │ │  ┌─────────────────┐  │ │  ┌─────────────────┐  │
│  │ vLLM Engine     │  │ │  │ vLLM Engine     │  │ │  │ vLLM Engine     │  │
│  │ + LMCache       │  │ │  │ + LMCache       │  │ │  │ + LMCache       │  │
│  │ + KVEvents      │  │ │  │ + KVEvents      │  │ │  │ + KVEvents      │  │
│  └────────┬────────┘  │ │  └────────┬────────┘  │ │  └────────┬────────┘  │
│           │           │ │           │           │ │           │           │
│  ┌────────▼────────┐  │ │  ┌────────▼────────┐  │ │  ┌────────▼────────┐  │
│  │ Tiered Storage  │  │ │  │ Tiered Storage  │  │ │  │ Tiered Storage  │  │
│  │ GPU→CPU→Disk    │  │ │  │ GPU→CPU→Disk    │  │ │  │ GPU→CPU→Disk    │  │
│  └─────────────────┘  │ │  └─────────────────┘  │ │  └─────────────────┘  │
└───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │         Redis Cluster           │
                    │  (Shared KV Cache + Metadata)   │
                    │                                 │
                    │  • LMCache shared storage       │
                    │  • Node health/metrics          │
                    │  • Configuration                │
                    └─────────────────────────────────┘
```

### 1.2 Component Summary

| Component | Technology | Purpose | Count |
|-----------|------------|---------|-------|
| Gateway | Python/aiohttp | Request routing, KV-cache indexing | 2-3 nodes |
| vLLM Workers | vLLM + LMCache | LLM inference, KV cache management | 8 nodes (1 GPU each) |
| Event Bus | ZeroMQ | KVEvent distribution | Embedded in workers |
| Shared Cache | Redis | Cross-node KV cache sharing | 1-3 nodes (cluster) |
| Storage | Local NVMe + Network | Tiered KV cache storage | Per worker |

---

## 2. Component Deep Dive

### 2.1 Gateway Node

The gateway is the entry point for all inference requests. It maintains a global view of KV cache distribution and makes routing decisions.

#### 2.1.1 Responsibilities

1. **Request Reception**: Accept HTTP requests from clients
2. **Prefix Hashing**: Compute block hashes from prompt tokens
3. **KV-Cache Index Maintenance**: Track block locations via KVEvents
4. **Routing Decision**: Score workers and select optimal target
5. **Request Forwarding**: Send inference request to selected worker
6. **Response Handling**: Stream or buffer response back to client

#### 2.1.2 Internal Architecture

```python
class GatewayNode:
    """
    Gateway node architecture.
    """
    
    def __init__(self):
        # KV-Cache Index: block_hash -> [(pod_id, tier, timestamp)]
        # Memory: ~100MB - 1GB for millions of blocks
        self.kv_cache_index: Dict[str, List[BlockLocation]] = {}
        
        # Worker connection pool (HTTP)
        # Maintains persistent connections to workers
        self.worker_pool: WorkerClientPool
        
        # ZMQ subscriber for KVEvents
        # Subscribes to all worker nodes
        self.event_subscriber: ZMQEventSubscriber
        
        # Tokenizer for block hash computation
        # Must match vLLM's tokenizer
        self.tokenizer: PreTrainedTokenizer
        
        # Metrics collector
        self.metrics: PrometheusMetrics

    async def handle_request(self, request: InferenceRequest) -> InferenceResponse:
        """Main request handling flow."""
        
        # 1. Tokenize and compute block hashes
        tokens = self.tokenizer.encode(request.prompt)
        block_hashes = self.compute_block_hashes(tokens, chunk_size=256)
        
        # 2. Query KV-cache index for each block
        block_locations = self.lookup_blocks(block_hashes)
        
        # 3. Score all healthy workers
        worker_scores = self.score_workers(block_locations)
        
        # 4. Select best worker
        target_worker = self.select_worker(worker_scores)
        
        # 5. Forward request
        response = await self.worker_pool.send_request(target_worker, request)
        
        return response
```

#### 2.1.3 Memory Requirements

| Data Structure | Size per Entry | Typical Count | Total Memory |
|----------------|----------------|---------------|--------------|
| KV-Cache Index | 80 bytes | 10 million blocks | 800 MB |
| Worker Connections | 50 KB | 10 workers | 500 KB |
| Active Requests | 1 KB | 1000 concurrent | 1 MB |
| **Total** | | | **~1 GB** |

### 2.2 vLLM Worker Node

Each worker runs a vLLM instance with LMCache integration and KVEvents publishing.

#### 2.2.1 Configuration

```python
# vLLM Worker Configuration
vllm_config = {
    # Model configuration
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "tensor_parallel_size": 1,  # 8B model fits on single GPU
    "max_model_len": 8192,
    "max_num_seqs": 64,
    "gpu_memory_utilization": 0.85,
    
    # KVEvents configuration (for ZeroMQ publishing)
    "kv_events_config": {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": "tcp://*:5557",
        "topic": "kv-events",
        "hwm": 100000,  # High water mark
        "max_queue_size": 100000
    },
    
    # LMCache configuration (for tiered storage)
    "kv_transfer_config": {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both"  # Both producer and consumer
    }
}

# LMCache Configuration (lmcache-config.yaml)
lmcache_config = {
    "chunk_size": 256,  # Tokens per block
    "local_cpu": True,  # Enable CPU RAM tier
    "local_disk": "file:///tmp/lmcache_kv",  # Enable disk tier
    "max_local_disk_size": 500,  # 500 GB
    "remote_url": "redis://redis-cluster:6379",  # Shared Redis
    "remote_serde": "cachegen"  # Compression for network transfer
}
```

#### 2.2.2 Tiered Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        vLLM Worker Tiered Storage                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Tier 1: GPU Memory (vLLM PagedAttention)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Capacity: 64 GB (after ~16GB model weights on 80GB A100)           │   │
│  │  Latency: 0.01 ms                                                   │   │
│  │  Managed by: vLLM's BlockSpaceManager                               │   │
│  │  Eviction: LRU → Tier 2                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │ Evict (async)                               │
│                              ▼                                              │
│  Tier 2: CPU RAM (LMCache)                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Capacity: 200 GB (configurable)                                    │   │
│  │  Latency: 1-5 ms (PCIe transfer to GPU)                            │   │
│  │  Managed by: LMCache CPU backend                                    │   │
│  │  Eviction: LRU → Tier 3 or Tier 4                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │ Evict (async)                               │
│                              ▼                                              │
│  Tier 3: Local Disk (LMCache)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Capacity: 500 GB - 2 TB (NVMe SSD)                                 │   │
│  │  Latency: 10-50 ms (read + decompress + transfer)                   │   │
│  │  Managed by: LMCache disk backend with index                        │   │
│  │  Eviction: LRU → Tier 4 or Delete                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │ Evict (async)                               │
│                              ▼                                              │
│  Tier 4: Shared Redis (LMCache)                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Capacity: 100-500 GB (cluster-wide)                                │   │
│  │  Latency: 5-20 ms (network + deserialize)                           │   │
│  │  Managed by: LMCache Redis connector                                │   │
│  │  Benefit: Shared across all workers                                 │   │
│  │  Eviction: TTL or LRU → Delete (recompute if needed)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. KV-Cache-Aware Routing

### 3.1 Block Hash Computation

KV cache is organized into blocks (chunks of tokens). Each block is identified by a hash of its token IDs.

```python
def compute_block_hashes(tokens: List[int], chunk_size: int = 256) -> List[str]:
    """
    Compute block hashes for KV cache lookup.
    
    Args:
        tokens: Token IDs from tokenizer
        chunk_size: Tokens per block (must match LMCache config)
    
    Returns:
        List of block hashes (SHA-256 hex strings)
    """
    block_hashes = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        
        # Hash includes:
        # - Token IDs (primary identifier)
        # - Position in sequence (for correctness)
        hash_input = f"{chunk}:{i}".encode('utf-8')
        block_hash = hashlib.sha256(hash_input).hexdigest()[:32]
        
        block_hashes.append(block_hash)
    
    return block_hashes

# Example:
# Prompt: "What is the capital of France?" (8 tokens)
# With chunk_size=256, this is 1 block
# block_hash = sha256([token_ids]).hexdigest()[:32]
```

### 3.2 Scoring Algorithm

```python
def score_workers(
    block_hashes: List[str],
    candidate_workers: List[str],
    kv_index: Dict[str, List[BlockLocation]],
    worker_metrics: Dict[str, WorkerMetrics]
) -> Dict[str, float]:
    """
    Score workers for routing decision.
    
    Scoring dimensions:
    1. KV-cache hit score (40% weight) - Most important
    2. Load score (30% weight) - Current utilization
    3. Latency score (20% weight) - Historical performance
    4. Capacity score (10% weight) - Available headroom
    """
    
    scores = {}
    
    for worker_id in candidate_workers:
        metrics = worker_metrics.get(worker_id)
        
        # 1. KV-cache hit score (0.0 - 1.0)
        cache_hits = 0
        cache_score_sum = 0.0
        
        for block_hash in block_hashes:
            if block_hash in kv_index:
                for location in kv_index[block_hash]:
                    if location.pod_id == worker_id:
                        # Score based on tier (faster = higher score)
                        tier_scores = {
                            "GPU": 1.0,   # Instant
                            "CPU": 0.7,   # ~5ms load time
                            "disk": 0.4,  # ~50ms load time
                            "redis": 0.3  # ~20ms network + deserialize
                        }
                        cache_score_sum += tier_scores.get(location.tier, 0.2)
                        cache_hits += 1
                        break
        
        cache_score = cache_score_sum / len(block_hashes) if block_hashes else 0.0
        
        # 2. Load score (0.0 - 1.0, higher = less loaded)
        utilization = metrics.active_sequences / metrics.max_sequences
        load_score = max(0.0, 1.0 - utilization)
        
        # 3. Latency score (0.0 - 1.0, based on recent p50 latency)
        target_latency = 100.0  # ms
        latency_score = min(1.0, target_latency / max(metrics.p50_latency_ms, 1.0))
        
        # 4. Capacity score (0.0 - 1.0)
        memory_available = 1.0 - (metrics.gpu_memory_used / metrics.gpu_memory_total)
        capacity_score = memory_available
        
        # Weighted combination
        total_score = (
            cache_score * 0.40 +
            load_score * 0.30 +
            latency_score * 0.20 +
            capacity_score * 0.10
        )
        
        scores[worker_id] = {
            'total': total_score,
            'cache': cache_score,
            'load': load_score,
            'latency': latency_score,
            'capacity': capacity_score,
            'cache_hits': cache_hits,
            'total_blocks': len(block_hashes)
        }
    
    return scores
```

### 3.3 Routing Decision Example

```
Request: "Translate to Spanish: Hello, how are you today?"
Tokens: 12 tokens → 1 block (chunk_size=256)
Block hash: 0xABCD1234...

KV-Cache Index:
  0xABCD1234 → [
    {pod: worker-1, tier: GPU},   # Best!
    {pod: worker-3, tier: disk}   # Slower
  ]

Worker Scores:
┌──────────┬───────┬───────┬─────────┬──────────┬─────────┐
│ Worker   │ Cache │ Load  │ Latency │ Capacity │ Total   │
├──────────┼───────┼───────┼─────────┼──────────┼─────────┤
│ worker-1 │ 1.00  │ 0.60  │ 0.90    │ 0.70     │ 0.83    │ ← Selected
│ worker-2 │ 0.00  │ 0.90  │ 0.95    │ 0.85     │ 0.56    │
│ worker-3 │ 0.40  │ 0.70  │ 0.85    │ 0.75     │ 0.60    │
└──────────┴───────┴───────┴─────────┴──────────┴─────────┘

Decision: Route to worker-1 (highest total score due to GPU cache hit)
```

---

## 4. Event-Driven Architecture with ZeroMQ

### 4.1 Why ZeroMQ?

| Requirement | ZeroMQ | Redis Pub/Sub | Kafka | WebSocket |
|-------------|--------|---------------|-------|-----------|
| Latency | ~10μs | ~1ms | ~5-10ms | ~1ms |
| Throughput | 1M+ msg/s | 100K msg/s | 1M+ msg/s | 100K msg/s |
| Broker required | No | Yes | Yes | No |
| Built into vLLM | Yes | No | No | No |
| Persistence | No | No | Yes | No |
| Complexity | Low | Low | High | Medium |

**Decision**: ZeroMQ is the optimal choice because:
1. **Native vLLM support**: KVEventsConfig uses ZMQ by default
2. **Lowest latency**: Critical for near-real-time index updates
3. **No broker**: Reduces infrastructure complexity
4. **High throughput**: Handles thousands of events per second

**Tradeoff**: No persistence means events are lost if gateway restarts. Mitigated by periodic full sync (see Section 9).

### 4.2 Event Types

vLLM emits three types of KVEvents:

```python
@dataclass
class BlockStoredEvent:
    """Emitted when a KV cache block is stored."""
    event_type: str = "BlockStored"
    block_hash: str          # SHA-256 hash of token chunk
    parent_block_hash: str   # For prefix tree (optional)
    token_ids: List[int]     # Actual token IDs
    block_size: int          # Tokens in block
    lora_id: Optional[str]   # LoRA adapter ID (if applicable)
    storage_medium: str      # "GPU", "CPU", "disk", "redis"

@dataclass
class BlockRemovedEvent:
    """Emitted when a KV cache block is evicted."""
    event_type: str = "BlockRemoved"
    block_hash: str
    storage_medium: str      # Tier it was removed from

@dataclass
class AllBlocksClearedEvent:
    """Emitted when all blocks are cleared (e.g., OOM recovery)."""
    event_type: str = "AllBlocksCleared"
```

### 4.3 Event Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KVEvent Flow                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  vLLM Worker (Publisher)                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  1. Inference request processed                                     │   │
│  │  2. KV cache block created in GPU                                   │   │
│  │  3. vLLM emits BlockStored event                                    │   │
│  │                                                                     │   │
│  │     zmq_socket.send_json({                                         │   │
│  │         "event_type": "BlockStored",                               │   │
│  │         "block_hash": "0xABCD1234",                                │   │
│  │         "storage_medium": "GPU",                                    │   │
│  │         "pod_id": "worker-1"                                       │   │
│  │     })                                                              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              │ ZMQ PUB (tcp://*:5557)                      │
│                              ▼                                              │
│  Gateway (Subscriber)                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  event = zmq_socket.recv_json()                                    │   │
│  │                                                                     │   │
│  │  if event["event_type"] == "BlockStored":                          │   │
│  │      kv_index[event["block_hash"]].append({                        │   │
│  │          "pod_id": event["pod_id"],                                │   │
│  │          "tier": event["storage_medium"]                           │   │
│  │      })                                                             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Event Subscriber Implementation

```python
class ZMQEventSubscriber:
    """
    Subscribes to KVEvents from all vLLM workers.
    Updates the KV-cache index in real-time.
    """
    
    def __init__(self, worker_endpoints: List[str]):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.worker_endpoints = worker_endpoints
        self.running = False
        
    async def start(self, kv_index: Dict[str, List[BlockLocation]]):
        """Start subscribing to KVEvents."""
        
        # Connect to all workers
        for endpoint in self.worker_endpoints:
            self.socket.connect(f"tcp://{endpoint}:5557")
            logger.info(f"Connected to KVEvent publisher at {endpoint}")
        
        # Subscribe to kv-events topic
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "kv-events")
        
        self.running = True
        
        while self.running:
            try:
                # Non-blocking receive with timeout
                if self.socket.poll(timeout=100):  # 100ms timeout
                    message = self.socket.recv_json(zmq.NOBLOCK)
                    await self._handle_event(message, kv_index)
                    
            except zmq.Again:
                continue  # No message available
            except Exception as e:
                logger.error(f"Error processing KVEvent: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_event(
        self, 
        event: dict, 
        kv_index: Dict[str, List[BlockLocation]]
    ):
        """Process a single KVEvent."""
        
        event_type = event.get("event_type")
        block_hash = event.get("block_hash")
        pod_id = event.get("pod_id")
        tier = event.get("storage_medium")
        
        if event_type == "BlockStored":
            if block_hash not in kv_index:
                kv_index[block_hash] = []
            
            # Update or add location
            updated = False
            for loc in kv_index[block_hash]:
                if loc.pod_id == pod_id:
                    loc.tier = tier
                    loc.timestamp = time.time()
                    updated = True
                    break
            
            if not updated:
                kv_index[block_hash].append(BlockLocation(
                    pod_id=pod_id,
                    tier=tier,
                    timestamp=time.time()
                ))
                
        elif event_type == "BlockRemoved":
            if block_hash in kv_index:
                kv_index[block_hash] = [
                    loc for loc in kv_index[block_hash]
                    if not (loc.pod_id == pod_id and loc.tier == tier)
                ]
                
                # Clean up empty entries
                if not kv_index[block_hash]:
                    del kv_index[block_hash]
                    
        elif event_type == "AllBlocksCleared":
            # Remove all entries for this pod
            for block_hash in list(kv_index.keys()):
                kv_index[block_hash] = [
                    loc for loc in kv_index[block_hash]
                    if loc.pod_id != pod_id
                ]
```

### 4.5 Event Volume Estimation

```
Assumptions:
- 100 concurrent requests
- Average 4000 tokens per prompt
- 256 tokens per block
- 100 output tokens average

Blocks created per request: 4000/256 = 16 blocks
Events per request (store): 16 events
Events per request (eviction, over time): ~8 events

At 100 req/s peak:
- Store events: 100 × 16 = 1,600 events/sec
- Eviction events: 100 × 8 = 800 events/sec
- Total: ~2,400 events/sec

ZMQ capacity: 1,000,000+ events/sec
Headroom: 400x
```

---

## 5. Tiered Storage with LMCache

### 5.1 LMCache Integration

LMCache integrates with vLLM via the KVConnector API:

```bash
# Start vLLM with LMCache
LMCACHE_CONFIG_FILE=/etc/lmcache/config.yaml \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'
```

### 5.2 LMCache Configuration

```yaml
# /etc/lmcache/config.yaml

# Block size (must be consistent across all nodes)
chunk_size: 256

# Tier 2: CPU RAM offloading
local_cpu: true

# Tier 3: Local disk offloading
local_disk: "file:///tmp/lmcache_kv"
max_local_disk_size: 500  # GB

# Tier 4: Shared Redis (cross-node)
remote_url: "redis://redis-cluster:6379"
remote_serde: "cachegen"  # Compressed serialization

# Advanced options
prefetch_enabled: true      # Preload from disk to CPU
async_put: true             # Non-blocking writes
compression_level: 1        # LZ4 fast compression
```

### 5.3 Storage Capacity Planning

```
Model: Llama-3.1-8B
KV cache per token: ~0.5 KB (with GQA, 8 KV heads, 128 dim)
Block size: 256 tokens
KV cache per block: 128 KB

Tier capacities and block counts:
┌────────────┬──────────┬─────────────┬──────────────┐
│ Tier       │ Capacity │ Blocks      │ Prefixes*    │
├────────────┼──────────┼─────────────┼──────────────┤
│ GPU        │ 64 GB    │ 512,000     │ 32,000       │
│ CPU        │ 200 GB   │ 1,600,000   │ 100,000      │
│ Disk       │ 500 GB   │ 4,000,000   │ 250,000      │
│ Redis      │ 200 GB   │ 1,600,000   │ 100,000      │
├────────────┼──────────┼─────────────┼──────────────┤
│ Total      │ 964 GB   │ 7,712,000   │ 482,000      │
└────────────┴──────────┴─────────────┴──────────────┘

* Assuming average prefix of 4000 tokens (16 blocks)
```

### 5.4 Cache Lookup Flow

```python
async def lookup_and_load_cache(
    block_hash: str,
    worker: VLLMWorker
) -> Optional[KVCacheBlock]:
    """
    LMCache lookup flow (executed on worker).
    No disk scanning - all hash-based O(1) lookups.
    """
    
    # Tier 1: GPU memory (managed by vLLM)
    if block_hash in worker.gpu_cache_index:
        return worker.gpu_cache[block_hash]  # Instant
    
    # Tier 2: CPU RAM (LMCache)
    if block_hash in worker.cpu_cache_index:
        block = worker.cpu_cache[block_hash]
        # Async copy to GPU
        await worker.copy_to_gpu(block)
        return block  # ~1-5ms
    
    # Tier 3: Local disk (LMCache)
    if block_hash in worker.disk_index:
        file_path = worker.disk_index[block_hash]
        block = await worker.load_from_disk(file_path)
        # Move through tiers: disk → CPU → GPU
        await worker.copy_to_cpu(block)
        await worker.copy_to_gpu(block)
        return block  # ~10-50ms
    
    # Tier 4: Shared Redis (LMCache)
    block = await worker.redis_client.get(f"kv:{block_hash}")
    if block:
        block = deserialize_block(block)
        # Move through tiers: Redis → CPU → GPU
        await worker.copy_to_cpu(block)
        await worker.copy_to_gpu(block)
        return block  # ~5-20ms
    
    # Cache miss - must recompute
    return None  # ~50-500ms to recompute
```

---

## 6. Communication Protocols

### 6.1 Protocol Summary

| Communication | Protocol | Port | Purpose |
|--------------|----------|------|---------|
| Client → Gateway | HTTP/HTTPS | 443 | Inference requests |
| Gateway → Worker | HTTP | 8000 | Inference forwarding |
| Worker → Gateway | ZMQ PUB/SUB | 5557 | KVEvents |
| Worker ↔ Redis | Redis Protocol | 6379 | Shared KV cache |
| Gateway ↔ Redis | Redis Protocol | 6379 | Metadata/health |

### 6.2 HTTP Request Flow (Gateway → Worker)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HTTP Request Flow                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Non-Streaming Request:                                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Gateway                              Worker                                │
│     │                                    │                                  │
│     │  POST /v1/completions              │                                  │
│     │  {                                 │                                  │
│     │    "model": "llama-8b",            │                                  │
│     │    "prompt": "...",                │                                  │
│     │    "max_tokens": 100               │                                  │
│     │  }                                 │                                  │
│     │ ─────────────────────────────────► │                                  │
│     │                                    │  Process (100ms - 10s)           │
│     │                                    │                                  │
│     │  HTTP 200 OK                       │                                  │
│     │  {                                 │                                  │
│     │    "choices": [...],               │                                  │
│     │    "usage": {...}                  │                                  │
│     │  }                                 │                                  │
│     │ ◄───────────────────────────────── │                                  │
│     │                                    │                                  │
│                                                                             │
│  Streaming Request (Server-Sent Events):                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Gateway                              Worker                                │
│     │                                    │                                  │
│     │  POST /v1/completions              │                                  │
│     │  {"stream": true, ...}             │                                  │
│     │ ─────────────────────────────────► │                                  │
│     │                                    │                                  │
│     │  HTTP 200 OK                       │                                  │
│     │  Content-Type: text/event-stream   │                                  │
│     │ ◄───────────────────────────────── │                                  │
│     │                                    │                                  │
│     │  data: {"delta": {"content": "The"}}                                 │
│     │ ◄───────────────────────────────── │                                  │
│     │                                    │                                  │
│     │  data: {"delta": {"content": " capital"}}                            │
│     │ ◄───────────────────────────────── │                                  │
│     │                                    │                                  │
│     │  data: [DONE]                      │                                  │
│     │ ◄───────────────────────────────── │                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Why HTTP for Inference (Not gRPC)?

| Factor | HTTP/REST | gRPC |
|--------|-----------|------|
| vLLM native support | Yes (OpenAI-compatible) | Requires wrapper |
| Streaming | SSE (simple) | Bidirectional (complex) |
| Debugging | Easy (curl, browser) | Requires tooling |
| Load balancer support | Universal | Limited |
| Latency overhead | ~1ms | ~0.5ms |
| Implementation effort | Low | Medium |

**Decision**: HTTP/REST with SSE streaming because:
1. vLLM provides OpenAI-compatible HTTP API out of the box
2. SSE is sufficient for our streaming use case (server → client only)
3. Easier debugging and monitoring
4. 0.5ms latency difference is negligible vs 100ms+ inference time

---

## 7. Performance Analysis

### 7.1 Latency Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Request Latency Breakdown                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Component                          Cache Hit    Cache Miss    Improvement  │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  1. Gateway receive                 1 ms         1 ms          -            │
│  2. Tokenization                    5 ms         5 ms          -            │
│  3. Block hash computation          2 ms         2 ms          -            │
│  4. KV-index lookup                 1 ms         1 ms          -            │
│  5. Worker scoring                  2 ms         2 ms          -            │
│  6. HTTP request to worker          5 ms         5 ms          -            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Subtotal (routing overhead)        16 ms        16 ms         -            │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  7. KV cache load (if needed)                                               │
│     - GPU hit                       0 ms         -             -            │
│     - CPU hit                       5 ms         -             -            │
│     - Disk hit                      50 ms        -             -            │
│     - Redis hit                     20 ms        -             -            │
│     - Full recompute                -            150 ms        -            │
│                                                                             │
│  8. Prefill (remaining tokens)      50 ms        200 ms        75% ↓        │
│  9. Decode (100 tokens @ 20ms/tok)  2000 ms      2000 ms       -            │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Total (GPU cache hit)              2066 ms      -             -            │
│  Total (CPU cache hit)              2071 ms      -             -            │
│  Total (Disk cache hit)             2116 ms      -             -            │
│  Total (Redis cache hit)            2086 ms      -             -            │
│  Total (Cache miss)                 -            2216 ms       -            │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  TTFT Improvement (cache hit):      66-150 ms faster (30-70% reduction)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Conservative Performance Estimates

| Metric | Without KV-Aware Routing | With KV-Aware Routing | Improvement |
|--------|--------------------------|----------------------|-------------|
| **TTFT (p50)** | 200 ms | 120-150 ms | 25-40% ↓ |
| **TTFT (p99)** | 500 ms | 300-400 ms | 20-40% ↓ |
| **Cache hit rate** | ~5% (random routing) | 40-60% (optimized) | 8-12x ↑ |
| **Throughput** | 180 req/s | 250-300 req/s | 40-65% ↑ |
| **GPU utilization** | 70% | 85-90% | 15-20% ↑ |

**Assumptions:**
- 8B model on 1x A100-80GB per worker
- 4000 token average prompt
- 100 token average output
- 50% prefix overlap in workload
- 8 worker nodes

### 7.3 Throughput Analysis

```
Single worker throughput (8B model on A100):
- Prefill: ~15,000 tokens/sec (batched, 8B is ~2x faster than 70B)
- Decode: ~150 tokens/sec per sequence (memory-bound)
- Max concurrent sequences: 128 (more GPU memory available)

With 8 workers:
- Total prefill capacity: 120,000 tokens/sec
- Total decode capacity: 1024 sequences × 150 tok/s = 153,600 tokens/sec

Request throughput (4000 input + 100 output tokens):
- Bottleneck: Prefill (for short outputs)
- Theoretical max: 120,000 / 4000 = 30 req/s per worker = 240 req/s total
- Practical (with overhead): ~180 req/s sustained

With KV cache hits (50% cache hit rate):
- Effective prefill tokens: 4000 × 0.5 = 2000 tokens
- Improved throughput: 120,000 / 2000 = 60 req/s per worker = 480 req/s theoretical
- Practical: ~300 req/s sustained
```

### 7.4 Memory Requirements

| Component | Per Node | 8 Nodes Total |
|-----------|----------|---------------|
| **Gateway** | | |
| KV-cache index | 1 GB | 3 GB (3 gateways) |
| Connection pool | 100 MB | 300 MB |
| **vLLM Worker** | | |
| Model weights | 16 GB | 128 GB |
| KV cache (GPU) | 64 GB | 512 GB |
| KV cache (CPU) | 200 GB | 1,600 GB |
| LMCache index | 100 MB | 800 MB |
| **Redis Cluster** | | |
| Shared KV cache | 200 GB | 200 GB (shared) |
| Metadata | 10 GB | 10 GB |

---

## 8. PACE ICE Deployment

### 8.1 Resource Allocation

```bash
#!/bin/bash
#SBATCH --job-name=distributed-inference
#SBATCH --partition=gpu-a100
#SBATCH --nodes=11
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

# Node allocation:
# - Nodes 0-1: Gateway (CPU-only would be ideal, but using GPU nodes)
# - Node 2: Redis cluster
# - Nodes 3-10: vLLM workers (8 nodes × 1 GPU = 8 GPUs)

# Get node hostnames
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))

GATEWAY_NODES=("${NODES[0]}" "${NODES[1]}")
REDIS_NODE="${NODES[2]}"
WORKER_NODES=("${NODES[@]:3}")

echo "Gateway nodes: ${GATEWAY_NODES[@]}"
echo "Redis node: ${REDIS_NODE}"
echo "Worker nodes: ${WORKER_NODES[@]}"
```

### 8.2 Network Configuration

```yaml
# PACE ICE Network Topology
#
# InfiniBand network provides:
# - ~200 Gbps bandwidth between nodes
# - ~1-2 μs latency
# - RDMA capability
#
# Network usage:
# - KVEvents (ZMQ): ~1 Mbps per worker
# - Inference requests (HTTP): ~10 Mbps per worker
# - Redis cache: ~100 Mbps per worker (burst)
# - Total per worker: ~111 Mbps
# - Headroom: 1800x

network_config:
  # Use InfiniBand for low-latency communication
  use_infiniband: true
  
  # ZMQ binds to IB interface
  zmq_interface: "ib0"
  
  # Redis uses IB for cross-node access
  redis_bind: "0.0.0.0"
```

### 8.3 Deployment Scripts

```bash
# deploy_redis.sh - Run on Redis node
#!/bin/bash

# Start Redis with InfiniBand
redis-server \
  --port 6379 \
  --bind 0.0.0.0 \
  --maxmemory 200gb \
  --maxmemory-policy allkeys-lru \
  --save "" \
  --appendonly no
```

```bash
# deploy_worker.sh - Run on each worker node
#!/bin/bash

NODE_ID=$1
REDIS_HOST=$2

# Load modules
module load anaconda3
module load cuda/12.1

# Activate environment
conda activate vllm

# Set LMCache config
export LMCACHE_CONFIG_FILE=/path/to/lmcache-config.yaml

# Set environment for InfiniBand
export UCX_NET_DEVICES=mlx5_0:1
export NCCL_IB_DISABLE=0

# Start vLLM worker
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --kv-events-config "{\"enable_kv_cache_events\": true, \"publisher\": \"zmq\", \"endpoint\": \"tcp://*:5557\", \"topic\": \"kv-events\"}"
```

```bash
# deploy_gateway.sh - Run on gateway nodes
#!/bin/bash

WORKER_HOSTS=$1  # Comma-separated list
REDIS_HOST=$2

python -m src.gateway.server \
  --host 0.0.0.0 \
  --port 8080 \
  --workers "$WORKER_HOSTS" \
  --redis-url "redis://${REDIS_HOST}:6379" \
  --zmq-port 5557
```

### 8.4 LMCache Configuration for PACE ICE

```yaml
# lmcache-config-pace.yaml

# Block size
chunk_size: 256

# Tier 2: CPU RAM (use node's 512GB RAM)
local_cpu: true

# Tier 3: Local disk (use /tmp - NVMe SSD)
# WARNING: /tmp is wiped when job ends
local_disk: "file:///tmp/lmcache_kv"
max_local_disk_size: 500

# Tier 4: Shared Redis
# Using InfiniBand for low latency
remote_url: "redis://${REDIS_HOST}:6379"
remote_serde: "cachegen"

# Performance tuning
prefetch_enabled: true
async_put: true
compression_level: 1
```

---

## 9. Failure Handling

### 9.1 Gateway Failure

```
Scenario: Gateway node crashes

Impact:
- Requests to that gateway fail
- KV-cache index lost (in-memory only)

Mitigation:
1. Load balancer detects failure (health check)
2. Routes traffic to remaining gateways
3. Crashed gateway restarts
4. Rebuilds index via:
   a. Periodic sync from workers (every 30s)
   b. Subscribes to ZMQ events again

Recovery time: 30-60 seconds for full index rebuild
```

### 9.2 Worker Failure

```
Scenario: vLLM worker crashes

Impact:
- In-flight requests to that worker fail
- KV cache on that worker lost

Mitigation:
1. Gateway detects failure (health check or timeout)
2. Marks worker as unhealthy
3. Removes worker's blocks from index
4. Reroutes pending requests to other workers
5. Worker restarts and re-registers

Recovery time: 5-10 seconds for failover
```

### 9.3 Redis Failure

```
Scenario: Redis cluster node fails

Impact:
- Shared KV cache partially unavailable
- Cross-node cache sharing degraded

Mitigation:
1. Redis Sentinel detects failure
2. Promotes replica to master
3. Workers reconnect automatically
4. Some cache entries lost (acceptable)

Recovery time: 5-10 seconds with Sentinel
```

### 9.4 Periodic Sync for Robustness

```python
class PeriodicSyncWorker:
    """
    Periodically syncs full cache state from workers.
    Catches missed ZMQ events and handles gateway restarts.
    """
    
    async def run(self, interval_seconds: int = 30):
        while True:
            await asyncio.sleep(interval_seconds)
            
            for worker in self.workers:
                try:
                    # Fetch full cache state from worker
                    state = await self.fetch_worker_state(worker)
                    
                    # Reconcile with current index
                    self.reconcile_index(worker.id, state)
                    
                except Exception as e:
                    logger.warning(f"Failed to sync with {worker.id}: {e}")
    
    def reconcile_index(self, worker_id: str, state: WorkerCacheState):
        """
        Reconcile index with worker's actual state.
        Handles:
        - Missed BlockStored events
        - Missed BlockRemoved events
        - Index corruption
        """
        
        # Get current index entries for this worker
        current_blocks = {
            hash for hash, locs in self.kv_index.items()
            if any(loc.pod_id == worker_id for loc in locs)
        }
        
        # Get actual blocks from worker
        actual_blocks = set(state.block_hashes)
        
        # Add missing entries
        for block_hash in actual_blocks - current_blocks:
            self._add_block(block_hash, worker_id, state.tiers[block_hash])
        
        # Remove stale entries
        for block_hash in current_blocks - actual_blocks:
            self._remove_block(block_hash, worker_id)
```

---

## 10. Monitoring and Observability

### 10.1 Key Metrics

```python
# Prometheus metrics for monitoring

# Gateway metrics
gateway_requests_total = Counter(
    'gateway_requests_total',
    'Total inference requests',
    ['status', 'worker']
)

gateway_routing_latency = Histogram(
    'gateway_routing_latency_seconds',
    'Time to make routing decision',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)

kv_cache_hit_rate = Gauge(
    'kv_cache_hit_rate',
    'KV cache hit rate by tier',
    ['tier']  # GPU, CPU, disk, redis, miss
)

kv_index_size = Gauge(
    'kv_index_size_blocks',
    'Number of blocks in KV cache index'
)

# Worker metrics (from vLLM)
vllm_active_sequences = Gauge(
    'vllm_active_sequences',
    'Number of active sequences',
    ['worker']
)

vllm_gpu_cache_usage = Gauge(
    'vllm_gpu_cache_usage_percent',
    'GPU KV cache usage',
    ['worker']
)

# LMCache metrics
lmcache_tier_size = Gauge(
    'lmcache_tier_size_bytes',
    'Size of each cache tier',
    ['worker', 'tier']
)
```

### 10.2 Alerting Rules

```yaml
# Prometheus alerting rules

groups:
  - name: distributed-inference
    rules:
      - alert: HighCacheMissRate
        expr: kv_cache_hit_rate{tier="miss"} > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High cache miss rate ({{ $value | humanizePercentage }})"
          
      - alert: WorkerUnhealthy
        expr: up{job="vllm-worker"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "vLLM worker {{ $labels.instance }} is down"
          
      - alert: HighRoutingLatency
        expr: histogram_quantile(0.99, gateway_routing_latency_seconds) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Routing latency p99 > 100ms"
```

### 10.3 Dashboard Panels

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Distributed Inference Dashboard                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────┐                  │
│  │   Request Rate          │  │   TTFT Latency          │                  │
│  │   [Line Chart]          │  │   [Line Chart p50/p99]  │                  │
│  │   Current: 95 req/s     │  │   p50: 120ms p99: 350ms │                  │
│  └─────────────────────────┘  └─────────────────────────┘                  │
│                                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────┐                  │
│  │   Cache Hit Rate        │  │   Worker Utilization    │                  │
│  │   [Stacked Bar]         │  │   [Heatmap]             │                  │
│  │   GPU: 25% CPU: 20%     │  │   worker-1: 85%         │                  │
│  │   Disk: 10% Miss: 45%   │  │   worker-2: 78%         │                  │
│  └─────────────────────────┘  └─────────────────────────┘                  │
│                                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────┐                  │
│  │   KV Index Size         │  │   Redis Memory          │                  │
│  │   [Gauge]               │  │   [Gauge]               │                  │
│  │   2.5M blocks           │  │   150GB / 200GB         │                  │
│  └─────────────────────────┘  └─────────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Design Tradeoffs

### 11.1 ZeroMQ vs Alternatives

| Decision | ZeroMQ | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Event bus | ✅ Selected | Redis Pub/Sub | Lower latency (10μs vs 1ms), native vLLM support |
| | | Kafka | No persistence needed, simpler setup |
| Tradeoff | No persistence | Durability | Periodic sync mitigates event loss |

### 11.2 HTTP vs gRPC for Inference

| Decision | HTTP/SSE | Alternative | Rationale |
|----------|----------|-------------|-----------|
| Inference API | ✅ Selected | gRPC | Native vLLM support, easier debugging |
| Streaming | SSE | gRPC streaming | Simpler, sufficient for server→client |
| Tradeoff | ~0.5ms extra latency | Type safety | Negligible vs inference time |

### 11.3 Centralized vs Distributed Index

| Decision | Centralized (per gateway) | Alternative | Rationale |
|----------|---------------------------|-------------|-----------|
| KV index | ✅ Selected | Distributed hash table | Simpler, sufficient for scale |
| Tradeoff | Memory per gateway | Complexity | ~1GB memory is acceptable |

### 11.4 Tiered Storage Order

| Decision | GPU→CPU→Disk→Redis | Alternative | Rationale |
|----------|---------------------|-------------|-----------|
| Tier order | ✅ Selected | GPU→Redis→Disk | Disk faster than network |
| | | GPU→CPU only | Miss cross-node sharing |
| Tradeoff | Complexity | Simplicity | Worth it for hit rate |

### 11.5 Block Size Selection

| Decision | 256 tokens | Alternative | Rationale |
|----------|------------|-------------|-----------|
| Chunk size | ✅ Selected | 128, 512, 1024 | Balance granularity vs overhead |
| Tradeoff | | | Smaller = better partial hits, more overhead |

---

## 12. Future Considerations

### 12.1 Disaggregated Prefill/Decode

Current architecture uses unified workers. Future optimization:

```
Prefill Nodes (A100)     Decode Nodes (RTX)
┌──────────────────┐     ┌──────────────────┐
│ High FLOPS       │     │ High memory BW   │
│ Process prompts  │ ──► │ Generate tokens  │
│ Generate KV cache│     │ Use KV cache     │
└──────────────────┘     └──────────────────┘

Expected improvement: 40-60% better resource utilization
```

### 12.2 Speculative Decoding

Combine with smaller draft model for faster decoding:

```
Draft Model (1B)         Target Model (8B)
┌──────────────────┐     ┌──────────────────┐
│ Generate 4 tokens│ ──► │ Verify tokens    │
│ Fast (~2ms)      │     │ Accept 3-4 tokens│
└──────────────────┘     └──────────────────┘

Expected improvement: 1.5-2x decode throughput
```

### 12.3 Prefix Tree Indexing

Current: Flat hash index
Future: Tree structure for partial prefix matching

```
Before: "The capital of France" → hash → lookup
After:  "The" → "capital" → "of" → "France" → lookup at any level

Expected improvement: Better partial cache hit detection
```

---

## Appendix A: API Reference

### A.1 Gateway API

```yaml
openapi: 3.0.0
info:
  title: Distributed Inference Gateway
  version: 1.0.0

paths:
  /v1/completions:
    post:
      summary: Generate text completion
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [model, prompt]
              properties:
                model:
                  type: string
                prompt:
                  type: string
                max_tokens:
                  type: integer
                  default: 100
                temperature:
                  type: number
                  default: 0.7
                stream:
                  type: boolean
                  default: false
      responses:
        '200':
          description: Completion response
          
  /health:
    get:
      summary: Health check
      responses:
        '200':
          description: Healthy

  /metrics:
    get:
      summary: Prometheus metrics
      responses:
        '200':
          description: Metrics in Prometheus format
```

### A.2 Worker Health Endpoint

```yaml
paths:
  /health:
    get:
      summary: Worker health and cache state
      responses:
        '200':
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  gpu_memory_used:
                    type: integer
                  gpu_memory_total:
                    type: integer
                  active_sequences:
                    type: integer
                  kv_cache_blocks:
                    type: integer
                  cache_tiers:
                    type: object
                    properties:
                      gpu:
                        type: integer
                      cpu:
                        type: integer
                      disk:
                        type: integer
```

---

## Appendix B: Configuration Reference

### B.1 Complete Configuration Example

```yaml
# config/production.yaml

gateway:
  host: "0.0.0.0"
  port: 8080
  workers: 4  # Async workers
  
  kv_index:
    max_entries: 10000000
    cleanup_interval_seconds: 300
    
  routing:
    cache_weight: 0.40
    load_weight: 0.30
    latency_weight: 0.20
    capacity_weight: 0.10
    
  connection_pool:
    max_connections_per_worker: 10
    keepalive_timeout_seconds: 30
    request_timeout_seconds: 300

workers:
  - host: "worker-1"
    port: 8000
    zmq_port: 5557
  - host: "worker-2"
    port: 8000
    zmq_port: 5557
  # ... additional workers

redis:
  url: "redis://redis-cluster:6379"
  max_connections: 100
  
monitoring:
  prometheus_port: 9090
  health_check_interval_seconds: 10
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **KV Cache** | Key-Value cache storing attention layer computations |
| **Block** | Fixed-size chunk of KV cache (e.g., 256 tokens) |
| **TTFT** | Time To First Token - latency until first output token |
| **ITL** | Inter-Token Latency - time between output tokens |
| **Prefill** | Processing input prompt (compute-bound) |
| **Decode** | Generating output tokens (memory-bound) |
| **PagedAttention** | vLLM's memory management for KV cache |
| **LMCache** | Library for tiered KV cache storage |
| **KVEvents** | vLLM's event system for cache state changes |
| **ZMQ** | ZeroMQ - high-performance messaging library |
| **SSE** | Server-Sent Events - HTTP streaming protocol |

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Author: Distributed Inference Team*
