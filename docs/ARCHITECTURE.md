# Architecture Documentation

## System Overview

This distributed inference server implements a non-Kubernetes approach to high-performance LLM serving with three core innovations:

1. **Prefix-Aware KV Cache Routing**: Routes requests to nodes with matching cached computations
2. **Multi-Tier Shared Cache**: Distributes and shares KV caches across GPU/CPU/Disk/Network layers
3. **Disaggregated Prefill/Decode**: Separates prompt processing from token generation for optimal resource utilization

## Architectural Decisions & Tradeoffs

### Design Choice: Multi-Service Architecture vs Monolith

**Decision**: Separate services for Gateway, Prefill, Decode, and Cache Management

**Rationale**:
- **Pros**: Independent scaling, specialized resource allocation, fault isolation
- **Cons**: Network overhead, complexity in coordination
- **Alternative Considered**: Monolithic design with internal routing
- **Why Rejected**: Would limit scaling flexibility and resource optimization

### Design Choice: Async Python vs Go/Rust

**Decision**: Python with asyncio for implementation

**Rationale**:
- **Pros**: Rich ML ecosystem, rapid development, excellent library support
- **Cons**: GIL limitations, potential performance overhead
- **Alternative Considered**: Go for performance, Rust for safety
- **Why Rejected**: Development velocity and ML integration outweigh performance concerns

### Design Choice: Redis + Custom TCP vs Pure Distributed Cache

**Decision**: Hybrid approach with Redis for metadata and custom TCP for bulk data

**Rationale**:
- **Pros**: Redis provides consistency and discovery, TCP optimizes large transfers
- **Cons**: Dual-protocol complexity
- **Alternative Considered**: Pure Redis, Pure custom protocol
- **Why Rejected**: Redis alone can't handle large KV efficiently, custom alone lacks metadata features

## Component Deep Dive

### Gateway Service
```
Request → Prefix Analyzer → Cache Registry Query → Routing Decision → Target Selection
```

**Key Responsibilities**:
- Extract and hash request prefixes for cache lookup
- Query distributed cache registry for hit probability
- Score available nodes based on cache availability and load
- Route requests with fallback logic for cache misses

**Scaling Strategy**:
- **Horizontal**: Multiple gateway instances behind load balancer
- **Vertical**: CPU-optimized nodes for prefix analysis
- **Bottlenecks**: Cache registry queries (mitigated by local caching)

### Cache System Architecture

```
L1: Node GPU Memory (1-10ms access)
    ↓
L2: Node CPU Memory (10-50ms access) 
    ↓
L3: Shared Network Cache (50-200ms access)
    ↓
L4: Cold Storage (200ms+ access)
```

**Cache Consistency Model**: Eventually consistent with invalidation

**Eviction Strategy**: 
- **L1/L2**: LRU with recency and frequency weighting
- **L3**: LRU with replication factor consideration
- **L4**: TTL-based with access pattern analysis

**Scaling Strategy**:
- **Capacity**: Add more cache nodes to L3 tier
- **Performance**: Increase replication of hot prefixes
- **Bottlenecks**: Network bandwidth (use compression and batching)

### Prefill/Decode Separation

**Prefill Cluster Characteristics**:
- **Hardware**: GPU-heavy nodes for parallel attention computation
- **Memory**: High GPU memory for large batch processing
- **Network**: High bandwidth for KV cache publishing
- **Scaling**: Scale based on prompt processing queue depth

**Decode Cluster Characteristics**:  
- **Hardware**: Balanced CPU+GPU for sequential generation
- **Memory**: Optimized for cache retrieval and small computation
- **Network**: Low latency for cache fetching
- **Scaling**: Scale based on active generation sessions

## Performance Optimization Strategies

### 1. Cache Locality Optimization
- **Prefix Clustering**: Group similar prefixes on same nodes
- **Geographic Affinity**: Route requests to nearby cache locations  
- **Temporal Locality**: Keep recently used caches in faster tiers

### 2. Network Optimization
- **Compression**: Use LZ4 for KV cache compression
- **Batching**: Combine multiple small cache transfers
- **Pipelining**: Overlap cache transfer with computation

### 3. Memory Management
- **Pool Allocation**: Pre-allocate memory pools for cache objects
- **Copy Avoidance**: Zero-copy transfers where possible
- **Garbage Collection**: Explicit memory management for large objects

## Failure Handling & Resilience

### Cache Miss Scenarios
- **Partial Miss**: Use available partial cache + recompute delta
- **Complete Miss**: Fallback to full computation with cache population
- **Cache Corruption**: Validate checksums, fallback to recomputation

### Node Failure Handling
- **Gateway Failure**: Client-side load balancing to other gateways
- **Cache Node Failure**: Automatic failover to replica nodes
- **Compute Node Failure**: Request re-routing to healthy nodes

### Network Partition Handling
- **Split Brain**: Use consensus algorithm for cache coordination
- **Partial Connectivity**: Local cache fallback with eventual consistency
- **Complete Isolation**: Full local operation with cache warming

## Monitoring & Observability

### Key Metrics
- **Performance**: Latency percentiles, throughput, cache hit rates
- **Resource**: GPU/CPU/Memory utilization, network bandwidth
- **Reliability**: Error rates, availability, failover frequency

### Alerting Strategy
- **Critical**: Service unavailability, high error rates (>5%)
- **Warning**: Performance degradation, cache hit rate drops
- **Info**: Capacity planning, usage pattern changes

### Debugging Support
- **Distributed Tracing**: Request flow across all services
- **Structured Logging**: Consistent log format with correlation IDs
- **Cache Analytics**: Cache hit/miss patterns and hotspot analysis

## Security Considerations

### Network Security
- **TLS**: All inter-service communication encrypted
- **Authentication**: Service-to-service mTLS certificates
- **Network Isolation**: Services in separate network segments

### Data Protection  
- **Encryption**: KV caches encrypted at rest and in transit
- **Access Control**: Role-based access to cache and compute resources
- **Audit Logging**: All cache access and modifications logged

### Input Validation
- **Request Sanitization**: Validate all incoming requests
- **Cache Validation**: Checksums for cache integrity
- **Resource Limits**: Prevent DoS through resource exhaustion