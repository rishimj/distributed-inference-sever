# Distributed LLM Inference Server

A high-performance distributed inference system for large language models with intelligent KV cache routing and sharing.

## 🚀 Features

- **Cache-Aware Routing**: Intelligent request routing based on KV cache availability
- **Disaggregated Architecture**: Separate prefill and decode processing for optimal resource utilization
- **Distributed Cache Management**: Share KV cache data across multiple worker nodes
- **Horizontal Scaling**: Scale prefill and decode workers independently
- **High Performance**: 3-10x latency reduction through cache reuse
- **Production Ready**: Comprehensive monitoring, health checks, and auto-recovery

## 📋 Performance Targets

- **<100ms TTFT** for cached prefixes
- **>1000 RPS** per node throughput
- **>80% cache hit rate** for common prefixes
- **<50% GPU memory** average utilization

## 🏗️ Architecture

```
┌─────────────┐
│   Gateway   │  ← Request entry point with intelligent routing
└──────┬──────┘
       │
   ┌───┴───┬───────┬───────┐
   │       │       │       │
┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐
│ W1  │ │ W2  │ │ W3  │ │ W4  │  ← vLLM worker nodes
└──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
   │       │       │       │
   └───┬───┴───┬───┴───┬───┘
       │       │       │
    ┌──▼───────▼───────▼──┐
    │   Cache Registry    │  ← Shared cache coordination (Redis)
    └─────────────────────┘
```

### Core Components

1. **Gateway Service** - Request routing and load balancing
2. **vLLM Workers** - LLM inference with KV cache
3. **Cache Registry** - Distributed cache metadata (Redis)
4. **Worker Manager** - Process lifecycle management
5. **Cache Sync Service** - Metadata synchronization
6. **Metrics Collector** - Prometheus-compatible monitoring

## 🛠️ Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 7.0+)
  - V100, T4, A10, A100, H100 recommended
  - Minimum 16GB GPU memory per worker
  - 40GB+ for larger models (70B parameters)
- **CPU**: 4+ cores per worker
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ for model weights

### Software

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Docker Runtime**: For GPU support
- **Python**: 3.11+ (if running without Docker)

## 📦 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd distributed-inference-server
```

### 2. Configure Environment

```bash
# Set your model (default: Llama-2-7b-hf)
export MODEL_NAME="meta-llama/Llama-2-7b-hf"

# Optional: Set Grafana password
export GRAFANA_PASSWORD="your-secure-password"
```

### 3. Start the System

```bash
./scripts/start_system.sh
```

This will start:
- Gateway service (port 8000)
- 2x vLLM workers (ports 8001, 8002)
- Redis cache registry (port 6379)
- Prometheus metrics (port 9090)
- Grafana dashboards (port 3000)

### 4. Test the System

```bash
./scripts/test_system.sh
```

### 5. Make an Inference Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time in a faraway land",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## 🔧 Configuration

### Worker Configuration

Edit `docker-compose.yml` to configure workers:

```yaml
environment:
  - MODEL_NAME=meta-llama/Llama-2-7b-hf
  - MAX_NUM_SEQS=256
  - MAX_MODEL_LEN=4096
  - GPU_MEMORY_UTILIZATION=0.8
```

### Scaling Workers

Add more workers in `docker-compose.yml`:

```yaml
worker-3:
  build:
    context: .
    dockerfile: Dockerfile.worker
  ports:
    - "8003:8001"
  environment:
    - MODEL_NAME=${MODEL_NAME}
    - CUDA_VISIBLE_DEVICES=2
    - WORKER_ID=worker-3
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['2']
            capabilities: [gpu]
```

### Cache Configuration

Configure cache behavior in `src/cache/registry.py`:

```python
cache_ttl_seconds = 300  # Cache entry lifetime
eviction_policy = "lru"  # Eviction strategy
```

## 📊 Monitoring

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default: admin/admin)

Pre-configured dashboards show:
- Request throughput and latency
- Cache hit rates and utilization
- Worker health and resource usage
- System-level performance

### Prometheus Metrics

Access raw metrics at:
- Gateway: `http://localhost:8000/metrics`
- Workers: `http://localhost:8001/metrics`, `http://localhost:8002/metrics`
- Prometheus: `http://localhost:9090`

### Key Metrics

- `distributed_inference_requests_total` - Total request count
- `distributed_inference_request_duration_milliseconds` - Request latency
- `distributed_inference_cache_hit_rate` - Cache effectiveness
- `distributed_inference_tokens_per_second` - Generation throughput
- `distributed_inference_worker_gpu_utilization` - GPU usage

## 🧪 Testing

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_routing_engine.py

# Run with coverage
pytest --cov=src tests/
```

### Integration Tests

```bash
# Run integration tests
pytest tests/test_end_to_end_integration.py

# Test with real workers (requires GPU)
pytest tests/test_disaggregated_system.py
```

### Load Testing

```bash
# Install load testing tool
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000
```

## 🚢 Deployment

### Docker Deployment

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f gateway worker-1

# Stop services
docker-compose down
```

### Kubernetes Deployment

See `docs/DEPLOYMENT.md` for Kubernetes manifests and deployment guide.

### Cloud Deployment

#### AWS

- Use EC2 instances with GPU (g4dn, p3, p4d)
- ECS/EKS for container orchestration
- ElastiCache for Redis

#### GCP

- Compute Engine with A100/V100 GPUs
- GKE for Kubernetes
- Memorystore for Redis

#### Azure

- NC/ND-series VMs with GPUs
- AKS for Kubernetes
- Azure Cache for Redis

## 🔍 Troubleshooting

### Workers not starting

Check GPU availability:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Cache misses high

- Increase cache TTL in configuration
- Add more workers to increase cache coverage
- Check prefix hashing strategy

### High latency

- Monitor GPU utilization (`nvidia-smi`)
- Check worker health endpoints
- Review Prometheus metrics for bottlenecks
- Increase worker count for load distribution

### Memory issues

- Reduce `GPU_MEMORY_UTILIZATION` (default 0.8)
- Decrease `MAX_NUM_SEQS` for fewer concurrent requests
- Use smaller models or quantization

## 📚 Documentation

- [Architecture Design](docs/ARCHITECTURE.md)
- [Disaggregated Flow](docs/DISAGGREGATED_FLOW.md)
- [Routing Algorithms](docs/ROUTING_ALGORITHMS.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Scaling Guide](docs/SCALING_GUIDE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [Redis](https://redis.io/) - Cache coordination
- [Prometheus](https://prometheus.io/) - Metrics collection
- [Grafana](https://grafana.com/) - Visualization

## 📞 Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: support@example.com

---

**Status**: Production-ready Beta  
**Version**: 0.1.0  
**Last Updated**: 2026-07-01
