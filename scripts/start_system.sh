#!/bin/bash
#
# Start the distributed inference system
#

set -e

echo "Starting Distributed Inference System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. System may not have GPU support."
    echo "Workers will require GPU to run vLLM."
fi

# Set default model if not specified
export MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-2-7b-hf}
echo "Using model: $MODEL_NAME"

# Create necessary directories
mkdir -p configs/grafana-dashboards
mkdir -p configs/grafana-datasources

# Start services
echo "Starting services with docker-compose..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to become healthy..."
sleep 10

# Check service health
echo "Checking service health..."

check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "✓ $service is healthy"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "  Waiting for $service... (attempt $attempt/$max_attempts)"
        sleep 2
    done
    
    echo "✗ $service failed to become healthy"
    return 1
}

check_service "Redis" "http://localhost:6379" || true
check_service "Gateway" "http://localhost:8000/health" || true
check_service "Worker 1" "http://localhost:8001/health" || true
check_service "Worker 2" "http://localhost:8002/health" || true
check_service "Prometheus" "http://localhost:9090/-/healthy" || true
check_service "Grafana" "http://localhost:3000/api/health" || true

echo ""
echo "==================================="
echo "System started successfully!"
echo "==================================="
echo ""
echo "Service URLs:"
echo "  Gateway:    http://localhost:8000"
echo "  Worker 1:   http://localhost:8001"
echo "  Worker 2:   http://localhost:8002"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f [service_name]"
echo ""
echo "To stop the system:"
echo "  ./scripts/stop_system.sh"
echo ""
