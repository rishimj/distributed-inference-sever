#!/bin/bash
#
# Start the demo version (no GPU required)
#

set -e

echo "Starting Distributed Inference Server Demo..."
echo ""
echo "⚠️  Demo Mode: Using mock workers (no GPU required)"
echo ""

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

# Start services in demo mode
echo "Starting services..."
docker-compose -f docker-compose.demo.yml up -d

# Wait for services
echo "Waiting for services to start..."
sleep 10

echo ""
echo "==================================="
echo "Demo started successfully!"
echo "==================================="
echo ""
echo "🌐 Web Demo:    http://localhost:8080"
echo "🔌 API:         http://localhost:8000"
echo "📊 Grafana:     http://localhost:3000 (admin/demo123)"
echo "📈 Prometheus:  http://localhost:9090"
echo ""
echo "Try it:"
echo "  curl -X POST http://localhost:8000/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"Hello world\", \"max_tokens\": 50}'"
echo ""
echo "To stop: docker-compose -f docker-compose.demo.yml down"
echo ""
