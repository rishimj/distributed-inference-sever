#!/bin/bash
#
# Stop the distributed inference system
#

set -e

echo "Stopping Distributed Inference System..."

# Stop all services
docker-compose down

echo "System stopped successfully!"
echo ""
echo "To remove all data (including volumes):"
echo "  docker-compose down -v"
echo ""
