#!/bin/bash
#
# Test the distributed inference system
#

set -e

GATEWAY_URL=${GATEWAY_URL:-http://localhost:8000}

echo "Testing Distributed Inference System..."
echo "Gateway URL: $GATEWAY_URL"
echo ""

# Test 1: Health check
echo "Test 1: Gateway health check..."
if curl -sf "$GATEWAY_URL/health" | jq . > /dev/null 2>&1; then
    echo "✓ Gateway is healthy"
else
    echo "✗ Gateway health check failed"
    exit 1
fi
echo ""

# Test 2: Worker status
echo "Test 2: Checking worker status..."
if curl -sf "$GATEWAY_URL/workers" | jq . > /dev/null 2>&1; then
    echo "✓ Workers registered"
    curl -s "$GATEWAY_URL/workers" | jq '.workers'
else
    echo "✗ Failed to get worker status"
    exit 1
fi
echo ""

# Test 3: Simple inference request
echo "Test 3: Running inference test..."
cat > /tmp/test_request.json << EOF
{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
}
EOF

if curl -sf -X POST "$GATEWAY_URL/generate" \
    -H "Content-Type: application/json" \
    -d @/tmp/test_request.json | jq . > /dev/null 2>&1; then
    echo "✓ Inference request succeeded"
    curl -s -X POST "$GATEWAY_URL/generate" \
        -H "Content-Type: application/json" \
        -d @/tmp/test_request.json | jq '.generated_text'
else
    echo "✗ Inference request failed"
    exit 1
fi
echo ""

# Test 4: Cache statistics
echo "Test 4: Checking cache statistics..."
if curl -sf "$GATEWAY_URL/cache/stats" | jq . > /dev/null 2>&1; then
    echo "✓ Cache statistics available"
    curl -s "$GATEWAY_URL/cache/stats" | jq .
else
    echo "✗ Failed to get cache statistics"
fi
echo ""

# Test 5: Metrics endpoint
echo "Test 5: Checking metrics endpoint..."
if curl -sf "$GATEWAY_URL/metrics" > /dev/null 2>&1; then
    echo "✓ Metrics endpoint accessible"
    echo "  Sample metrics:"
    curl -s "$GATEWAY_URL/metrics" | grep "distributed_inference" | head -5
else
    echo "✗ Metrics endpoint failed"
fi
echo ""

echo "==================================="
echo "All tests completed!"
echo "==================================="
