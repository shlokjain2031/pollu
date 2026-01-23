#!/bin/bash

# Test script for concurrent prediction requests
# Shows cache filling up, hits/misses, and health monitoring

BASE_URL="http://localhost:8080"
NUM_REQUESTS=10
NUM_PARALLEL=3

echo "=== Testing Coordinator with Concurrent Requests ==="
echo ""
echo "Configuration:"
echo "  Base URL: $BASE_URL"
echo "  Total Requests: $NUM_REQUESTS"
echo "  Parallel Requests: $NUM_PARALLEL"
echo ""

# Function to make a prediction request
make_prediction() {
    local grid_id=$1
    local timestamp=$2
    local request_num=$3
    
    echo "[Request $request_num] Sending prediction for grid=$grid_id timestamp=$timestamp"
    
    curl -s -X POST "$BASE_URL/predict" \
        -H "Content-Type: application/json" \
        -d "{\"grid_id\":\"$grid_id\",\"timestamp_hour\":\"$timestamp\"}" \
        -w "\n" | jq '{
            grid_id: .grid_id,
            timestamp: .timestamp_hour,
            pm25: .pm25,
            source: .source,
            error: .error
        }'
}

# Function to show health status
show_health() {
    echo ""
    echo "=== Health Status ==="
    curl -s "$BASE_URL/health" | jq '{
        status: .status,
        queue_length: .queue_length,
        num_workers: .num_workers,
        cache: .cache,
        dedup: .dedup
    }'
    echo ""
}

# Test 1: Sequential requests (same grid, different timestamps)
echo ""
echo "=== Test 1: Sequential Requests (Same Grid, Different Times) ==="
echo "Making 5 sequential requests for same grid..."
echo ""

for i in {1..5}; do
    make_prediction "g00000001" "2026-01-19T14:0${i}:00Z" $i
    sleep 0.5
done

show_health

# Test 2: Concurrent requests (different grids, same timestamp)
echo ""
echo "=== Test 2: Concurrent Requests (Different Grids) ==="
echo "Making $NUM_REQUESTS concurrent requests in groups of $NUM_PARALLEL..."
echo ""

for i in $(seq 1 $NUM_PARALLEL $NUM_REQUESTS); do
    echo "--- Batch starting at request $i ---"
    
    # Launch multiple requests in parallel
    for j in $(seq 0 $((NUM_PARALLEL - 1))); do
        req_num=$((i + j))
        if [ $req_num -le $NUM_REQUESTS ]; then
            grid_id=$(printf "g%08d" $req_num)
            (make_prediction "$grid_id" "2026-01-19T14:00:00Z" $req_num) &
        fi
    done
    
    # Wait for this batch to complete
    wait
    sleep 1
done

show_health

# Test 3: Cache hit test - repeat same requests
echo ""
echo "=== Test 3: Cache Hit Test (Repeat Requests) ==="
echo "Making 5 requests for grid_00000001 (should hit cache)..."
echo ""

for i in {1..5}; do
    make_prediction "g00000001" "2026-01-19T14:00:00Z" "cache-hit-$i"
    sleep 0.3
done

show_health

# Test 4: Heavy concurrent load
echo ""
echo "=== Test 4: Heavy Concurrent Load ==="
echo "Launching 20 concurrent requests across different grids..."
echo ""

for i in $(seq 1 20); do
    grid_id=$(printf "g%08d" $i)
    (
        echo "[Heavy Load $i] Submitting grid=$grid_id"
        curl -s -X POST "$BASE_URL/predict" \
            -H "Content-Type: application/json" \
            -d "{\"grid_id\":\"$grid_id\",\"timestamp_hour\":\"2026-01-19T14:00:00Z\"}" 2>/dev/null | jq -c '{grid_id: .grid_id, pm25: .pm25, error: .error}'
    ) &
done

echo "Waiting for all requests to complete..."
wait

show_health

# Final summary
echo ""
echo "=== Test Summary ==="
echo "Total requests sent: $((5 + NUM_REQUESTS + 5 + 20))"
echo "Check health endpoint to see:"
echo "  - cache.total_entries: How many predictions are cached"
echo "  - dedup.active_computations: Concurrent computations"
echo "  - dedup.total_followers: Requests waiting on same computation"
echo "  - queue_length: Jobs in queue"
