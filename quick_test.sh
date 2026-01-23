#!/bin/bash

# Quick concurrent test - run this to see the system in action

echo "🚀 Sending 10 concurrent prediction requests..."
echo ""

# Send 10 concurrent requests
for i in {1..10}; do
    (
        grid_id=$(printf "g%08d" $i)
        echo "Request $i: Predicting for $grid_id at 2026-01-19T14:00:00Z"
        curl -s -X POST http://localhost:8080/predict \
            -H "Content-Type: application/json" \
            -d "{\"grid_id\":\"$grid_id\",\"timestamp_hour\":\"2026-01-19T14:00:00Z\"}" | jq -c '{grid_id, pm25, error}'
    ) &
done

wait

echo ""
echo "✅ All requests completed!"
echo ""
echo "📊 Health Status:"
curl -s http://localhost:8080/health | jq .
echo ""
echo "💾 Cache Statistics:"
curl -s http://localhost:8080/health | jq '.cache'
echo ""
echo "🔄 Deduplication Stats:"
curl -s http://localhost:8080/health | jq '.dedup'
