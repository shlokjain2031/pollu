#!/bin/bash
# Bring up the full Delhi PM2.5 backend: Redis + kashi + Go coordinator.
#
#   scripts/run_backend.sh          # start everything
#   scripts/run_backend.sh stop     # stop everything
#
# Logs: /tmp/{kashi,coordinator}.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv/bin"
COORD="$ROOT/takshashila/distributed/coordinator"

stop() {
  pkill -f "uvicorn kashi.service" 2>/dev/null || true
  lsof -ti:8080 | xargs kill 2>/dev/null || true  # coordinator, however it was launched
  redis-cli shutdown nosave 2>/dev/null || true
  echo "stopped"
}

if [ "${1:-}" = "stop" ]; then stop; exit 0; fi

# Redis
if ! redis-cli ping >/dev/null 2>&1; then
  # Cache only: no persistence, bounded memory with LRU eviction.
  redis-server --daemonize yes --port 6379 --save '' --dir /tmp \
    --maxmemory 512mb --maxmemory-policy allkeys-lru
  sleep 1
fi
echo "redis: $(redis-cli ping)"

# Feature lookup table (build once)
[ -f "$ROOT/kashi/data/delhi_features.parquet" ] || "$VENV/python" -m kashi.build_features

# Kashi inference service (multiple workers parallelize concurrent cold inference)
cd "$ROOT"
KASHI_WORKERS="${KASHI_WORKERS:-4}"
"$VENV/uvicorn" kashi.service:app --host 127.0.0.1 --port 9999 --workers "$KASHI_WORKERS" >/tmp/kashi.log 2>&1 &
for _ in $(seq 1 30); do
  curl -sf localhost:9999/health >/dev/null 2>&1 && break || sleep 1
done
echo "kashi: $(curl -s localhost:9999/health)"

# Coordinator
cd "$COORD"
go build -o coordinator .
./coordinator >/tmp/coordinator.log 2>&1 &
sleep 2
echo "coordinator: $(curl -s localhost:8080/health | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
echo "ready — POST http://localhost:8080/predict"
