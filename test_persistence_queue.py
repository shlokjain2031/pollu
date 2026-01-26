#!/usr/bin/env python3
"""Test if PersistenceQueue is working."""

import time
from datetime import datetime, timezone
from takshashila.core.persistence import PersistenceQueue
import duckdb

# Clear predictions table
db = duckdb.connect('takshashila/duckdb/predictions.duckdb')
db.execute('DELETE FROM predictions')
print("✓ Cleared predictions table")

# Create a queue
queue = PersistenceQueue(
    db_path='takshashila/duckdb/predictions.duckdb',
    batch_size=10,
    batch_timeout_seconds=2.0
)

# Add some test predictions
print("\nAdding 25 test predictions to queue...")
for i in range(25):
    queue.put(
        grid_id=f"g{i:08d}",
        timestamp_hour=datetime(2026, 1, 26, 15, 0, 0, tzinfo=timezone.utc),
        pm25_value=10.0 + i,
        model_version="test_v1",
        schema_hash="test_hash"
    )

print("Waiting for background writes (3 seconds)...")
time.sleep(3)

# Shutdown the queue (flushes remaining)
print("Shutting down queue...")
queue.shutdown()

# Check results
db = duckdb.connect('takshashila/duckdb/predictions.duckdb')
count = db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
print(f"\n✅ PersistenceQueue wrote {count:,} predictions to DuckDB!")

result = db.execute("""
    SELECT grid_id, pm25_prediction, model_version FROM predictions ORDER BY grid_id LIMIT 5
""").fetchall()
print("\nSample predictions:")
for row in result:
    print(f"  {row[0]}: {row[1]:.1f} µg/m³ ({row[2]})")
