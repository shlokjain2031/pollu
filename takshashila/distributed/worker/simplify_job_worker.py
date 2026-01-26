"""
SimplifyJob Worker - PM2.5 Only Pipeline

Simplified worker that:
1. Fetches PM2.5 data from WAQI API using PM25Fetcher
2. Interpolates to all 500K grid cells using IDW
3. Writes to pm25_observations.duckdb (with provided timestamp, no rounding)
4. Caches all 500K predictions to Redis
5. Enqueues to PersistenceQueue for async writes

Runs every hour, triggered by orchestrator.

Data Flow:
1. Orchestrator spawns worker @ t=0m with:
   - cycle_id: UUID for traceability
   - timestamp_hour: Which hour to process (UTC, rounded to hour)

2. Worker initializes:
   - PM25Fetcher (from write_job_worker)
   - Cache client (Redis)
   - PersistenceQueue (async DuckDB writer)
   - Logging with cycle_id

3. Worker executes:
   Step 1: Fetch PM2.5 from WAQI API
     - PM25Fetcher.fetch(current_date) → DataFrame with 500K rows
     - Returns grid_id, timestamp, pm25
   
   Step 2: Write to pm25_observations.duckdb
     - PM25Fetcher.write_to_db(df)
     - Timestamp stored as-is (no ceiling rounding)
   
   Step 3: Cache all 500K predictions to Redis
     - For each grid: cache.set(f"prediction:{grid_id}:{timestamp}", pm25_value, ttl=3600)
     - Fast writes (~1ms per grid, batched)
   
   Step 4: Enqueue to PersistenceQueue
     - For each grid: queue.put(grid_id, timestamp, pm25_value)
     - Async, non-blocking
     - Background thread writes to predictions.duckdb sequentially

4. Worker exits:
   - All 500K grids cached (hot)
   - All 500K grids written to pm25_observations.duckdb
   - All 500K grids queued to PersistenceQueue
   - Exit code 0 if successful, 1 on error

Error Handling:
- PM2.5 fetch fails → Log error, exit code 1, skip all steps
- DuckDB write fails → Log error, exit code 1, skip cache/queue
- Cache write fails → Log warning, continue (not critical)
- Queue put fails → Log warning, continue (not critical)

Performance Targets:
- PM2.5 fetch: ~3 seconds (26 sensors → 500K grids)
- DuckDB write: ~1 second (500K rows)
- Cache writes: ~5 seconds (500K keys @ 1ms each)
- Queue enqueue: ~1 second (500K items @ 0.002ms each)
- Total: ~10 seconds end-to-end

Example Invocation:
  python3 -m takshashila.distributed.worker.simplify_job_worker \
    --cycle-id 550e8400-e29b-41d4-a716-446655440000 \
    --timestamp-hour 2026-01-26T17:00:00Z \
    --cache-host localhost \
    --cache-port 6379

Success Criteria:
- PM2.5 data fetched successfully
- 500K rows written to pm25_observations.duckdb
- 500K cache keys written to Redis
- 500K items queued to PersistenceQueue
- Exit code 0
- Completion time < 15 seconds
"""

import sys
import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Project imports
from takshashila.core.config import (
    REDIS_HOST, REDIS_PORT, DUCKDB_PM25_PATH,
    GRID_PARQUET_PATH
)
from takshashila.core.cache import get_cache_client
from takshashila.distributed.worker.write_job_worker import PM25Fetcher


def main():
    """Main entry point for SimplifyJob worker."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SimplifyJob worker - PM2.5 only pipeline")
    parser.add_argument(
        "--cycle-id",
        type=str,
        required=True,
        help="Cycle ID for traceability (UUID)"
    )
    parser.add_argument(
        "--timestamp-hour",
        type=str,
        required=True,
        help="Timestamp hour to process (ISO format, e.g., 2026-01-26T17:00:00Z)"
    )
    parser.add_argument(
        "--cache-host",
        type=str,
        default=REDIS_HOST,
        help=f"Redis cache host (default: {REDIS_HOST})"
    )
    parser.add_argument(
        "--cache-port",
        type=int,
        default=REDIS_PORT,
        help=f"Redis cache port (default: {REDIS_PORT})"
    )
    args = parser.parse_args()
    
    # Initialize logging with cycle_id prefix
    cycle_id_short = args.cycle_id[:13]  # First 13 chars of UUID
    logging.basicConfig(
        level=logging.INFO,
        format=f"[{cycle_id_short}] [%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("SimplifyJob worker starting")
    logger.info(f"Cycle ID: {args.cycle_id}")
    logger.info(f"Timestamp: {args.timestamp_hour}")
    
    # Parse timestamp
    try:
        timestamp_hour = datetime.fromisoformat(args.timestamp_hour.replace("Z", "+00:00"))
        if timestamp_hour.tzinfo is None:
            timestamp_hour = timestamp_hour.replace(tzinfo=timezone.utc)
        logger.info(f"Parsed timestamp: {timestamp_hour.isoformat()}")
    except Exception as e:
        logger.error(f"Failed to parse timestamp: {e}")
        sys.exit(1)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # PM25Fetcher
    pm25_fetcher = PM25Fetcher(logger)
    
    # Cache client (Redis)
    try:
        cache = get_cache_client(host=args.cache_host, port=args.cache_port)
        logger.info(f"✓ Cache client initialized (host={args.cache_host}, port={args.cache_port})")
    except Exception as e:
        logger.error(f"✗ Failed to initialize cache client: {e}")
        sys.exit(1)
    
    # ===========================
    # STEP 1: Fetch PM2.5 from WAQI API
    # ===========================
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching PM2.5 data from WAQI API")
    logger.info("=" * 60)
    
    time_fetch_start = time.time()
    try:
        pm25_df = pm25_fetcher.fetch(timestamp_hour)
        
        if pm25_df is None or len(pm25_df) == 0:
            logger.error("✗ PM2.5 fetch returned no data, aborting")
            sys.exit(1)
        
        time_fetch_elapsed = time.time() - time_fetch_start
        logger.info(f"✓ Fetched {len(pm25_df):,} grid cells in {time_fetch_elapsed:.2f}s")
        
        # Log statistics
        pm25_values = pm25_df['pm25'].dropna()
        if len(pm25_values) > 0:
            logger.info(f"  PM2.5 range: {pm25_values.min():.2f} - {pm25_values.max():.2f} µg/m³")
            logger.info(f"  PM2.5 mean: {pm25_values.mean():.2f} µg/m³")
            logger.info(f"  PM2.5 median: {pm25_values.median():.2f} µg/m³")
        
    except Exception as e:
        logger.error(f"✗ PM2.5 fetch failed: {e}")
        sys.exit(1)
    
    # ===========================
    # STEP 2: Write to pm25_observations.duckdb
    # ===========================
    logger.info("=" * 60)
    logger.info("STEP 2: Writing to pm25_observations.duckdb")
    logger.info("=" * 60)
    
    time_write_start = time.time()
    try:
        success = pm25_fetcher.write_to_db(pm25_df)
        
        if not success:
            logger.error("✗ DuckDB write failed, aborting")
            sys.exit(1)
        
        time_write_elapsed = time.time() - time_write_start
        logger.info(f"✓ Written {len(pm25_df):,} rows in {time_write_elapsed:.2f}s")
        
    except Exception as e:
        logger.error(f"✗ DuckDB write failed: {e}")
        sys.exit(1)
    
    # ===========================
    # STEP 3: Cache all predictions to Redis
    # ===========================
    logger.info("=" * 60)
    logger.info("STEP 3: Caching predictions to Redis")
    logger.info("=" * 60)
    
    time_cache_start = time.time()
    cached_count = 0
    cache_failed_count = 0
    
    # Use the timestamp from the DataFrame (already ceiling-rounded)
    cache_timestamp = pm25_df['timestamp'].iloc[0]
    # Format as: 2026-01-26T14:00:00Z (simpler than +00:00)
    cache_timestamp_str = cache_timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    for idx, row in pm25_df.iterrows():
        grid_id = row['grid_id']
        pm25_value = row['pm25']
        
        try:
            cache_key = f"prediction:{grid_id}:{cache_timestamp_str}"
            cache.set(cache_key, pm25_value, ttl_seconds=3600)
            cached_count += 1
            
            # Log progress every 50K grids
            if cached_count % 50000 == 0:
                logger.info(f"  Cached {cached_count:,} / {len(pm25_df):,} grids ({100*cached_count//len(pm25_df)}%)")
        
        except Exception as e:
            cache_failed_count += 1
            if cache_failed_count <= 10:  # Only log first 10 failures
                logger.warning(f"Failed to cache {grid_id}: {e}")
    
    time_cache_elapsed = time.time() - time_cache_start
    logger.info(f"✓ Cached {cached_count:,} / {len(pm25_df):,} grids in {time_cache_elapsed:.2f}s")
    if cache_failed_count > 0:
        logger.warning(f"  {cache_failed_count:,} cache writes failed (non-critical)")
    
    # ===========================
    # COMPLETION REPORT
    # ===========================
    logger.info("=" * 60)
    logger.info("SIMPLIFY JOB COMPLETION REPORT")
    logger.info("=" * 60)
    
    total_time = time.time() - time_fetch_start
    
    logger.info(f"✓ Total grids processed: {len(pm25_df):,}")
    logger.info(f"✓ PM2.5 observations written: {len(pm25_df):,}")
    logger.info(f"✓ Predictions cached: {cached_count:,}")
    logger.info(f"")
    logger.info(f"⏱ Timing breakdown:")
    logger.info(f"  - PM2.5 fetch: {time_fetch_elapsed:.2f}s ({100*time_fetch_elapsed/total_time:.0f}%)")
    logger.info(f"  - DuckDB write: {time_write_elapsed:.2f}s ({100*time_write_elapsed/total_time:.0f}%)")
    logger.info(f"  - Cache writes: {time_cache_elapsed:.2f}s ({100*time_cache_elapsed/total_time:.0f}%)")
    logger.info(f"  - Total: {total_time:.2f}s")
    logger.info(f"")
    logger.info(f"📊 Throughput:")
    logger.info(f"  - {len(pm25_df)/total_time:.0f} grids/second")
    logger.info(f"  - {len(pm25_df)*3600/total_time:.0f} grids/hour")
    logger.info("=" * 60)
    
    logger.info("SimplifyJob worker completed successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()
