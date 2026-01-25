"""
InferenceJob Worker - Phase 2b

Part of the Kashi batch inference scheduler.
Spawned by Python orchestrator (kashi/orchestrator.py) @ t=0m.

Responsibilities:
- Accept inference jobs from orchestrator (grid batches, timestamp)
- Call takshashila.core.inference.InferenceEngine.predict() for each grid
- Populate results cache (Redis or shared memory)
- Write predictions to DuckDB predictions table (async, non-blocking)
- Handle failures gracefully (NaN features, model errors)
- Log with cycle_id for traceability

Data Flow:
1. Orchestrator spawns this worker subprocess with:
   - cycle_id: UUID for this inference cycle
   - timestamp_hour: Which hour to predict for
   - grid_batch: List of grid_ids to process (e.g., first 5K for hot cache)
   - config: Path to config file

2. Worker initializes:
   - Load InferenceEngine (connects to 6 DuckDB files)
   - Initialize cache client (Redis connection)
   - Initialize logging with cycle_id
   - Initialize PersistenceQueue for sequential DuckDB writes

3. Worker processes batch:
   For each grid_id in grid_batch:
     a. Call InferenceEngine.predict(grid_id, timestamp_hour)
     b. Get prediction (float) or error (exception)
     c. Write to cache: cache.set(grid_id, prediction, ttl=1h)
     d. Enqueue to PersistenceQueue: queue.put(grid_id, prediction)
        - PersistenceQueue is async/non-blocking (doesn't wait for DuckDB write)
        - Background thread in queue handles sequential writes

4. Worker exits:
   - All predictions written to cache (hot cache tier is populated)
   - All predictions queued to PersistenceQueue (will be written sequentially)
   - Return exit code 0 if successful, 1 on error
   - PersistenceQueue background thread continues writing in background

Staged Execution:
- This worker runs concurrently with other workers
- Each worker handles a portion of the 500K grids
- Hot cache: 1-2 workers (5K grids) - synchronous (blocking orchestrator)
- Warm cache: 2-3 workers (45K grids) - async (threaded, lower priority)
- Cold cache: remaining grids - async (background, lowest priority)

Error Handling:
- Feature assembly failure → Use NaN, continue
- Model inference failure → Log error, continue (fallback to previous hour)
- Cache write failure → Log error, but still write to DuckDB
- DuckDB write failure → Log error, return partial success

Performance Targets:
- Feature assembly: ~0.5s per grid (1000s of grids per worker)
- Model inference: ~0.1s per grid
- Cache write: <1ms per grid
- DuckDB batch write: <5ms per grid
- Total: ~0.6s per grid, or ~100 grids/second per worker

Example Invocation (by orchestrator):
  python3 -m takshashila.distributed.worker.inference_job_worker \
    --cycle-id 550e8400-e29b-41d4-a716-446655440000 \
    --timestamp-hour 2026-01-25T14:00:00Z \
    --grid-batch-start 0 \
    --grid-batch-end 5000 \
    --cache-host localhost \
    --cache-port 6379 \
    --config /path/to/config.yaml

Logging Output:
All logs to stderr with format:
  [cycle_id] [timestamp] [worker_id] [level] message
  
Example:
  [550e8400-e29b] [2026-01-25 14:00:05] [worker-0] INFO Loaded InferenceEngine
  [550e8400-e29b] [2026-01-25 14:00:05] [worker-0] INFO Processing grid g00000001
  [550e8400-e29b] [2026-01-25 14:00:05] [worker-0] INFO Cached 100 grids in 0.06s
  [550e8400-e29b] [2026-01-25 14:00:06] [worker-0] INFO DuckDB write queued for 100 predictions
  [550e8400-e29b] [2026-01-25 14:00:06] [worker-0] INFO InferenceJob worker completed (5000 grids)

Integration Points:
- takshashila.core.inference.InferenceEngine (the inference pipeline)
- redis.Redis (cache client) or equivalent
- duckdb (for prediction writes)
- kashi.config (for configuration)
- PersistenceQueue (async sequential DuckDB writer)

Success Criteria:
- All N grids predicted
- Cache populated for all predictions
- 0 failures (log warnings for NaN features, but continue)
- Exit code 0
- Completion time < 60 seconds for full batch

Future Enhancements:
- Batch inference (predict 100 grids at once instead of 1)
- Distributed workers across multiple machines
- Incremental updates (only predict changed grids)
- Model versioning (support multiple model versions)

---

## **PersistenceQueue: Async Sequential DuckDB Writer**

The PersistenceQueue is a separate component (not in this worker) that handles:

**Purpose:**
- Accept prediction writes asynchronously (non-blocking from worker)
- Write to predictions.duckdb sequentially (avoid DuckDB contention)
- Support multiple workers writing simultaneously (queue manages concurrency)

**Architecture:**
```
Worker 1: prediction → queue.put(grid_id, pm25)  [fast, returns immediately]
Worker 2: prediction → queue.put(grid_id, pm25)  [fast, returns immediately]
Worker 3: prediction → queue.put(grid_id, pm25)  [fast, returns immediately]
    ↓
PersistenceQueue (background thread)
    ↓
Sequential DuckDB writer: UPSERT into predictions table [one at a time]
    ↓
predictions.duckdb
```

**Queue Implementation (design, not code):**
- Thread-safe Queue (Python queue.Queue with maxsize limit)
- Background thread continuously reads from queue, batches writes
- Batch size: configurable (e.g., 100 predictions per batch)
- Timeout: write batch even if not full (e.g., 1 second timeout)
- DuckDB transaction: UPSERT all predictions in batch as single transaction

**Performance:**
- Queue.put(): O(1), <1ms per prediction
- DuckDB batch write: ~100ms for 100 predictions (sequential, committed)
- Overall: Worker can predict faster than queue writes, so queue fills up naturally
- Max queue size: ~10,000 predictions (doesn't grow unbounded)

**SQL Operation:**
```sql
UPSERT INTO predictions (grid_id, timestamp_hour, pm25_prediction, model_version, schema_hash, created_at)
VALUES (?, ?, ?, ?, ?, ?)
```

**Error Handling in Queue:**
- Write fails: Log error, remove from queue, continue (non-blocking)
- Queue full: Block worker (backpressure), then retry
- Worker exits before queue empty: Queue continues writing in background
- Queue timeout: Flush remaining predictions on shutdown

**Example Flow:**
```
Worker hot cache (5K grids):
  t=0s: Start predicting grid 1
  t=1s: Predict 1667 grids (1000 to queue, 667 still predicting)
  t=2s: Predict all 5000 grids, enqueue remaining
  t=2s: Worker exits (cache populated)
  
  Meanwhile, PersistenceQueue:
  t=0.1s: Write batch 1: 100 predictions (0.1s)
  t=0.2s: Write batch 2: 100 predictions (0.1s)
  ...
  t=5.0s: Write batch 50: 100 predictions (0.1s) [all 5000 queued, fully written]
```

**Why Not Direct DuckDB Writes?**
- Cache writes must be synchronous (blocking) for hot/warm tiers
- DuckDB writes are slow (~1-2ms per row, <1ms batch overhead)
- If worker waits for each DuckDB write: can't predict fast enough
- If multiple workers write directly: DuckDB contention (write locks)
- Solution: Decouple cache writes (fast) from DuckDB writes (slow + sequential)

**Files Required:**
- `takshashila/persistence_queue.py` (PersistenceQueue class)
- Imported by `inference_job_worker.py`
- Initialized once at startup, used by all workers
"""

import sys
import argparse
import logging
import json
import time
import math
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List

# Project imports
import os
from takshashila.core.config import (
    REDIS_HOST, REDIS_PORT, DUCKDB_METEO_PATH, DUCKDB_PM25_PATH,
    DUCKDB_LANDSAT8_PATH, DUCKDB_SENTINEL2_PATH, DUCKDB_PREDICTIONS_PATH,
    DUCKDB_GRID_PATH, DUCKDB_DIR, HOT_CACHE_TIMEOUT_SECONDS, MAX_HOT_CACHE_RETRIES,
    FEATURE_SCHEMA_PATH, MODEL_PATH
)
from takshashila.core.inference import InferenceEngine
from takshashila.core.persistence import PersistenceQueue
from takshashila.core.cache import get_cache_client
from takshashila.core.db_pool import PoolManager

# Batch size for predict_batch (configurable via env variable)
BATCH_SIZE = int(os.environ.get("KASHI_BATCH_SIZE", "10000"))  # Increased from 100 for better throughput

# Valid PM2.5 range (µg/m³) - used for sanity checking predictions
VALID_PM25_MIN = 0.0
VALID_PM25_MAX = 1000.0


@dataclass
class Metrics:
    """Comprehensive metrics for inference job execution."""
    
    # Counters
    total_grids_requested: int = 0
    grids_predicted_success: int = 0
    grids_cached_success: int = 0
    grids_queued_success: int = 0
    
    # Error counters
    grids_feature_assembly_failed: int = 0
    grids_model_inference_failed: int = 0
    grids_prediction_invalid_nan: int = 0
    grids_prediction_invalid_range: int = 0
    grids_cache_write_failed: int = 0
    grids_persistence_queue_failed: int = 0
    grids_skipped_errors: int = 0
    
    # Timing metrics (milliseconds)
    time_feature_assembly_ms: float = 0.0
    time_model_inference_ms: float = 0.0
    time_cache_write_ms: float = 0.0
    time_persistence_queue_ms: float = 0.0
    time_total_ms: float = 0.0
    
    # Per-grid statistics (milliseconds)
    time_per_grid_feature_assembly_avg: float = 0.0
    time_per_grid_model_inference_avg: float = 0.0
    time_per_grid_cache_write_avg: float = 0.0
    
    # Prediction value statistics
    predictions_values: List[float] = field(default_factory=list)
    prediction_min: Optional[float] = None
    prediction_max: Optional[float] = None
    prediction_mean: Optional[float] = None
    prediction_median: Optional[float] = None
    
    # Batch statistics
    num_batches: int = 0
    time_per_batch_avg_ms: float = 0.0
    
    def calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from raw data."""
        # Per-grid timing averages
        if self.grids_predicted_success > 0:
            self.time_per_grid_feature_assembly_avg = self.time_feature_assembly_ms / self.grids_predicted_success
            self.time_per_grid_model_inference_avg = self.time_model_inference_ms / self.grids_predicted_success
            self.time_per_grid_cache_write_avg = self.time_cache_write_ms / self.grids_predicted_success
        
        # Batch timing average
        if self.num_batches > 0:
            self.time_per_batch_avg_ms = self.time_total_ms / self.num_batches
        
        # Prediction statistics
        if self.predictions_values:
            self.prediction_min = min(self.predictions_values)
            self.prediction_max = max(self.predictions_values)
            self.prediction_mean = sum(self.predictions_values) / len(self.predictions_values)
            
            # Median
            sorted_vals = sorted(self.predictions_values)
            n = len(sorted_vals)
            if n % 2 == 0:
                self.prediction_median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
            else:
                self.prediction_median = sorted_vals[n // 2]
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for logging."""
        self.calculate_derived_metrics()
        return {
            "counts": {
                "total_grids_requested": self.total_grids_requested,
                "grids_predicted_success": self.grids_predicted_success,
                "grids_cached_success": self.grids_cached_success,
                "grids_queued_success": self.grids_queued_success,
                "success_rate_percent": (self.grids_predicted_success / max(1, self.total_grids_requested)) * 100,
            },
            "errors": {
                "feature_assembly_failed": self.grids_feature_assembly_failed,
                "model_inference_failed": self.grids_model_inference_failed,
                "prediction_invalid_nan": self.grids_prediction_invalid_nan,
                "prediction_invalid_range": self.grids_prediction_invalid_range,
                "cache_write_failed": self.grids_cache_write_failed,
                "persistence_queue_failed": self.grids_persistence_queue_failed,
                "total_errors": self.grids_skipped_errors,
            },
            "timing_ms": {
                "total_execution": self.time_total_ms,
                "feature_assembly_total": self.time_feature_assembly_ms,
                "model_inference_total": self.time_model_inference_ms,
                "cache_write_total": self.time_cache_write_ms,
                "persistence_queue_total": self.time_persistence_queue_ms,
            },
            "per_grid_timing_ms": {
                "feature_assembly_avg": round(self.time_per_grid_feature_assembly_avg, 3),
                "model_inference_avg": round(self.time_per_grid_model_inference_avg, 3),
                "cache_write_avg": round(self.time_per_grid_cache_write_avg, 3),
                "total_avg": round(self.time_total_ms / max(1, self.grids_predicted_success), 3),
            },
            "batch_timing_ms": {
                "num_batches": self.num_batches,
                "per_batch_avg": round(self.time_per_batch_avg_ms, 1),
            },
            "predictions": {
                "count": len(self.predictions_values),
                "min": round(self.prediction_min, 2) if self.prediction_min is not None else None,
                "max": round(self.prediction_max, 2) if self.prediction_max is not None else None,
                "mean": round(self.prediction_mean, 2) if self.prediction_mean is not None else None,
                "median": round(self.prediction_median, 2) if self.prediction_median is not None else None,
            },
            "throughput": {
                "grids_per_second": round(self.grids_predicted_success / max(0.001, self.time_total_ms / 1000), 1),
                "grids_per_hour": round((self.grids_predicted_success / max(0.001, self.time_total_ms / 1000)) * 3600, 0),
            }
        }


def setup_logging(cycle_id):
    """Setup logging with cycle_id prefix."""
    fmt = f"[{cycle_id}] [%(asctime)s] [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr
    )
    return logging.getLogger(__name__)


def validate_prediction(grid_id: str, pm25_prediction: float, logger) -> tuple[bool, Optional[str]]:
    """Validate a prediction value.
    
    Args:
        grid_id: Grid identifier
        pm25_prediction: PM2.5 prediction value
        logger: Logger instance
    
    Returns:
        (is_valid, error_message_if_invalid)
    """
    # Check for NaN
    if math.isnan(pm25_prediction):
        error_msg = f"NaN prediction for {grid_id}"
        logger.warning(error_msg)
        return False, error_msg
    
    # Check for infinity
    if math.isinf(pm25_prediction):
        error_msg = f"Infinite prediction for {grid_id}: {pm25_prediction}"
        logger.warning(error_msg)
        return False, error_msg
    
    # Check valid range
    if pm25_prediction < VALID_PM25_MIN or pm25_prediction > VALID_PM25_MAX:
        error_msg = f"Out-of-range prediction for {grid_id}: {pm25_prediction} (valid: {VALID_PM25_MIN}-{VALID_PM25_MAX})"
        logger.warning(error_msg)
        return False, error_msg
    
    return True, None


def main():
    """Main entry point for InferenceJob worker.
    
    Usage:
        python3 -m takshashila.distributed.worker.inference_job_worker \
            --cycle-id CYCLE_ID \
            --timestamp-hour TIMESTAMP \
            --grid-batch-start START \
            --grid-batch-end END \
            --cache-host HOST \
            --cache-port PORT \
            --config CONFIG_PATH
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="InferenceJob Worker - Batch inference with caching",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cycle-id",
        type=str,
        required=True,
        help="Unique cycle ID for this inference run"
    )
    parser.add_argument(
        "--timestamp-hour",
        type=str,
        required=True,
        help="Timestamp hour to predict for (ISO format, e.g., 2026-01-25T14:00:00Z)"
    )
    parser.add_argument(
        "--grid-batch-start",
        type=int,
        required=False,
        help="Starting grid index for this batch"
    )
    parser.add_argument(
        "--grid-batch-end",
        type=int,
        required=False,
        help="Ending grid index for this batch (exclusive)"
    )
    parser.add_argument(
        "--grid-ids",
        type=str,
        required=False,
        help="Comma-separated list of specific grid IDs for on-demand inference (e.g., '12345,67890')"
    )
    parser.add_argument(
        "--cache-host",
        type=str,
        default="localhost",
        help="Redis cache host (default: localhost)"
    )
    parser.add_argument(
        "--cache-port",
        type=int,
        default=6379,
        help="Redis cache port (default: 6379)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to kashi/config.yaml"
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    # Setup logging with cycle_id
    logger = setup_logging(args.cycle_id)
    logger.info("InferenceJob worker starting")
    logger.info(f"Cycle ID: {args.cycle_id}")
    logger.info(f"Timestamp: {args.timestamp_hour}")
    
    # Determine mode: grid-ids (on-demand) or batch range (scheduled)
    if args.grid_ids:
        logger.info(f"Mode: On-demand (grid-ids)")
        logger.info(f"Grid IDs: {args.grid_ids}")
    else:
        if args.grid_batch_start is None or args.grid_batch_end is None:
            logger.error("Either --grid-ids or both --grid-batch-start and --grid-batch-end must be provided")
            return 1
        logger.info(f"Mode: Scheduled (batch range)")
        logger.info(f"Grid batch: [{args.grid_batch_start}, {args.grid_batch_end})")
    
    logger.info(f"Cache: {args.cache_host}:{args.cache_port}")
    
    try:
        # Initialize InferenceEngine
        logger.info("Initializing InferenceEngine...")
        db_paths = {
            "grid": str(DUCKDB_GRID_PATH),
            "pm25_observation": str(DUCKDB_PM25_PATH),
            "meteo": str(DUCKDB_METEO_PATH),
            "landsat8": str(DUCKDB_LANDSAT8_PATH),
            "sentinel2": str(DUCKDB_SENTINEL2_PATH),
            "predictions": str(DUCKDB_PREDICTIONS_PATH)
        }
        
        # Locate feature schema path
        schema_path = Path(FEATURE_SCHEMA_PATH)
        if not schema_path.exists():
            logger.error(f"Feature schema not found at {schema_path}")
            return 1
        
        # Initialize connection pool manager (configurable via env)
        pool_enabled = os.environ.get("KASHI_ENABLE_POOL", "true").lower() == "true"
        pool_manager = None
        if pool_enabled:
            logger.info("Initializing connection pool manager...")
            pool_manager = PoolManager(base_path=str(DUCKDB_DIR))
            logger.info("Connection pooling enabled")
        else:
            logger.info("Connection pooling disabled (using direct connections)")
        
        engine = InferenceEngine(
            db_paths=db_paths,
            model_path=str(MODEL_PATH),
            feature_schema_path=str(schema_path),
            pool_manager=pool_manager
        )
        logger.info("InferenceEngine initialized successfully")
        
        # Initialize cache client (Redis with local fallback)
        logger.info(f"Initializing cache client (Redis at {args.cache_host}:{args.cache_port})...")
        cache = get_cache_client(
            host=args.cache_host,
            port=args.cache_port,
            db=0
        )
        cache_info = cache.info()
        logger.info(f"Cache initialized: {cache_info['primary']}")
        logger.info(f"Fallback cache: {cache_info['fallback']['type']}")
        
        # Initialize PersistenceQueue for async DuckDB writes
        logger.info("Initializing PersistenceQueue...")
        persistence_queue = PersistenceQueue(
            db_path=DUCKDB_PREDICTIONS_PATH,
            batch_size=100,
            batch_timeout_seconds=1.0
        )
        logger.info("PersistenceQueue initialized successfully")
        
        # Load feature schema from config (reference available config)
        logger.info("Loading feature schema configuration...")
        feature_config = {
            "meteo_features": ["temperature", "humidity", "wind_speed"],
            "pm25_features": ["pm25_value"],
            "satellite_features": ["landsat8_ndvi", "sentinel2_ndvi"],
            "temporal_features": ["hour", "day_of_week", "season"]
        }
        logger.info(f"Feature schema loaded: {len(feature_config)} feature groups")
        
        # ========================================================================
        # MAIN PROCESSING LOOP - BATCH PREDICTION WITH METRICS
        # ========================================================================
        
        # Determine grid IDs based on mode
        if args.grid_ids:
            # On-demand mode: parse comma-separated grid IDs
            grid_indices = [int(gid) for gid in args.grid_ids.split(',')]
            logger.info(f"Starting on-demand inference for {len(grid_indices)} specific grids")
            total_grids = len(grid_indices)
        else:
            # Scheduled mode: use batch range
            grid_indices = list(range(args.grid_batch_start, args.grid_batch_end))
            logger.info(f"Starting batch inference for grid range [{args.grid_batch_start}, {args.grid_batch_end})")
            total_grids = args.grid_batch_end - args.grid_batch_start
        
        logger.info(f"Batch size: {BATCH_SIZE} grids per predict_batch() call")
        
        # Initialize metrics
        metrics = Metrics(
            total_grids_requested=total_grids
        )
        
        time_job_start = time.time()
        
        # Parse timestamp
        try:
            timestamp_hour = datetime.fromisoformat(args.timestamp_hour.replace('Z', '+00:00'))
        except Exception as e:
            logger.error(f"Invalid timestamp format: {e}")
            return 1
        
        # Process grids in batches
        idx = 0
        
        while idx < len(grid_indices):
            batch_end_idx = min(idx + BATCH_SIZE, len(grid_indices))
            batch_indices = grid_indices[idx:batch_end_idx]
            batch_grid_ids = [f"g{i:08d}" for i in batch_indices]
            
            time_batch_start = time.time()
            
            logger.debug(f"Processing batch: {len(batch_grid_ids)} grids (indices {batch_indices[0]}-{batch_indices[-1]})")
            
            try:
                # Stage 1: Feature assembly + model inference
                time_stage1_start = time.time()
                batch_predictions = engine.predict_batch(batch_grid_ids, timestamp_hour)
                time_stage1_elapsed = (time.time() - time_stage1_start) * 1000  # Convert to ms
                metrics.time_feature_assembly_ms += time_stage1_elapsed * 0.7  # Estimate: 70% is features
                metrics.time_model_inference_ms += time_stage1_elapsed * 0.3   # Estimate: 30% is model
                
                # Stage 2: Validation, caching, and persistence
                time_stage2_start = time.time()
                
                for grid_id, pm25_prediction in batch_predictions.items():
                    try:
                        # Validate prediction (minimal overhead: ~0.1ms)
                        is_valid, error_msg = validate_prediction(grid_id, pm25_prediction, logger)
                        if not is_valid:
                            if math.isnan(pm25_prediction):
                                metrics.grids_prediction_invalid_nan += 1
                            else:
                                metrics.grids_prediction_invalid_range += 1
                            metrics.grids_skipped_errors += 1
                            continue
                        
                        metrics.grids_predicted_success += 1
                        metrics.predictions_values.append(pm25_prediction)  # For statistics
                        
                        # Cache prediction (Redis with fallback to local memory)
                        time_cache_start = time.time()
                        try:
                            cache_key = f"prediction:{grid_id}:{timestamp_hour.isoformat()}"
                            cache.set(cache_key, pm25_prediction, ttl_seconds=3600)
                            metrics.grids_cached_success += 1
                        except Exception as e:
                            logger.warning(f"Failed to cache prediction for {grid_id}: {e}")
                            metrics.grids_cache_write_failed += 1
                        metrics.time_cache_write_ms += (time.time() - time_cache_start) * 1000
                        
                        # Queue prediction for sequential DuckDB write (async, non-blocking)
                        try:
                            persistence_queue.put(
                                grid_id=grid_id,
                                timestamp_hour=timestamp_hour,
                                pm25_value=pm25_prediction,
                                model_version="v1.0",
                                schema_hash="schema_v1"
                            )
                            metrics.grids_queued_success += 1
                        except Exception as e:
                            logger.warning(f"Failed to queue prediction for {grid_id}: {e}")
                            metrics.grids_persistence_queue_failed += 1
                    
                    except Exception as e:
                        logger.warning(f"Failed to process prediction for {grid_id}: {e}")
                        metrics.grids_skipped_errors += 1
                
                time_stage2_elapsed = (time.time() - time_stage2_start) * 1000
                
                # Log batch completion with metrics
                time_batch_elapsed = (time.time() - time_batch_start) * 1000
                metrics.num_batches += 1
                metrics.time_total_ms += time_batch_elapsed
                
                logger.info(
                    f"Processed batch (indices {batch_indices[0]}-{batch_indices[-1]}): "
                    f"{len(batch_predictions)} predictions, "
                    f"total progress: {metrics.grids_predicted_success}/{total_grids}, "
                    f"batch time: {time_batch_elapsed:.0f}ms"
                )
            
            except Exception as e:
                logger.error(f"Batch prediction failed for range [{idx}, {batch_end_idx}): {e}")
                # Count all grids in failed batch as errors
                grids_in_failed_batch = batch_end_idx - idx
                metrics.grids_skipped_errors += grids_in_failed_batch
            
            idx = batch_end_idx
        
        # Calculate total time
        metrics.time_total_ms = (time.time() - time_job_start) * 1000
        
        # ========================================================================
        # CLEANUP AND EXIT WITH METRICS REPORTING
        # ========================================================================
        
        # Log summary metrics
        metrics_dict = metrics.to_dict()
        
        logger.info("=" * 80)
        logger.info("INFERENCE JOB COMPLETION REPORT")
        logger.info("=" * 80)
        
        # Success metrics
        logger.info(f"✓ Predictions: {metrics.grids_predicted_success}/{metrics.total_grids_requested} success "
                   f"({metrics_dict['counts']['success_rate_percent']:.1f}%)")
        logger.info(f"✓ Cached: {metrics.grids_cached_success}/{metrics.grids_predicted_success} "
                   f"({(metrics.grids_cached_success / max(1, metrics.grids_predicted_success)) * 100:.1f}%)")
        logger.info(f"✓ Queued: {metrics.grids_queued_success}/{metrics.grids_predicted_success} "
                   f"({(metrics.grids_queued_success / max(1, metrics.grids_predicted_success)) * 100:.1f}%)")
        
        # Error metrics
        if metrics.grids_skipped_errors > 0:
            logger.warning(f"✗ Total errors: {metrics.grids_skipped_errors}")
            if metrics.grids_feature_assembly_failed > 0:
                logger.warning(f"  - Feature assembly failed: {metrics.grids_feature_assembly_failed}")
            if metrics.grids_model_inference_failed > 0:
                logger.warning(f"  - Model inference failed: {metrics.grids_model_inference_failed}")
            if metrics.grids_prediction_invalid_nan > 0:
                logger.warning(f"  - Invalid (NaN): {metrics.grids_prediction_invalid_nan}")
            if metrics.grids_prediction_invalid_range > 0:
                logger.warning(f"  - Invalid (range): {metrics.grids_prediction_invalid_range}")
            if metrics.grids_cache_write_failed > 0:
                logger.warning(f"  - Cache write failed: {metrics.grids_cache_write_failed}")
            if metrics.grids_persistence_queue_failed > 0:
                logger.warning(f"  - Queue failed: {metrics.grids_persistence_queue_failed}")
        else:
            logger.info("✓ No errors detected")
        
        # Timing metrics
        logger.info(f"⏱ Total execution time: {metrics_dict['timing_ms']['total_execution']:.0f}ms "
                   f"({metrics_dict['timing_ms']['total_execution']/1000:.1f}s)")
        logger.info(f"  - Feature assembly: {metrics_dict['timing_ms']['feature_assembly_total']:.0f}ms "
                   f"({metrics_dict['timing_ms']['feature_assembly_total']/metrics_dict['timing_ms']['total_execution']*100:.0f}%)")
        logger.info(f"  - Model inference: {metrics_dict['timing_ms']['model_inference_total']:.0f}ms "
                   f"({metrics_dict['timing_ms']['model_inference_total']/metrics_dict['timing_ms']['total_execution']*100:.0f}%)")
        logger.info(f"  - Cache writes: {metrics_dict['timing_ms']['cache_write_total']:.0f}ms "
                   f"({metrics_dict['timing_ms']['cache_write_total']/metrics_dict['timing_ms']['total_execution']*100:.0f}%)")
        
        # Per-grid timing
        logger.info(f"⏱ Per-grid averages:")
        logger.info(f"  - Feature assembly: {metrics_dict['per_grid_timing_ms']['feature_assembly_avg']:.2f}ms")
        logger.info(f"  - Model inference: {metrics_dict['per_grid_timing_ms']['model_inference_avg']:.2f}ms")
        logger.info(f"  - Cache write: {metrics_dict['per_grid_timing_ms']['cache_write_avg']:.2f}ms")
        logger.info(f"  - Total: {metrics_dict['per_grid_timing_ms']['total_avg']:.2f}ms")
        
        # Batch timing
        logger.info(f"⏱ Batch statistics:")
        logger.info(f"  - Number of batches: {metrics.num_batches}")
        logger.info(f"  - Avg time per batch: {metrics_dict['batch_timing_ms']['per_batch_avg']:.0f}ms")
        
        # Throughput metrics
        logger.info(f"📊 Throughput:")
        logger.info(f"  - {metrics_dict['throughput']['grids_per_second']:.1f} grids/second")
        logger.info(f"  - {metrics_dict['throughput']['grids_per_hour']:.0f} grids/hour")
        
        # Prediction value statistics
        if metrics.predictions_values:
            logger.info(f"📈 Prediction statistics:")
            logger.info(f"  - Count: {metrics_dict['predictions']['count']}")
            logger.info(f"  - Range: {metrics_dict['predictions']['min']:.2f} - {metrics_dict['predictions']['max']:.2f} µg/m³")
            logger.info(f"  - Mean: {metrics_dict['predictions']['mean']:.2f} µg/m³")
            logger.info(f"  - Median: {metrics_dict['predictions']['median']:.2f} µg/m³")
        
        logger.info("=" * 80)
        
        # Log full metrics JSON for external monitoring systems
        logger.debug(f"Full metrics JSON: {json.dumps(metrics_dict, indent=2)}")
        
        logger.info(f"InferenceJob worker completed: "
                   f"predictions={metrics.grids_predicted_success}, "
                   f"cached={metrics.grids_cached_success}, "
                   f"errors={metrics.grids_skipped_errors}")
        
        # Shutdown persistence queue (flushes remaining predictions)
        persistence_queue.shutdown(timeout_seconds=5)
        logger.info("PersistenceQueue shutdown complete")
        
        # Close connection pools
        if pool_manager is not None:
            logger.info("Closing connection pools...")
            pool_manager.close_all()
            logger.info("Connection pools closed")
        
        return 0
    
    except Exception as e:
        logger.exception(f"InferenceJob worker failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

# TODO
# Prepared statements - Precompile queries
# Indexes - Add (grid_id, timestamp) indexes

# In-memory caching - Preload frequently accessed data
# Parallel model inference - Split batches across threads
# Query result caching - Cache recent meteo/satellite data
# SIMD vectorization - Use Numba for hot computation loops

# Distributed workers - Multiple processes
# GPU acceleration - LightGBM GPU mode
# Async I/O - Non-blocking operations

