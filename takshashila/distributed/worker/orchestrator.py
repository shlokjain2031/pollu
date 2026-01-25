"""
Orchestrator - Worker Coordinator

Manages the timing and spawning of WriterJob and InferenceJob workers.
Runs as a daemon subprocess of the Go Coordinator.
Implements the Kashi batch inference scheduler logic.

Responsibilities:
- Schedule WriterJob @ t=-2m (2 minutes before hour)
- Wait for WriterJob completion (must finish by t=-1m)
- Schedule InferenceJob @ t=0m (top of hour)
- Spawn InferenceJob workers in stages (hot → warm → cold)
- Wait for hot cache to be populated before returning to coordinator
- Log all execution with cycle_id
- Handle failures (retries, stale cache fallback)
- Coordinate between multiple workers (shared cache, persistence queue)

Timing Guarantees:
- WriterJob: t=-2m to t=-1m (120 seconds, typical 20-40 seconds)
- InferenceJob hot: t=0m to t=1m:05s (65 seconds, typical 45-55 seconds)
- InferenceJob warm: t=1m:05s to t=1m:35s (30 seconds, async)
- InferenceJob cold: t=1m:35s to t=1m:55s (20 seconds, async, background)

Architecture:
```
Orchestrator (Python daemon, started by Go Coordinator)
├── APScheduler (cron jobs for t=-2m and t=0m)
├── JobState (tracks cycle progress)
├── WriterJob subprocess
│   └── write_job_worker.py (fetches from 4 APIs, writes to DuckDB)
└── InferenceJob subprocesses (up to N workers)
    ├── Worker 1: inference_job_worker.py (hot cache, 5K grids)
    ├── Worker 2: inference_job_worker.py (warm cache, 45K grids, async)
    └── Worker 3+: inference_job_worker.py (cold cache, remaining, async)
```

Main Entry Point:
- `python3 -m takshashila.distributed.worker.orchestrator`
- Reads config from environment or kashi/config.yaml
- Initializes APScheduler with UTC timezone
- Registers two jobs: run_writer_job (t=-2m), run_inference_job (t=0m)
- Runs forever (until SIGTERM from coordinator)

Cycle Management:
- cycle_id: UUID generated for each hourly cycle
- Each job (WriterJob, InferenceJob workers) logs with same cycle_id
- Allows tracing entire inference pipeline through logs

WriterJob Execution:
1. @ t=-2m, run_writer_job() is called
2. Generate cycle_id = uuid.uuid4()
3. Spawn subprocess: python3 -m takshashila.distributed.worker.write_job_worker
   --cycle-id {cycle_id}
   --timestamp-to-write {rounded_to_hour}
   --config {config_path}
4. Wait for subprocess with timeout=120s
5. If exit code 0: Success, log "WriterJob completed"
6. If exit code non-zero: Log error, skip InferenceJob (use stale cache)
7. If timeout: Kill process, log error, skip InferenceJob
8. Store result in JobState for reference

InferenceJob Execution:
1. @ t=0m, run_inference_job() is called (only if WriterJob succeeded)
2. Use same cycle_id from WriterJob
3. Load all 500K grid IDs from grid.duckdb (cached in memory)
4. Initialize shared components:
   - Redis cache client (for cache.set writes)
   - PersistenceQueue (for DuckDB writes)
5. Spawn workers in stages:

   Stage 1 (Hot Cache, Synchronous):
   - Spawn 1 worker: grid batch [0:5000]
   - Call: python3 -m takshashila.distributed.worker.inference_job_worker
     --cycle-id {cycle_id}
     --timestamp-hour {hour}
     --grid-batch-start 0
     --grid-batch-end 5000
     --cache-host localhost
     --cache-port 6379
   - WAIT for completion (blocking orchestrator)
   - Max time: ~60 seconds
   - If timeout or error: Retry up to 2 times, then fail fast
   - At t≈1m:05s: Hot cache (5K grids) ready for queries

   Stage 2 (Warm Cache, Async):
   - Spawn 2-3 workers concurrently (non-blocking orchestrator)
   - Worker 2: grid batch [5000:50000]
   - Worker 3: grid batch [50000:150000]
   - Worker 4: grid batch [150000:250000]
   - Call with same parameters, different grid ranges
   - DON'T wait for completion (return immediately to coordinator)
   - Expected completion: t≈1m:35s
   - If error: Log but continue (warm cache not critical)

   Stage 3 (Cold Cache, Background):
   - Spawn remaining workers (non-blocking)
   - Worker 5+: remaining grid batches
   - Expected completion: t≈1m:55s
   - If error: Log but continue (cold cache not critical)

6. Return to main loop after hot cache is ready (t≈1m:05s)

Error Handling:
- WriterJob failure: Skip InferenceJob, use stale predictions from DuckDB
- Hot cache failure: Retry up to 2 times, fail if all retries exhausted
- Warm/cold cache failure: Log warning, continue (hot cache sufficient)
- Worker crash: Log error, attempt respawn if not at max retries
- Timeout: Kill process, log error, move to next stage

Graceful Shutdown:
- Receive SIGTERM from coordinator
- Stop scheduling new jobs
- Wait for current jobs to complete (max 60 seconds)
- Kill any remaining worker subprocesses
- Close connections (Redis, DuckDB)
- Exit with code 0

Configuration:
- Reads from kashi/config.yaml:
  - schedule_hour (default: 0)
  - schedule_minute (default: 0)
  - writer_timeout_seconds (default: 300)
  - inference_timeout_seconds (default: 300)
  - num_inference_workers (for warm/cold stages)
  - grid_batch_size (how many grids per worker)
  - cache_host, cache_port (Redis)
- Falls back to environment variables:
  - KASHI_SCHEDULE_HOUR
  - KASHI_WRITER_TIMEOUT
  - KASHI_CACHE_HOST
  - etc.

Logging:
- All logs to stderr with format: [cycle_id] [timestamp] [level] message
- Log levels: INFO (normal), WARNING (partial failure), ERROR (critical)
- Example:
  [550e8400-e29b] [2026-01-25 13:58:00] INFO WriterJob scheduled for t=-2m
  [550e8400-e29b] [2026-01-25 13:58:00] INFO WriterJob starting (PID: 12345)
  [550e8400-e29b] [2026-01-25 13:58:30] INFO WriterJob completed (meteo: 500K, pm25: 26)
  [550e8400-e29b] [2026-01-25 14:00:00] INFO InferenceJob scheduled for t=0m
  [550e8400-e29b] [2026-01-25 14:00:00] INFO InferenceJob Stage 1 (hot cache) starting
  [550e8400-e29b] [2026-01-25 14:00:05] INFO InferenceJob hot cache worker started (PID: 12346)
  [550e8400-e29b] [2026-01-25 14:01:05] INFO InferenceJob hot cache completed (5000 grids cached)
  [550e8400-e29b] [2026-01-25 14:01:05] INFO InferenceJob Stage 2 (warm cache) starting async
  [550e8400-e29b] [2026-01-25 14:01:05] INFO InferenceJob Stage 3 (cold cache) starting async
  [550e8400-e29b] [2026-01-25 14:01:05] INFO Cycle completed, hot cache ready

Integration with Go Coordinator:
- Coordinator starts orchestrator as subprocess: python3 -m takshashila.distributed.worker.orchestrator
- Orchestrator runs in foreground, logs to stderr
- Coordinator receives SIGTERM when shutting down, forwards to orchestrator
- Orchestrator manages all child processes (workers)
- Coordinator doesn't directly spawn writers/inference workers (orchestrator does)

Monitoring & Metrics:
- Emit timing metrics: writer_duration, hot_cache_duration, total_cycle_duration
- Emit counters: writer_success, writer_failure, hot_cache_success, hot_cache_failure
- Emit gauges: cycle_id (for log correlation), worker_count, queue_size (from PersistenceQueue)

Concurrency & Resource Management:
- Max concurrent workers: configured (default: 4)
- Hot cache: 1 worker (must finish before returning)
- Warm cache: 2-3 workers (concurrent)
- Cold cache: remaining workers (concurrent)
- All workers share: Redis cache, PersistenceQueue
- No contention: separate DuckDB files (meteo, pm25, landsat8, sentinel2, predictions)

Future Enhancements:
- Dynamic hot/warm/cold distribution based on query patterns
- Worker health monitoring (restart failed workers)
- Metrics export (Prometheus)
- Incremental cache updates (only update changed grids)
- Model versioning (support A/B testing)
"""

import sys
import os
import argparse
import logging
import json
import uuid
import signal
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
import pytz

# APScheduler imports
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Project imports
from takshashila.core.config import (
    SCHEDULE_HOUR, SCHEDULE_MINUTE, WRITER_START_OFFSET_MINUTES,
    WRITER_JOB_TIMEOUT_SECONDS, HOT_CACHE_TIMEOUT_SECONDS,
    MAX_WRITER_JOB_RETRIES, MAX_HOT_CACHE_RETRIES,
    NUM_INFERENCE_WORKERS_HOT, NUM_INFERENCE_WORKERS_WARM, NUM_INFERENCE_WORKERS_COLD,
    GRID_BATCH_SIZE_HOT, GRID_BATCH_SIZE_WARM, GRID_BATCH_SIZE_COLD,
    REDIS_HOST, REDIS_PORT, DUCKDB_GRID_PATH, ORCHESTRATOR_LOG_FILE, WORKER_LOG_DIR
)


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config():
    """Load configuration from takshashila.core.config with environment overrides.
    
    Configuration comes from takshashila/core/config.py with optional
    environment variable overrides.
    
    Priority:
    1. Environment variables (KASHI_*)
    2. Defaults from takshashila.core.config
    
    Returns:
        dict: Configuration dictionary with all settings
    """
    config = {
        "schedule_hour": int(os.environ.get("KASHI_SCHEDULE_HOUR", SCHEDULE_HOUR)),
        "schedule_minute": int(os.environ.get("KASHI_SCHEDULE_MINUTE", SCHEDULE_MINUTE)),
        "writer_timeout_seconds": int(os.environ.get("KASHI_WRITER_TIMEOUT", WRITER_JOB_TIMEOUT_SECONDS)),
        "hot_cache_timeout_seconds": int(os.environ.get("KASHI_HOT_CACHE_TIMEOUT", HOT_CACHE_TIMEOUT_SECONDS)),
        "max_writer_retries": int(os.environ.get("KASHI_MAX_WRITER_RETRIES", MAX_WRITER_JOB_RETRIES)),
        "max_hot_cache_retries": int(os.environ.get("KASHI_MAX_HOT_RETRIES", MAX_HOT_CACHE_RETRIES)),
        "cache_host": os.environ.get("KASHI_CACHE_HOST", REDIS_HOST),
        "cache_port": int(os.environ.get("KASHI_CACHE_PORT", REDIS_PORT)),
        "num_hot_workers": int(os.environ.get("KASHI_NUM_HOT_WORKERS", NUM_INFERENCE_WORKERS_HOT)),
        "num_warm_workers": int(os.environ.get("KASHI_NUM_WARM_WORKERS", NUM_INFERENCE_WORKERS_WARM)),
        "num_cold_workers": int(os.environ.get("KASHI_NUM_COLD_WORKERS", NUM_INFERENCE_WORKERS_COLD)),
        "grid_batch_size_hot": int(os.environ.get("KASHI_GRID_BATCH_SIZE_HOT", GRID_BATCH_SIZE_HOT)),
        "grid_batch_size_warm": int(os.environ.get("KASHI_GRID_BATCH_SIZE_WARM", GRID_BATCH_SIZE_WARM)),
        "grid_batch_size_cold": int(os.environ.get("KASHI_GRID_BATCH_SIZE_COLD", GRID_BATCH_SIZE_COLD)),
    }
    return config


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file=None, cycle_id=None):
    """Setup logging with optional cycle_id prefix.
    
    Args:
        log_file: Path to log file (optional, logs to stderr if not provided)
        cycle_id: Optional cycle_id to include in all log messages
    """
    log_file = log_file or ORCHESTRATOR_LOG_FILE
    
    # Create log directory if needed
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    if cycle_id:
        fmt = f"[{cycle_id}] [%(asctime)s] [%(levelname)s] %(message)s"
    else:
        fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),  # Always log to stderr
            logging.FileHandler(log_file)  # Also log to file
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# ORCHESTRATOR CLASS
# ============================================================================

class Orchestrator:
    """Main orchestrator for WriterJob and InferenceJob scheduling."""
    
    def __init__(self, config):
        """Initialize orchestrator.
        
        Args:
            config: Configuration dictionary from load_config()
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scheduler = None
        self.cycle_state = {}  # Track state across cycles
        
        # Initialize APScheduler with UTC timezone
        self.scheduler = BackgroundScheduler(timezone=pytz.UTC)
        self.logger.info("APScheduler initialized with UTC timezone")
    
    def register_jobs(self):
        """Register WriterJob and InferenceJob with scheduler.
        
        WriterJob: Scheduled @ t=-2m (2 minutes before hour)
        InferenceJob: Scheduled @ t=0m (top of hour)
        """
        schedule_hour = self.config["schedule_hour"]
        schedule_minute = self.config["schedule_minute"]
        
        # WriterJob @ t=-2m
        writer_hour = schedule_hour
        writer_minute = (schedule_minute - 2) % 60
        if schedule_minute < 2:
            writer_hour = (schedule_hour - 1) % 24
        
        self.scheduler.add_job(
            self.run_writer_job,
            CronTrigger(hour=writer_hour, minute=writer_minute, timezone=pytz.UTC),
            id="writer_job",
            name="WriterJob (t=-2m)"
        )
        self.logger.info(f"Registered WriterJob @ {writer_hour:02d}:{writer_minute:02d} UTC")
        
        # InferenceJob @ t=0m
        self.scheduler.add_job(
            self.run_inference_job,
            CronTrigger(hour=schedule_hour, minute=schedule_minute, timezone=pytz.UTC),
            id="inference_job",
            name="InferenceJob (t=0m)"
        )
        self.logger.info(f"Registered InferenceJob @ {schedule_hour:02d}:{schedule_minute:02d} UTC")
    
    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            self.logger.info("Orchestrator started, waiting for scheduled jobs")
    
    def shutdown(self):
        """Gracefully shutdown the scheduler."""
        self.logger.info("Shutting down orchestrator...")
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
        self.logger.info("Orchestrator shutdown complete")
    
    def run_writer_job(self):
        """Execute WriterJob (placeholder, implements TODO)."""
        # TODO: Implement run_writer_job() function
        pass
    
    def run_inference_job(self):
        """Execute InferenceJob - spawns workers in stages (hot → warm → cold).
        
        Stage 1 (Hot Cache, Synchronous):
        - Spawn 1 worker for first 5K grids
        - Wait for completion (blocking)
        - Must finish before returning to coordinator
        
        Stage 2 (Warm Cache, Async):
        - Spawn warm cache workers (non-blocking)
        - Return immediately after spawning
        
        Stage 3 (Cold Cache, Async):
        - Spawn cold cache workers (non-blocking, background)
        - Return immediately after spawning
        """
        cycle_id = str(uuid.uuid4())
        self.logger.info(f"InferenceJob starting (cycle_id: {cycle_id})")
        
        try:
            # Get current timestamp (rounded to hour)
            timestamp_hour = datetime.now(pytz.UTC).replace(minute=0, second=0, microsecond=0)
            
            # Stage 1: Hot Cache (Synchronous, must complete before returning)
            self.logger.info(f"InferenceJob Stage 1 (Hot Cache) starting")
            grid_batch_start = 0
            grid_batch_end = self.config.get("grid_batch_size_hot", GRID_BATCH_SIZE_HOT)
            
            try:
                result = self._spawn_inference_worker(
                    cycle_id=cycle_id,
                    timestamp_hour=timestamp_hour.isoformat(),
                    grid_batch_start=grid_batch_start,
                    grid_batch_end=grid_batch_end,
                    wait=True,  # Blocking
                    timeout=self.config.get("hot_cache_timeout_seconds", HOT_CACHE_TIMEOUT_SECONDS)
                )
                
                if result["exit_code"] == 0:
                    self.logger.info(f"InferenceJob hot cache completed successfully")
                else:
                    self.logger.error(f"InferenceJob hot cache failed with exit code {result['exit_code']}")
                    return
            
            except Exception as e:
                self.logger.error(f"InferenceJob hot cache error: {e}")
                return
            
            # Stage 2: Warm Cache (Async, non-blocking)
            self.logger.info(f"InferenceJob Stage 2 (Warm Cache) starting async")
            grid_batch_start = self.config.get("grid_batch_size_hot", GRID_BATCH_SIZE_HOT)
            grid_batch_end = grid_batch_start + self.config.get("grid_batch_size_warm", GRID_BATCH_SIZE_WARM)
            
            Thread(
                target=self._spawn_inference_worker,
                args=(cycle_id, timestamp_hour.isoformat(), grid_batch_start, grid_batch_end),
                kwargs={"wait": False},
                daemon=True
            ).start()
            
            # Stage 3: Cold Cache (Async, background)
            self.logger.info(f"InferenceJob Stage 3 (Cold Cache) starting async")
            grid_batch_start = self.config.get("grid_batch_size_hot", GRID_BATCH_SIZE_HOT) + self.config.get("grid_batch_size_warm", GRID_BATCH_SIZE_WARM)
            # Remaining grids
            
            Thread(
                target=self._spawn_inference_worker,
                args=(cycle_id, timestamp_hour.isoformat(), grid_batch_start, 509209),  # Total grid count
                kwargs={"wait": False},
                daemon=True
            ).start()
            
            self.logger.info(f"InferenceJob cycle completed (hot cache ready for serving)")
        
        except Exception as e:
            self.logger.exception(f"InferenceJob failed: {e}")
    
    def _spawn_inference_worker(self, cycle_id, timestamp_hour, grid_batch_start, grid_batch_end, wait=True, timeout=None):
        """Spawn a single inference worker subprocess.
        
        Args:
            cycle_id: Unique cycle ID
            timestamp_hour: ISO format timestamp
            grid_batch_start: Starting grid index
            grid_batch_end: Ending grid index
            wait: Whether to wait for completion (blocking)
            timeout: Timeout in seconds (if wait=True)
        
        Returns:
            dict: Result with exit_code, stdout, stderr
        """
        cmd = [
            sys.executable, "-m", "takshashila.distributed.worker.inference_job_worker",
            "--cycle-id", cycle_id,
            "--timestamp-hour", timestamp_hour,
            "--grid-batch-start", str(grid_batch_start),
            "--grid-batch-end", str(grid_batch_end),
            "--cache-host", self.config.get("cache_host", "localhost"),
            "--cache-port", str(self.config.get("cache_port", 6379))
        ]
        
        self.logger.info(f"Spawning worker: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if wait:
                stdout, stderr = process.communicate(timeout=timeout)
                result = {
                    "exit_code": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr
                }
                self.logger.info(f"Worker completed with exit code {process.returncode}")
                # Log worker output
                if stderr:
                    self.logger.info(f"Worker stderr:\n{stderr}")
                if stdout:
                    self.logger.info(f"Worker stdout:\n{stdout}")
                return result
            else:
                # Non-blocking spawn
                self.logger.info(f"Worker spawned (PID: {process.pid})")
                return {"exit_code": 0, "pid": process.pid}
        
        except subprocess.TimeoutExpired:
            process.kill()
            self.logger.error(f"Worker timeout after {timeout} seconds, killed")
            return {"exit_code": 1, "error": "timeout"}
        except Exception as e:
            self.logger.error(f"Failed to spawn worker: {e}")
            return {"exit_code": 1, "error": str(e)}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for orchestrator.
    
    Usage:
        python3 -m takshashila.distributed.worker.orchestrator
    
    Configuration comes from takshashila/core/config.py with optional
    environment variable overrides (KASHI_*).
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Kashi Orchestrator - Batch Inference Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m takshashila.distributed.worker.orchestrator
  
Environment Variables:
  KASHI_SCHEDULE_HOUR       Hour to schedule jobs (default: 0)
  KASHI_SCHEDULE_MINUTE     Minute to schedule jobs (default: 0)
  KASHI_CACHE_HOST          Redis host (default: localhost)
  KASHI_CACHE_PORT          Redis port (default: 6379)
        """
    )
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config()
        logger = setup_logging()
        logger.info("Configuration loaded from takshashila.core.config")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Initialize orchestrator
    try:
        orchestrator = Orchestrator(config)
        orchestrator.register_jobs()
        logger.info(f"Orchestrator initialized (schedule: {config['schedule_hour']:02d}:{config['schedule_minute']:02d} UTC)")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        sys.exit(1)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        orchestrator.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start scheduler
    try:
        orchestrator.start()
        logger.info("Orchestrator running, press Ctrl+C to stop")
        # Keep running until interrupted
        while True:
            signal.pause()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        orchestrator.shutdown()
    except Exception as e:
        logger.exception(f"Unexpected error in orchestrator: {e}")
        orchestrator.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()


# TODO: Implement run_writer_job() function
# TODO: Implement hot cache worker spawning and waiting
# TODO: Implement warm cache workers (async)
# TODO: Implement cold cache workers (async, background)
# TODO: Initialize shared components (Redis, PersistenceQueue)
# TODO: Implement error handling and retries
# TODO: Implement graceful shutdown (SIGTERM)
# TODO: Implement logging with cycle_id
# TODO: Implement cycle state tracking
# TODO: Implement timeout handling for subprocesses
# TODO: Implement health monitoring for workers
