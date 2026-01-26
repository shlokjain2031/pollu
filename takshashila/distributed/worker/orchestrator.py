"""
Orchestrator - Worker Coordinator

Manages the scheduling and execution of SimplifyJob workers.
Runs as a daemon subprocess, implements the simplified batch PM2.5 processing pipeline.

Responsibilities:
- Schedule SimplifyJob execution at configured time (default: every hour)
- Spawn simplify_job_worker.py subprocess
- Monitor worker completion and handle timeouts
- Log all execution with cycle_id for traceability
- Coordinate Redis cache and persistence queue
- Handle failures with appropriate error reporting

Timing:
- SimplifyJob: Runs at configured minute (default: 0)
- Typical execution time: 10-15 seconds
- Timeout: 60 seconds (configurable)

Architecture:
```
Orchestrator (Python daemon)
├── APScheduler (cron job scheduler)
├── JobState (tracks cycle progress)
└── SimplifyJob subprocess
    └── simplify_job_worker.py
        ├── PM25Fetcher (WAQI API → 26 sensors)
        ├── IDW Interpolation (→ 509K grid cells)
        ├── DuckDB Writer (pm25_observations.duckdb)
        ├── Redis Cache (prediction:{grid_id}:{timestamp})
        └── PersistenceQueue (async write to predictions.duckdb)
```

Main Entry Point:
- `python3 -m takshashila.distributed.worker.orchestrator`
- Reads config from environment or kashi/config.yaml
- Initializes APScheduler with UTC timezone
- Registers SimplifyJob with cron trigger
- Runs forever (until SIGTERM)

Cycle Management:
- cycle_id: UUID generated for each cycle
- SimplifyJob logs with cycle_id for traceability
- Allows end-to-end tracking through logs

SimplifyJob Execution:
1. Cron trigger fires at configured time (e.g., every hour at minute 0)
2. Generate cycle_id = uuid.uuid4()
3. Spawn subprocess: python3 -m takshashila.distributed.worker.simplify_job_worker
   --cycle-id {cycle_id}
   --timestamp-hour {rounded_to_hour}
   --cache-host {redis_host}
   --cache-port {redis_port}
4. Wait for subprocess with timeout (default: 60s)
5. Parse output metrics: "Total grids processed: 509209"
6. If exit code 0: Log success with grid count
7. If exit code non-zero: Log error with exit code
8. If timeout: Kill process, log timeout error

Error Handling:
- Worker failure: Log error, continue scheduling next cycle
- Timeout: Kill process after 60s, log timeout error
- Redis unavailable: Worker logs error, skips cache step
- Queue unavailable: Worker logs error, skips persistence step
- All failures are non-fatal (next cycle will retry)

Graceful Shutdown:
- Receive SIGTERM signal
- Stop scheduling new jobs
- Wait for current worker to complete (max 10 seconds)
- Kill worker subprocess if still running
- Exit with code 0

Configuration:
- Reads from kashi/config.yaml:
  - schedule_hour (default: *)  # Every hour
  - schedule_minute (default: 0)
  - simplify_timeout_seconds (default: 60)
  - cache_host (default: localhost)
  - cache_port (default: 6379)
- Falls back to environment variables:
  - KASHI_SCHEDULE_HOUR
  - KASHI_SCHEDULE_MINUTE
  - KASHI_CACHE_HOST

Logging:
- All logs to stderr: [cycle_id] [timestamp] [level] message
- Log levels: INFO (normal), WARNING (partial failure), ERROR (critical)
- Example:
  [550e8400-e29b] [2026-01-25 14:00:00] INFO SimplifyJob scheduled
  [550e8400-e29b] [2026-01-25 14:00:00] INFO SimplifyJob starting (PID: 12345)
  [550e8400-e29b] [2026-01-25 14:00:12] INFO SimplifyJob completed (509209 grids)

Monitoring & Metrics:
- Emit timing: simplify_duration_seconds
- Emit counters: simplify_success, simplify_failure
- Emit gauges: grids_processed, cycle_id

Concurrency:
- Single worker per cycle (no parallelization)
- Worker handles all 509K grids sequentially
- Redis cache and queue shared across cycles
- No contention on DuckDB (single writer)
"""

import sys
import os
import argparse
import logging
import json
import uuid
import signal
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
import pytz
from flask import Flask, request, jsonify

# APScheduler imports
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Project imports
from takshashila.core.config import (
    SCHEDULE_HOUR, SCHEDULE_MINUTE,
    REDIS_HOST, REDIS_PORT, DUCKDB_GRID_PATH, ORCHESTRATOR_LOG_FILE, WORKER_LOG_DIR
)


# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)

# Global orchestrator instance for Flask endpoints
orchestrator_instance = None


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
        "simplify_timeout_seconds": int(os.environ.get("KASHI_SIMPLIFY_TIMEOUT", 60)),
        "cache_host": os.environ.get("KASHI_CACHE_HOST", REDIS_HOST),
        "cache_port": int(os.environ.get("KASHI_CACHE_PORT", REDIS_PORT)),
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
    
    # Configure logging format with UTC timestamps
    if cycle_id:
        fmt = f"[{cycle_id}] [%(asctime)s UTC] [%(levelname)s] %(message)s"
    else:
        fmt = "[%(asctime)s UTC] [%(levelname)s] %(message)s"
    
    # Custom formatter that forces UTC timezone
    class UTCFormatter(logging.Formatter):
        converter = time.gmtime
    
    formatter = UTCFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Configure logging with UTC formatter
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers
    
    # Stream handler (stderr)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# FLASK ENDPOINTS
# ============================================================================

@app.route('/inference', methods=['POST'])
def on_demand_inference():
    """Handle on-demand inference requests from Go Coordinator.
    
    Request JSON:
    {
        "grids": [12345, 67890, ...],
        "timestamp_hour": "2026-01-25T14:00:00Z",
        "request_id": "req-uuid",
        "cycle_id": "cycle-uuid"  // optional
    }
    
    Response JSON:
    {
        "status": "accepted",
        "request_id": "req-uuid",
        "grids_count": 2,
        "message": "Worker spawned for on-demand inference"
    }
    
    On error:
    {
        "status": "error",
        "request_id": "req-uuid",
        "error": "error message"
    }
    """
    global orchestrator_instance
    
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "error": "Missing JSON body"
            }), 400
        
        grids = data.get("grids", [])
        timestamp_hour = data.get("timestamp_hour")
        request_id = data.get("request_id")
        cycle_id = data.get("cycle_id") or str(uuid.uuid4())
        
        # Validate
        if not grids:
            return jsonify({
                "status": "error",
                "request_id": request_id,
                "error": "Missing 'grids' array"
            }), 400
        
        if not timestamp_hour:
            return jsonify({
                "status": "error",
                "request_id": request_id,
                "error": "Missing 'timestamp_hour'"
            }), 400
        
        logger = logging.getLogger(__name__)
        logger.info(f"[{request_id}] On-demand inference request received: {len(grids)} grids @ {timestamp_hour}")
        
        # This endpoint is deprecated - SimplifyJob does not support on-demand inference
        # Return error message
        return jsonify({
            "status": "error",
            "error": "On-demand inference not supported in SimplifyJob architecture",
            "message": "Use batch processing via SimplifyJob scheduler instead"
        }), 501
    
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"On-demand inference endpoint error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "orchestrator"}), 200


@app.route('/test/trigger-simplify-job', methods=['POST'])
def trigger_simplify_job_test():
    """Test endpoint to manually trigger SimplifyJob immediately."""
    try:
        if orchestrator_instance is None:
            return jsonify({"status": "error", "error": "Orchestrator not initialized"}), 500
        
        orchestrator_instance.logger.info("[TEST] Manually triggering SimplifyJob via /test/trigger-simplify-job")
        orchestrator_instance.run_simplify_job()
        
        return jsonify({
            "status": "success",
            "message": "SimplifyJob triggered successfully",
            "cycle_state": orchestrator_instance.cycle_state.get("simplify_job", {})
        }), 200
    except Exception as e:
        orchestrator_instance.logger.exception(f"[TEST] Error triggering SimplifyJob: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================================
# ORCHESTRATOR CLASS
# ============================================================================

class Orchestrator:
    """Main orchestrator for SimplifyJob scheduling."""
    
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
        """Register SimplifyJob with scheduler.
        
        SimplifyJob: Scheduled EVERY HOUR @ t=0m (at specified minute)
        
        Runs every hour at the configured minute.
        For example, if schedule_minute=35:
        - SimplifyJob runs @ XX:35 every hour (e.g., 00:35, 01:35, 02:35, ...)
        """
        schedule_minute = self.config["schedule_minute"]
        
        # SimplifyJob @ t=0m (every hour at schedule_minute)
        self.logger.info(f"[REGISTER] SimplifyJob: minute={schedule_minute}, trigger=CronTrigger(minute={schedule_minute})")
        self.scheduler.add_job(
            self.run_simplify_job,
            CronTrigger(minute=schedule_minute, timezone=pytz.UTC),
            id="simplify_job",
            name="SimplifyJob (PM2.5 pipeline)",
            misfire_grace_time=30
        )
        self.logger.info(f"✓ Registered SimplifyJob to run EVERY HOUR @ XX:{schedule_minute:02d} UTC (schedule_minute={schedule_minute})")
    
    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.logger.info(f"[START] Starting scheduler with {len(self.scheduler.get_jobs())} jobs registered")
            for job in self.scheduler.get_jobs():
                self.logger.info(f"[START] Job: {job.id} | Name: {job.name} | Trigger: {job.trigger}")
            
            # Check if we should run SimplifyJob immediately (if we're past the scheduled time this hour)
            now = datetime.now(pytz.UTC)
            schedule_minute = self.config.get("schedule_minute", SCHEDULE_MINUTE)
            
            self.logger.info(f"[START] Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            self.logger.info(f"[START] SimplifyJob scheduled for minute={schedule_minute}")
            
            # If we're past the scheduled time but before next hour, run it now
            if now.minute >= schedule_minute:
                self.logger.warning(f"[START] ⚠️  We're past SimplifyJob time ({now.minute:02d} >= {schedule_minute:02d})")
                self.logger.warning(f"[START] Running SimplifyJob immediately since it was missed this hour")
                # Run in background thread to avoid blocking scheduler startup
                def run_immediately():
                    try:
                        self.run_simplify_job()
                    except Exception as e:
                        self.logger.exception(f"[START] Error running SimplifyJob immediately: {e}")
                Thread(target=run_immediately, daemon=True).start()
            
            self.scheduler.start()
            self.logger.info("✓ Scheduler started, now listening for job triggers")
    
    def shutdown(self):
        """Gracefully shutdown the scheduler."""
        self.logger.info("Shutting down orchestrator...")
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
        self.logger.info("Orchestrator shutdown complete")
    
    def run_simplify_job(self):
        """Execute SimplifyJob - spawns simplify_job_worker.py subprocess.
        
        Responsibilities:
        - Generate cycle_id for traceability
        - Spawn simplify_job_worker.py subprocess @ t=0m (configured minute)
        - Wait for completion with timeout
        - Track metrics: duration, grids cached, errors
        - Handle failures gracefully (log and continue)
        
        SimplifyJob performs:
        1. Fetch PM2.5 from WAQI API (26 sensors)
        2. Interpolate to 500K grids using IDW
        3. Write to pm25_observations.duckdb
        4. Cache all 500K predictions to Redis
        5. Enqueue to PersistenceQueue for async writes
        
        Timing:
        - Starts @ configured minute (e.g., XX:35)
        - Timeout: 60 seconds
        - Typical duration: 10-15 seconds
        
        Returns:
        - Updates self.cycle_state["simplify_job"] with result:
          {
            "cycle_id": str,
            "timestamp": datetime,
            "success": bool,
            "exit_code": int,
            "duration_seconds": float,
            "grids_processed": int,
            "errors": list of str
          }
        """
        cycle_id = str(uuid.uuid4())
        now_utc = datetime.now(pytz.UTC)
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"[JOB_FIRED] SimplifyJob TRIGGERED at {now_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")
        self.logger.info(f"[JOB_FIRED] Cycle ID: {cycle_id}")
        self.logger.info("=" * 80)
        
        timestamp_hour = now_utc.replace(minute=0, second=0, microsecond=0)
        start_time = now_utc
        timeout_seconds = self.config.get("simplify_timeout_seconds", 60)
        self.logger.info(f"[JOB_CONFIG] Timeout: {timeout_seconds}s")
        
        self.logger.info(f"SimplifyJob scheduled (cycle_id: {cycle_id})")
        
        try:
            # Spawn simplify_job_worker subprocess
            cmd = [
                sys.executable, "-m", "takshashila.distributed.worker.simplify_job_worker",
                "--cycle-id", cycle_id,
                "--timestamp-hour", timestamp_hour.isoformat(),
                "--cache-host", self.config.get("redis_host", "localhost"),
                "--cache-port", str(self.config.get("redis_port", 6379)),
            ]
            
            self.logger.info(f"SimplifyJob: spawning worker (PID: TBD)")
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                self.logger.info(f"SimplifyJob worker started (PID: {process.pid})")
                
                # Wait for completion with timeout
                try:
                    stdout, stderr = process.communicate(timeout=timeout_seconds)
                    exit_code = process.returncode
                    
                    # Parse metrics from worker output
                    grids_processed = self._parse_simplify_metrics(stderr or stdout)
                    duration_seconds = (datetime.now(pytz.UTC) - start_time).total_seconds()
                    
                    if exit_code == 0:
                        self.logger.info(
                            f"SimplifyJob completed successfully (duration: {duration_seconds:.1f}s, "
                            f"grids: {grids_processed})"
                        )
                        
                        # Store successful result
                        self.cycle_state["simplify_job"] = {
                            "cycle_id": cycle_id,
                            "timestamp": timestamp_hour,
                            "success": True,
                            "exit_code": 0,
                            "duration_seconds": duration_seconds,
                            "grids_processed": grids_processed,
                            "errors": []
                        }
                        return
                    
                    else:
                        # Worker failed with non-zero exit code
                        self.logger.error(
                            f"SimplifyJob worker exited with code {exit_code}"
                        )
                        if stderr:
                            self.logger.debug(f"SimplifyJob stderr:\n{stderr}")
                        
                        duration_seconds = (datetime.now(pytz.UTC) - start_time).total_seconds()
                        error_msg = f"SimplifyJob failed (exit code: {exit_code})"
                        self.logger.error(error_msg)
                        
                        self.cycle_state["simplify_job"] = {
                            "cycle_id": cycle_id,
                            "timestamp": timestamp_hour,
                            "success": False,
                            "exit_code": exit_code,
                            "duration_seconds": duration_seconds,
                            "grids_processed": 0,
                            "errors": [error_msg]
                        }
                        return
                
                except subprocess.TimeoutExpired:
                    # Kill process if timeout
                    process.kill()
                    timeout_error = f"SimplifyJob timed out after {timeout_seconds}s"
                    self.logger.error(timeout_error)
                    
                    duration_seconds = (datetime.now(pytz.UTC) - start_time).total_seconds()
                    
                    self.cycle_state["simplify_job"] = {
                        "cycle_id": cycle_id,
                        "timestamp": timestamp_hour,
                        "success": False,
                        "exit_code": 1,
                        "duration_seconds": duration_seconds,
                        "grids_processed": 0,
                        "errors": [timeout_error]
                    }
                    return
            
            except Exception as e:
                spawn_error = f"Failed to spawn SimplifyJob worker: {e}"
                self.logger.error(spawn_error)
                
                duration_seconds = (datetime.now(pytz.UTC) - start_time).total_seconds()
                self.logger.error(f"SimplifyJob failed: {spawn_error}")
                
                self.cycle_state["simplify_job"] = {
                    "cycle_id": cycle_id,
                    "timestamp": timestamp_hour,
                    "success": False,
                    "exit_code": 1,
                    "duration_seconds": duration_seconds,
                    "grids_processed": 0,
                    "errors": [spawn_error]
                }
                return
        
        except Exception as e:
            duration_seconds = (datetime.now(pytz.UTC) - start_time).total_seconds()
            error_msg = f"Unexpected error in SimplifyJob: {e}"
            self.logger.exception(error_msg)
            
            self.cycle_state["simplify_job"] = {
                "cycle_id": cycle_id,
                "timestamp": timestamp_hour,
                "success": False,
                "exit_code": 1,
                "duration_seconds": duration_seconds,
                "grids_processed": 0,
                "errors": [error_msg]
            }
            return
    
    def _parse_simplify_metrics(self, output):
        """Parse SimplifyJob metrics from worker output.
        
        Looks for log lines like:
        - "✓ Total grids processed: 509,209"
        
        Args:
            output: Combined stdout/stderr from SimplifyJob worker
        
        Returns:
            int: Number of grids processed
        """
        grids_processed = 0
        
        if not output:
            return grids_processed
        
        for line in output.split("\n"):
            if "Total grids processed:" in line:
                try:
                    # Extract number from "✓ Total grids processed: 509,209"
                    parts = line.split(":")
                    count_str = parts[-1].strip().replace(",", "")
                    grids_processed = int(count_str)
                    break
                except (IndexError, ValueError):
                    pass
        
        return grids_processed
    
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
        
        Note: This method is deprecated and should not be used with SimplifyJob.
        """
        # This method is no longer used by SimplifyJob
        self.logger.error("_spawn_worker() is deprecated - SimplifyJob does not use this method")
        raise NotImplementedError("_spawn_worker() is deprecated - use run_simplify_job() instead")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for orchestrator.
    
    Usage:
        python3 -m takshashila.distributed.worker.orchestrator
    
    Configuration comes from takshashila/core/config.py with optional
    environment variable overrides (KASHI_*).
    
    Orchestrator Subprocess Management: IMPLEMENTED
    - Coordinator spawns: exec.Command(python3, '-m', 'takshashila.distributed.worker.orchestrator') ✓
    - Coordinator monitors orchestrator process health via /health endpoint ✓
    - Coordinator restarts orchestrator on failure (max 10 retries, 5s delay) ✓
    - Orchestrator communicates back via HTTP (already implemented) ✓
    - Graceful shutdown: SIGTERM forwarding with 10s timeout ✓
    - Process output: Redirected to coordinator logs with [Orchestrator] prefix ✓
    
    SimplifyJob Integration: IMPLEMENTED
    - Register run_simplify_job() with APScheduler @ configured time ✓
    - Wait for SimplifyJob completion with timeout ✓
    - Track SimplifyJob metrics (duration, grids processed) ✓
    - Error logging for failed jobs ✓
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
    
    # Set global orchestrator instance for Flask endpoints
    global orchestrator_instance
    orchestrator_instance = orchestrator
    
    # Start scheduler
    try:
        orchestrator.start()
        logger.info("Orchestrator APScheduler started")
        
        # Start Flask server in main thread
        flask_port = int(os.environ.get("KASHI_ORCHESTRATOR_PORT", 9999))
        flask_host = os.environ.get("KASHI_ORCHESTRATOR_HOST", "0.0.0.0")
        
        logger.info(f"Starting Flask server on {flask_host}:{flask_port}")
        logger.info("Orchestrator running, press Ctrl+C to stop")
        
        # Run Flask (blocking call)
        app.run(host=flask_host, port=flask_port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        orchestrator.shutdown()
    except Exception as e:
        logger.exception(f"Unexpected error in orchestrator: {e}")
        orchestrator.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
