"""
Phase 2a Python Worker

Reads jobs from stdin (JSON lines), calls Phase 1 inference engine,
returns results to stdout (JSON lines).
"""

import sys
import json
import logging
import os
from pathlib import Path
from datetime import datetime

# Set up paths - Phase 1 is a package, we need to add parent directory
# Use absolute path based on this file's location
WORKER_SCRIPT = Path(__file__).resolve()
WORKER_DIR = WORKER_SCRIPT.parent
TAKSHASHILA_ROOT = WORKER_DIR.parent
POLLU_ROOT = TAKSHASHILA_ROOT.parent

sys.path.insert(0, str(TAKSHASHILA_ROOT))

# Now import Phase 1 as a package
from core.inference import InferenceEngine

# Get worker ID from environment
WORKER_ID = os.environ.get("WORKER_ID", "worker-unknown")

# Use absolute paths based on actual file locations
DUCKDB_PATH = TAKSHASHILA_ROOT / "duckdb" / "predictions.duckdb"
MODEL_PATH = POLLU_ROOT / "dwarka" / "models" / "prod_v1" / "model.txt"
FEATURE_SCHEMA_PATH = POLLU_ROOT / "dwarka" / "models" / "prod_v1" / "feature_schema.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=f'[{WORKER_ID}] %(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Log to stderr to keep stdout clean for JSON
)
logger = logging.getLogger(__name__)


class Worker:
    """Python worker that processes jobs from Go coordinator."""
    
    def __init__(self):
        """Initialize the worker with Phase 1 inference engine."""
        # Validate paths exist before initializing
        if not DUCKDB_PATH.exists():
            raise FileNotFoundError(f"DuckDB file not found: {DUCKDB_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not FEATURE_SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Feature schema file not found: {FEATURE_SCHEMA_PATH}")
        
        logger.info(f"Initializing worker (DB: {DUCKDB_PATH}, Model: {MODEL_PATH}, Schema: {FEATURE_SCHEMA_PATH})")
        
        try:
            self.engine = InferenceEngine(
                db_path=str(DUCKDB_PATH),
                model_path=str(MODEL_PATH),
                feature_schema_path=str(FEATURE_SCHEMA_PATH)
            )
            logger.info("Worker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize worker: {e}")
            raise
    
    def process_job(self, job):
        """
        Process a single job.
        
        Args:
            job: Dict with keys: grid_id, timestamp_hour, request_id
                timestamp_hour is an ISO 8601 string (e.g., "2026-01-19T16:00:00Z")
            
        Returns:
            Dict with keys: request_id, pm25, error, worker_id
        """
        request_id = job.get("request_id", "unknown")
        grid_id = job.get("grid_id")
        timestamp_hour_str = job.get("timestamp_hour")
        
        logger.info(f"Processing job {request_id}: grid_id={grid_id}, timestamp={timestamp_hour_str}")
        
        try:
            # Parse timestamp from ISO 8601 string to datetime
            timestamp_hour = datetime.fromisoformat(timestamp_hour_str.replace("Z", "+00:00"))
            
            # Call Phase 1 inference engine (returns float directly)
            pm25_value = self.engine.predict(grid_id, timestamp_hour)
            
            logger.info(f"Job {request_id} completed: pm25={pm25_value:.2f}")
            
            return {
                "request_id": request_id,
                "pm25": pm25_value,
                "error": "",
                "worker_id": WORKER_ID
            }
            
        except Exception as e:
            logger.error(f"Job {request_id} failed: {e}")
            
            # Classify error type
            error_type, error_msg = self._classify_error(e, grid_id, timestamp_hour_str)
            
            return {
                "request_id": request_id,
                "pm25": 0.0,
                "error": error_msg,
                "error_type": error_type,
                "worker_id": WORKER_ID
            }
    
    def _classify_error(self, exc, grid_id, timestamp_hour):
        """
        Classify an exception into one of 6 error types.
        
        Returns:
            Tuple of (error_type: str, error_message: str)
        """
        error_str = str(exc).lower()
        
        # InvalidRequest: malformed input
        if "invalid" in error_str or "malformed" in error_str or "parse" in error_str:
            return ("InvalidRequest", f"Invalid request parameters: {exc}")
        
        # GridNotFound: grid doesn't exist in database
        if "grid" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return ("GridNotFound", f"Grid {grid_id} not found")
        
        # DataMissing: data missing for grid/timestamp (PM2.5, meteo, satellite)
        if any(keyword in error_str for keyword in ["no data", "missing", "insufficient", "empty result"]):
            return ("DataMissing", f"Insufficient data for grid {grid_id} at {timestamp_hour}")
        
        # WorkerError: database errors, file I/O, etc.
        if any(keyword in error_str for keyword in ["database", "connection", "file", "i/o", "disk"]):
            return ("WorkerError", f"Worker internal error: {exc}")
        
        # Unknown: anything else
        return ("Unknown", f"Unknown error: {exc}")
    
    def run(self):
        """Main loop: read jobs from stdin, process, write results to stdout."""
        logger.info("Worker starting main loop")
        
        try:
            # Read jobs from stdin (one JSON object per line)
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse job
                    job = json.loads(line)
                    
                    # Process job
                    result = self.process_job(job)
                    
                    # Write result to stdout (one JSON object per line)
                    result_json = json.dumps(result)
                    print(result_json, flush=True)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse job JSON: {e}")
                    error_result = {
                        "request_id": "unknown",
                        "pm25": 0.0,
                        "error": f"Invalid job JSON: {e}",
                        "error_type": "InvalidRequest",
                        "worker_id": WORKER_ID
                    }
                    print(json.dumps(error_result), flush=True)
                    
                except Exception as e:
                    logger.error(f"Unexpected error processing job: {e}")
                    error_result = {
                        "request_id": "unknown",
                        "pm25": 0.0,
                        "error": f"Worker error: {e}",
                        "error_type": "WorkerError",
                        "worker_id": WORKER_ID
                    }
                    print(json.dumps(error_result), flush=True)
                    
        except KeyboardInterrupt:
            logger.info("Worker received shutdown signal")
        except Exception as e:
            logger.error(f"Worker main loop error: {e}")
            raise
        finally:
            logger.info("Worker shutting down")


def main():
    """Entry point for the worker."""
    try:
        worker = Worker()
        worker.run()
    except Exception as e:
        logger.error(f"Worker failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
