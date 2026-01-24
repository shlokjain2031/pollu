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
DISTRIBUTED_DIR = WORKER_DIR.parent
TAKSHASHILA_ROOT = DISTRIBUTED_DIR.parent  # Go up 2 levels: worker -> distributed -> takshashila
POLLU_ROOT = TAKSHASHILA_ROOT.parent

# Add Takshashila to path so we can import core as a package
# This is critical for subprocess spawning
sys.path.insert(0, str(TAKSHASHILA_ROOT))

# Now import Phase 1 as a package
from core.inference import InferenceEngine

# Get worker ID from environment
WORKER_ID = os.environ.get("WORKER_ID", "worker-unknown")

# Use absolute paths based on actual file locations
DUCKDB_DIR = TAKSHASHILA_ROOT / "duckdb"
DUCKDB_GRID_PATH = DUCKDB_DIR / "grid.duckdb"
DUCKDB_PM25_PATH = DUCKDB_DIR / "pm25_observations.duckdb"
DUCKDB_METEO_PATH = DUCKDB_DIR / "meteo.duckdb"
DUCKDB_LANDSAT8_PATH = DUCKDB_DIR / "landsat8.duckdb"
DUCKDB_SENTINEL2_PATH = DUCKDB_DIR / "sentinel2.duckdb"
DUCKDB_PREDICTIONS_PATH = DUCKDB_DIR / "predictions.duckdb"
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
        required_files = [
            DUCKDB_GRID_PATH, DUCKDB_PM25_PATH, DUCKDB_METEO_PATH,
            DUCKDB_LANDSAT8_PATH, DUCKDB_SENTINEL2_PATH, DUCKDB_PREDICTIONS_PATH,
            MODEL_PATH, FEATURE_SCHEMA_PATH
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        logger.info(f"Initializing worker (DB dir: {DUCKDB_DIR}, Model: {MODEL_PATH}, Schema: {FEATURE_SCHEMA_PATH})")
        
        try:
            # Build db_paths dictionary
            db_paths = {
                "grid": str(DUCKDB_GRID_PATH),
                "pm25_observation": str(DUCKDB_PM25_PATH),
                "meteo": str(DUCKDB_METEO_PATH),
                "landsat8": str(DUCKDB_LANDSAT8_PATH),
                "sentinel2": str(DUCKDB_SENTINEL2_PATH),
                "predictions": str(DUCKDB_PREDICTIONS_PATH)
            }
            
            self.engine = InferenceEngine(
                db_paths=db_paths,
                model_path=str(MODEL_PATH),
                feature_schema_path=str(FEATURE_SCHEMA_PATH)
            )
            logger.info("Worker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize worker: {e}")
            raise
    
    def process_message(self, msg):
        """
        Part 1: Route message to appropriate handler based on type field.
        
        Handles:
        - "job": Regular inference job
        - "health_check": Health check ping
        """
        msg_type = msg.get("type")
        
        # Part 1: Handle health check
        if msg_type == "health_check":
            return {
                "type": "health_response",
                "pong": True
            }
        
        # Default: treat as job for backward compatibility
        if msg_type is None or msg_type == "job":
            return self.process_job(msg)
        
        # Unknown message type
        logger.warning(f"Unknown message type: {msg_type}")
        return {
            "request_id": msg.get("request_id", "unknown"),
            "pm25": 0.0,
            "error": f"Unknown message type: {msg_type}",
            "error_type": "InvalidRequest",
            "worker_id": WORKER_ID
        }
    
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
                    # Parse message (job or health check)
                    msg = json.loads(line)
                    
                    # Part 1: Route message based on type field
                    result = self.process_message(msg)
                    
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
