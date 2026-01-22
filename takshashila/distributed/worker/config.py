"""
Phase 2a Worker Configuration

Hardcoded defaults for the Python worker.
"""

import os
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CORE_PATH = PROJECT_ROOT / "core"

# Add Phase 1 to Python path (so we can import inference module)
sys.path.insert(0, str(CORE_PATH))

DB_PATH = CORE_PATH / "takshashila.db"
MODEL_PATH = CORE_PATH.parent / "artifacts" / "v1" / "aqi" / "prod_v1.txt"

# Worker settings
WORKER_ID = os.environ.get("WORKER_ID", "worker-unknown")

# Database settings (for Phase 2b)
DB_POOL_SIZE = 4
