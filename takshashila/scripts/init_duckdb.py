#!/usr/bin/env python3
"""Initialize DuckDB schema and load grid reference data.

Usage:
    python3 takshashika/scripts/init_duckdb.py
    cd takshashika && python3 scripts/init_duckdb.py

Creates:
    - DuckDB file at takshashika/duckdb/predictions.duckdb
    - All schema tables (grids, features_*, predictions)
    - Loads 509,209 grid cells from Mumbai parquet
    - Populates features_spatial from grids
"""
import sys
import logging
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from takshashika.core.schema import initialize_schema
from takshashika.core.config import DUCKDB_PATH, GRID_PARQUET_PATH
from takshashika.core.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Initialize DuckDB schema and load grid data."""
    logger.info("=" * 80)
    logger.info("Initializing DuckDB schema for takshashika Phase 1")
    logger.info("=" * 80)
    
    logger.info(f"DuckDB path: {DUCKDB_PATH}")
    logger.info(f"Grid parquet: {GRID_PARQUET_PATH}")
    
    if DUCKDB_PATH.exists():
        logger.warning(f"DuckDB file already exists: {DUCKDB_PATH}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            logger.info("Aborted")
            return
        
        DUCKDB_PATH.unlink()
        logger.info("Deleted existing DuckDB file")
    
    # Initialize schema
    initialize_schema(DUCKDB_PATH, GRID_PARQUET_PATH)
    
    logger.info("=" * 80)
    logger.info("Schema initialization complete!")
    logger.info(f"DuckDB file created: {DUCKDB_PATH}")
    logger.info(f"File size: {DUCKDB_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run populate_mock_features.py to load test data")
    logger.info("2. Run tests: pytest takshashika/tests/")
    logger.info("")


if __name__ == "__main__":
    main()
