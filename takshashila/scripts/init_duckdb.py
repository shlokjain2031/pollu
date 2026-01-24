#!/usr/bin/env python3
"""Initialize DuckDB schema and load grid reference data.

Usage:
    python3 takshashila/scripts/init_duckdb.py
    cd takshashila && python3 scripts/init_duckdb.py

Creates:
    - DuckDB files in takshashila/duckdb/ (one per table):
      * grid.duckdb
      * pm25_observations.duckdb
      * meteo.duckdb
      * landsat8.duckdb
      * sentinel2.duckdb
      * predictions.duckdb
    - All schema tables in their respective files
    - Loads 509,209 grid cells from Mumbai parquet into grid.duckdb
"""
import sys
import logging
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from takshashila.core.schema import initialize_schema
from takshashila.core.config import (
    DUCKDB_DIR, DUCKDB_GRID_PATH, DUCKDB_PM25_PATH, DUCKDB_METEO_PATH,
    DUCKDB_LANDSAT8_PATH, DUCKDB_SENTINEL2_PATH, DUCKDB_PREDICTIONS_PATH,
    GRID_PARQUET_PATH
)
from takshashila.core.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Initialize DuckDB schema and load grid data."""
    logger.info("=" * 80)
    logger.info("Initializing DuckDB schema for takshashila (separate files per table)")
    logger.info("=" * 80)
    
    logger.info(f"DuckDB directory: {DUCKDB_DIR}")
    logger.info(f"Grid parquet: {GRID_PARQUET_PATH}")
    
    # Check if any files exist
    db_files = [
        DUCKDB_GRID_PATH, DUCKDB_PM25_PATH, DUCKDB_METEO_PATH,
        DUCKDB_LANDSAT8_PATH, DUCKDB_SENTINEL2_PATH, DUCKDB_PREDICTIONS_PATH
    ]
    existing_files = [f for f in db_files if f.exists()]
    
    if existing_files:
        logger.warning(f"Found {len(existing_files)} existing DuckDB files:")
        for f in existing_files:
            logger.warning(f"  - {f}")
        response = input("Overwrite all? (y/N): ")
        if response.lower() != 'y':
            logger.info("Aborted")
            return
        
        # Delete existing files
        for f in existing_files:
            f.unlink()
            logger.info(f"Deleted {f}")
    
    # Ensure directory exists
    DUCKDB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize schema across all files
    initialize_schema(
        grid_path=DUCKDB_GRID_PATH,
        pm25_path=DUCKDB_PM25_PATH,
        meteo_path=DUCKDB_METEO_PATH,
        landsat_path=DUCKDB_LANDSAT8_PATH,
        sentinel_path=DUCKDB_SENTINEL2_PATH,
        predictions_path=DUCKDB_PREDICTIONS_PATH,
        grid_parquet_path=GRID_PARQUET_PATH
    )
    
    logger.info("=" * 80)
    logger.info("Schema initialization complete!")
    logger.info(f"DuckDB files created in: {DUCKDB_DIR}")
    
    total_size = sum(f.stat().st_size for f in db_files if f.exists()) / 1024 / 1024
    logger.info(f"Total size: {total_size:.2f} MB")
    
    logger.info("")
    logger.info("Files created:")
    for f in db_files:
        if f.exists():
            size_mb = f.stat().st_size / 1024 / 1024
            logger.info(f"  - {f.name}: {size_mb:.2f} MB")
    
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run populate_mock_features.py to load test data")
    logger.info("2. Run tests: pytest takshashila/tests/")
    logger.info("")


if __name__ == "__main__":
    main()
