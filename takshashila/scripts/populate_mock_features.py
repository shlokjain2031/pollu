#!/usr/bin/env python3
"""Populate DuckDB with mock raw observational data for testing.

Optimized for fast inserts at scale (100k-500k grids).

Populates N unique grid IDs with:
- pm25_observations: hourly values for past 2 days (48 readings per grid)
- meteo: hourly values for past 2 days with 8 raw meteorological variables (48 readings per grid)
- landsat8: single date 5 days ago with raw spectral bands B1-B11
- sentinel2: single date 2 days ago with spectral bands B2, B4, B8, AOT, QA10

Usage:
    # Default: 500k grids
    python3 takshashila/scripts/populate_mock_features.py
    
    # Custom grid count
    NUM_GRIDS=100000 python3 takshashila/scripts/populate_mock_features.py
    
    cd takshashila && python3 scripts/populate_mock_features.py
"""
import sys
import logging
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import random
import duckdb
import time

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from takshashila.core.config import (
    DUCKDB_PM25_PATH, DUCKDB_METEO_PATH, DUCKDB_LANDSAT8_PATH, DUCKDB_SENTINEL2_PATH
)
from takshashila.core.utils import setup_logging, floor_to_hour

setup_logging()
logger = logging.getLogger(__name__)

# Base timestamp for mock data (now) - normalized to floor of hour
# Set this to match the inference hour you're testing (e.g., 14:00:00 UTC for t=14h predictions)
BASE_TIMESTAMP = datetime(
    2026, 1, 25, 14, 0, 0, tzinfo=timezone.utc
)

# Number of unique grid IDs to populate (configurable via env)
NUM_GRIDS = int(os.environ.get("NUM_GRIDS", "5000"))

# Batch size for inserts (tune for memory vs speed)
# Larger batches = faster but more memory. 50k rows = ~2-5MB per table
BATCH_SIZE = 500

GRID_IDS = [f"g{i:08d}" for i in range(1, NUM_GRIDS + 1)]


def generate_pm25_observation(grid_id: str, timestamp: datetime) -> dict:
    """Generate realistic PM2.5 raw observation."""
    # Typical Mumbai PM2.5 range: 30-120 µg/m³
    pm25_value = random.uniform(30, 120)
    return {
        "grid_id": grid_id,
        "timestamp": timestamp,
        "pm25": pm25_value
    }


def generate_meteo_observation(grid_id: str, timestamp: datetime) -> dict:
    """Generate realistic meteorological raw observation (8 variables)."""
    temp = random.uniform(20, 35)
    humidity = random.uniform(40, 90)
    dewpoint = temp - random.uniform(2, 8)
    pressure = random.uniform(1000, 1020)
    wind_speed = random.uniform(0, 15)
    wind_direction = random.uniform(0, 360)
    precip = random.uniform(0, 2)
    cloud_cover = random.uniform(0, 100)
    
    return {
        "grid_id": grid_id,
        "timestamp": timestamp,
        "temperature_2m": temp,
        "relative_humidity_2m": humidity,
        "dewpoint_2m": dewpoint,
        "surface_pressure": pressure,
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_direction,
        "precipitation": precip,
        "cloud_cover": cloud_cover
    }


def generate_landsat8_observation(grid_id: str, timestamp: datetime) -> dict:
    """Generate Landsat8 raw spectral bands (B1-B11)."""
    # Landsat8 bands are digital numbers (0-65535 for 16-bit)
    # Normalized to 0-1 for easier visualization
    return {
        "grid_id": grid_id,
        "timestamp": timestamp,
        "B1": random.uniform(0.01, 0.1),      # Coastal/Aerosol
        "B2": random.uniform(0.02, 0.15),     # Blue
        "B3": random.uniform(0.03, 0.2),      # Green
        "B4": random.uniform(0.02, 0.25),     # Red
        "B5": random.uniform(0.05, 0.4),      # NIR
        "B6": random.uniform(0.1, 0.5),       # SWIR1
        "B7": random.uniform(0.05, 0.3),      # SWIR2
        "B8": random.uniform(0.02, 0.35),     # Panchromatic
        "B9": random.uniform(0.0, 0.05),      # Cirrus
        "B10": random.uniform(0.0, 0.1),      # TIRS1
        "B11": random.uniform(0.0, 0.08)      # TIRS2
    }


def generate_sentinel2_observation(grid_id: str, timestamp: datetime) -> dict:
    """Generate Sentinel2 spectral bands (B2, B4, B8, AOT, QA10)."""
    return {
        "grid_id": grid_id,
        "timestamp": timestamp,
        "B2": random.uniform(0.02, 0.15),     # Blue
        "B4": random.uniform(0.02, 0.25),     # Red
        "B8": random.uniform(0.05, 0.4),      # NIR
        "AOT": random.uniform(0.0, 0.5),      # Aerosol Optical Thickness
        "QA10": random.randint(0, 3000)       # Quality Assessment band
    }


def populate_raw_observations():
    """Populate all raw observation tables with mock data in separate files.
    
    Optimized for fast bulk inserts at scale (100k-500k grids):
    - Batch inserts in chunks (BATCH_SIZE)
    - Pre-allocate lists to reduce memory churn
    - Use executemany for efficient bulk insert
    - Parallel connections (one per table)
    - Progress tracking for long operations
    """
    
    base_hour = BASE_TIMESTAMP.replace(minute=0, second=0, microsecond=0)
    total_grid_chunks = (NUM_GRIDS + BATCH_SIZE - 1) // BATCH_SIZE
    
    # === POPULATE PM2.5 OBSERVATIONS ===
    logger.info(f"Populating pm25_observation: {NUM_GRIDS:,} grids × 48 readings = {NUM_GRIDS*48:,} rows")
    conn_pm25 = duckdb.connect(str(DUCKDB_PM25_PATH))
    try:
        conn_pm25.execute("DELETE FROM pm25_observation")
        
        pm25_total = 0
        for chunk_idx in range(total_grid_chunks):
            start_idx = chunk_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, NUM_GRIDS)
            chunk_grids = GRID_IDS[start_idx:end_idx]
            
            # Pre-allocate for efficiency
            pm25_batch = []
            pm25_batch.extend([
                (grid_id, (base_hour - timedelta(hours=hour_offset)).replace(tzinfo=None), random.uniform(30, 120))
                for grid_id in chunk_grids
                for hour_offset in range(48)
            ])
            
            conn_pm25.executemany(
                "INSERT INTO pm25_observation (grid_id, timestamp, pm25) VALUES (?, ?, ?)",
                pm25_batch
            )
            pm25_total += len(pm25_batch)
            
            if (chunk_idx + 1) % max(1, total_grid_chunks // 10) == 0:
                logger.info(f"  pm25_observation: {pm25_total:,} rows inserted ({(chunk_idx+1)*100//total_grid_chunks}%)")
        
        conn_pm25.commit()
        pm25_count = conn_pm25.execute("SELECT COUNT(*) FROM pm25_observation").fetchone()[0]
        
        if pm25_count <= 5:
            print(f"=== SAMPLE ROWS: pm25_observation ===")
            sample = conn_pm25.execute("SELECT * FROM pm25_observation LIMIT 5").fetchdf()
            print(sample)
            print()
        
        logger.info(f"✓ pm25_observation: {pm25_count:,} rows")
    finally:
        conn_pm25.close()
    
    # === POPULATE METEOROLOGICAL DATA ===
    logger.info(f"Populating meteo: {NUM_GRIDS:,} grids × 48 readings = {NUM_GRIDS*48:,} rows")
    conn_meteo = duckdb.connect(str(DUCKDB_METEO_PATH))
    try:
        conn_meteo.execute("DELETE FROM meteo")
        
        meteo_total = 0
        for chunk_idx in range(total_grid_chunks):
            start_idx = chunk_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, NUM_GRIDS)
            chunk_grids = GRID_IDS[start_idx:end_idx]
            
            meteo_batch = []
            for grid_id in chunk_grids:
                for hour_offset in range(48):
                    timestamp = (base_hour - timedelta(hours=hour_offset)).replace(tzinfo=None)
                    temp = random.uniform(20, 35)
                    humidity = random.uniform(40, 90)
                    dewpoint = temp - random.uniform(2, 8)
                    pressure = random.uniform(1000, 1020)
                    wind_speed = random.uniform(0, 15)
                    wind_direction = random.uniform(0, 360)
                    precip = random.uniform(0, 2)
                    cloud_cover = random.uniform(0, 100)
                    
                    meteo_batch.append((
                        grid_id, timestamp, temp, humidity, dewpoint, pressure,
                        wind_speed, wind_direction, precip, cloud_cover
                    ))
            
            conn_meteo.executemany(
                """INSERT INTO meteo (grid_id, timestamp, temperature_2m, relative_humidity_2m, 
                                       dewpoint_2m, surface_pressure, wind_speed_10m, wind_direction_10m, 
                                       precipitation, cloud_cover) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                meteo_batch
            )
            meteo_total += len(meteo_batch)
            
            if (chunk_idx + 1) % max(1, total_grid_chunks // 10) == 0:
                logger.info(f"  meteo: {meteo_total:,} rows inserted ({(chunk_idx+1)*100//total_grid_chunks}%)")
        
        conn_meteo.commit()
        meteo_count = conn_meteo.execute("SELECT COUNT(*) FROM meteo").fetchone()[0]
        logger.info(f"✓ meteo: {meteo_count:,} rows")
    finally:
        conn_meteo.close()
    
    # === POPULATE LANDSAT8 DATA ===
    logger.info(f"Populating landsat8: {NUM_GRIDS:,} grids × 1 date (5 days ago) = {NUM_GRIDS:,} rows")
    conn_landsat = duckdb.connect(str(DUCKDB_LANDSAT8_PATH))
    try:
        conn_landsat.execute("DELETE FROM landsat8")
        
        landsat_timestamp = (base_hour - timedelta(days=5)).replace(tzinfo=None)
        landsat_total = 0
        
        for chunk_idx in range(total_grid_chunks):
            start_idx = chunk_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, NUM_GRIDS)
            chunk_grids = GRID_IDS[start_idx:end_idx]
            
            landsat_batch = []
            for grid_id in chunk_grids:
                landsat_batch.append((
                    grid_id, landsat_timestamp,
                    random.uniform(0.01, 0.1),   # B1
                    random.uniform(0.02, 0.15),  # B2
                    random.uniform(0.03, 0.2),   # B3
                    random.uniform(0.02, 0.25),  # B4
                    random.uniform(0.05, 0.4),   # B5
                    random.uniform(0.1, 0.5),    # B6
                    random.uniform(0.05, 0.3),   # B7
                    random.uniform(0.02, 0.35),  # B8
                    random.uniform(0.0, 0.05),   # B9
                    random.uniform(0.0, 0.1),    # B10
                    random.uniform(0.0, 0.08)    # B11
                ))
            
            conn_landsat.executemany(
                """INSERT INTO landsat8 (grid_id, timestamp, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                landsat_batch
            )
            landsat_total += len(landsat_batch)
            
            if (chunk_idx + 1) % max(1, total_grid_chunks // 10) == 0:
                logger.info(f"  landsat8: {landsat_total:,} rows inserted ({(chunk_idx+1)*100//total_grid_chunks}%)")
        
        conn_landsat.commit()
        landsat_count = conn_landsat.execute("SELECT COUNT(*) FROM landsat8").fetchone()[0]
        logger.info(f"✓ landsat8: {landsat_count:,} rows")
    finally:
        conn_landsat.close()
    
    # === POPULATE SENTINEL2 DATA ===
    logger.info(f"Populating sentinel2: {NUM_GRIDS:,} grids × 1 date (2 days ago) = {NUM_GRIDS:,} rows")
    conn_sentinel = duckdb.connect(str(DUCKDB_SENTINEL2_PATH))
    try:
        conn_sentinel.execute("DELETE FROM sentinel2")
        
        sentinel_timestamp = (base_hour - timedelta(days=2)).replace(tzinfo=None)
        sentinel_total = 0
        
        for chunk_idx in range(total_grid_chunks):
            start_idx = chunk_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, NUM_GRIDS)
            chunk_grids = GRID_IDS[start_idx:end_idx]
            
            sentinel_batch = []
            for grid_id in chunk_grids:
                sentinel_batch.append((
                    grid_id, sentinel_timestamp,
                    random.uniform(0.02, 0.15),   # B2
                    random.uniform(0.02, 0.25),   # B4
                    random.uniform(0.05, 0.4),    # B8
                    random.uniform(0.0, 0.5),     # AOT
                    random.randint(0, 3000)       # QA10
                ))
            
            conn_sentinel.executemany(
                """INSERT INTO sentinel2 (grid_id, timestamp, B2, B4, B8, AOT, QA10)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                sentinel_batch
            )
            sentinel_total += len(sentinel_batch)
            
            if (chunk_idx + 1) % max(1, total_grid_chunks // 10) == 0:
                logger.info(f"  sentinel2: {sentinel_total:,} rows inserted ({(chunk_idx+1)*100//total_grid_chunks}%)")
        
        conn_sentinel.commit()
        sentinel_count = conn_sentinel.execute("SELECT COUNT(*) FROM sentinel2").fetchone()[0]
        logger.info(f"✓ sentinel2: {sentinel_count:,} rows")
    finally:
        conn_sentinel.close()
    
    # Print summary
    logger.info("=" * 80)
    logger.info("Raw observation data populated successfully!")
    logger.info(f"  pm25_observation: {pm25_count:,} rows ({NUM_GRIDS:,} grids × 48 hours)")
    logger.info(f"  meteo: {meteo_count:,} rows ({NUM_GRIDS:,} grids × 48 hours)")
    logger.info(f"  landsat8: {landsat_count:,} rows ({NUM_GRIDS:,} grids × 1 date)")
    logger.info(f"  sentinel2: {sentinel_count:,} rows ({NUM_GRIDS:,} grids × 1 date)")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("Populating mock raw observations for Phase 1 testing")
    logger.info(f"Configuration: {NUM_GRIDS:,} grids, {BATCH_SIZE:,} batch size")
    logger.info("=" * 80)
    
    # Check all files exist
    required_files = [DUCKDB_PM25_PATH, DUCKDB_METEO_PATH, DUCKDB_LANDSAT8_PATH, DUCKDB_SENTINEL2_PATH]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        logger.error("Missing DuckDB files:")
        for f in missing_files:
            logger.error(f"  - {f}")
        logger.error("Run init_duckdb.py first!")
        return 1
    
    start_time = time.time()
    populate_raw_observations()
    elapsed = time.time() - start_time
    
    total_rows = (NUM_GRIDS * 48) + (NUM_GRIDS * 48) + NUM_GRIDS + NUM_GRIDS
    logger.info("")
    logger.info(f"⏱ Completed in {elapsed:.1f} seconds")
    logger.info(f"📊 {total_rows:,} total rows inserted")
    logger.info(f"🚀 {total_rows/elapsed:,.0f} rows/second throughput")
    logger.info("")
    logger.info("Next step: Run tests with pytest takshashila/tests/ -v")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
