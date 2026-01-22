#!/usr/bin/env python3
"""Populate DuckDB with mock raw observational data for testing.

Populates 1000 unique grid IDs with:
- pm25_observations: hourly values for past 7 days (168 readings per grid)
- meteo: hourly values for past 7 days with 8 raw meteorological variables (168 readings per grid)
- landsat8: single date 5 days ago with raw spectral bands B1-B11
- sentinel2: single date 2 days ago with spectral bands B2, B4, B8, AOT, QA10

Usage:
    python3 takshashika/scripts/populate_mock_features.py
    cd takshashika && python3 scripts/populate_mock_features.py
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
import random
import duckdb

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from takshashika.core.config import DUCKDB_PATH
from takshashika.core.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Base timestamp for mock data (now)
BASE_TIMESTAMP = datetime(2026, 1, 21, 12, 0, 0, tzinfo=timezone.utc)

# Number of unique grid IDs to populate
NUM_GRIDS = 1000

# Generate 1000 unique grid IDs
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


def populate_raw_observations(db_path: Path):
    """Populate all raw observation tables with mock data."""
    conn = duckdb.connect(str(db_path))
    
    try:
        # Clear existing mock data
        logger.info("Clearing existing observation data...")
        conn.execute("DELETE FROM pm25_observation")
        conn.execute("DELETE FROM meteo")
        conn.execute("DELETE FROM landsat8")
        conn.execute("DELETE FROM sentinel2")
        
        # === POPULATE PM2.5 OBSERVATIONS ===
        # Hourly values for past 7 days (168 readings per grid)
        logger.info(f"Populating pm25_observation: {NUM_GRIDS} grids × 168 hourly readings...")
        pm25_batch = []
        for grid_id in GRID_IDS:
            for hour_offset in range(168):  # 7 days × 24 hours
                timestamp = BASE_TIMESTAMP - timedelta(hours=hour_offset)
                obs = generate_pm25_observation(grid_id, timestamp)
                pm25_batch.append((obs["grid_id"], obs["timestamp"], obs["pm25"]))
        
        conn.executemany(
            "INSERT INTO pm25_observation (grid_id, timestamp, pm25) VALUES (?, ?, ?)",
            pm25_batch
        )
        conn.commit()
        pm25_count = conn.execute("SELECT COUNT(*) FROM pm25_observation").fetchone()[0]
        logger.info(f"  pm25_observation: {pm25_count} rows")
        
        # === POPULATE METEOROLOGICAL DATA ===
        # Hourly values for past 7 days (168 readings per grid)
        logger.info(f"Populating meteo: {NUM_GRIDS} grids × 168 hourly readings...")
        meteo_batch = []
        for grid_id in GRID_IDS:
            for hour_offset in range(168):  # 7 days × 24 hours
                timestamp = BASE_TIMESTAMP - timedelta(hours=hour_offset)
                obs = generate_meteo_observation(grid_id, timestamp)
                meteo_batch.append((
                    obs["grid_id"],
                    obs["timestamp"],
                    obs["temperature_2m"],
                    obs["relative_humidity_2m"],
                    obs["dewpoint_2m"],
                    obs["surface_pressure"],
                    obs["wind_speed_10m"],
                    obs["wind_direction_10m"],
                    obs["precipitation"],
                    obs["cloud_cover"]
                ))
        
        conn.executemany(
            """INSERT INTO meteo (grid_id, timestamp, temperature_2m, relative_humidity_2m, 
                                   dewpoint_2m, surface_pressure, wind_speed_10m, wind_direction_10m, 
                                   precipitation, cloud_cover) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            meteo_batch
        )
        conn.commit()
        meteo_count = conn.execute("SELECT COUNT(*) FROM meteo").fetchone()[0]
        logger.info(f"  meteo: {meteo_count} rows")
        
        # === POPULATE LANDSAT8 DATA ===
        # Single date: 5 days ago
        logger.info(f"Populating landsat8: {NUM_GRIDS} grids × 1 date (5 days ago)...")
        landsat_timestamp = BASE_TIMESTAMP - timedelta(days=5)
        landsat_batch = []
        for grid_id in GRID_IDS:
            obs = generate_landsat8_observation(grid_id, landsat_timestamp)
            landsat_batch.append((
                obs["grid_id"],
                obs["timestamp"],
                obs["B1"], obs["B2"], obs["B3"], obs["B4"], obs["B5"],
                obs["B6"], obs["B7"], obs["B8"], obs["B9"], obs["B10"], obs["B11"]
            ))
        
        conn.executemany(
            """INSERT INTO landsat8 (grid_id, timestamp, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            landsat_batch
        )
        conn.commit()
        landsat_count = conn.execute("SELECT COUNT(*) FROM landsat8").fetchone()[0]
        logger.info(f"  landsat8: {landsat_count} rows")
        
        # === POPULATE SENTINEL2 DATA ===
        # Single date: 2 days ago
        logger.info(f"Populating sentinel2: {NUM_GRIDS} grids × 1 date (2 days ago)...")
        sentinel_timestamp = BASE_TIMESTAMP - timedelta(days=2)
        sentinel_batch = []
        for grid_id in GRID_IDS:
            obs = generate_sentinel2_observation(grid_id, sentinel_timestamp)
            sentinel_batch.append((
                obs["grid_id"],
                obs["timestamp"],
                obs["B2"],
                obs["B4"],
                obs["B8"],
                obs["AOT"],
                obs["QA10"]
            ))
        
        conn.executemany(
            """INSERT INTO sentinel2 (grid_id, timestamp, B2, B4, B8, AOT, QA10)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            sentinel_batch
        )
        conn.commit()
        sentinel_count = conn.execute("SELECT COUNT(*) FROM sentinel2").fetchone()[0]
        logger.info(f"  sentinel2: {sentinel_count} rows")
        
        # Print summary
        logger.info("=" * 80)
        logger.info("Raw observation data populated successfully!")
        logger.info(f"  pm25_observation: {pm25_count:,} rows ({NUM_GRIDS} grids × 168 hours)")
        logger.info(f"  meteo: {meteo_count:,} rows ({NUM_GRIDS} grids × 168 hours)")
        logger.info(f"  landsat8: {landsat_count:,} rows ({NUM_GRIDS} grids × 1 date [5 days ago])")
        logger.info(f"  sentinel2: {sentinel_count:,} rows ({NUM_GRIDS} grids × 1 date [2 days ago])")
        logger.info("=" * 80)
    
    finally:
        conn.close()


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("Populating mock raw observations for Phase 1 testing")
    logger.info("=" * 80)
    
    if not DUCKDB_PATH.exists():
        logger.error(f"DuckDB file not found: {DUCKDB_PATH}")
        logger.error("Run reinitialize_duckdb.py first!")
        return 1
    
    populate_raw_observations(DUCKDB_PATH)
    
    logger.info("")
    logger.info("Next step: Run tests with pytest takshashika/tests/ -v")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
