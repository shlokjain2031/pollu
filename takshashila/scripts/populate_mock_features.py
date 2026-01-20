#!/usr/bin/env python3
"""Populate DuckDB with adversarial mock feature data for testing.

Creates ~30 grid cells with:
- Valid complete features (15 grids)
- Missing PM2.5 features (3 grids)
- Missing meteo features (3 grids)
- Missing satellite features (3 grids)
- Edge timestamp cases (3 grids)
- Invalid grid_ids (not inserted, for testing only)

Usage:
    python3 Takshashila/scripts/populate_mock_features.py
    cd Takshashila && python3 scripts/populate_mock_features.py
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import random
import duckdb

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Takshashila.phase1_minimal_inference.config import DUCKDB_PATH
from Takshashila.phase1_minimal_inference.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Test grids (adversarial selection)
VALID_GRIDS = [
    "g00000001", "g00000042", "g00001337", "g00002000", "g00005000",
    "g00010000", "g00020000", "g00050000", "g00100000", "g00150000",
    "g00200000", "g00250000", "g00300000", "g00400000", "g00500000"
]

MISSING_PM25_GRIDS = ["g00000100", "g00000101", "g00000102"]
MISSING_METEO_GRIDS = ["g00000200", "g00000201", "g00000202"]
MISSING_SAT_GRIDS = ["g00000300", "g00000301", "g00000302"]
EDGE_TIMESTAMP_GRIDS = ["g00000500", "g00000501", "g00000502"]

# Test timestamp
TEST_TIMESTAMP = datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)


def generate_pm25_features(grid_id: str, timestamp: datetime) -> dict:
    """Generate realistic PM2.5 lag features."""
    base_pm25 = random.uniform(30, 120)  # Typical Mumbai range
    return {
        "grid_id": grid_id,
        "timestamp_hour": timestamp,
        "pm25_t_minus_1h": base_pm25 * random.uniform(0.9, 1.1),
        "pm25_t_minus_2h": base_pm25 * random.uniform(0.85, 1.15),
        "pm25_t_minus_3h": base_pm25 * random.uniform(0.8, 1.2),
        "pm25_t_minus_6h": base_pm25 * random.uniform(0.7, 1.3),
        "pm25_t_minus_12h": base_pm25 * random.uniform(0.6, 1.4),
        "pm25_t_minus_24h": base_pm25 * random.uniform(0.5, 1.5)
    }


def generate_meteo_features(grid_id: str, timestamp: datetime) -> dict:
    """Generate realistic meteorological features."""
    temp = random.uniform(20, 35)
    humidity = random.uniform(40, 90)
    pressure = random.uniform(1000, 1020)
    wind_u = random.uniform(-5, 5)
    wind_v = random.uniform(-5, 5)
    
    return {
        "grid_id": grid_id,
        "timestamp_hour": timestamp,
        "temperature_2m": temp,
        "temperature_2m_lag1h": temp * random.uniform(0.98, 1.02),
        "temperature_2m_lag3h": temp * random.uniform(0.95, 1.05),
        "temperature_2m_lag12h": temp * random.uniform(0.9, 1.1),
        "temperature_2m_lag24h": temp * random.uniform(0.85, 1.15),
        "relative_humidity_2m": humidity,
        "relative_humidity_2m_lag1h": humidity * random.uniform(0.98, 1.02),
        "relative_humidity_2m_lag3h": humidity * random.uniform(0.95, 1.05),
        "relative_humidity_2m_lag6h": humidity * random.uniform(0.9, 1.1),
        "relative_humidity_2m_lag12h": humidity * random.uniform(0.85, 1.15),
        "relative_humidity_2m_lag24h": humidity * random.uniform(0.8, 1.2),
        "dewpoint_2m": temp - 5,
        "dewpoint_2m_lag1h": temp - 5.2,
        "dewpoint_2m_lag3h": temp - 5.5,
        "dewpoint_2m_lag6h": temp - 6.0,
        "dewpoint_2m_lag12h": temp - 6.5,
        "dewpoint_2m_lag24h": temp - 7.0,
        "surface_pressure": pressure,
        "surface_pressure_lag1h": pressure + random.uniform(-1, 1),
        "surface_pressure_lag3h": pressure + random.uniform(-2, 2),
        "surface_pressure_lag6h": pressure + random.uniform(-3, 3),
        "surface_pressure_lag12h": pressure + random.uniform(-4, 4),
        "surface_pressure_lag24h": pressure + random.uniform(-5, 5),
        "wind_u_10m": wind_u,
        "wind_u_10m_lag1h": wind_u * random.uniform(0.9, 1.1),
        "wind_u_10m_lag3h": wind_u * random.uniform(0.8, 1.2),
        "wind_u_10m_lag6h": wind_u * random.uniform(0.7, 1.3),
        "wind_u_10m_lag12h": wind_u * random.uniform(0.6, 1.4),
        "wind_u_10m_lag24h": wind_u * random.uniform(0.5, 1.5),
        "wind_v_10m": wind_v,
        "wind_v_10m_lag1h": wind_v * random.uniform(0.9, 1.1),
        "wind_v_10m_lag3h": wind_v * random.uniform(0.8, 1.2),
        "wind_v_10m_lag6h": wind_v * random.uniform(0.7, 1.3),
        "wind_v_10m_lag12h": wind_v * random.uniform(0.6, 1.4),
        "wind_v_10m_lag24h": wind_v * random.uniform(0.5, 1.5),
        "cloud_cover": random.uniform(0, 100),
        "cloud_cover_lag6h": random.uniform(0, 100),
        "cloud_cover_lag24h": random.uniform(0, 100),
        "vpd": random.uniform(0.5, 3.0),
        "temp_dewpoint_diff": random.uniform(2, 10),
        "precip_last_1h": random.uniform(0, 5),
        "precip_last_6h": random.uniform(0, 20),
        "precip_last_24h": random.uniform(0, 50)
    }


def generate_satellite_features(grid_id: str, timestamp: datetime) -> dict:
    """Generate realistic satellite imagery features."""
    return {
        "grid_id": grid_id,
        "timestamp_hour": timestamp,
        "ndvi": random.uniform(0.1, 0.6),
        "mndwi": random.uniform(-0.5, 0.5),
        "ndbi": random.uniform(-0.3, 0.5),
        "bsi": random.uniform(0.0, 0.5),
        "ibi": random.uniform(0.0, 0.5),
        "scene_cloud_cover": random.uniform(0, 30),
        "evi": random.uniform(0.1, 0.6),
        "ui_light": random.uniform(0.0, 1.0),
        "si": random.uniform(0.0, 0.5),
        "local_ndvi_std": random.uniform(0.01, 0.15),
        "ndvi_s2": random.uniform(0.1, 0.6),
        "days_since_landsat": random.uniform(0, 16),
        "days_since_sentinel": random.uniform(0, 5)
    }


def populate_features(db_path: Path):
    """Populate all feature tables with mock data."""
    conn = duckdb.connect(str(db_path))
    
    try:
        # Clear existing mock data
        logger.info("Clearing existing feature data...")
        conn.execute("DELETE FROM features_pm25")
        conn.execute("DELETE FROM features_meteo")
        conn.execute("DELETE FROM features_satellite")
        
        # Populate valid grids (complete features)
        logger.info(f"Populating {len(VALID_GRIDS)} valid grids...")
        for grid_id in VALID_GRIDS:
            pm25_data = generate_pm25_features(grid_id, TEST_TIMESTAMP)
            meteo_data = generate_meteo_features(grid_id, TEST_TIMESTAMP)
            sat_data = generate_satellite_features(grid_id, TEST_TIMESTAMP)
            
            conn.execute("""
                INSERT INTO features_pm25 
                (grid_id, timestamp_hour, pm25_t_minus_1h, pm25_t_minus_2h, 
                 pm25_t_minus_3h, pm25_t_minus_6h, pm25_t_minus_12h, pm25_t_minus_24h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [pm25_data["grid_id"], pm25_data["timestamp_hour"],
                  pm25_data["pm25_t_minus_1h"], pm25_data["pm25_t_minus_2h"],
                  pm25_data["pm25_t_minus_3h"], pm25_data["pm25_t_minus_6h"],
                  pm25_data["pm25_t_minus_12h"], pm25_data["pm25_t_minus_24h"]])
            
            conn.execute(f"""
                INSERT INTO features_meteo VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            """, [meteo_data["grid_id"], meteo_data["timestamp_hour"]] + [
                meteo_data[k] for k in [
                    "temperature_2m", "temperature_2m_lag1h", "temperature_2m_lag3h",
                    "temperature_2m_lag12h", "temperature_2m_lag24h",
                    "relative_humidity_2m", "relative_humidity_2m_lag1h", "relative_humidity_2m_lag3h",
                    "relative_humidity_2m_lag6h", "relative_humidity_2m_lag12h", "relative_humidity_2m_lag24h",
                    "dewpoint_2m", "dewpoint_2m_lag1h", "dewpoint_2m_lag3h",
                    "dewpoint_2m_lag6h", "dewpoint_2m_lag12h", "dewpoint_2m_lag24h",
                    "surface_pressure", "surface_pressure_lag1h", "surface_pressure_lag3h",
                    "surface_pressure_lag6h", "surface_pressure_lag12h", "surface_pressure_lag24h",
                    "wind_u_10m", "wind_u_10m_lag1h", "wind_u_10m_lag3h",
                    "wind_u_10m_lag6h", "wind_u_10m_lag12h", "wind_u_10m_lag24h",
                    "wind_v_10m", "wind_v_10m_lag1h", "wind_v_10m_lag3h",
                    "wind_v_10m_lag6h", "wind_v_10m_lag12h", "wind_v_10m_lag24h",
                    "cloud_cover", "cloud_cover_lag6h", "cloud_cover_lag24h",
                    "vpd", "temp_dewpoint_diff",
                    "precip_last_1h", "precip_last_6h", "precip_last_24h"
                ]
            ])
            
            conn.execute("""
                INSERT INTO features_satellite VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            """, [sat_data["grid_id"], sat_data["timestamp_hour"],
                  sat_data["ndvi"], sat_data["mndwi"], sat_data["ndbi"],
                  sat_data["bsi"], sat_data["ibi"], sat_data["scene_cloud_cover"],
                  sat_data["evi"], sat_data["ui_light"], sat_data["si"],
                  sat_data["local_ndvi_std"], sat_data["ndvi_s2"],
                  sat_data["days_since_landsat"], sat_data["days_since_sentinel"]])
        
        # Populate grids with missing PM2.5 features (only meteo + sat)
        logger.info(f"Populating {len(MISSING_PM25_GRIDS)} grids with missing PM2.5...")
        for grid_id in MISSING_PM25_GRIDS:
            meteo_data = generate_meteo_features(grid_id, TEST_TIMESTAMP)
            sat_data = generate_satellite_features(grid_id, TEST_TIMESTAMP)
            
            conn.execute(f"""
                INSERT INTO features_meteo VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            """, [meteo_data["grid_id"], meteo_data["timestamp_hour"]] + [
                meteo_data[k] for k in [
                    "temperature_2m", "temperature_2m_lag1h", "temperature_2m_lag3h",
                    "temperature_2m_lag12h", "temperature_2m_lag24h",
                    "relative_humidity_2m", "relative_humidity_2m_lag1h", "relative_humidity_2m_lag3h",
                    "relative_humidity_2m_lag6h", "relative_humidity_2m_lag12h", "relative_humidity_2m_lag24h",
                    "dewpoint_2m", "dewpoint_2m_lag1h", "dewpoint_2m_lag3h",
                    "dewpoint_2m_lag6h", "dewpoint_2m_lag12h", "dewpoint_2m_lag24h",
                    "surface_pressure", "surface_pressure_lag1h", "surface_pressure_lag3h",
                    "surface_pressure_lag6h", "surface_pressure_lag12h", "surface_pressure_lag24h",
                    "wind_u_10m", "wind_u_10m_lag1h", "wind_u_10m_lag3h",
                    "wind_u_10m_lag6h", "wind_u_10m_lag12h", "wind_u_10m_lag24h",
                    "wind_v_10m", "wind_v_10m_lag1h", "wind_v_10m_lag3h",
                    "wind_v_10m_lag6h", "wind_v_10m_lag12h", "wind_v_10m_lag24h",
                    "cloud_cover", "cloud_cover_lag6h", "cloud_cover_lag24h",
                    "vpd", "temp_dewpoint_diff",
                    "precip_last_1h", "precip_last_6h", "precip_last_24h"
                ]
            ])
            
            conn.execute("""
                INSERT INTO features_satellite VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            """, [sat_data["grid_id"], sat_data["timestamp_hour"],
                  sat_data["ndvi"], sat_data["mndwi"], sat_data["ndbi"],
                  sat_data["bsi"], sat_data["ibi"], sat_data["scene_cloud_cover"],
                  sat_data["evi"], sat_data["ui_light"], sat_data["si"],
                  sat_data["local_ndvi_std"], sat_data["ndvi_s2"],
                  sat_data["days_since_landsat"], sat_data["days_since_sentinel"]])
        
        # Populate grids with missing meteo features (only PM2.5 + sat)
        logger.info(f"Populating {len(MISSING_METEO_GRIDS)} grids with missing meteo...")
        for grid_id in MISSING_METEO_GRIDS:
            pm25_data = generate_pm25_features(grid_id, TEST_TIMESTAMP)
            sat_data = generate_satellite_features(grid_id, TEST_TIMESTAMP)
            
            conn.execute("""
                INSERT INTO features_pm25 
                (grid_id, timestamp_hour, pm25_t_minus_1h, pm25_t_minus_2h, 
                 pm25_t_minus_3h, pm25_t_minus_6h, pm25_t_minus_12h, pm25_t_minus_24h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [pm25_data["grid_id"], pm25_data["timestamp_hour"],
                  pm25_data["pm25_t_minus_1h"], pm25_data["pm25_t_minus_2h"],
                  pm25_data["pm25_t_minus_3h"], pm25_data["pm25_t_minus_6h"],
                  pm25_data["pm25_t_minus_12h"], pm25_data["pm25_t_minus_24h"]])
            
            conn.execute("""
                INSERT INTO features_satellite VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            """, [sat_data["grid_id"], sat_data["timestamp_hour"],
                  sat_data["ndvi"], sat_data["mndwi"], sat_data["ndbi"],
                  sat_data["bsi"], sat_data["ibi"], sat_data["scene_cloud_cover"],
                  sat_data["evi"], sat_data["ui_light"], sat_data["si"],
                  sat_data["local_ndvi_std"], sat_data["ndvi_s2"],
                  sat_data["days_since_landsat"], sat_data["days_since_sentinel"]])
        
        # Populate grids with missing satellite features (only PM2.5 + meteo)
        logger.info(f"Populating {len(MISSING_SAT_GRIDS)} grids with missing satellite...")
        for grid_id in MISSING_SAT_GRIDS:
            pm25_data = generate_pm25_features(grid_id, TEST_TIMESTAMP)
            meteo_data = generate_meteo_features(grid_id, TEST_TIMESTAMP)
            
            conn.execute("""
                INSERT INTO features_pm25 
                (grid_id, timestamp_hour, pm25_t_minus_1h, pm25_t_minus_2h, 
                 pm25_t_minus_3h, pm25_t_minus_6h, pm25_t_minus_12h, pm25_t_minus_24h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [pm25_data["grid_id"], pm25_data["timestamp_hour"],
                  pm25_data["pm25_t_minus_1h"], pm25_data["pm25_t_minus_2h"],
                  pm25_data["pm25_t_minus_3h"], pm25_data["pm25_t_minus_6h"],
                  pm25_data["pm25_t_minus_12h"], pm25_data["pm25_t_minus_24h"]])
            
            conn.execute(f"""
                INSERT INTO features_meteo VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            """, [meteo_data["grid_id"], meteo_data["timestamp_hour"]] + [
                meteo_data[k] for k in [
                    "temperature_2m", "temperature_2m_lag1h", "temperature_2m_lag3h",
                    "temperature_2m_lag12h", "temperature_2m_lag24h",
                    "relative_humidity_2m", "relative_humidity_2m_lag1h", "relative_humidity_2m_lag3h",
                    "relative_humidity_2m_lag6h", "relative_humidity_2m_lag12h", "relative_humidity_2m_lag24h",
                    "dewpoint_2m", "dewpoint_2m_lag1h", "dewpoint_2m_lag3h",
                    "dewpoint_2m_lag6h", "dewpoint_2m_lag12h", "dewpoint_2m_lag24h",
                    "surface_pressure", "surface_pressure_lag1h", "surface_pressure_lag3h",
                    "surface_pressure_lag6h", "surface_pressure_lag12h", "surface_pressure_lag24h",
                    "wind_u_10m", "wind_u_10m_lag1h", "wind_u_10m_lag3h",
                    "wind_u_10m_lag6h", "wind_u_10m_lag12h", "wind_u_10m_lag24h",
                    "wind_v_10m", "wind_v_10m_lag1h", "wind_v_10m_lag3h",
                    "wind_v_10m_lag6h", "wind_v_10m_lag12h", "wind_v_10m_lag24h",
                    "cloud_cover", "cloud_cover_lag6h", "cloud_cover_lag24h",
                    "vpd", "temp_dewpoint_diff",
                    "precip_last_1h", "precip_last_6h", "precip_last_24h"
                ]
            ])
        
        # Populate edge timestamp grids (complete features)
        logger.info(f"Populating {len(EDGE_TIMESTAMP_GRIDS)} edge timestamp grids...")
        for grid_id in EDGE_TIMESTAMP_GRIDS:
            pm25_data = generate_pm25_features(grid_id, TEST_TIMESTAMP)
            meteo_data = generate_meteo_features(grid_id, TEST_TIMESTAMP)
            sat_data = generate_satellite_features(grid_id, TEST_TIMESTAMP)
            
            conn.execute("""
                INSERT INTO features_pm25 
                (grid_id, timestamp_hour, pm25_t_minus_1h, pm25_t_minus_2h, 
                 pm25_t_minus_3h, pm25_t_minus_6h, pm25_t_minus_12h, pm25_t_minus_24h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [pm25_data["grid_id"], pm25_data["timestamp_hour"],
                  pm25_data["pm25_t_minus_1h"], pm25_data["pm25_t_minus_2h"],
                  pm25_data["pm25_t_minus_3h"], pm25_data["pm25_t_minus_6h"],
                  pm25_data["pm25_t_minus_12h"], pm25_data["pm25_t_minus_24h"]])
            
            conn.execute(f"""
                INSERT INTO features_meteo VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            """, [meteo_data["grid_id"], meteo_data["timestamp_hour"]] + [
                meteo_data[k] for k in [
                    "temperature_2m", "temperature_2m_lag1h", "temperature_2m_lag3h",
                    "temperature_2m_lag12h", "temperature_2m_lag24h",
                    "relative_humidity_2m", "relative_humidity_2m_lag1h", "relative_humidity_2m_lag3h",
                    "relative_humidity_2m_lag6h", "relative_humidity_2m_lag12h", "relative_humidity_2m_lag24h",
                    "dewpoint_2m", "dewpoint_2m_lag1h", "dewpoint_2m_lag3h",
                    "dewpoint_2m_lag6h", "dewpoint_2m_lag12h", "dewpoint_2m_lag24h",
                    "surface_pressure", "surface_pressure_lag1h", "surface_pressure_lag3h",
                    "surface_pressure_lag6h", "surface_pressure_lag12h", "surface_pressure_lag24h",
                    "wind_u_10m", "wind_u_10m_lag1h", "wind_u_10m_lag3h",
                    "wind_u_10m_lag6h", "wind_u_10m_lag12h", "wind_u_10m_lag24h",
                    "wind_v_10m", "wind_v_10m_lag1h", "wind_v_10m_lag3h",
                    "wind_v_10m_lag6h", "wind_v_10m_lag12h", "wind_v_10m_lag24h",
                    "cloud_cover", "cloud_cover_lag6h", "cloud_cover_lag24h",
                    "vpd", "temp_dewpoint_diff",
                    "precip_last_1h", "precip_last_6h", "precip_last_24h"
                ]
            ])
            
            conn.execute("""
                INSERT INTO features_satellite VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            """, [sat_data["grid_id"], sat_data["timestamp_hour"],
                  sat_data["ndvi"], sat_data["mndwi"], sat_data["ndbi"],
                  sat_data["bsi"], sat_data["ibi"], sat_data["scene_cloud_cover"],
                  sat_data["evi"], sat_data["ui_light"], sat_data["si"],
                  sat_data["local_ndvi_std"], sat_data["ndvi_s2"],
                  sat_data["days_since_landsat"], sat_data["days_since_sentinel"]])
        
        conn.commit()
        
        # Print summary
        pm25_count = conn.execute("SELECT COUNT(*) FROM features_pm25").fetchone()[0]
        meteo_count = conn.execute("SELECT COUNT(*) FROM features_meteo").fetchone()[0]
        sat_count = conn.execute("SELECT COUNT(*) FROM features_satellite").fetchone()[0]
        
        logger.info("=" * 80)
        logger.info("Mock feature data populated successfully!")
        logger.info(f"  features_pm25: {pm25_count} rows")
        logger.info(f"  features_meteo: {meteo_count} rows")
        logger.info(f"  features_satellite: {sat_count} rows")
        logger.info("=" * 80)
    
    finally:
        conn.close()


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("Populating mock features for Phase 1 testing")
    logger.info("=" * 80)
    
    if not DUCKDB_PATH.exists():
        logger.error(f"DuckDB file not found: {DUCKDB_PATH}")
        logger.error("Run init_duckdb.py first!")
        return 1
    
    populate_features(DUCKDB_PATH)
    
    logger.info("")
    logger.info("Next step: Run tests with pytest Takshashila/tests/")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
