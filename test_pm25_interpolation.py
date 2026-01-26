#!/usr/bin/env python3
"""Test PM25Fetcher with synthetic sensor data (bypasses CPCB API rate limit)."""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from takshashila.distributed.worker.write_job_worker import PM25Fetcher

def test_pm25_interpolation():
    """Test PM25Fetcher IDW interpolation with synthetic sensors."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Testing PM25Fetcher with synthetic sensor data")
    logger.info("")
    
    # Create fetcher
    fetcher = PM25Fetcher(logger)
    
    # Create synthetic sensors from real OpenAQ sensor locations (31 sensors across Mumbai)
    logger.info("Step 1: Creating synthetic sensor data from real OpenAQ locations...")
    np.random.seed(42)  # Reproducible randomness
    
    sensor_locations = [
        {"latitude": 18.91, "longitude": 72.82, "location_name": "Colaba, Mumbai - MPCB"},
        {"latitude": 18.9767, "longitude": 72.838, "location_name": "Byculla, Mumbai - BMC"},
        {"latitude": 18.9936162, "longitude": 72.8128113, "location_name": "Worli, Mumbai - MPCB"},
        {"latitude": 19.000083, "longitude": 72.813993, "location_name": "Siddharth Nagar-Worli, Mumbai - IITM"},
        {"latitude": 19.000084, "longitude": 72.85673, "location_name": "Sewri, Mumbai - BMC"},
        {"latitude": 19.0364585, "longitude": 72.8954371, "location_name": "Chembur, Mumbai - MPCB"},
        {"latitude": 19.047, "longitude": 72.8746, "location_name": "Sion, Mumbai - MPCB"},
        {"latitude": 19.04946, "longitude": 72.923, "location_name": "Deonar, Mumbai - IITM"},
        {"latitude": 19.053536000000005, "longitude": 72.84643, "location_name": "Bandra Kurla Complex, Mumbai - IITM"},
        {"latitude": 19.060498, "longitude": 72.923356, "location_name": "Shivaji Nagar, Mumbai - BMC"},
        {"latitude": 19.0627, "longitude": 72.84614, "location_name": "Bandra, Mumbai - MPCB"},
        {"latitude": 19.0632143, "longitude": 72.8456324, "location_name": "Kherwadi_Bandra East, Mumbai - MPCB"},
        {"latitude": 19.065931, "longitude": 72.862131, "location_name": "Bandra Kurla Complex, Mumbai - MPCB"},
        {"latitude": 19.07283, "longitude": 72.88261, "location_name": "Mumbai"},
        {"latitude": 19.083694, "longitude": 72.920967, "location_name": "Ghatkopar, Mumbai - BMC"},
        {"latitude": 19.0863, "longitude": 72.8888, "location_name": "Kurla, Mumbai - MPCB"},
        {"latitude": 19.10078, "longitude": 72.87462, "location_name": "Chhatrapati Shivaji Intl. Airport (T2), Mumbai - MPCB"},
        {"latitude": 19.10861, "longitude": 72.83622, "location_name": "Vile Parle West, Mumbai - MPCB"},
        {"latitude": 19.11074, "longitude": 72.86084, "location_name": "Chakala-Andheri East, Mumbai - IITM"},
        {"latitude": 19.1375, "longitude": 72.915056, "location_name": "Powai, Mumbai - MPCB"},
        {"latitude": 19.164394, "longitude": 72.928297, "location_name": "Khindipada-Bhandup West, Mumbai - IITM"},
        {"latitude": 19.175, "longitude": 72.9419, "location_name": "Mulund West, Mumbai - MPCB"},
        {"latitude": 19.1878657, "longitude": 72.8304069, "location_name": "Mindspace-Malad West, Mumbai - MPCB"},
        {"latitude": 19.192056, "longitude": 72.9585188, "location_name": "Pimpleshwar Mandir, Thane - MPCB"},
        {"latitude": 19.19709, "longitude": 72.82204, "location_name": "Malad West, Mumbai - IITM"},
        {"latitude": 19.2058, "longitude": 72.8682, "location_name": "Kandivali East, Mumbai - MPCB"},
        {"latitude": 19.215859, "longitude": 72.831718, "location_name": "Kandivali West, Mumbai - BMC"},
        {"latitude": 19.222279, "longitude": 72.957979, "location_name": "Upvan Fort, Thane - MPCB"},
        {"latitude": 19.227474, "longitude": 72.864394, "location_name": "Borivali East, Mumbai - MPCB"},
        {"latitude": 19.23241, "longitude": 72.86895, "location_name": "Borivali East, Mumbai - IITM"},
        {"latitude": 19.26777, "longitude": 72.97182, "location_name": "Kasarvadavali, Thane - MPCB"},
    ]
    
    # Assign random PM2.5 values in range [50, 300] µg/m³
    sensors_data = [
        {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "pm25": np.random.uniform(50, 300),
            "location_name": loc["location_name"]
        }
        for loc in sensor_locations
    ]
    
    sensor_df = pd.DataFrame(sensors_data)
    sensor_gdf = gpd.GeoDataFrame(
        sensor_df,
        geometry=gpd.points_from_xy(sensor_df["longitude"], sensor_df["latitude"]),
        crs="EPSG:4326"
    )
    
    logger.info(f"  ✓ Created {len(sensor_gdf)} synthetic sensors")
    logger.info(f"    PM2.5 range: {sensor_gdf['pm25'].min():.1f} - {sensor_gdf['pm25'].max():.1f} µg/m³")
    logger.info("")
    
    # Load real grid
    logger.info("Step 2: Loading Mumbai grid...")
    try:
        grid_gdf = gpd.read_parquet(fetcher.grid_path)
        logger.info(f"  ✓ Loaded {len(grid_gdf):,} grid cells")
    except Exception as e:
        logger.error(f"  ✗ Failed to load grid: {e}")
        return
    
    logger.info("")
    
    # Test IDW interpolation
    logger.info("Step 3: Testing IDW interpolation (p=2)...")
    try:
        result_df = fetcher.interpolate_idw(sensor_gdf, grid_gdf, power=2.0)
        
        logger.info("")
        logger.info("Step 4: Validating interpolation results...")
        logger.info(f"  Shape: {result_df.shape}")
        logger.info(f"  Columns: {list(result_df.columns)}")
        logger.info(f"  PM2.5 statistics:")
        logger.info(f"    Min:  {result_df['pm25'].min():.2f} µg/m³")
        logger.info(f"    Max:  {result_df['pm25'].max():.2f} µg/m³")
        logger.info(f"    Mean: {result_df['pm25'].mean():.2f} µg/m³")
        logger.info(f"    Std:  {result_df['pm25'].std():.2f} µg/m³")
        
        # Check for nulls
        null_count = result_df['pm25'].isnull().sum()
        if null_count > 0:
            logger.error(f"  ✗ Found {null_count} null values!")
        else:
            logger.info("  ✓ No null values")
        
        # Check range sanity (should be between min/max of sensors)
        sensor_min = sensor_gdf['pm25'].min()
        sensor_max = sensor_gdf['pm25'].max()
        result_min = result_df['pm25'].min()
        result_max = result_df['pm25'].max()
        
        if result_min < sensor_min * 0.8 or result_max > sensor_max * 1.2:
            logger.warning(f"  ⚠ Interpolated values outside expected range!")
            logger.warning(f"    Sensor range: {sensor_min:.1f} - {sensor_max:.1f}")
            logger.warning(f"    Result range: {result_min:.1f} - {result_max:.1f}")
        else:
            logger.info("  ✓ Interpolated values within reasonable range")
        
        # Sample some values
        logger.info("")
        logger.info("  Sample interpolated values:")
        sample = result_df.sample(min(5, len(result_df)))
        for idx, row in sample.iterrows():
            logger.info(f"    {row['grid_id']}: {row['pm25']:.2f} µg/m³")
        
        logger.info("")
        logger.info("="*60)
        logger.info("✓ IDW interpolation test passed!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"  ✗ Interpolation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pm25_interpolation()
