#!/usr/bin/env python3
"""Test PM25Fetcher with WAQI API."""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from takshashila.distributed.worker.write_job_worker import PM25Fetcher

def test_pm25_waqi():
    """Test PM25Fetcher with WAQI API."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Testing PM25Fetcher with WAQI API")
    logger.info("")
    
    # Create fetcher
    fetcher = PM25Fetcher(logger)
    
    # Test WAQI API fetch
    logger.info("Step 1: Fetching from WAQI API...")
    sensor_gdf = fetcher.fetch_from_cpcb_api()
    
    if sensor_gdf is None or len(sensor_gdf) == 0:
        logger.error("✗ Failed to fetch sensor data from WAQI API")
        return
    
    logger.info(f"✓ Fetched {len(sensor_gdf)} sensors")
    logger.info(f"  PM2.5 range: {sensor_gdf['pm25'].min():.1f} - {sensor_gdf['pm25'].max():.1f}")
    logger.info("")
    
    # Load grid
    logger.info("Step 2: Loading Mumbai grid...")
    try:
        import geopandas as gpd
        grid_gdf = gpd.read_parquet(fetcher.grid_path)
        logger.info(f"  ✓ Loaded {len(grid_gdf):,} grid cells")
    except Exception as e:
        logger.error(f"  ✗ Failed to load grid: {e}")
        return
    
    logger.info("")
    
    # Test IDW interpolation
    logger.info("Step 3: Testing IDW interpolation (p=4)...")
    try:
        result_df = fetcher.interpolate_idw(sensor_gdf, grid_gdf, power=4.0)
        
        logger.info("")
        logger.info("Step 4: Validating interpolation results...")
        logger.info(f"  Shape: {result_df.shape}")
        logger.info(f"  PM2.5 statistics:")
        logger.info(f"    Min:  {result_df['pm25'].min():.2f}")
        logger.info(f"    Max:  {result_df['pm25'].max():.2f}")
        logger.info(f"    Mean: {result_df['pm25'].mean():.2f}")
        logger.info(f"    Std:  {result_df['pm25'].std():.2f}")
        
        # Check for nulls
        null_count = result_df['pm25'].isnull().sum()
        if null_count > 0:
            logger.error(f"  ✗ Found {null_count} null values!")
        else:
            logger.info("  ✓ No null values")
        
        logger.info("")
        logger.info("Step 5: Writing to DuckDB...")
        
        # Add timestamp for DB write
        from datetime import datetime, timezone
        current_date = datetime.now(timezone.utc)
        result_df['timestamp'] = current_date
        result_df = result_df[['grid_id', 'timestamp', 'pm25']]
        
        # Write to DB
        success = fetcher.write_to_db(result_df)
        
        if success:
            logger.info("")
            logger.info("✓ WAQI PM25Fetcher test PASSED (fetch + interpolation + DB write)!")
        else:
            logger.error("✗ Failed to write to DuckDB")
        
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")

if __name__ == "__main__":
    test_pm25_waqi()
