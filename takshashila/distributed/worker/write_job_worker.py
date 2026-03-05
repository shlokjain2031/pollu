"""
WriterJob Worker - Phase 2b

Part of the Takshashila batch inference scheduler.
Spawned by Python orchestrator (distributed/worker/orchestrator.py) @ t=-2m.

Responsibilities:
- Fetch meteorological data from OpenMeteo API
- Fetch PM2.5 observations from CPCB API
- Fetch Landsat8 satellite data from Google Earth Engine (if available)
- Fetch Sentinel2 satellite data from Google Earth Engine (if available)
- Write all fetched data to DuckDB tables (sequential, minimal memory)
- Handle partial failures gracefully (skip unavailable sources)
- Log with cycle_id for traceability
- Exit with code 0 on success, non-zero on critical failure

Timing Guarantee:
- Starts @ t=-2m (2 minutes before hour)
- Must finish by t=-1m (1 minute before hour)
- 120-second window is sufficient (typical: 20-40 seconds)

Data Flow:
1. Orchestrator spawns this worker @ t=-2m with:
   - cycle_id: UUID for this cycle
   - timestamp_to_write: Which hour's data to write (rounded to nearest hour)
   - config: Path to config file with API credentials, timeouts, DuckDB paths

2. Worker initializes:
   - Load API credentials from environment or config
   - Validate DuckDB file paths exist (create if needed)
   - Initialize logging with cycle_id
   - Set up timeout budgets for each fetcher

3. Worker fetches sequentially:

   a. MeteoFetcher (target: 1-2 seconds)
      - Calls OpenMeteo API endpoint
      - Fetches hourly meteorology for "now" (temperature, humidity, wind, etc.)
      - Returns list of dicts: [{"grid_id": "...", "temperature_2m": ..., ...}, ...]
      - Note: OpenMeteo returns tile-level data (~3-6 tiles for Mumbai)
      - Broadcast to all 500K grids (one value per tile)
      - Timeout: 15 seconds (if exceeds, skip and log error)

   b. PM25Fetcher (target: 2-5 seconds)
      - Calls CPCB API (data.gov.in Real-time Air Quality Index endpoint)
      - Fetches latest PM2.5 observations from ~26 monitoring stations
      - Returns list of dicts: [{"latitude": ..., "longitude": ..., "avg_value": ...}, ...]
      - Use avg_value field only (discard min/max)
      - API filters: filters%5Bcity%5D=Mumbai&filters%5Bpollutant_id%5D=PM2.5
      - Timeout: 15 seconds (if exceeds, skip and log error)

   c. LandsatFetcher (target: 0 or 10-15 seconds)
      - Calls Google Earth Engine API
      - Checks if new Landsat8 data is available (available ~1 per 16 days)
      - If available: Batch fetch 500 batches × 1000 cells concurrently
      - Returns 500K rows × 11 bands (B1-B11)
      - If not available: Returns empty list (0 seconds)
      - Timeout: 60 seconds
      - Graceful failure: Missing data OK (use next available)

   d. SentinelFetcher (target: 0 or 15-25 seconds)
      - Calls Google Earth Engine API
      - Checks if new Sentinel2 data is available (available ~4/5 days, more frequent)
      - If available: Batch fetch 500 batches × 1000 cells concurrently
      - Returns 500K rows × 5 bands (B2, B4, B8, AOT, QA10)
      - If not available: Returns empty list (0 seconds)
      - Timeout: 60 seconds
      - Graceful failure: Missing data OK (use next available)

4. Worker writes to DuckDB (sequential, separate files):
   - Each fetch completes and writes before the next source begins
   - Each write targets a separate DuckDB file
   - Each write: INSERT into table with fetched data

   a. Write meteo → meteo.duckdb
      - 500K rows × ~20 columns
      - Duration: ~2-3 seconds
      - Includes grid_id, timestamp, all meteo fields

   b. Write PM2.5 → pm25_observation.duckdb
      - ~10-26 rows (one per monitoring station)
      - Duration: <1 second
      - Includes latitude, longitude, avg_value, timestamp

   c. Write Landsat8 → landsat8.duckdb (if available)
      - 500K rows × 11 bands
      - Duration: ~2-3 seconds
      - Includes grid_id, timestamp, B1-B11

   d. Write Sentinel2 → sentinel2.duckdb (if available)
      - 500K rows × 5 bands
      - Duration: ~2-3 seconds
      - Includes grid_id, timestamp, B2, B4, B8, AOT, QA10

5. Worker exits:
   - All fetches completed (or timed out/failed)
   - All fetched data written to DuckDB
   - Commit all transactions
   - Return exit code 0 if successful, 1 on critical failure
   - Log summary: "WriterJob completed: X rows written to {meteo, pm25, landsat8, sentinel2}"

Error Handling:
- API fetch timeout (>15s) → Log error, skip that source, continue
- API returns 0 rows → Log warning, continue (OK for satellites)
- DuckDB table doesn't exist → Create it with schema
- DuckDB write fails → Retry once, fail job if second attempt fails
- Partial write (some sources succeeded, others failed) → Exit code 0 (partial success acceptable)
- All sources failed → Exit code 1 (critical failure)

Performance Targets:
- OpenMeteo: 1-2 seconds (typical)
- CPCB: 2-5 seconds (typical)
- Landsat8: 0 seconds (unavailable) or 10-15 seconds (available, batch fetch)
- Sentinel2: 0 seconds (unavailable) or 15-25 seconds (available, batch fetch)
- DuckDB writes: ~8-10 seconds total (sequential)
  - meteo write (2-3s)
  - PM2.5 write (<1s)
  - Landsat8 write (2-3s, if available)
  - Sentinel2 write (2-3s, if available)
  - Total write time = sum of individual writes
- **Total WriterJob duration: 30-60 seconds (sequential execution)**
- 120-second window provides 3x buffer

Example Invocation (by orchestrator):
  python3 -m takshashila.distributed.worker.write_job_worker \
    --cycle-id 550e8400-e29b-41d4-a716-446655440000 \
    --timestamp-to-write 2026-01-25T14:00:00Z \
    --config /path/to/takshashila/config.yaml

Logging Output:
All logs to stderr with format:
  [cycle_id] [timestamp] [level] message

Example:
  [550e8400-e29b] [2026-01-25 14:00:05] INFO WriterJob starting
  [550e8400-e29b] [2026-01-25 14:00:05] INFO Fetching from 4 sources sequentially (timeout: 60s)
  [550e8400-e29b] [2026-01-25 14:00:07] INFO Fetched meteo: 500000 rows
  [550e8400-e29b] [2026-01-25 14:00:09] INFO Fetched PM2.5: 26 rows
  [550e8400-e29b] [2026-01-25 14:00:15] INFO Landsat8 not available today (expected 1/16 days)
  [550e8400-e29b] [2026-01-25 14:00:32] INFO Fetched Sentinel2: 500000 rows
  [550e8400-e29b] [2026-01-25 14:00:32] INFO All fetches completed in 27.3s
  [550e8400-e29b] [2026-01-25 14:00:32] INFO Writing to DuckDB (sequential)
  [550e8400-e29b] [2026-01-25 14:00:35] INFO Written meteo: 500000 rows (2.1s)
  [550e8400-e29b] [2026-01-25 14:00:36] INFO Written PM2.5: 26 rows (0.2s)
  [550e8400-e29b] [2026-01-25 14:00:39] INFO Written Sentinel2: 500000 rows (2.8s)
  [550e8400-e29b] [2026-01-25 14:00:39] INFO All DuckDB writes completed in 7.1s (sequential)
  [550e8400-e29b] [2026-01-25 14:00:35] INFO WriterJob completed successfully (2.8s writes, 30.1s total)

Integration Points:
- OpenMeteo API (https://archive-api.open-meteo.com/v1/archive)
- CPCB API (data.gov.in)
- Google Earth Engine API (geemap or ee library)
- DuckDB (for writing data)
- takshashila.core.config (for paths, credentials, timeouts)

Success Criteria:
- All fetches completed or gracefully failed
- Meteo data written: 500K rows
- PM2.5 data written: 10-26 rows
- Exit code 0
- Completion time < 40 seconds
- Log summary with row counts

Future Enhancements:
- Incremental updates (only fetch new data since last hour)
- Data validation (check bounds, data types, expected counts)
- Retry logic with exponential backoff
- Metrics emission (fetch time, row counts, errors)
- Caching of satellite queries (avoid re-fetching if no new data)
- Distributed fetching across multiple machines
"""

# TODO: Implement MeteoFetcher class
# TODO: Implement PM25Fetcher class (CPCB API)
# TODO: Implement SentinelFetcher class (GEE API)
# TODO: Implement parallel fetch with ThreadPoolExecutor
# TODO: Implement DuckDB schema creation if tables don't exist
# TODO: Implement Parallel DuckDB writes, each thread owns one database file
# TODO: Implement error handling and logging
# TODO: Implement timeout budgets per fetcher
# TODO: Implement graceful degradation (partial success)
# TODO: Implement exit codes and final logging

import sys
import logging
import duckdb
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import time

import ee
import geopandas as gpd
import pandas as pd

from takshashila.core.config import (
    GEE_PROJECT,
    GRID_PARQUET_PATH,
    DUCKDB_LANDSAT8_PATH,
    DUCKDB_SENTINEL2_PATH,
    DUCKDB_METEO_PATH,
    DUCKDB_PM25_PATH,
    GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT
)

import requests
import numpy as np


class LandsatFetcher:
    """Fetch Landsat8 band values for all grid cells via Google Earth Engine.
    
    Checks if latest DB record is >16 days old before fetching.
    If no new Landsat data available, silently skips insert.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.db_path = DUCKDB_LANDSAT8_PATH
        self.grid_path = GRID_PARQUET_PATH
        self.collection_name = "LANDSAT/LC08/C02/T1_TOA"
        self.bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
        
    def should_fetch(self, current_date: datetime) -> bool:
        """Check if latest record in DB is >16 days old.
        
        Args:
            current_date: Current UTC datetime to use for comparison
            
        Returns:
            True if should fetch (latest record >16 days or no data), False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            
            # Check if table exists
            table_exists = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'landsat8'"
            ).fetchone()[0] > 0
            
            if not table_exists:
                conn.close()
                self.logger.info("  Landsat8 table doesn't exist, will create and fetch")
                return True
            
            # Get latest timestamp
            result = conn.execute(
                "SELECT MAX(timestamp) FROM landsat8"
            ).fetchone()
            conn.close()
            
            if result[0] is None:
                self.logger.info("  No Landsat8 records found, will fetch")
                return True
            
            # Convert to timezone-aware UTC datetime
            latest_timestamp = pd.to_datetime(result[0])
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
            
            # Compare dates only (not time component)
            latest_date = latest_timestamp.date()
            current_date_only = current_date.date()
            
            days_diff = (current_date_only - latest_date).days
            
            if days_diff >= 16:
                self.logger.info(f"  Latest Landsat8 record is {days_diff} days old (>16), will fetch")
                return True
            else:
                self.logger.info(f"  Latest Landsat8 record is {days_diff} days old (<=16), skip fetch")
                return False
                
        except Exception as e:
            self.logger.warning(f"  Error checking Landsat8 DB: {e}, will attempt fetch")
            return True
    
    def get_landsat8_image(self, date_str: str, bbox: tuple) -> Optional[ee.Image]:
        """Query Landsat-8 for given date and bbox.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            bbox: (west, south, east, north) in EPSG:4326
            
        Returns:
            ee.Image if found, None if no image available
        """
        west, south, east, north = bbox
        geometry = ee.Geometry.Rectangle([west, south, east, north])
        
        collection = (
            ee.ImageCollection(self.collection_name)
            .filterBounds(geometry)
            .filterDate(date_str, ee.Date(date_str).advance(1, "day"))
            .sort("system:time_start")
        )
        
        count = collection.size().getInfo()
        if count == 0:
            return None
        
        if count == 1:
            image = collection.first()
            scene_cloud_cover = image.get('CLOUD_COVER')
        else:
            image = collection.mosaic()
            scene_cloud_cover = collection.aggregate_mean('CLOUD_COVER')
        
        image = image.set('scene_cloud_cover', scene_cloud_cover)
        image = image.set('scene_count', count)
        
        return image.select(self.bands)
    
    def fetch_band_values_batched(
        self,
        image: ee.Image,
        grid_gdf: gpd.GeoDataFrame,
        batch_size: int = 500,
        max_workers: int = 4,
        tile_scale: int = 2,
    ) -> pd.DataFrame:
        """Fetch band values for all grid cells using batched reduceRegions.
        
        Args:
            image: Landsat-8 image with bands B1-B11
            grid_gdf: Grid with columns: grid_id, geometry (Point)
            batch_size: Number of cells per EE request
            max_workers: Number of concurrent workers
            tile_scale: Earth Engine tileScale
            
        Returns:
            DataFrame with columns: grid_id, B1, B2, ..., B11, scene_cloud_cover
        """
        # Ensure WGS84
        if grid_gdf.crs and grid_gdf.crs.to_string() != "EPSG:4326":
            grid_wgs84 = grid_gdf.to_crs("EPSG:4326")
        else:
            grid_wgs84 = grid_gdf
        
        total_batches = (len(grid_wgs84) + batch_size - 1) // batch_size
        self.logger.info(f"  Fetching {len(grid_wgs84):,} cells in {total_batches} batches")
        
        def fetch_batch(batch_idx: int) -> tuple[int, list[dict]]:
            """Fetch a single batch."""
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(grid_wgs84))
            batch_gdf = grid_wgs84.iloc[start_idx:end_idx]
            
            # Convert to EE FeatureCollection
            features = []
            for idx, row in batch_gdf.iterrows():
                pt = row.geometry
                feat = ee.Feature(ee.Geometry.Point([pt.x, pt.y]))
                feat = feat.set({
                    "grid_id": row["grid_id"],
                    "scene_cloud_cover": image.get("scene_cloud_cover"),
                })
                features.append(feat)
            
            fc = ee.FeatureCollection(features)
            
            # Reduce regions at 30m scale
            reduced = image.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=30,
                tileScale=tile_scale,
            )
            
            # Fetch with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result_list = reduced.getInfo()["features"]
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"EE getInfo failed after {max_retries} attempts: {e}")
                    time.sleep(2 ** attempt)
            
            batch_results = [feat["properties"] for feat in result_list]
            return (batch_idx, batch_results)
        
        # Fetch all batches concurrently
        all_results = []
        effective_workers = min(max_workers, total_batches if total_batches > 0 else 1, 8)
        
        completed_batches = 0
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(fetch_batch, batch_idx): batch_idx
                for batch_idx in range(total_batches)
            }
            
            for future in as_completed(futures):
                batch_idx, batch_results = future.result()
                all_results.extend(batch_results)
                completed_batches += 1
                if completed_batches % max(1, total_batches // 10) == 0:  # Log every 10%
                    self.logger.info(f"  Progress: {completed_batches}/{total_batches} batches completed ({100*completed_batches//total_batches}%)")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Sort bands numerically
        band_cols = [c for c in df.columns if c.startswith("B") and c[1:].isdigit()]
        band_cols = sorted(band_cols, key=lambda c: int(c[1:]))
        
        keep_cols = ["grid_id", "scene_cloud_cover"] + band_cols
        df = df[keep_cols]
        
        # Reindex to match grid order
        grid_order = grid_gdf[["grid_id"]].copy()
        grid_order["_order"] = range(len(grid_order))
        df = grid_order.merge(df, on="grid_id", how="left").sort_values("_order").drop(columns=["_order"])
        
        self.logger.info(f"  ✓ Fetched {len(df):,} cells with {len(band_cols)} bands")
        return df
    
    def fetch(self, current_date: datetime, bbox: tuple = (72.7763, 18.8939, 72.9797, 19.2701)) -> Optional[pd.DataFrame]:
        """Fetch Landsat8 data for current date if needed.
        
        Args:
            current_date: Current UTC datetime
            bbox: Mumbai bounding box (west, south, east, north)
            
        Returns:
            DataFrame with fetched data, or None if skip
        """
        # Check if we should fetch
        if not self.should_fetch(current_date):
            return None
        
        # Initialize Earth Engine with service account credentials
        try:
            key_path = '/Users/shlokjain/pollu/takshashila/distributed/worker/pollu-485506-73502dcc4936.json'
            credentials = ee.ServiceAccountCredentials(GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT, key_path)
            ee.Initialize(credentials=credentials, project=GEE_PROJECT)
            self.logger.info(f"  ✓ Earth Engine initialized (project: {GEE_PROJECT})")
        except Exception as e:
            self.logger.error(f"  ✗ Failed to initialize Earth Engine: {e}")
            return None
        
        # Load grid
        try:
            grid_gdf = gpd.read_parquet(self.grid_path)
            self.logger.info(f"  ✓ Loaded {len(grid_gdf):,} grid cells from {self.grid_path}")
        except Exception as e:
            self.logger.error(f"  ✗ Failed to load grid: {e}")
            return None
        
        # Query Landsat for current date
        date_str = current_date.strftime("%Y-%m-%d")
        self.logger.info(f"  Querying Landsat-8 for {date_str}...")
        
        try:
            image = self.get_landsat8_image(date_str, bbox)
            if image is None:
                self.logger.info(f"  No Landsat-8 image available for {date_str} (expected - ~1 per 16 days)")
                return None
            
            self.logger.info(f"  ✓ Found Landsat-8 image for {date_str}")
        except Exception as e:
            self.logger.error(f"  ✗ Error querying Landsat: {e}")
            return None
        
        # Fetch band values
        try:
            df = self.fetch_band_values_batched(image, grid_gdf, batch_size=500, max_workers=6)
            
            # Add timestamp column
            df['timestamp'] = current_date
            
            # Reorder: grid_id, timestamp, B1-B11, scene_cloud_cover
            cols = ['grid_id', 'timestamp'] + self.bands + ['scene_cloud_cover']
            df = df[cols]
            
            return df
            
        except Exception as e:
            self.logger.error(f"  ✗ Error fetching band values: {e}")
            return None
    
    def write_to_db(self, df: pd.DataFrame) -> bool:
        """Write fetched data to DuckDB.
        
        Args:
            df: DataFrame with Landsat data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Round timestamp to ceiling of hour (UTC)
            df = df.copy()
            df['timestamp'] = df['timestamp'].dt.ceil('H')
            
            conn = duckdb.connect(str(self.db_path))
            
            # Create table if doesn't exist (match schema.py)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS landsat8 (
                    grid_id VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    B1 DOUBLE, B2 DOUBLE, B3 DOUBLE, B4 DOUBLE, B5 DOUBLE,
                    B6 DOUBLE, B7 DOUBLE, B8 DOUBLE, B9 DOUBLE, B10 DOUBLE, B11 DOUBLE,
                    scene_cloud_cover DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (grid_id, timestamp)
                )
            """)
            
            # Insert data
            conn.execute("INSERT INTO landsat8 SELECT * FROM df")
            conn.close()
            
            self.logger.info(f"  ✓ Written {len(df):,} rows to {self.db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"  ✗ Error writing to DuckDB: {e}")
            return False


class Sentinel2Fetcher:
    """Fetch Sentinel2 band values for all grid cells via Google Earth Engine.
    
    Checks if latest DB record is >5 days old before fetching.
    If no new Sentinel2 data available, silently skips insert.
    Fetches for just one date (current date).
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.db_path = DUCKDB_SENTINEL2_PATH
        self.grid_path = GRID_PARQUET_PATH
        self.collection_name = "COPERNICUS/S2_SR_HARMONIZED"
        self.bands = ["B2", "B4", "B8", "AOT", "QA10"]
        
    def should_fetch(self, current_date: datetime) -> bool:
        """Check if latest record in DB is >5 days old.
        
        Args:
            current_date: Current UTC datetime to use for comparison
            
        Returns:
            True if should fetch (latest record >5 days or no data), False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            
            # Check if table exists
            table_exists = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'sentinel2'"
            ).fetchone()[0] > 0
            
            if not table_exists:
                conn.close()
                self.logger.info("  Sentinel2 table doesn't exist, will create and fetch")
                return True
            
            # Get latest timestamp
            result = conn.execute(
                "SELECT MAX(timestamp) FROM sentinel2"
            ).fetchone()
            conn.close()
            
            if result[0] is None:
                self.logger.info("  No Sentinel2 records found, will fetch")
                return True
            
            # Convert to timezone-aware UTC datetime
            latest_timestamp = pd.to_datetime(result[0])
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
            
            # Compare dates only (not time component)
            latest_date = latest_timestamp.date()
            current_date_only = current_date.date()
            
            days_diff = (current_date_only - latest_date).days
            
            if days_diff >= 5:
                self.logger.info(f"  Latest Sentinel2 record is {days_diff} days old (>5), will fetch")
                return True
            else:
                self.logger.info(f"  Latest Sentinel2 record is {days_diff} days old (<=5), skip fetch")
                return False
                
        except Exception as e:
            self.logger.warning(f"  Error checking Sentinel2 DB: {e}, will attempt fetch")
            return True
    
    def get_sentinel2_image(self, date_str: str, bbox: tuple) -> Optional[ee.Image]:
        """Query Sentinel-2 for given date and bbox.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            bbox: (west, south, east, north) in EPSG:4326
            
        Returns:
            ee.Image if found, None if no image available
        """
        west, south, east, north = bbox
        geometry = ee.Geometry.Rectangle([west, south, east, north])
        
        collection = (
            ee.ImageCollection(self.collection_name)
            .filterBounds(geometry)
            .filterDate(date_str, ee.Date(date_str).advance(1, "day"))
            .sort("system:time_start")
        )
        
        count = collection.size().getInfo()
        if count == 0:
            return None
        
        if count == 1:
            image = collection.first()
        else:
            # Multiple scenes (tiles): mosaic
            image = collection.mosaic()
        
        return image.select(self.bands)
    
    def fetch_band_values_batched(
        self,
        image: ee.Image,
        grid_gdf: gpd.GeoDataFrame,
        batch_size: int = 500,
        max_workers: int = 4,
        tile_scale: int = 2,
    ) -> pd.DataFrame:
        """Fetch band values for all grid cells using batched reduceRegions.
        
        Args:
            image: Sentinel-2 image with bands B2, B4, B8, AOT, QA10
            grid_gdf: Grid with columns: grid_id, geometry (Point)
            batch_size: Number of cells per EE request
            max_workers: Number of concurrent workers
            tile_scale: Earth Engine tileScale
            
        Returns:
            DataFrame with columns: grid_id, B2, B4, B8, AOT, QA10
        """
        # Ensure WGS84
        if grid_gdf.crs and grid_gdf.crs.to_string() != "EPSG:4326":
            grid_wgs84 = grid_gdf.to_crs("EPSG:4326")
        else:
            grid_wgs84 = grid_gdf
        
        total_batches = (len(grid_wgs84) + batch_size - 1) // batch_size
        self.logger.info(f"  Fetching {len(grid_wgs84):,} cells in {total_batches} batches")
        
        def fetch_batch(batch_idx: int) -> tuple[int, list[dict]]:
            """Fetch a single batch."""
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(grid_wgs84))
            batch_gdf = grid_wgs84.iloc[start_idx:end_idx]
            
            # Convert to EE FeatureCollection
            features = []
            for idx, row in batch_gdf.iterrows():
                pt = row.geometry
                feat = ee.Feature(ee.Geometry.Point([pt.x, pt.y]))
                feat = feat.set({"grid_id": row["grid_id"]})
                features.append(feat)
            
            fc = ee.FeatureCollection(features)
            
            # Reduce regions at 30m scale (aggregates 3x3 10m pixels → mean per 30m cell)
            reduced = image.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=30,
                tileScale=tile_scale,
            )
            
            # Fetch with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result_list = reduced.getInfo()["features"]
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"EE getInfo failed after {max_retries} attempts: {e}")
                    time.sleep(2 ** attempt)
            
            batch_results = [feat["properties"] for feat in result_list]
            return (batch_idx, batch_results)
        
        # Fetch all batches concurrently
        all_results = []
        effective_workers = min(max_workers, total_batches if total_batches > 0 else 1, 8)
        
        completed_batches = 0
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(fetch_batch, batch_idx): batch_idx
                for batch_idx in range(total_batches)
            }
            
            for future in as_completed(futures):
                batch_idx, batch_results = future.result()
                all_results.extend(batch_results)
                completed_batches += 1
                if completed_batches % max(1, total_batches // 10) == 0:  # Log every 10%
                    self.logger.info(f"  Progress: {completed_batches}/{total_batches} batches completed ({100*completed_batches//total_batches}%)")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Rescale AOT if needed (Sentinel-2 stores AOT as x1000 integer)
        if "AOT" in df.columns:
            try:
                aot_max = float(df["AOT"].dropna().max())
                if aot_max > 10.0:
                    # AOT stored as per-thousand (x1000)
                    df["AOT"] = df["AOT"] / 1000.0
                    self.logger.info("  ⚑ Rescaled AOT by /1000 to convert to unit range (~0-2)")
            except Exception:
                pass
        
        # Sort bands: B-bands numerically, then AOT, then QA10
        def band_sort_key(col: str) -> tuple:
            if col.startswith("B") and col[1:].isdigit():
                return (0, int(col[1:]))
            elif col == "AOT":
                return (1, 0)
            elif col == "QA10":
                return (2, 0)
            else:
                return (3, 0)
        
        band_cols = [c for c in df.columns if c in self.bands]
        band_cols = sorted(band_cols, key=band_sort_key)
        
        keep_cols = ["grid_id"] + band_cols
        df = df[keep_cols]
        
        # Reindex to match grid order
        grid_order = grid_gdf[["grid_id"]].copy()
        grid_order["_order"] = range(len(grid_order))
        df = grid_order.merge(df, on="grid_id", how="left").sort_values("_order").drop(columns=["_order"])
        
        self.logger.info(f"  ✓ Fetched {len(df):,} cells with {len(band_cols)} bands")
        return df
    
    def fetch(self, current_date: datetime, bbox: tuple = (72.7763, 18.8939, 72.9797, 19.2701)) -> Optional[pd.DataFrame]:
        """Fetch Sentinel2 data for current date if needed.
        
        Args:
            current_date: Current UTC datetime
            bbox: Mumbai bounding box (west, south, east, north)
            
        Returns:
            DataFrame with fetched data, or None if skip
        """
        # Check if we should fetch
        if not self.should_fetch(current_date):
            return None
        
        # Initialize Earth Engine with service account credentials
        try:
            service_account = '771194812335-compute@developer.gserviceaccount.com'
            key_path = '/Users/shlokjain/pollu/takshashila/distributed/worker/pollu-485506-73502dcc4936.json'
            credentials = ee.ServiceAccountCredentials(service_account, key_path)
            ee.Initialize(credentials=credentials, project=GEE_PROJECT)
            self.logger.info(f"  ✓ Earth Engine initialized (project: {GEE_PROJECT})")
        except Exception as e:
            self.logger.error(f"  ✗ Failed to initialize Earth Engine: {e}")
            return None
        
        # Load grid
        try:
            grid_gdf = gpd.read_parquet(self.grid_path)
            self.logger.info(f"  ✓ Loaded {len(grid_gdf):,} grid cells from {self.grid_path}")
        except Exception as e:
            self.logger.error(f"  ✗ Failed to load grid: {e}")
            return None
        
        # Query Sentinel-2 for current date
        date_str = current_date.strftime("%Y-%m-%d")
        self.logger.info(f"  Querying Sentinel-2 for {date_str}...")
        
        try:
            image = self.get_sentinel2_image(date_str, bbox)
            if image is None:
                self.logger.info(f"  No Sentinel-2 image available for {date_str} (expected - ~every 4-5 days)")
                return None
            
            self.logger.info(f"  ✓ Found Sentinel-2 image for {date_str}")
        except Exception as e:
            self.logger.error(f"  ✗ Error querying Sentinel-2: {e}")
            return None
        
        # Fetch band values
        try:
            df = self.fetch_band_values_batched(image, grid_gdf, batch_size=500, max_workers=6)
            
            # Add timestamp column
            df['timestamp'] = current_date
            
            # Reorder: grid_id, timestamp, B2, B4, B8, AOT, QA10
            cols = ['grid_id', 'timestamp'] + self.bands
            df = df[cols]
            
            return df
            
        except Exception as e:
            self.logger.error(f"  ✗ Error fetching band values: {e}")
            return None
    
    def write_to_db(self, df: pd.DataFrame) -> bool:
        """Write fetched data to DuckDB.
        
        Args:
            df: DataFrame with Sentinel2 data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Round timestamp to ceiling of hour (UTC)
            df = df.copy()
            df['timestamp'] = df['timestamp'].dt.ceil('H')
            
            conn = duckdb.connect(str(self.db_path))
            
            # Create table if doesn't exist (match schema.py)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS sentinel2 (
                    grid_id VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    B2 DOUBLE,
                    B4 DOUBLE,
                    B8 DOUBLE,
                    AOT DOUBLE,
                    QA10 INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (grid_id, timestamp)
                )
            """)
            
            # Insert data
            conn.execute("INSERT INTO sentinel2 SELECT * FROM df")
            conn.close()
            
            self.logger.info(f"  ✓ Written {len(df):,} rows to {self.db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"  ✗ Error writing to DuckDB: {e}")
            return False


class MeteoFetcher:
    """Fetch meteorology from Open-Meteo API and map to grid cells.
    
    Checks if latest DB record is >1 hour old before fetching.
    Uses tile-based mapping for efficiency (Mumbai = ~3-6 tiles at 9km resolution).
    Fetches for just one hour (current hour).
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.db_path = DUCKDB_METEO_PATH
        self.grid_path = GRID_PARQUET_PATH
        self.tile_size_m = 9000  # ECMWF IFS resolution
        self.variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "cloud_cover",
        ]
        
    def should_fetch(self, current_date: datetime) -> bool:
        """Check if latest record in DB is >1 hour old.
        
        Args:
            current_date: Current UTC datetime to use for comparison
            
        Returns:
            True if should fetch (latest record >1 hour or no data), False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            
            # Check if table exists
            table_exists = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'meteo'"
            ).fetchone()[0] > 0
            
            if not table_exists:
                conn.close()
                self.logger.info("  Meteo table doesn't exist, will create and fetch")
                return True
            
            # Get latest timestamp
            result = conn.execute(
                "SELECT MAX(timestamp) FROM meteo"
            ).fetchone()
            conn.close()
            
            if result[0] is None:
                self.logger.info("  No Meteo records found, will fetch")
                return True
            
            # Convert to timezone-aware UTC datetime
            latest_timestamp = pd.to_datetime(result[0])
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
            
            # Compare hours (round to hour)
            latest_hour = latest_timestamp.replace(minute=0, second=0, microsecond=0)
            current_hour = current_date.replace(minute=0, second=0, microsecond=0)
            
            hours_diff = (current_hour - latest_hour).total_seconds() / 3600
            
            if hours_diff >= 1:
                self.logger.info(f"  Latest Meteo record is {hours_diff:.1f} hours old (>1), will fetch")
                return True
            else:
                self.logger.info(f"  Latest Meteo record is {hours_diff:.1f} hours old (<=1), skip fetch")
                return False
                
        except Exception as e:
            self.logger.warning(f"  Error checking Meteo DB: {e}, will attempt fetch")
            return True
    
    def compute_meteo_tile_id(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Compute meteo tile ID by snapping coordinates to tile grid."""
        tile_col = int(x_m // self.tile_size_m)
        tile_row = int(y_m // self.tile_size_m)
        return (tile_col, tile_row)
    
    def assign_meteo_tiles(self, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Assign each grid cell to a meteo tile."""
        # Project to UTM if needed
        if grid_gdf.crs and grid_gdf.crs.to_string() == "EPSG:4326":
            grid_proj = grid_gdf.to_crs("EPSG:32643")  # UTM 43N for Mumbai
            x_coords = grid_proj.geometry.x
            y_coords = grid_proj.geometry.y
        else:
            x_coords = grid_gdf.geometry.x
            y_coords = grid_gdf.geometry.y
        
        # Compute tile IDs
        tile_cols = (x_coords // self.tile_size_m).astype(int)
        tile_rows = (y_coords // self.tile_size_m).astype(int)
        tile_ids = tile_cols.astype(str) + "_" + tile_rows.astype(str)
        
        # Create result DataFrame
        result = grid_gdf[["grid_id"]].copy()
        result["meteo_tile_id"] = tile_ids.values
        result["x_m"] = x_coords.values
        result["y_m"] = y_coords.values
        
        return result
    
    def get_unique_tile_centroids(self, grid_with_tiles: pd.DataFrame) -> dict[str, tuple[float, float]]:
        """Get unique meteo tile centroids in WGS84."""
        from pyproj import Transformer
        
        unique_tiles = grid_with_tiles["meteo_tile_id"].unique()
        transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
        
        tile_centroids = {}
        for tile_id in unique_tiles:
            parts = tile_id.split("_")
            tile_col = int(parts[0])
            tile_row = int(parts[1])
            
            # Compute tile centroid in projected coords
            centroid_x = (tile_col + 0.5) * self.tile_size_m
            centroid_y = (tile_row + 0.5) * self.tile_size_m
            
            # Transform to WGS84
            lon, lat = transformer.transform(centroid_x, centroid_y)
            tile_centroids[tile_id] = (lat, lon)
        
        return tile_centroids
    
    def fetch_openmeteo_hourly(
        self,
        date_str: str,
        hour: int,
        latitude: float,
        longitude: float,
    ) -> dict:
        """Fetch hourly meteorology from Open-Meteo for a single location and hour.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            hour: Hour in UTC (0-23)
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            
        Returns:
            Dict with meteorology variables
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": ",".join(self.variables),
            "timezone": "UTC",
            "models": "ecmwf_ifs",
        }
        
        # Retry with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Open-Meteo fetch failed after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
        
        data = response.json()
        hourly = data["hourly"]
        
        # Extract values for the specific hour
        if hour < len(hourly["time"]):
            result = {}
            for var in self.variables:
                if var in hourly:
                    result[var] = hourly[var][hour]
            return result
        
        # Return NaN if hour not found
        return {var: np.nan for var in self.variables}
    
    def fetch_all_tiles(
        self,
        tile_centroids: dict[str, tuple[float, float]],
        date_str: str,
        hour: int,
    ) -> dict[str, dict]:
        """Fetch meteorology for all unique tiles."""
        self.logger.info(f"  Fetching hourly data for {len(tile_centroids)} tiles...")
        
        tile_data = {}
        for i, (tile_id, (lat, lon)) in enumerate(tile_centroids.items(), 1):
            self.logger.info(f"    [{i}/{len(tile_centroids)}] Tile {tile_id} at ({lat:.4f}, {lon:.4f})")
            
            values = self.fetch_openmeteo_hourly(
                date_str=date_str,
                hour=hour,
                latitude=lat,
                longitude=lon,
            )
            tile_data[tile_id] = values
        
        return tile_data
    
    def build_grid_meteo_dataframe(
        self,
        grid_with_tiles: pd.DataFrame,
        tile_data: dict[str, dict],
    ) -> pd.DataFrame:
        """Build per-grid-cell meteorology DataFrame."""
        records = []
        for _, row in grid_with_tiles.iterrows():
            grid_id = row["grid_id"]
            tile_id = row["meteo_tile_id"]
            
            record = {"grid_id": grid_id}
            
            if tile_id in tile_data:
                record.update(tile_data[tile_id])
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def fetch(self, current_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch Meteo data for current hour if needed.
        
        Args:
            current_date: Current UTC datetime
            
        Returns:
            DataFrame with fetched data, or None if skip
        """
        # Check if we should fetch
        if not self.should_fetch(current_date):
            return None
        
        # Load grid
        try:
            grid_gdf = gpd.read_parquet(self.grid_path)
            self.logger.info(f"  ✓ Loaded {len(grid_gdf):,} grid cells from {self.grid_path}")
        except Exception as e:
            self.logger.error(f"  ✗ Failed to load grid: {e}")
            return None
        
        # Assign meteo tiles
        self.logger.info(f"  Assigning meteo tiles (tile size: {self.tile_size_m}m)...")
        grid_with_tiles = self.assign_meteo_tiles(grid_gdf)
        unique_tiles = grid_with_tiles["meteo_tile_id"].nunique()
        self.logger.info(f"  ✓ Grid spans {unique_tiles} unique meteo tiles")
        
        # Get tile centroids
        tile_centroids = self.get_unique_tile_centroids(grid_with_tiles)
        self.logger.info(f"  ✓ Computed {len(tile_centroids)} tile centroids")
        
        # Fetch data for current hour
        date_str = current_date.strftime("%Y-%m-%d")
        hour = current_date.hour
        self.logger.info(f"  Querying Open-Meteo for {date_str} hour {hour}...")
        
        try:
            tile_data = self.fetch_all_tiles(tile_centroids, date_str, hour)
            self.logger.info(f"  ✓ Fetched data for {len(tile_data)} tiles")
        except Exception as e:
            self.logger.error(f"  ✗ Error fetching from Open-Meteo: {e}")
            return None
        
        # Build grid DataFrame
        try:
            df = self.build_grid_meteo_dataframe(
                grid_with_tiles[["grid_id", "meteo_tile_id"]],
                tile_data,
            )
            
            # Add timestamp column
            df['timestamp'] = current_date
            
            # Reorder: grid_id, timestamp, then all meteo variables
            cols = ['grid_id', 'timestamp'] + self.variables
            df = df[cols]
            
            return df
            
        except Exception as e:
            self.logger.error(f"  ✗ Error building grid DataFrame: {e}")
            return None
    
    def write_to_db(self, df: pd.DataFrame) -> bool:
        """Write fetched data to DuckDB.
        
        Args:
            df: DataFrame with Meteo data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Round timestamp to ceiling of hour (UTC)
            df = df.copy()
            df['timestamp'] = df['timestamp'].dt.ceil('H')
            
            conn = duckdb.connect(str(self.db_path))
            
            # Create table if doesn't exist (match schema.py)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS meteo (
                    grid_id VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    temperature_2m DOUBLE,
                    relative_humidity_2m DOUBLE,
                    dewpoint_2m DOUBLE,
                    surface_pressure DOUBLE,
                    wind_speed_10m DOUBLE,
                    wind_direction_10m DOUBLE,
                    precipitation DOUBLE,
                    cloud_cover DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (grid_id, timestamp)
                )
            """)
            
            # Insert data (DuckDB will auto-populate created_at)
            conn.execute("INSERT INTO meteo (grid_id, timestamp, temperature_2m, relative_humidity_2m, dewpoint_2m, surface_pressure, wind_speed_10m, wind_direction_10m, precipitation, cloud_cover) SELECT * FROM df")
            conn.close()
            
            self.logger.info(f"  ✓ Written {len(df):,} rows to {self.db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"  ✗ Error writing to DuckDB: {e}")
            return False


class PM25Fetcher:
    """Fetch PM2.5 observations from WAQI API and interpolate to grid cells using IDW.
    
    Note: Currently uses WAQI API (returns AQI values treated as PM2.5) as temporary replacement.
    
    Checks if latest DB record is >1 hour old before fetching.
    Uses Inverse Distance Weighting (IDW) with p=2 to interpolate from ~20-30 sensors to all grid cells.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.db_path = DUCKDB_PM25_PATH
        self.grid_path = GRID_PARQUET_PATH
        # WAQI API: https://waqi.info/
        self.waqi_api_url = "https://api.waqi.info/v2/map/bounds"
        self.waqi_token = "5c88c4c3a4397350d3afdd1e47115fa2b1a3878e"
        # Mumbai bbox: south=18.8939, west=72.7763, north=19.2701, east=72.9797
        self.bbox_latlng = "18.8939,72.7763,19.2701,72.9797"  # south,west,north,east
        
    def should_fetch(self, current_date: datetime) -> bool:
        """Check if latest record in DB is >1 hour old.
        
        Args:
            current_date: Current UTC datetime to use for comparison
            
        Returns:
            True if should fetch (latest record >1 hour or no data), False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            
            # Check if table exists
            table_exists = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'pm25_observation'"
            ).fetchone()[0] > 0
            
            if not table_exists:
                conn.close()
                self.logger.info("  PM2.5 table doesn't exist, will create and fetch")
                return True
            
            # Get latest timestamp
            result = conn.execute(
                "SELECT MAX(timestamp) FROM pm25_observation"
            ).fetchone()
            conn.close()
            
            if result[0] is None:
                self.logger.info("  No PM2.5 records found, will fetch")
                return True
            
            # Convert to timezone-aware UTC datetime
            latest_timestamp = pd.to_datetime(result[0])
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
            
            # Compare hours (round to hour)
            latest_hour = latest_timestamp.replace(minute=0, second=0, microsecond=0)
            current_hour = current_date.replace(minute=0, second=0, microsecond=0)
            
            hours_diff = (current_hour - latest_hour).total_seconds() / 3600
            
            if hours_diff >= 1:
                self.logger.info(f"  Latest PM2.5 record is {hours_diff:.1f} hours old (>=1), will fetch")
                return True
            else:
                self.logger.info(f"  Latest PM2.5 record is {hours_diff:.1f} hours old (<1), skip fetch")
                return False
                
        except Exception as e:
            self.logger.warning(f"  Error checking PM2.5 DB: {e}, will attempt fetch")
            return True
    
    def fetch_from_cpcb_api(self) -> Optional[gpd.GeoDataFrame]:
        """Fetch latest PM2.5 observations from WAQI API (temporary replacement for CPCB).
        
        WAQI returns AQI values which we treat as PM2.5 for now.
        
        Returns:
            GeoDataFrame with columns: latitude, longitude, pm25, geometry
            None if API call fails or returns no data
        """
        params = {
            "latlng": self.bbox_latlng,  # south,west,north,east
            "networks": "all",
            "token": self.waqi_token,
        }
        
        # Retry with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(self.waqi_api_url, params=params, timeout=15)
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"  ✗ WAQI API failed after {max_retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)
        
        try:
            data = response.json()
            
            # Check API response status
            if data.get("status") != "ok":
                self.logger.warning(f"  WAQI API returned status: {data.get('status')}")
                return None
            
            # Extract data
            results = data.get("data", [])
            if len(results) == 0:
                self.logger.warning("  No PM2.5 data returned from WAQI API")
                return None
            
            self.logger.info(f"  ✓ Fetched {len(results)} sensor records from WAQI API")
            
            # Parse into DataFrame
            sensors = []
            for result in results:
                try:
                    lat = float(result.get("lat", 0))
                    lon = float(result.get("lon", 0))
                    aqi_str = result.get("aqi", "0")
                    
                    # AQI value comes as string, convert to float
                    try:
                        pm25_value = float(aqi_str)
                    except (ValueError, TypeError):
                        continue
                    
                    # Validate coordinates (Mumbai bounding box)
                    if not (18.8 <= lat <= 19.3 and 72.7 <= lon <= 73.0):
                        continue
                    
                    # Validate PM2.5 value (AQI typically 0-500)
                    if pm25_value < 0 or pm25_value > 1000:
                        continue
                    
                    sensors.append({
                        "latitude": lat,
                        "longitude": lon,
                        "pm25": pm25_value,
                    })
                except (ValueError, KeyError, TypeError):
                    continue
            
            if len(sensors) == 0:
                self.logger.warning("  No valid PM2.5 sensors after filtering")
                return None
            
            # Convert to GeoDataFrame
            df = pd.DataFrame(sensors)
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                crs="EPSG:4326"
            )
            
            self.logger.info(f"  ✓ Validated {len(gdf)} sensors with valid PM2.5 data")
            return gdf
            
        except Exception as e:
            self.logger.error(f"  ✗ Error parsing CPCB API response: {e}")
            return None
    
    def interpolate_idw(
        self,
        sensor_gdf: gpd.GeoDataFrame,
        grid_gdf: gpd.GeoDataFrame,
        power: float = 2.0
    ) -> pd.DataFrame:
        """Interpolate PM2.5 values to grid cells using Inverse Distance Weighting.
        
        Args:
            sensor_gdf: Sensors with columns [latitude, longitude, pm25, geometry]
            grid_gdf: Grid cells with columns [grid_id, geometry]
            power: IDW power parameter (default=2 for distance-squared weighting)
            
        Returns:
            DataFrame with columns [grid_id, pm25]
        """
        self.logger.info(f"  Interpolating {len(sensor_gdf)} sensors to {len(grid_gdf):,} grid cells (IDW p={power})...")
        
        # Project to UTM 43N for accurate distance calculation
        sensor_utm = sensor_gdf.to_crs("EPSG:32643")
        grid_utm = grid_gdf.to_crs("EPSG:32643")
        
        # Extract coordinates as numpy arrays
        sensor_coords = np.array([[pt.x, pt.y] for pt in sensor_utm.geometry])  # Shape: (n_sensors, 2)
        grid_coords = np.array([[pt.x, pt.y] for pt in grid_utm.geometry])      # Shape: (n_grids, 2)
        sensor_values = sensor_gdf["pm25"].values  # Shape: (n_sensors,)
        
        # Compute pairwise distances: distance from each grid cell to each sensor
        # Using vectorized broadcasting for efficiency
        # Shape: (n_grids, n_sensors)
        from scipy.spatial.distance import cdist
        distances = cdist(grid_coords, sensor_coords, metric='euclidean')
        
        # Handle zero distances (grid cell exactly at sensor location)
        # Set to small value to avoid division by zero
        distances = np.maximum(distances, 1.0)  # Minimum 1 meter
        
        # Compute IDW weights: w_i = 1 / d_i^p
        weights = 1.0 / (distances ** power)  # Shape: (n_grids, n_sensors)
        
        # Compute weighted average: PM2.5 = sum(w_i * PM2.5_i) / sum(w_i)
        weighted_sum = np.sum(weights * sensor_values, axis=1)  # Shape: (n_grids,)
        weight_sum = np.sum(weights, axis=1)                     # Shape: (n_grids,)
        pm25_interpolated = weighted_sum / weight_sum
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            "grid_id": grid_gdf["grid_id"].values,
            "pm25": pm25_interpolated
        })
        
        self.logger.info(f"  ✓ Interpolated PM2.5 for {len(result_df):,} grid cells")
        self.logger.info(f"    PM2.5 range: {pm25_interpolated.min():.2f} - {pm25_interpolated.max():.2f} µg/m³")
        
        return result_df
    
    def fetch(self, current_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch PM2.5 data for current hour if needed.
        
        Args:
            current_date: Current UTC datetime
            
        Returns:
            DataFrame with fetched data, or None if skip
        """
        # Check if we should fetch
        if not self.should_fetch(current_date):
            return None
        
        # Fetch sensor data from CPCB API
        self.logger.info("  Querying CPCB API for PM2.5 data...")
        sensor_gdf = self.fetch_from_cpcb_api()
        
        if sensor_gdf is None or len(sensor_gdf) == 0:
            self.logger.warning("  No sensor data available, skipping PM2.5 fetch")
            return None
        
        # Load grid
        try:
            grid_gdf = gpd.read_parquet(self.grid_path)
            self.logger.info(f"  ✓ Loaded {len(grid_gdf):,} grid cells from {self.grid_path}")
        except Exception as e:
            self.logger.error(f"  ✗ Failed to load grid: {e}")
            return None
        
        # Interpolate PM2.5 to all grid cells
        try:
            df = self.interpolate_idw(sensor_gdf, grid_gdf, power=2.0)
            
            # Add timestamp column
            df['timestamp'] = current_date
            
            # Reorder: grid_id, timestamp, pm25
            df = df[['grid_id', 'timestamp', 'pm25']]
            
            return df
            
        except Exception as e:
            self.logger.error(f"  ✗ Error interpolating PM2.5: {e}")
            return None
    
    def write_to_db(self, df: pd.DataFrame) -> bool:
        """Write fetched data to DuckDB.
        
        Args:
            df: DataFrame with PM2.5 data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Round timestamp to ceiling of hour (UTC)
            df = df.copy()
            df['timestamp'] = df['timestamp'].dt.ceil('H')
            
            conn = duckdb.connect(str(self.db_path))
            
            # Create table if doesn't exist
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS pm25_observation (
                    grid_id VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    pm25 DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (grid_id, timestamp)
                )
            """)
            
            # Insert data (DuckDB will auto-populate created_at)
            conn.execute("INSERT INTO pm25_observation (grid_id, timestamp, pm25) SELECT * FROM df")
            conn.close()
            
            self.logger.info(f"  ✓ Written {len(df):,} rows to {self.db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"  ✗ Error writing to DuckDB: {e}")
            return False


def main():
    """Main entry point for WriterJob worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WriterJob worker for batch inference")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("WriterJob worker spawned successfully")
    
    # Get current UTC time
    current_date = datetime.now(timezone.utc)
    
    logger.info(f"Current UTC time: {current_date.isoformat()}")
    
    # Initialize and run MeteoFetcher
    logger.info("Starting Meteo fetch...")
    meteo_fetcher = MeteoFetcher(logger)
    
    meteo_df = meteo_fetcher.fetch(current_date)
    
    if meteo_df is not None:
        success = meteo_fetcher.write_to_db(meteo_df)
        if success:
            logger.info("✓ Meteo fetch and write completed")
        else:
            logger.error("✗ Meteo write failed")
    else:
        logger.info("Meteo fetch skipped (no new data or not needed)")
    
    # Initialize and run LandsatFetcher
    logger.info("Starting Landsat8 fetch...")
    landsat_fetcher = LandsatFetcher(logger)
    
    landsat_df = landsat_fetcher.fetch(current_date)
    
    if landsat_df is not None:
        success = landsat_fetcher.write_to_db(landsat_df)
        if success:
            logger.info("✓ Landsat8 fetch and write completed")
        else:
            logger.error("✗ Landsat8 write failed")
    else:
        logger.info("Landsat8 fetch skipped (no new data or not needed)")
    
    # Initialize and run Sentinel2Fetcher
    logger.info("Starting Sentinel2 fetch...")
    sentinel2_fetcher = Sentinel2Fetcher(logger)
    
    sentinel2_df = sentinel2_fetcher.fetch(current_date)
    
    if sentinel2_df is not None:
        success = sentinel2_fetcher.write_to_db(sentinel2_df)
        if success:
            logger.info("✓ Sentinel2 fetch and write completed")
        else:
            logger.error("✗ Sentinel2 write failed")
    else:
        logger.info("Sentinel2 fetch skipped (no new data or not needed)")
    
    logger.info("WriterJob worker completed")
    sys.exit(0)

if __name__ == "__main__":
    main()

