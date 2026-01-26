"""
WriterJob Worker - Phase 2b

Part of the Kashi batch inference scheduler.
Spawned by Python orchestrator (kashi/orchestrator.py) @ t=-2m.

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

3. Worker fetches in parallel (ThreadPoolExecutor, max_workers=4):

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

4. Worker writes to DuckDB (parallel, separate files):
   - All 4 fetches happen in parallel (max time = slowest, typically Sentinel2)
   - Then writes happen in parallel via ThreadPoolExecutor (max_workers=4)
   - Each write thread handles one separate DuckDB file (no locking contention)
   - Each write: INSERT into table with fetched data

   a. Write meteo → meteo.duckdb (Thread 1)
      - 500K rows × ~20 columns
      - Duration: ~2-3 seconds
      - Includes grid_id, timestamp, all meteo fields
      - Independent file: no locking with other writes

   b. Write PM2.5 → pm25_observations.duckdb (Thread 2)
      - ~10-26 rows (one per monitoring station)
      - Duration: <1 second
      - Includes latitude, longitude, avg_value, timestamp
      - Independent file: no locking with other writes

   c. Write Landsat8 → landsat8.duckdb (Thread 3, if available)
      - 500K rows × 11 bands
      - Duration: ~2-3 seconds
      - Includes grid_id, timestamp, B1-B11
      - Independent file: no locking with other writes

   d. Write Sentinel2 → sentinel2.duckdb (Thread 4, if available)
      - 500K rows × 5 bands
      - Duration: ~2-3 seconds
      - Includes grid_id, timestamp, B2, B4, B8, AOT, QA10
      - Independent file: no locking with other writes

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
- DuckDB writes: 2-3 seconds max (parallel, 4 threads, no contention)
  - Thread 1: meteo write (2-3s)
  - Thread 2: PM2.5 write (<1s, shorter)
  - Thread 3: Landsat8 write (2-3s, if available)
  - Thread 4: Sentinel2 write (2-3s, if available)
  - Total write time = max(2-3s, <1s, 2-3s, 2-3s) = 2-3 seconds (parallel)
- **Total WriterJob duration: 20-40 seconds (95% of runs <40 seconds)**
- 120-second window provides 3x buffer

Example Invocation (by orchestrator):
  python3 -m takshashila.distributed.worker.write_job_worker \
    --cycle-id 550e8400-e29b-41d4-a716-446655440000 \
    --timestamp-to-write 2026-01-25T14:00:00Z \
    --config /path/to/kashi/config.yaml

Logging Output:
All logs to stderr with format:
  [cycle_id] [timestamp] [level] message

Example:
  [550e8400-e29b] [2026-01-25 14:00:05] INFO WriterJob starting
  [550e8400-e29b] [2026-01-25 14:00:05] INFO Fetching from 4 sources in parallel (timeout: 60s)
  [550e8400-e29b] [2026-01-25 14:00:07] INFO Fetched meteo: 500000 rows
  [550e8400-e29b] [2026-01-25 14:00:09] INFO Fetched PM2.5: 26 rows
  [550e8400-e29b] [2026-01-25 14:00:15] INFO Landsat8 not available today (expected 1/16 days)
  [550e8400-e29b] [2026-01-25 14:00:32] INFO Fetched Sentinel2: 500000 rows
  [550e8400-e29b] [2026-01-25 14:00:32] INFO All fetches completed in 27.3s
  [550e8400-e29b] [2026-01-25 14:00:32] INFO Writing to DuckDB (parallel, 4 threads)
  [550e8400-e29b] [2026-01-25 14:00:35] INFO Written meteo: 500000 rows (2.1s, thread 1)
  [550e8400-e29b] [2026-01-25 14:00:35] INFO Written PM2.5: 26 rows (0.2s, thread 2)
  [550e8400-e29b] [2026-01-25 14:00:35] INFO Written Sentinel2: 500000 rows (2.8s, thread 3)
  [550e8400-e29b] [2026-01-25 14:00:35] INFO All DuckDB writes completed in 2.8s (parallel)
  [550e8400-e29b] [2026-01-25 14:00:35] INFO WriterJob completed successfully (2.8s writes, 30.1s total)

Integration Points:
- OpenMeteo API (https://archive-api.open-meteo.com/v1/archive)
- CPCB API (data.gov.in)
- Google Earth Engine API (geemap or ee library)
- DuckDB (for writing data)
- kashi.config (for paths, credentials, timeouts)

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
)


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
            
            if days_diff > 16:
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
            df = self.fetch_band_values_batched(image, grid_gdf, batch_size=500, max_workers=4)
            
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
            
            conn = duckdb.connect(str(self.db_path))
            
            # Create table if doesn't exist
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS landsat8 (
                    grid_id VARCHAR,
                    timestamp TIMESTAMP,
                    B1 DOUBLE,
                    B2 DOUBLE,
                    B3 DOUBLE,
                    B4 DOUBLE,
                    B5 DOUBLE,
                    B6 DOUBLE,
                    B7 DOUBLE,
                    B8 DOUBLE,
                    B9 DOUBLE,
                    B10 DOUBLE,
                    B11 DOUBLE,
                    scene_cloud_cover DOUBLE
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


def main():
    """Main entry point for WriterJob worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WriterJob worker for batch inference")
    parser.add_argument(
        "--current-date",
        type=str,
        default=None,
        help="Override current date (YYYY-MM-DD HH:MM:SS) for testing",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("WriterJob worker spawned successfully")
    
    # Get current UTC time
    if args.current_date:
        current_date = datetime.fromisoformat(args.current_date.replace(" ", "T"))
        if current_date.tzinfo is None:
            current_date = current_date.replace(tzinfo=timezone.utc)
        logger.info(f"Using test date: {current_date.isoformat()}")
    else:
        current_date = datetime.now(timezone.utc)
    
    logger.info(f"Current UTC time: {current_date.isoformat()}")
    
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
    
    logger.info("WriterJob worker completed")
    sys.exit(0)

if __name__ == "__main__":
    main()

