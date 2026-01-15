#!/usr/bin/env python3
"""
Fetch MODIS MAIAC AOD at native 1km resolution.

SCIENTIFIC APPROACH:
- MODIS MAIAC retrieves AOD at ~1km resolution
- We sample at native resolution (no fake resampling to 30m)
- Each 30m grid cell deterministically maps to its parent 1km pixel at training time
- AOD is treated as a coarse-scale covariate (like meteorology)

For each date, we:
1. Generate MODIS pixel centroids for Mumbai bbox (~924 pixels)
2. Sample AOD values at pixel centroids via single reduceRegions call
3. Write parquet with schema: pixel_id, pixel_row, pixel_col, aod_047, aod_055, ...
4. Store in artifacts/v1/modis_aod/mumbai/{date}.parquet

PERFORMANCE:
- ~924 pixels per date (vs 509,209 grid cells previously)
- 1 API call per date (~1-2 seconds)
- Full 2019-2024 dataset: ~73 minutes (vs 20 days previously)

TRAINING TIME JOIN:
- Each 30m grid cell computes its MODIS pixel_id via O(1) coordinate math
- Join on pixel_id to inherit AOD values
- ~1089 grid cells share each MODIS pixel (33x33 at 30m)

Prerequisites:
  - Earth Engine authenticated
  - Grid artifact exists (for bbox extraction)

Usage:
  python -m patliputra.modis_aod_signals_v2 \
    --city mumbai \
    --project pollu-483104 \
    --start-date 2019-01-01 \
    --end-date 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import tempfile
import shutil
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import numpy as np
import pandas as pd

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from manifest.modis_aod_manifest import ModisAodManifest
from patliputra.logging_utils import setup_logging


# Mumbai bounding box (EPSG:4326)
MUMBAI_BBOX = (72.7763, 18.8939, 72.9797, 19.2701)  # west, south, east, north

# MODIS MAIAC pixel size at ~19°N latitude (degrees)
# 1000m / 105,300m (longitude at 19°N) ≈ 0.00949°
# 1000m / 111,000m (latitude) ≈ 0.00901°
MODIS_PIXEL_SIZE_LON = 0.00949  # degrees
MODIS_PIXEL_SIZE_LAT = 0.00901  # degrees

# MODIS MAIAC bands to fetch (Terra satellite only)
MODIS_AOD_BANDS = [
    "Optical_Depth_047",  # AOD at 470nm
    "Optical_Depth_055",  # AOD at 550nm
    "AOD_Uncertainty",    # Uncertainty estimate
    "AOD_QA",             # Quality flags (int32)
    "Column_WV",          # Column water vapor
]

# Output column names (shorter for parquet)
OUTPUT_COLUMNS = {
    "Optical_Depth_047": "aod_047",
    "Optical_Depth_055": "aod_055",
    "AOD_Uncertainty": "aod_uncertainty",
    "AOD_QA": "aod_qa",
    "Column_WV": "column_wv",
}

# MODIS MAIAC scale factors (from product documentation)
# AOD values are stored as integers, need to multiply by 0.001
# https://lpdaac.usgs.gov/products/mcd19a2v061/
SCALE_FACTORS = {
    "aod_047": 0.001,
    "aod_055": 0.001,
    "aod_uncertainty": 0.0001,  # Uncertainty scale factor
    "column_wv": 0.001,  # Column water vapor in cm
}


def generate_modis_pixel_grid(bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Generate MODIS pixel grid for bounding box at native 1km resolution.
    
    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in EPSG:4326 degrees
    
    Returns
    -------
    pd.DataFrame
        Columns: pixel_id, pixel_row, pixel_col, pixel_lon, pixel_lat
        Rows: One per MODIS pixel (~924 for Mumbai)
    """
    west, south, east, north = bbox
    
    # Calculate grid dimensions
    ncols = int(np.ceil((east - west) / MODIS_PIXEL_SIZE_LON))
    nrows = int(np.ceil((north - south) / MODIS_PIXEL_SIZE_LAT))
    
    print(f"MODIS pixel grid: {nrows} rows x {ncols} cols = {nrows * ncols} pixels")
    
    pixels = []
    for row in range(nrows):
        for col in range(ncols):
            # Pixel center coordinates (top-left origin, row increases southward)
            pixel_lon = west + (col + 0.5) * MODIS_PIXEL_SIZE_LON
            pixel_lat = north - (row + 0.5) * MODIS_PIXEL_SIZE_LAT
            
            # Validate pixel center is within bbox
            if pixel_lon > east or pixel_lat < south:
                continue
            
            pixels.append({
                'pixel_row': row,
                'pixel_col': col,
                'pixel_id': f'{row}_{col}',
                'pixel_lon': pixel_lon,
                'pixel_lat': pixel_lat,
            })
    
    df = pd.DataFrame(pixels)
    print(f"Generated {len(df)} MODIS pixel centroids")
    return df


def get_modis_pixel_id(lon: float, lat: float, bbox: Tuple[float, float, float, float]) -> str:
    """
    Deterministically map (lon, lat) to MODIS pixel_id.
    
    This is O(1) coordinate math - no lookup table needed.
    Used at training time to join 30m grid cells to MODIS pixels.
    
    Parameters
    ----------
    lon, lat : float
        Coordinates in EPSG:4326 degrees
    bbox : tuple
        (west, south, east, north) bounding box
    
    Returns
    -------
    str
        Pixel ID in format "{row}_{col}"
    """
    west, south, east, north = bbox
    
    # Column: distance from western edge (floor assignment)
    pixel_col = int((lon - west) / MODIS_PIXEL_SIZE_LON)
    
    # Row: distance from northern edge (image origin is top-left)
    pixel_row = int((north - lat) / MODIS_PIXEL_SIZE_LAT)
    
    return f'{pixel_row}_{pixel_col}'


def generate_date_range(start_date: str, end_date: str) -> List[str]:
    """Generate list of dates from start to end (inclusive) as YYYY-MM-DD strings."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return dates


def get_modis_aod_image(date_str: str, bbox: Tuple[float, float, float, float]) -> ee.Image:
    """
    Query MODIS MAIAC AOD for given date and bbox (Terra only).
    
    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format
    bbox : tuple
        (west, south, east, north) in EPSG:4326
    
    Returns
    -------
    ee.Image
        MODIS AOD image with selected bands
    
    Raises
    ------
    RuntimeError
        If no image found for date
    """
    west, south, east, north = bbox
    geometry = ee.Geometry.Rectangle([west, south, east, north])
    
    # Query MODIS MAIAC granules (Terra only via SATELLITE property)
    collection = (
        ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
        .filterBounds(geometry)
        .filterDate(date_str, ee.Date(date_str).advance(1, "day"))
        .filter(ee.Filter.eq("SATELLITE", "A"))  # Terra = A
        .sort("system:time_start")
    )
    
    count = collection.size().getInfo()
    if count == 0:
        raise RuntimeError(f"No MODIS AOD image (Terra) found for {date_str}")
    
    # Mosaic multiple granules if present
    if count == 1:
        image = collection.first()
    else:
        image = collection.mosaic()
    
    # Select AOD bands
    return image.select(MODIS_AOD_BANDS)


def fetch_aod_at_pixel_centroids(
    image: ee.Image,
    pixel_grid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fetch AOD values at MODIS pixel centroids via single reduceRegions call.
    
    Parameters
    ----------
    image : ee.Image
        MODIS AOD image with bands
    pixel_grid : pd.DataFrame
        MODIS pixel grid with pixel_id, pixel_lon, pixel_lat
    
    Returns
    -------
    pd.DataFrame
        Columns: pixel_id, pixel_row, pixel_col, aod_047, aod_055, ...
    """
    # Convert pixel grid to EE FeatureCollection
    features = []
    for _, row in pixel_grid.iterrows():
        feat = ee.Feature(
            ee.Geometry.Point([row['pixel_lon'], row['pixel_lat']])
        )
        feat = feat.set({
            'pixel_id': row['pixel_id'],
            'pixel_row': row['pixel_row'],
            'pixel_col': row['pixel_col'],
        })
        features.append(feat)
    
    fc = ee.FeatureCollection(features)
    
    # Sample image at pixel centroids (SINGLE API CALL for all ~924 pixels)
    reduced = image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),  # Point sampling at native resolution
        scale=1000,  # MODIS native resolution
        tileScale=1,
    )
    
    # Fetch results with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result_list = reduced.getInfo()["features"]
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"EE getInfo failed after {max_retries} attempts: {e}")
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    
    # Extract properties
    data = []
    for feat in result_list:
        props = feat["properties"]
        data.append(props)
    
    df = pd.DataFrame(data)
    
    # Rename columns to shorter output names
    rename_map = {}
    for orig_col, new_col in OUTPUT_COLUMNS.items():
        if orig_col in df.columns:
            rename_map[orig_col] = new_col
    df = df.rename(columns=rename_map)
    
    # Ensure expected columns exist
    expected_cols = ['pixel_id', 'pixel_row', 'pixel_col'] + list(OUTPUT_COLUMNS.values())
    keep_cols = [c for c in expected_cols if c in df.columns]
    df = df[keep_cols]
    
    # Apply scale factors to AOD bands
    for col, scale_factor in SCALE_FACTORS.items():
        if col in df.columns:
            df[col] = df[col] * scale_factor
    
    # Convert aod_qa to int32
    if 'aod_qa' in df.columns:
        df['aod_qa'] = df['aod_qa'].astype('Int32')
    
    return df


def write_artifact_atomic(
    df: pd.DataFrame,
    artifact_path: Path,
    metadata: dict,
    manifest: ModisAodManifest,
    date_str: str,
) -> None:
    """
    Write parquet artifact atomically and register in manifest.
    """
    request_id = f"modis_aod_{manifest.city}_{date_str}"
    manifest.mark_producing(date_str, request_id)
    
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=artifact_path.parent,
            suffix=".parquet.tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
            df.to_parquet(tmp_path, engine="pyarrow", index=False)
        
        # Atomic rename
        shutil.move(str(tmp_path), str(artifact_path))
        
        # Write metadata sidecar
        meta_path = artifact_path.with_suffix(".parquet.meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Mark success
        manifest.mark_ok(date_str, str(artifact_path))
        
    except Exception as e:
        manifest.mark_error(date_str, str(e))
        raise


def create_date_batches(dates: List[str], batch_size: int = 5) -> List[List[str]]:
    """
    Group dates into batches for parallel processing.
    
    Parameters
    ----------
    dates : List[str]
        List of dates in YYYY-MM-DD format
    batch_size : int
        Number of dates per batch (default: 5)
    
    Returns
    -------
    List[List[str]]
        List of date batches
    """
    batches = []
    for i in range(0, len(dates), batch_size):
        batches.append(dates[i:i + batch_size])
    return batches


def process_date_batch(
    date_batch: List[str],
    pixel_grid: pd.DataFrame,
    bbox: tuple,
    output_dir: Path,
    manifest: ModisAodManifest,
) -> Dict[str, str]:
    """
    Process a batch of dates in a single Earth Engine API call.
    
    This stacks multiple dates' images into a multi-band image and samples
    once, significantly reducing API calls.
    
    Parameters
    ----------
    date_batch : List[str]
        List of dates to process (e.g., ['2024-01-01', '2024-01-02', ...])
    pixel_grid : pd.DataFrame
        MODIS pixel grid with centroids
    bbox : tuple
        Bounding box
    output_dir : Path
        Output directory for parquet files
    manifest : ModisAodManifest
        Manifest for tracking
    
    Returns
    -------
    Dict[str, str]
        Status for each date: {date: 'success' | 'skipped' | 'no_data' | 'error: msg'}
    """
    results = {}
    
    # Check which dates need processing
    dates_to_process = []
    for date_str in date_batch:
        artifact_path = output_dir / f"{date_str}.parquet"
        if artifact_path.exists():
            results[date_str] = 'skipped'
        else:
            dates_to_process.append(date_str)
    
    if not dates_to_process:
        return results
    
    try:
        # Get images for all dates in batch
        images_dict = {}
        for date_str in dates_to_process:
            try:
                image = get_modis_aod_image(date_str, bbox)
                images_dict[date_str] = image
            except RuntimeError as e:
                if "No MODIS AOD image" in str(e):
                    results[date_str] = 'no_data'
                else:
                    results[date_str] = f'error: {e}'
        
        if not images_dict:
            return results
        
        # Stack all images into multi-band image
        # Each date's 5 bands become: aod_047_DATE, aod_055_DATE, etc.
        stacked_bands = []
        band_to_date = {}  # Maps band name -> (date, original_band)
        
        for date_str, image in images_dict.items():
            for band_name in MODIS_AOD_BANDS:
                # Rename band to include date
                new_band_name = f"{band_name}_{date_str.replace('-', '')}"
                stacked_bands.append(image.select([band_name]).rename([new_band_name]))
                band_to_date[new_band_name] = (date_str, band_name)
        
        # Combine all bands into single image
        if len(stacked_bands) == 1:
            stacked_image = stacked_bands[0]
        else:
            stacked_image = stacked_bands[0]
            for band_img in stacked_bands[1:]:
                stacked_image = stacked_image.addBands(band_img)
        
        # Sample at pixel centroids (SINGLE API CALL for all dates!)
        df_multi = fetch_aod_at_pixel_centroids_multidates(stacked_image, pixel_grid, band_to_date)
        
        # Split multi-date dataframe into per-date dataframes and write
        for date_str in images_dict.keys():
            date_key = date_str.replace('-', '')
            
            # Extract columns for this date
            date_cols = ['pixel_id', 'pixel_row', 'pixel_col']
            for orig_band, out_col in OUTPUT_COLUMNS.items():
                multi_col = f"{orig_band}_{date_key}"
                if multi_col in df_multi.columns:
                    date_cols.append(multi_col)
            
            df_date = df_multi[date_cols].copy()
            
            # Rename columns back to standard names
            rename_map = {}
            for orig_band, out_col in OUTPUT_COLUMNS.items():
                multi_col = f"{orig_band}_{date_key}"
                if multi_col in df_date.columns:
                    rename_map[multi_col] = out_col
            df_date = df_date.rename(columns=rename_map)
            
            # Apply scale factors to AOD bands
            for col, scale_factor in SCALE_FACTORS.items():
                if col in df_date.columns:
                    df_date[col] = df_date[col] * scale_factor
            
            # Convert aod_qa to Int32
            if 'aod_qa' in df_date.columns:
                df_date['aod_qa'] = df_date['aod_qa'].astype('Int32')
            
            # Count valid values
            valid_count = df_date['aod_055'].notna().sum() if 'aod_055' in df_date.columns else 0
            
            # Metadata
            metadata = {
                "date": date_str,
                "city": manifest.city,
                "source": "MODIS/061/MCD19A2_GRANULES",
                "satellite": "Terra",
                "resolution_m": 1000,
                "bands": list(OUTPUT_COLUMNS.values()),
                "total_pixels": len(df_date),
                "valid_pixels": int(valid_count),
                "null_fraction": round(1 - valid_count / len(df_date), 3) if len(df_date) > 0 else 1.0,
                "bbox": bbox,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "batch_processing": True,
            }
            
            # Write artifact
            artifact_path = output_dir / f"{date_str}.parquet"
            try:
                write_artifact_atomic(df_date, artifact_path, metadata, manifest, date_str)
                results[date_str] = 'success'
            except Exception as e:
                results[date_str] = f'error: {e}'
        
        return results
    
    except Exception as e:
        # Batch failed entirely
        for date_str in dates_to_process:
            if date_str not in results:
                results[date_str] = f'batch_error: {e}'
        return results


def fetch_aod_at_pixel_centroids_multidates(
    stacked_image: ee.Image,
    pixel_grid: pd.DataFrame,
    band_to_date: Dict[str, tuple],
) -> pd.DataFrame:
    """
    Fetch multi-date AOD values from stacked image.
    
    Similar to fetch_aod_at_pixel_centroids but handles multi-band image
    where bands are named like 'Optical_Depth_055_20240101'.
    
    Parameters
    ----------
    stacked_image : ee.Image
        Multi-band image with bands for multiple dates
    pixel_grid : pd.DataFrame
        MODIS pixel grid
    band_to_date : Dict[str, tuple]
        Maps band name to (date_str, original_band_name)
    
    Returns
    -------
    pd.DataFrame
        All columns from all dates, pivot format
    """
    # Convert pixel grid to EE FeatureCollection
    features = []
    for _, row in pixel_grid.iterrows():
        feat = ee.Feature(
            ee.Geometry.Point([row['pixel_lon'], row['pixel_lat']])
        )
        feat = feat.set({
            'pixel_id': row['pixel_id'],
            'pixel_row': row['pixel_row'],
            'pixel_col': row['pixel_col'],
        })
        features.append(feat)
    
    fc = ee.FeatureCollection(features)
    
    # Sample all bands at once
    reduced = stacked_image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=1000,
        tileScale=1,
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
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    
    # Extract properties
    data = []
    for feat in result_list:
        props = feat["properties"]
        data.append(props)
    
    df = pd.DataFrame(data)
    
    return df


def main():
    setup_logging("modis_aod_v2")
    
    parser = argparse.ArgumentParser(
        description="Fetch MODIS MAIAC AOD at native 1km resolution (batched & parallel)"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="mumbai",
        help="City name",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory for artifacts",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default="72.7763,18.8939,72.9797,19.2701",
        help="Bounding box as west,south,east,north in EPSG:4326",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Earth Engine project ID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of dates per batch (default: 5, reduces API calls)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent batch workers (default: 4)",
    )
    
    args = parser.parse_args()
    
    # Parse bbox
    bbox_parts = [float(x.strip()) for x in args.bbox.split(",")]
    if len(bbox_parts) != 4:
        print("Error: bbox must be west,south,east,north")
        sys.exit(1)
    bbox = tuple(bbox_parts)
    
    # Initialize manifest
    manifest = ModisAodManifest(
        city=args.city,
        artifacts_root=args.artifacts_root,
        version="v1"
    )
    print(f"Manifest initialized: {manifest.db_path}")
    
    # Initialize Earth Engine
    try:
        ee.Initialize(project=args.project)
        print(f"Earth Engine initialized (project: {args.project})")
    except Exception as e:
        print(f"Failed to initialize Earth Engine: {e}")
        sys.exit(1)
    
    # Generate MODIS pixel grid (one-time per run)
    print(f"\nGenerating MODIS pixel grid for bbox {bbox}...")
    pixel_grid = generate_modis_pixel_grid(bbox)
    print(f"Total MODIS pixels: {len(pixel_grid)}")
    
    # Save pixel grid reference (useful for debugging/documentation)
    pixel_grid_path = args.artifacts_root / "v1" / "modis_aod" / args.city / "modis_pixel_grid.parquet"
    pixel_grid_path.parent.mkdir(parents=True, exist_ok=True)
    pixel_grid.to_parquet(pixel_grid_path, index=False)
    print(f"Pixel grid saved: {pixel_grid_path}")
    
    # Generate date range
    dates = generate_date_range(args.start_date, args.end_date)
    print(f"\nProcessing {len(dates)} dates from {args.start_date} to {args.end_date}")
    
    # Output directory
    output_dir = args.artifacts_root / "v1" / "modis_aod" / args.city
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create date batches
    date_batches = create_date_batches(dates, batch_size=args.batch_size)
    print(f"\nCreated {len(date_batches)} batches ({args.batch_size} dates per batch)")
    print(f"Parallel workers: {args.max_workers}")
    
    # Track progress
    new_count = 0
    skipped_count = 0
    no_data_count = 0
    failed_count = 0
    failed_dates = []
    
    start_time = time.time()
    processed_batches = 0
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all batch jobs
        futures = {
            executor.submit(
                process_date_batch,
                batch,
                pixel_grid,
                bbox,
                output_dir,
                manifest
            ): batch
            for batch in date_batches
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            batch = futures[future]
            processed_batches += 1
            
            try:
                batch_results = future.result()
                
                # Aggregate stats
                for date_str, status in batch_results.items():
                    if status == 'success':
                        new_count += 1
                    elif status == 'skipped':
                        skipped_count += 1
                    elif status == 'no_data':
                        no_data_count += 1
                    elif status.startswith('error') or status.startswith('batch_error'):
                        failed_count += 1
                        failed_dates.append((date_str, status))
                
                # Progress update
                if processed_batches % 10 == 0 or processed_batches == len(date_batches):
                    elapsed = time.time() - start_time
                    total_processed = new_count + skipped_count + no_data_count + failed_count
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    remaining_dates = len(dates) - total_processed
                    eta_sec = remaining_dates / rate if rate > 0 else 0
                    
                    print(f"[Batch {processed_batches}/{len(date_batches)}] "
                          f"New: {new_count}, Skipped: {skipped_count}, "
                          f"No data: {no_data_count}, Failed: {failed_count} | "
                          f"Rate: {rate:.1f} dates/sec | ETA: {eta_sec/60:.1f} min")
            
            except Exception as e:
                # Entire batch failed
                for date_str in batch:
                    failed_count += 1
                    failed_dates.append((date_str, f"batch exception: {e}"))
    
    # Summary
    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed_total/60:.1f} minutes")
    print(f"  - New: {new_count}")
    print(f"  - Skipped (existing): {skipped_count}")
    print(f"  - No data: {no_data_count}")
    print(f"  - Failed: {failed_count}")
    print(f"Output: {output_dir}")
    
    if failed_dates:
        print(f"\nFailed dates ({len(failed_dates)}):")
        for date_str, error in failed_dates[:10]:
            print(f"  - {date_str}: {error[:60]}")
        if len(failed_dates) > 10:
            print(f"  ... and {len(failed_dates) - 10} more")
        
        # Write failed log
        failed_log = output_dir / "failed_dates.log"
        with open(failed_log, "w") as f:
            f.write(f"# Failed MODIS AOD fetches (batched processing)\n")
            f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            for date_str, error in failed_dates:
                f.write(f"{date_str} | {error}\n")
        print(f"\nFailed log: {failed_log}")


if __name__ == "__main__":
    main()
