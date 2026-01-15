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


def main():
    setup_logging("modis_aod_v2")
    
    parser = argparse.ArgumentParser(
        description="Fetch MODIS MAIAC AOD at native 1km resolution"
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
    
    # Track progress
    new_count = 0
    skipped_count = 0
    failed_count = 0
    failed_dates = []
    
    start_time = time.time()
    
    for idx, date_str in enumerate(dates, start=1):
        artifact_path = output_dir / f"{date_str}.parquet"
        
        # Skip if exists
        if artifact_path.exists():
            skipped_count += 1
            if idx % 100 == 0:
                print(f"[{idx}/{len(dates)}] Skipped {skipped_count} existing, processed {new_count} new")
            continue
        
        try:
            # Get MODIS image for date
            image = get_modis_aod_image(date_str, bbox)
            
            # Fetch AOD at pixel centroids (SINGLE API CALL)
            df = fetch_aod_at_pixel_centroids(image, pixel_grid)
            
            # Count valid values
            valid_count = df['aod_055'].notna().sum() if 'aod_055' in df.columns else 0
            
            # Prepare metadata
            metadata = {
                "date": date_str,
                "city": args.city,
                "source": "MODIS/061/MCD19A2_GRANULES",
                "satellite": "Terra",
                "resolution_m": 1000,
                "bands": list(OUTPUT_COLUMNS.values()),
                "total_pixels": len(df),
                "valid_pixels": int(valid_count),
                "null_fraction": round(1 - valid_count / len(df), 3) if len(df) > 0 else 1.0,
                "bbox": bbox,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            
            # Write artifact
            write_artifact_atomic(df, artifact_path, metadata, manifest, date_str)
            
            new_count += 1
            
            # Progress update
            if idx % 50 == 0 or idx == len(dates):
                elapsed = time.time() - start_time
                rate = new_count / elapsed if elapsed > 0 else 0
                remaining = (len(dates) - idx) / rate if rate > 0 else 0
                print(f"[{idx}/{len(dates)}] New: {new_count}, Skipped: {skipped_count}, "
                      f"Failed: {failed_count} | Rate: {rate:.1f} dates/sec | "
                      f"ETA: {remaining/60:.1f} min")
        
        except RuntimeError as e:
            # No image for this date (expected for some dates)
            if "No MODIS AOD image" in str(e):
                skipped_count += 1
            else:
                failed_count += 1
                failed_dates.append((date_str, str(e)))
        except Exception as e:
            failed_count += 1
            failed_dates.append((date_str, str(e)))
    
    # Summary
    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed_total/60:.1f} minutes")
    print(f"  - New: {new_count}")
    print(f"  - Skipped (existing or no data): {skipped_count}")
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
            f.write(f"# Failed MODIS AOD fetches\n")
            f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            for date_str, error in failed_dates:
                f.write(f"{date_str} | {error}\n")
        print(f"\nFailed log: {failed_log}")


if __name__ == "__main__":
    main()
