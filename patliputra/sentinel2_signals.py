#!/usr/bin/env python3
"""
Fetch Sentinel-2 raw band values for grid cells and write per-date Parquet.

NO intermediate TIFFs are stored. For each date, we:
1. Load the canonical grid
2. Query Sentinel-2 SR Harmonized image collection
3. Batch-fetch band values (B2, B4, B8, AOT, QA10) for all grid cells via reduceRegions
4. Write one parquet per date with schema: grid_id, B2, B4, B8, AOT, QA10
5. Store in artifacts/v1/sentinel2/{city}/{date}.parquet

Sentinel-2 bands: all native 10m, sampled at scale=30m to aggregate to grid resolution.
B2 (Blue), B4 (Red), B8 (NIR), AOT (Aerosol Optical Thickness), QA10 (Quality).

Prerequisites:
  - Earth Engine authenticated (ee.Authenticate() or service account)
  - Grid artifact exists (artifacts/v1/grids/{city}/grid.parquet)
  
Usage:
  python patliputra/copernicus_signals.py \
    --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
    --city mumbai \
    --project your-ee-project-id \
    --start-date 2018-01-01 \
    --end-date 2018-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
import tempfile
import shutil
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import pandas as pd
import geopandas as gpd

from manifest.sentinel2_manifest import Sentinel2Manifest


def enumerate_sentinel2_dates(
    start_date: str,
    end_date: str,
    bbox: tuple,
) -> List[str]:
    """
    Query Sentinel-2 collection and return all available dates.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    bbox : tuple
        (west, south, east, north) in EPSG:4326
    (No cloud cover filtering is applied.)

    Returns
    -------
    List[str]
        List of dates in YYYY-MM-DD format
    """
    west, south, east, north = bbox
    geometry = ee.Geometry.Rectangle([west, south, east, north])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .sort("system:time_start")
    )

    # Get image info
    images = collection.toList(collection.size()).getInfo()
    
    dates = []
    for img in images:
        # Extract date from system:index or system:time_start
        time_start = img["properties"]["system:time_start"]
        date_obj = datetime.fromtimestamp(time_start / 1000, tz=timezone.utc)
        date_str = date_obj.strftime("%Y-%m-%d")
        dates.append(date_str)
    
    # Deduplicate (multiple tiles same day)
    dates = sorted(set(dates))
    print(f"✓ Found {len(dates)} Sentinel-2 dates between {start_date} and {end_date}")
    
    return dates


def get_sentinel2_image(date_str: str, bbox: tuple) -> ee.Image:
    """
    Query Sentinel-2 SR Harmonized for given date and bbox.

    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format
    bbox : tuple
        (west, south, east, north) in EPSG:4326

    Returns
    -------
    ee.Image
        Sentinel-2 image with bands B2, B4, B8, AOT, QA10

    Raises
    ------
    RuntimeError
        If no image found for date
    """
    west, south, east, north = bbox
    geometry = ee.Geometry.Rectangle([west, south, east, north])

    # Query Sentinel-2 SR Harmonized
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(date_str, ee.Date(date_str).advance(1, "day"))
        .sort("system:time_start")
    )

    count = collection.size().getInfo()
    if count == 0:
        raise RuntimeError(f"No Sentinel-2 image found for {date_str}")

    if count == 1:
        image = collection.first()
    else:
        # Multiple scenes (tiles): mosaic
        image = collection.mosaic()

    # Select bands: B2, B4, B8 (10m), AOT (10m), QA10 (10m quality)
    bands = ["B2", "B4", "B8", "AOT", "QA10"]
    return image.select(bands)


def fetch_band_values_batched(
    image: ee.Image,
    grid_gdf: gpd.GeoDataFrame,
    batch_size: int = 1000,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Fetch raw band values for all grid cells using batched reduceRegions (concurrent).

    For each grid cell (30m point), samples Sentinel-2 bands at 30m scale.
    Since Sentinel-2 is 10m native, this aggregates 3x3 = 9 pixels per grid cell (mean).
    Batches are fetched concurrently to improve performance.

    Parameters
    ----------
    image : ee.Image
        Sentinel-2 image with bands B2, B4, B8, AOT, QA10
    grid_gdf : gpd.GeoDataFrame
        Grid with columns: grid_id, geometry (Point)
    batch_size : int
        Number of cells per EE request (default 1000)
    max_workers : int
        Number of concurrent workers (default 10)

    Returns
    -------
    pd.DataFrame
        Columns: grid_id, B2, B4, B8, AOT, QA10
    """
    print(f"  Fetching band values for {len(grid_gdf):,} cells (batch_size={batch_size}, workers={max_workers})...")
    
    # Ensure grid is in WGS84 for EE
    if grid_gdf.crs and grid_gdf.crs.to_string() != "EPSG:4326":
        grid_wgs84 = grid_gdf.to_crs("EPSG:4326")
    else:
        grid_wgs84 = grid_gdf

    total_batches = (len(grid_wgs84) + batch_size - 1) // batch_size
    
    def fetch_batch(batch_idx: int) -> tuple[int, list[dict]]:
        """Fetch a single batch of grid cells."""
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
            scale=30,  # Target grid resolution
            tileScale=1,
        )

        # Fetch results with retry/backoff for transient EE failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_list = reduced.getInfo()["features"]
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"EE getInfo failed after {max_retries} attempts: {e}")
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                time.sleep(wait_time)
        
        # Extract properties
        batch_results = []
        for feat in result_list:
            batch_results.append(feat["properties"])
        
        return (batch_idx, batch_results)
    
    # Submit all batches concurrently
    all_results = []
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch jobs
        futures = {
            executor.submit(fetch_batch, batch_idx): batch_idx
            for batch_idx in range(total_batches)
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            batch_idx, batch_results = future.result()
            all_results.extend(batch_results)
            completed_count += 1
            
            if completed_count % 10 == 0 or completed_count == total_batches:
                print(f"    Completed {completed_count}/{total_batches} batches ({len(all_results):,}/{len(grid_wgs84):,} cells)")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Some Sentinel-2 properties (AOT) may be stored in scaled integer units
    # (e.g. AOT ~ 0.6 stored as 600). Detect and rescale to human units.
    if "AOT" in df.columns:
        try:
            aot_max = float(df["AOT"].dropna().max())
            if aot_max > 10.0:
                # assume AOT stored as per-thousand (x1000)
                df["AOT"] = df["AOT"] / 1000.0
                print("  ⚑ Rescaled AOT by /1000 to convert to unit range (~0-2)")
        except Exception:
            pass

    # Band columns: B2, B4, B8, AOT, QA10
    # Sort numerically to preserve expected order
    band_cols = [c for c in df.columns if c in ["B2", "B4", "B8", "AOT", "QA10"]]
    
    # Custom sort: B-bands numerically, then AOT, then QA10
    def band_sort_key(col: str) -> tuple:
        if col.startswith("B") and col[1:].isdigit():
            return (0, int(col[1:]))
        elif col == "AOT":
            return (1, 0)
        elif col == "QA10":
            return (2, 0)
        else:
            return (3, 0)
    
    band_cols = sorted(band_cols, key=band_sort_key)

    # Ensure we have grid_id
    if "grid_id" not in df.columns:
        raise RuntimeError("grid_id missing from EE results")

    # Keep only grid_id + band columns
    keep_cols = ["grid_id"] + band_cols
    df = df[keep_cols]
    
    # Reindex to match canonical grid order (deterministic row ordering)
    grid_order = grid_gdf[["grid_id"]].copy()
    grid_order["_order"] = range(len(grid_order))
    df = grid_order.merge(df, on="grid_id", how="left").sort_values("_order").drop(columns=["_order"])
    
    print(f"  ✓ Fetched {len(df):,} cells with {len(band_cols)} bands (reindexed to grid order)")
    return df


def write_artifact_atomic(
    df: pd.DataFrame,
    artifact_path: Path,
    metadata: dict,
    manifest: Sentinel2Manifest,
    date_str: str,
) -> None:
    """
    Write parquet artifact atomically via temp file + rename, then register in manifest.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write
    artifact_path : Path
        Final artifact path
    metadata : dict
        Metadata to write alongside (as {artifact_path}.meta.json)
    manifest : Sentinel2Manifest
        Sentinel2 manifest instance for registration
    date_str : str
        Date (YYYY-MM-DD)
    """
    # Mark producing
    request_id = f"sentinel2_{manifest.city}_{date_str}"
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
        
        # Write metadata
        meta_path = artifact_path.with_suffix(".parquet.meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Mark ok in manifest
        manifest.mark_ok(date_str, artifact_path)
        
        print(f"  ✓ Wrote {artifact_path} ({len(df):,} rows)")
        
    except Exception as e:
        # Mark error in manifest
        manifest.mark_error(date_str, str(e))
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Sentinel-2 raw band values for grid cells (per-date Parquet)"
    )
    parser.add_argument(
        "--grid-artifact",
        type=Path,
        required=True,
        help="Path to grid artifact parquet (artifacts/v1/grids/{city}/grid.parquet)",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="mumbai",
        help="City name (used in output path)",
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
        "--batch-size",
        type=int,
        default=1000,
        help="Number of grid cells per EE request",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of concurrent workers for batch fetching (default: 10)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Earth Engine project ID",
    )

    args = parser.parse_args()

    # Initialize Sentinel-2 manifest for this city
    manifest = Sentinel2Manifest(
        city=args.city,
        artifacts_root=args.artifacts_root,
        version="v1"
    )
    print(f"✓ Sentinel-2 manifest initialized: {manifest.db_path}")

    # Initialize Earth Engine
    try:
        ee.Initialize(project=args.project)
        print(f"✓ Earth Engine initialized (project: {args.project})")
    except Exception as e:
        print(f"✗ Failed to initialize Earth Engine: {e}")
        print("  Run: python scripts/auth_earth_engine.py")
        sys.exit(1)

    # Parse bbox
    bbox_parts = [float(x.strip()) for x in args.bbox.split(",")]
    if len(bbox_parts) != 4:
        print("Error: bbox must be west,south,east,north")
        sys.exit(1)
    bbox = tuple(bbox_parts)

    # Load grid artifact
    if not args.grid_artifact.exists():
        print(f"✗ Grid artifact not found: {args.grid_artifact}")
        sys.exit(1)

    print(f"Loading grid from {args.grid_artifact}...")
    grid_gdf = gpd.read_parquet(args.grid_artifact)
    print(f"  ✓ Loaded {len(grid_gdf):,} grid cells")
    
    # Check required columns
    if "grid_id" not in grid_gdf.columns:
        print("✗ Grid must have 'grid_id' column")
        sys.exit(1)
    if "geometry" not in grid_gdf.columns:
        print("✗ Grid must have 'geometry' column")
        sys.exit(1)

    # Enumerate Sentinel-2 dates
    print(f"\nEnumerating Sentinel-2 dates from {args.start_date} to {args.end_date}...")
    dates = enumerate_sentinel2_dates(
        args.start_date,
        args.end_date,
        bbox,
    )
    print(f"✓ Found {len(dates)} dates to process")

    # Prepare output directory
    output_dir = args.artifacts_root / "v1" / "sentinel2" / args.city
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")

    # Track progress
    failed_dates = []
    processed_count = 0

    # Process each date
    for idx, date_str in enumerate(dates, start=1):
        print(f"\n[{idx}/{len(dates)}] Processing {date_str}")

        # Use canonical artifacts layout: artifacts/v1/sentinel2/{city}/{date}.parquet
        artifact_path = args.artifacts_root / "v1" / "sentinel2" / args.city / f"{date_str}.parquet"

        # Skip if already exists
        if artifact_path.exists():
            print(f"  ✓ Already exists: {artifact_path}")
            processed_count += 1
            continue

        try:
            # Get Sentinel-2 image
            print(f"  Querying Sentinel-2 collection...")
            image = get_sentinel2_image(date_str, bbox)
            print(f"  ✓ Image found")

            # Fetch band values for all grid cells
            df = fetch_band_values_batched(
                image=image,
                grid_gdf=grid_gdf,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
            )

            # Prepare metadata
            metadata = {
                "date": date_str,
                "city": args.city,
                "source": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": ["B2", "B4", "B8", "AOT", "QA10"],
                "native_resolution_m": 10,
                "sampling_scale_m": 30,
                "grid_cells": len(df),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "bbox": bbox,
            }

            # Write artifact
            write_artifact_atomic(df, artifact_path, metadata, manifest, date_str)
            processed_count += 1

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_dates.append((date_str, str(e)))
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Processed {processed_count}/{len(dates)} dates")
    print(f"  - Successful: {processed_count}")
    print(f"  - Failed: {len(failed_dates)}")
    print(f"Output: {output_dir}")

    if failed_dates:
        print(f"\n⚠ Failed dates:")
        for date_str, error in failed_dates:
            print(f"  - {date_str}: {error[:80]}")
        
        # Write failed log
        failed_log = output_dir / "failed_dates.log"
        with open(failed_log, "w") as f:
            f.write("# Failed Sentinel-2 fetches\n")
            f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            for date_str, error in failed_dates:
                f.write(f"{date_str} | {error}\n")
        print(f"\nFailed log: {failed_log}")


if __name__ == "__main__":
    main()
