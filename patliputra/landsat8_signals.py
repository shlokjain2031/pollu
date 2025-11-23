#!/usr/bin/env python3
"""
Download Landsat-8 TOA images for each date in landsat8_image_dates.txt,
sample them at grid points, and compute spectral indices (NDVI, MNDWI, NDBI, IBI).

Uses raster_to_grid_df from patliputra.utils.raster_sampling for sampling,
then computes spectral indices inline.

Prerequisites:
  - Earth Engine initialized (ee.Initialize())
  - Grid parquet file exists (you need to create this first)
  
Usage:
  python patliputra/download_and_process_landsat.py \
    --dates-file patliputra/landsat8_image_dates.txt \
    --grid-parquet path/to/your/grid.parquet \
    --output-dir cache/landsat_processed \
    --cache-dir cache/landsat_downloads \
    --project your-ee-project-id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Import the raster_to_grid_df function from raster_sampling
from patliputra.utils.raster_sampling import raster_to_grid_df


def read_dates(dates_file: Path) -> List[str]:
    """Read dates from text file, one per line."""
    with open(dates_file, "r") as f:
        dates = [line.strip() for line in f if line.strip()]
    return sorted(set(dates))  # unique and sorted


def split_bbox_into_tiles(bbox: tuple, tile_size_deg: float = 0.05) -> List[tuple]:
    """
    Split a bounding box into smaller tiles to avoid Earth Engine download limits.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in EPSG:4326
    tile_size_deg : float
        Size of each tile in degrees (default 0.05 ≈ 5.5km at Mumbai latitude)

    Returns
    -------
    List[tuple]
        List of tile bboxes (west, south, east, north)
    """
    west, south, east, north = bbox
    tiles = []

    x = west
    while x < east:
        y = south
        while y < north:
            tile_west = x
            tile_south = y
            tile_east = min(x + tile_size_deg, east)
            tile_north = min(y + tile_size_deg, north)
            tiles.append((tile_west, tile_south, tile_east, tile_north))
            y = tile_north
        x = tile_east

    return tiles


def download_landsat_tile(
    image: ee.Image,
    tile_bbox: tuple,
    tile_path: Path,
) -> Path:
    """
    Download a single tile of a Landsat image.

    Parameters
    ----------
    image : ee.Image
        Earth Engine image
    tile_bbox : tuple
        (west, south, east, north) for this tile
    tile_path : Path
        Where to save the tile

    Returns
    -------
    Path
        Path to downloaded tile
    """
    import requests

    west, south, east, north = tile_bbox
    geometry = ee.Geometry.Rectangle([west, south, east, north])

    url = image.getDownloadURL(
        {"scale": 30, "region": geometry, "filePerBand": False, "format": "GeoTIFF"}
    )

    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    tile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tile_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return tile_path


def download_landsat_image(
    date_str: str,
    bbox: tuple,
    output_path: Path,
    tile_cache_dir: Path,
) -> Path:
    """
    Download a Landsat-8 TOA image for the given date using tiled approach.

    Downloads the image in smaller tiles to avoid Earth Engine 50MB limit,
    then merges them into a single GeoTIFF.

    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format
    bbox : tuple
        (west, south, east, north) in EPSG:4326
    output_path : Path
        Where to save the final merged GeoTIFF
    tile_cache_dir : Path
        Directory to cache individual tiles

    Returns
    -------
    Path
        Path to merged file
    """
    import rasterio
    from rasterio.merge import merge as rio_merge
    import requests

    west, south, east, north = bbox
    geometry = ee.Geometry.Rectangle([west, south, east, north])

    # Query Landsat-8 Collection 2 Level-1 TOA
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
        .filterBounds(geometry)
        .filterDate(date_str, ee.Date(date_str).advance(1, "day"))
        .filter(ee.Filter.lt("CLOUD_COVER", 15))
        .sort("system:time_start")
    )

    # Get first image (or mosaic if multiple)
    count = collection.size().getInfo()
    if count == 0:
        raise RuntimeError(f"No Landsat-8 image found for {date_str}")

    if count == 1:
        image = collection.first()
    else:
        # If multiple scenes, mosaic them
        image = collection.mosaic()

    # Select all TOA bands (B1-B11)
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
    image = image.select(bands)

    # Split into tiles
    tiles = split_bbox_into_tiles(bbox, tile_size_deg=0.05)
    print(f"  Downloading {len(tiles)} tiles for {date_str}...")

    # Download each tile in parallel
    tile_paths = []
    tile_dir = tile_cache_dir / f"tiles_{date_str}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    def download_tile_task(idx: int, tile_bbox: tuple) -> tuple[int, Path | None]:
        """Download a single tile and return (index, path or None)."""
        tile_path = tile_dir / f"tile_{idx:03d}.tif"

        if tile_path.exists():
            return (idx, tile_path)

        try:
            download_landsat_tile(image, tile_bbox, tile_path)
            return (idx, tile_path)
        except Exception as e:
            print(f"    ✗ Tile {idx+1}/{len(tiles)} failed: {e}")
            return (idx, None)

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_tile_task, idx, tile_bbox): idx
            for idx, tile_bbox in enumerate(tiles)
        }

        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            idx, tile_path = future.result()
            if tile_path:
                tile_paths.append(tile_path)
                status = "cached" if tile_path.exists() else "downloaded"
                print(
                    f"    ✓ Tile {idx+1}/{len(tiles)} {status} ({completed_count}/{len(tiles)} complete)"
                )

    if not tile_paths:
        raise RuntimeError(f"Failed to download any tiles for {date_str}")

    # Merge tiles into single GeoTIFF
    print(f"  Merging {len(tile_paths)} tiles...")
    src_files = [rasterio.open(p) for p in tile_paths]

    try:
        mosaic, out_transform = rio_merge(src_files)

        # Get metadata from first tile
        out_meta = src_files[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "deflate",
            }
        )

        # Write merged file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        print(f"  → Merged to {output_path}")

    finally:
        for src in src_files:
            src.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and process Landsat-8 images for dates in a text file"
    )
    parser.add_argument(
        "--dates-file",
        type=Path,
        default=Path("patliputra/landsat8_image_dates.txt"),
        help="Path to dates file (one date per line, YYYY-MM-DD)",
    )
    parser.add_argument(
        "--grid-parquet",
        type=Path,
        required=True,
        help="Path to grid GeoParquet file (must have grid_id and geometry columns)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/landsat_processed"),
        help="Directory to save processed parquet files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/landsat_downloads"),
        help="Directory to cache downloaded GeoTIFFs",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default="72.7763,18.8939,72.9797,19.2701",
        help="Bounding box as west,south,east,north in EPSG:4326",
    )
    parser.add_argument(
        "--target-crs",
        type=str,
        default="EPSG:32643",
        help="Target CRS for reprojection",
    )
    parser.add_argument(
        "--target-res", type=float, default=30.0, help="Target resolution in meters"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Earth Engine project ID",
    )

    args = parser.parse_args()

    # Initialize Earth Engine
    try:
        ee.Initialize(project=args.project)
        print(f"Earth Engine initialized with project: {args.project}")
    except Exception as e:
        print(f"Failed to initialize Earth Engine: {e}")
        print("Make sure you have authenticated with: earthengine authenticate")
        sys.exit(1)

    # Parse bbox
    bbox_parts = [float(x.strip()) for x in args.bbox.split(",")]
    if len(bbox_parts) != 4:
        print("Error: bbox must be west,south,east,north")
        sys.exit(1)
    bbox = tuple(bbox_parts)

    # Read dates
    if not args.dates_file.exists():
        print(f"Error: dates file not found: {args.dates_file}")
        sys.exit(1)

    dates = read_dates(args.dates_file)
    print(f"Found {len(dates)} dates to process")

    # Check grid exists
    if not args.grid_parquet.exists():
        print(f"Error: grid parquet not found: {args.grid_parquet}")
        print(
            "You need to create a grid parquet first with grid_id and geometry columns"
        )
        sys.exit(1)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Process each date
    for idx, date_str in enumerate(dates, start=1):
        print(f"\n[{idx}/{len(dates)}] Processing {date_str}")

        # Define paths
        tif_path = args.cache_dir / f"landsat8_{date_str}.tif"
        out_parquet = args.output_dir / f"landsat8_{date_str}_signals.parquet"

        # Skip if already processed
        if out_parquet.exists():
            print(f"  ✓ Already processed: {out_parquet}")
            continue

        try:
            # Download if not cached
            if not tif_path.exists():
                tif_path = download_landsat_image(
                    date_str, bbox, tif_path, args.cache_dir
                )
            else:
                print(f"  ✓ Using cached: {tif_path}")

            # Process raster using raster_sampling
            print(f"  → Processing raster...")
            df = raster_to_grid_df(
                raster_path=tif_path,
                grid_parquet_path=args.grid_parquet,
                out_parquet=None,  # Don't write yet - we'll add indices first
                cache_dir=args.cache_dir,
                target_crs=args.target_crs,
                target_res=args.target_res,
                use_vrt=True,
                parallel=True,
            )

            # Compute spectral indices (NDVI, MNDWI, NDBI, IBI)
            print(f"  → Computing spectral indices...")
            
            # NDVI = (NIR - Red) / (NIR + Red) = (B5 - B4) / (B5 + B4)
            if "toa_b5" in df.columns and "toa_b4" in df.columns:
                b5 = df["toa_b5"].to_numpy(dtype="float64")
                b4 = df["toa_b4"].to_numpy(dtype="float64")
                denom = b5 + b4
                ndvi = np.where(
                    np.isfinite(denom) & (np.abs(denom) > 0), (b5 - b4) / denom, np.nan
                )
                df["ndvi"] = ndvi

            # MNDWI = (Green - SWIR1) / (Green + SWIR1) = (B3 - B6) / (B3 + B6)
            if "toa_b3" in df.columns and "toa_b6" in df.columns:
                b3 = df["toa_b3"].to_numpy(dtype="float64")
                b6 = df["toa_b6"].to_numpy(dtype="float64")
                denom = b3 + b6
                mndwi = np.where(
                    np.isfinite(denom) & (np.abs(denom) > 0), (b3 - b6) / denom, np.nan
                )
                df["mndwi"] = mndwi

            # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) = (B6 - B5) / (B6 + B5)
            if "toa_b6" in df.columns and "toa_b5" in df.columns:
                b6 = df["toa_b6"].to_numpy(dtype="float64")
                b5 = df["toa_b5"].to_numpy(dtype="float64")
                denom = b6 + b5
                ndbi = np.where(
                    np.isfinite(denom) & (np.abs(denom) > 0), (b6 - b5) / denom, np.nan
                )
                df["ndbi"] = ndbi

            # IBI = (NDBI - (NDVI + MNDWI)/2) / (NDBI + (NDVI + MNDWI)/2)
            # Canonical Index of Built-up Areas per Xu (2008)
            if "ndbi" in df.columns and "ndvi" in df.columns and "mndwi" in df.columns:
                a = df["ndbi"].to_numpy(dtype="float64")
                b = df["ndvi"].to_numpy(dtype="float64")
                c = df["mndwi"].to_numpy(dtype="float64")
                combo = (b + c) / 2.0
                numer = a - combo
                denom = a + combo
                ibi = np.where(
                    np.isfinite(denom) & (np.abs(denom) > 0), numer / denom, np.nan
                )
                df["ibi"] = ibi

            # Write to parquet
            print(f"  → Writing to {out_parquet}...")
            gdf_out = gpd.GeoDataFrame(
                df.drop(columns=["geometry"]), 
                geometry=df["geometry"], 
                crs=args.target_crs
            )
            gdf_out.to_parquet(out_parquet, engine="pyarrow", index=False)
            print(f"  ✓ Saved {len(df)} rows to {out_parquet}")

        except Exception as e:
            print(f"  ✗ Failed to process {date_str}: {e}")
            continue

    print(f"\n✓ Completed processing {len(dates)} dates")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
