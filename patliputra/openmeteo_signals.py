#!/usr/bin/env python3
"""
Fetch meteorological data from Open-Meteo and map to 30m grid.

Strategy: Coarse-to-Fine Tile Mapping
-------------------------------------
Open-Meteo provides point data from reanalysis models at coarse resolution:
    - ECMWF IFS: ~9km (default, 2017-present)
    - ERA5-Land: ~11km (1950-present)
    - ERA5: ~25km (1940-present)

Since any lat/lon within a ~9km pixel returns the same value, we:
1. Compute "meteo tile ID" for each 30m grid cell by snapping to reanalysis grid
2. Fetch unique tile centroids from Open-Meteo (Mumbai fits in ~3-6 tiles)
3. Broadcast tile values to all 30m cells within each tile

Artifact Layout
---------------
    artifacts/v1/openmeteo/{city}/
    ├── manifest/openmeteo_manifest.db    # Per-city manifest
    ├── {date}.parquet                    # Daily snapshot: grid_id + overpass + aggregates
    ├── {date}_hourly.parquet             # Hourly series: grid_id + datetime + meteo (24h × cells)
    └── {date}.parquet.meta.json          # Metadata sidecar

Variables (hourly, matched to overpass time + daily aggregates):
    - temperature_2m: Air temperature at 2m (°C)
    - relative_humidity_2m: Relative humidity at 2m (%)
    - dewpoint_2m: Dewpoint temperature at 2m (°C)
    - surface_pressure: Surface pressure (hPa)
    - wind_speed_10m: Wind speed at 10m (m/s)
    - wind_direction_10m: Wind direction at 10m (°)
    - precipitation: Total precipitation (mm)
    - cloud_cover: Cloud cover (%)

Usage:
  # Daily snapshot mode (default)
  python patliputra/openmeteo_fetch.py \
    --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
    --city mumbai \
    --start-date 2018-01-01 \
    --end-date 2018-01-31

  # Hourly time series mode (24 rows per cell per date)
  python patliputra/openmeteo_fetch.py \
    --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
    --city mumbai \
    --start-date 2018-01-01 \
    --end-date 2018-01-31 \
    --hourly

References:
    - https://open-meteo.com/en/docs/historical-weather-api
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

# Add project root to path for imports
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from manifest.openmeteo_manifest import OpenMeteoManifest


# Reanalysis model resolutions in meters
MODEL_RESOLUTION_M = {
    "ecmwf_ifs": 9000,      # ~9km
    "era5_land": 11000,     # ~11km
    "era5": 25000,          # ~25km
}

# Default meteorology variables
DEFAULT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]


def compute_meteo_tile_id(
    x_m: float,
    y_m: float,
    tile_size_m: float,
) -> Tuple[int, int]:
    """
    Compute meteo tile ID by snapping coordinates to tile grid.
    
    Parameters
    ----------
    x_m : float
        X coordinate in meters (projected CRS)
    y_m : float
        Y coordinate in meters (projected CRS)
    tile_size_m : float
        Tile size in meters (e.g., 9000 for ECMWF IFS)
    
    Returns
    -------
    Tuple[int, int]
        (tile_col, tile_row) IDs
    """
    tile_col = int(x_m // tile_size_m)
    tile_row = int(y_m // tile_size_m)
    return (tile_col, tile_row)


def assign_meteo_tiles(
    grid_gdf: gpd.GeoDataFrame,
    tile_size_m: float,
) -> pd.DataFrame:
    """
    Assign each grid cell to a meteo tile and compute tile centroids.
    
    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Grid with columns: grid_id, geometry (Point), x_m, y_m (or projected coords)
    tile_size_m : float
        Tile size in meters
    
    Returns
    -------
    pd.DataFrame
        Grid cells with meteo_tile_id column and unique tile info
    """
    # Ensure we have projected coordinates
    if "x_m" not in grid_gdf.columns or "y_m" not in grid_gdf.columns:
        # Project to UTM if needed
        if grid_gdf.crs and grid_gdf.crs.to_string() == "EPSG:4326":
            grid_proj = grid_gdf.to_crs("EPSG:32643")  # UTM 43N for Mumbai
            x_coords = grid_proj.geometry.x
            y_coords = grid_proj.geometry.y
        else:
            x_coords = grid_gdf.geometry.x
            y_coords = grid_gdf.geometry.y
    else:
        x_coords = grid_gdf["x_m"]
        y_coords = grid_gdf["y_m"]
    
    # Compute tile IDs
    tile_cols = (x_coords // tile_size_m).astype(int)
    tile_rows = (y_coords // tile_size_m).astype(int)
    tile_ids = tile_cols.astype(str) + "_" + tile_rows.astype(str)
    
    # Create result DataFrame
    result = grid_gdf[["grid_id"]].copy()
    result["meteo_tile_id"] = tile_ids.values
    result["x_m"] = x_coords.values
    result["y_m"] = y_coords.values
    
    return result


def get_unique_tile_centroids(
    grid_with_tiles: pd.DataFrame,
    tile_size_m: float,
    source_crs: str = "EPSG:32643",
) -> Dict[str, Tuple[float, float]]:
    """
    Get unique meteo tile centroids in WGS84.
    
    Parameters
    ----------
    grid_with_tiles : pd.DataFrame
        Grid with meteo_tile_id, x_m, y_m columns
    tile_size_m : float
        Tile size in meters
    source_crs : str
        CRS of x_m, y_m coordinates
    
    Returns
    -------
    Dict[str, Tuple[float, float]]
        Mapping of tile_id -> (lat, lon) centroid
    """
    from pyproj import Transformer
    
    # Get unique tiles
    unique_tiles = grid_with_tiles["meteo_tile_id"].unique()
    
    # Create transformer from projected to WGS84
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    
    tile_centroids = {}
    for tile_id in unique_tiles:
        # Parse tile col/row from ID
        parts = tile_id.split("_")
        tile_col = int(parts[0])
        tile_row = int(parts[1])
        
        # Compute tile centroid in projected coords
        centroid_x = (tile_col + 0.5) * tile_size_m
        centroid_y = (tile_row + 0.5) * tile_size_m
        
        # Transform to WGS84
        lon, lat = transformer.transform(centroid_x, centroid_y)
        tile_centroids[tile_id] = (lat, lon)
    
    return tile_centroids


def fetch_openmeteo_hourly(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    variables: List[str] | None = None,
    model: str = "ecmwf_ifs",
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch hourly meteorology from Open-Meteo for a single location.
    
    Parameters
    ----------
    start_date : str
        Start date YYYY-MM-DD
    end_date : str
        End date YYYY-MM-DD
    latitude : float
        Latitude in degrees
    longitude : float
        Longitude in degrees
    variables : List[str] | None
        Variables to fetch (default: all)
    model : str
        Reanalysis model (ecmwf_ifs, era5_land, era5)
    max_retries : int
        Max retry attempts
    
    Returns
    -------
    pd.DataFrame
        Hourly meteorology with datetime index
    """
    if variables is None:
        variables = DEFAULT_VARIABLES
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "models": model,
    }
    
    # Retry with backoff
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Open-Meteo fetch failed after {max_retries} attempts: {e}")
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    
    data = response.json()
    hourly = data["hourly"]
    times = pd.to_datetime(hourly["time"])
    
    df = pd.DataFrame({"datetime": times})
    for var in variables:
        if var in hourly:
            df[var] = hourly[var]
    
    df.set_index("datetime", inplace=True)
    return df


def fetch_all_tiles_hourly(
    tile_centroids: Dict[str, Tuple[float, float]],
    start_date: str,
    end_date: str,
    variables: List[str] | None = None,
    model: str = "ecmwf_ifs",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch hourly meteorology for all unique tiles.
    
    Parameters
    ----------
    tile_centroids : Dict[str, Tuple[float, float]]
        Mapping of tile_id -> (lat, lon)
    start_date : str
        Start date YYYY-MM-DD
    end_date : str
        End date YYYY-MM-DD
    variables : List[str] | None
        Variables to fetch
    model : str
        Reanalysis model
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of tile_id -> hourly DataFrame
    """
    print(f"  Fetching hourly data for {len(tile_centroids)} tiles...")
    
    tile_data = {}
    for i, (tile_id, (lat, lon)) in enumerate(tile_centroids.items(), 1):
        print(f"    [{i}/{len(tile_centroids)}] Tile {tile_id} at ({lat:.4f}, {lon:.4f})")
        
        df = fetch_openmeteo_hourly(
            start_date=start_date,
            end_date=end_date,
            latitude=lat,
            longitude=lon,
            variables=variables,
            model=model,
        )
        tile_data[tile_id] = df
    
    return tile_data


def extract_overpass_values(
    hourly_df: pd.DataFrame,
    date_str: str,
    overpass_hour_utc: int = 5,
) -> Dict[str, float]:
    """
    Extract meteorology values at overpass time for a date.
    
    Parameters
    ----------
    hourly_df : pd.DataFrame
        Hourly meteorology with datetime index
    date_str : str
        Date in YYYY-MM-DD format
    overpass_hour_utc : int
        Overpass hour in UTC (default 5 for Mumbai/Landsat)
    
    Returns
    -------
    Dict[str, float]
        Variable values at overpass time
    """
    overpass_dt = pd.Timestamp(f"{date_str} {overpass_hour_utc:02d}:00:00")
    
    if overpass_dt in hourly_df.index:
        return hourly_df.loc[overpass_dt].to_dict()
    
    # Find nearest hour within ±3 hours
    time_diffs = abs(hourly_df.index - overpass_dt)
    min_diff_idx = time_diffs.argmin()
    min_diff = time_diffs[min_diff_idx]
    
    if min_diff <= pd.Timedelta(hours=3):
        result = hourly_df.iloc[min_diff_idx].to_dict()
        result["_hours_offset"] = min_diff.total_seconds() / 3600
        return result
    
    # No data within 3 hours
    return {col: np.nan for col in hourly_df.columns}


def compute_daily_aggregates(
    hourly_df: pd.DataFrame,
    date_str: str,
) -> Dict[str, float]:
    """
    Compute 24-hour aggregate statistics for a date.
    
    Parameters
    ----------
    hourly_df : pd.DataFrame
        Hourly meteorology with datetime index
    date_str : str
        Date in YYYY-MM-DD format
    
    Returns
    -------
    Dict[str, float]
        Daily aggregate values
    """
    date = pd.Timestamp(date_str)
    day_mask = hourly_df.index.date == date.date()
    day_data = hourly_df[day_mask]
    
    if len(day_data) == 0:
        return {}
    
    aggs = {}
    
    # Temperature
    if "temperature_2m" in day_data.columns:
        aggs["temp_mean"] = day_data["temperature_2m"].mean()
        aggs["temp_max"] = day_data["temperature_2m"].max()
        aggs["temp_min"] = day_data["temperature_2m"].min()
    
    # Humidity
    if "relative_humidity_2m" in day_data.columns:
        aggs["rh_mean"] = day_data["relative_humidity_2m"].mean()
    
    # Wind
    if "wind_speed_10m" in day_data.columns:
        aggs["wind_speed_mean"] = day_data["wind_speed_10m"].mean()
        aggs["wind_speed_max"] = day_data["wind_speed_10m"].max()
    
    # Precipitation
    if "precipitation" in day_data.columns:
        aggs["precip_total"] = day_data["precipitation"].sum()
    
    # Cloud cover
    if "cloud_cover" in day_data.columns:
        aggs["cloud_cover_mean"] = day_data["cloud_cover"].mean()
    
    # Pressure
    if "surface_pressure" in day_data.columns:
        aggs["pressure_mean"] = day_data["surface_pressure"].mean()
    
    return aggs


def build_grid_meteo_dataframe(
    grid_with_tiles: pd.DataFrame,
    tile_data: Dict[str, pd.DataFrame],
    date_str: str,
    overpass_hour_utc: int = 5,
) -> pd.DataFrame:
    """
    Build per-grid-cell meteorology DataFrame for a single date.
    
    Parameters
    ----------
    grid_with_tiles : pd.DataFrame
        Grid with grid_id, meteo_tile_id columns
    tile_data : Dict[str, pd.DataFrame]
        Tile hourly data
    date_str : str
        Date in YYYY-MM-DD format
    overpass_hour_utc : int
        Overpass hour in UTC
    
    Returns
    -------
    pd.DataFrame
        Per-grid-cell meteorology: grid_id + meteo columns
    """
    # Extract values for each tile
    tile_values = {}
    for tile_id, hourly_df in tile_data.items():
        overpass = extract_overpass_values(hourly_df, date_str, overpass_hour_utc)
        daily = compute_daily_aggregates(hourly_df, date_str)
        
        # Merge overpass and daily
        combined = {**overpass, **daily}
        tile_values[tile_id] = combined
    
    # Build DataFrame by mapping grid cells to tile values
    records = []
    for _, row in grid_with_tiles.iterrows():
        grid_id = row["grid_id"]
        tile_id = row["meteo_tile_id"]
        
        record = {"grid_id": grid_id, "meteo_tile_id": tile_id}
        
        if tile_id in tile_values:
            record.update(tile_values[tile_id])
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Reorder columns: grid_id first, then meteo_tile_id, then sorted meteo vars
    first_cols = ["grid_id", "meteo_tile_id"]
    other_cols = sorted([c for c in df.columns if c not in first_cols])
    df = df[first_cols + other_cols]
    
    return df


def build_grid_meteo_hourly(
    grid_with_tiles: pd.DataFrame,
    tile_data: Dict[str, pd.DataFrame],
    date_str: str,
) -> pd.DataFrame:
    """
    Build per-grid-cell hourly meteorology DataFrame for a single date.
    
    Expands hourly time series to all 30m grid cells (24 rows per cell).
    
    Parameters
    ----------
    grid_with_tiles : pd.DataFrame
        Grid with grid_id, meteo_tile_id columns
    tile_data : Dict[str, pd.DataFrame]
        Tile hourly data (indexed by datetime)
    date_str : str
        Date in YYYY-MM-DD format
    
    Returns
    -------
    pd.DataFrame
        Hourly meteorology: grid_id, meteo_tile_id, datetime, + meteo vars
        (grid_cells × 24 rows)
    """
    date = pd.Timestamp(date_str)
    
    # Extract hourly data for the target date from each tile
    tile_hourly_data = {}
    for tile_id, hourly_df in tile_data.items():
        day_mask = hourly_df.index.date == date.date()
        day_data = hourly_df[day_mask].copy()
        
        if len(day_data) > 0:
            tile_hourly_data[tile_id] = day_data
    
    # Build records by expanding each grid cell to 24 hourly rows
    records = []
    
    for _, grid_row in grid_with_tiles.iterrows():
        grid_id = grid_row["grid_id"]
        tile_id = grid_row["meteo_tile_id"]
        
        if tile_id not in tile_hourly_data:
            continue
        
        tile_hours = tile_hourly_data[tile_id]
        
        # Create one row per hour for this grid cell
        for timestamp, hour_data in tile_hours.iterrows():
            record = {
                "grid_id": grid_id,
                "meteo_tile_id": tile_id,
                "datetime": timestamp,
            }
            record.update(hour_data.to_dict())
            records.append(record)
    
    if not records:
        # Return empty DataFrame with expected schema
        return pd.DataFrame(columns=["grid_id", "meteo_tile_id", "datetime"])
    
    df = pd.DataFrame(records)
    
    # Reorder columns: grid_id, meteo_tile_id, datetime first
    first_cols = ["grid_id", "meteo_tile_id", "datetime"]
    other_cols = sorted([c for c in df.columns if c not in first_cols])
    df = df[first_cols + other_cols]
    
    # Sort by grid_id then datetime for efficient access patterns
    df = df.sort_values(["grid_id", "datetime"]).reset_index(drop=True)
    
    return df


def write_artifact_atomic(
    df: pd.DataFrame,
    artifact_path: Path,
    metadata: dict,
    manifest: OpenMeteoManifest,
    date_str: str,
) -> None:
    """
    Write parquet artifact atomically and register in manifest.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to write
    artifact_path : Path
        Final artifact path
    metadata : dict
        Metadata sidecar content
    manifest : OpenMeteoManifest
        Manifest instance
    date_str : str
        Date (YYYY-MM-DD)
    """
    request_id = f"openmeteo_{manifest.city}_{date_str}"
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
        
        # Mark ok
        manifest.mark_ok(date_str, artifact_path)
        
    except Exception as e:
        manifest.mark_error(date_str, str(e))
        raise


def process_single_date(
    date_str: str,
    grid_with_tiles: pd.DataFrame,
    tile_data: Dict[str, pd.DataFrame],
    tile_centroids: Dict[str, Tuple[float, float]],
    output_dir: Path,
    manifest: OpenMeteoManifest,
    args: argparse.Namespace,
    tile_size_m: float,
) -> Tuple[str, bool, str]:
    """
    Process a single date (for concurrent execution).
    
    Returns
    -------
    Tuple[str, bool, str]
        (date_str, success, message)
    """
    try:
        # Choose artifact path based on mode
        if args.hourly:
            artifact_path = output_dir / f"{date_str}_hourly.parquet"
        else:
            artifact_path = output_dir / f"{date_str}.parquet"
        
        # Skip if exists
        if artifact_path.exists():
            return (date_str, True, "already exists")
        
        # Build grid meteorology DataFrame
        if args.hourly:
            df = build_grid_meteo_hourly(
                grid_with_tiles=grid_with_tiles,
                tile_data=tile_data,
                date_str=date_str,
            )
            mode_info = f"hourly ({len(df):,} rows)"
        else:
            df = build_grid_meteo_dataframe(
                grid_with_tiles=grid_with_tiles,
                tile_data=tile_data,
                date_str=date_str,
                overpass_hour_utc=args.overpass_hour,
            )
            mode_info = f"daily ({len(df):,} rows)"
        
        # Prepare metadata
        metadata = {
            "date": date_str,
            "city": args.city,
            "source": "open-meteo.com",
            "model": args.model,
            "native_resolution_m": tile_size_m,
            "grid_cells": len(grid_with_tiles),
            "unique_tiles": len(tile_centroids),
            "variables": DEFAULT_VARIABLES,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        if args.hourly:
            metadata["mode"] = "hourly"
            metadata["hours"] = len(df) // len(grid_with_tiles) if len(df) > 0 else 0
        else:
            metadata["mode"] = "daily_snapshot"
            metadata["overpass_hour_utc"] = args.overpass_hour
        
        # Write artifact
        write_artifact_atomic(df, artifact_path, metadata, manifest, date_str)
        
        return (date_str, True, mode_info)
        
    except Exception as e:
        return (date_str, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Open-Meteo meteorology and map to 30m grid"
    )
    parser.add_argument(
        "--grid-artifact",
        type=Path,
        required=True,
        help="Path to grid artifact parquet",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="mumbai",
        help="City name",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Root artifacts directory",
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
        "--model",
        type=str,
        default="ecmwf_ifs",
        choices=["ecmwf_ifs", "era5_land", "era5"],
        help="Reanalysis model (default: ecmwf_ifs at 9km)",
    )
    parser.add_argument(
        "--overpass-hour",
        type=int,
        default=5,
        help="Satellite overpass hour UTC (default: 5 for Mumbai)",
    )
    parser.add_argument(
        "--hourly",
        action="store_true",
        help="Write hourly time series (24 rows per grid cell per date)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max concurrent workers for date processing (default: 4)",
    )
    
    args = parser.parse_args()
    
    # Initialize manifest
    manifest = OpenMeteoManifest(
        city=args.city,
        artifacts_root=args.artifacts_root,
        version="v1",
    )
    print(f"✓ Open-Meteo manifest initialized: {manifest.db_path}")
    
    # Load grid
    if not args.grid_artifact.exists():
        print(f"✗ Grid artifact not found: {args.grid_artifact}")
        sys.exit(1)
    
    print(f"Loading grid from {args.grid_artifact}...")
    grid_gdf = gpd.read_parquet(args.grid_artifact)
    print(f"  ✓ Loaded {len(grid_gdf):,} grid cells")
    
    # Get tile size for selected model
    tile_size_m = MODEL_RESOLUTION_M[args.model]
    print(f"  Model: {args.model} ({tile_size_m}m native resolution)")
    
    # Assign meteo tiles to grid cells
    print(f"Assigning meteo tiles (tile size: {tile_size_m}m)...")
    grid_with_tiles = assign_meteo_tiles(grid_gdf, tile_size_m)
    unique_tiles = grid_with_tiles["meteo_tile_id"].nunique()
    print(f"  ✓ Grid spans {unique_tiles} unique meteo tiles")
    
    # Get tile centroids in WGS84
    tile_centroids = get_unique_tile_centroids(grid_with_tiles, tile_size_m)
    print(f"  ✓ Computed {len(tile_centroids)} tile centroids")
    
    # Fetch hourly data for all tiles (full date range)
    # Add buffer day on each end for daily aggregates
    start_dt = pd.Timestamp(args.start_date) - pd.Timedelta(days=1)
    end_dt = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
    
    print(f"\nFetching hourly data from {start_dt.date()} to {end_dt.date()}...")
    tile_data = fetch_all_tiles_hourly(
        tile_centroids=tile_centroids,
        start_date=str(start_dt.date()),
        end_date=str(end_dt.date()),
        model=args.model,
    )
    
    # Generate date list
    dates = pd.date_range(args.start_date, args.end_date, freq="D")
    print(f"\n✓ Processing {len(dates)} dates with {args.max_workers} workers")
    
    # Prepare output directory
    output_dir = args.artifacts_root / "v1" / "openmeteo" / args.city
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process dates concurrently
    failed_dates = []
    processed_count = 0
    
    date_strings = [date.strftime("%Y-%m-%d") for date in dates]
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all dates
        futures = {
            executor.submit(
                process_single_date,
                date_str,
                grid_with_tiles,
                tile_data,
                tile_centroids,
                output_dir,
                manifest,
                args,
                tile_size_m,
            ): date_str
            for date_str in date_strings
        }
        
        # Process results as they complete
        for idx, future in enumerate(as_completed(futures), start=1):
            date_str = futures[future]
            
            try:
                result_date, success, message = future.result()
                
                if success:
                    processed_count += 1
                    if message == "already exists":
                        print(f"[{idx}/{len(dates)}] {result_date}: ✓ Already exists")
                    else:
                        print(f"[{idx}/{len(dates)}] {result_date}: ✓ {message}")
                else:
                    failed_dates.append((result_date, message))
                    print(f"[{idx}/{len(dates)}] {result_date}: ✗ {message[:80]}")
                    
            except Exception as e:
                failed_dates.append((date_str, str(e)))
                print(f"[{idx}/{len(dates)}] {date_str}: ✗ {str(e)[:80]}")
    
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


if __name__ == "__main__":
    main()
