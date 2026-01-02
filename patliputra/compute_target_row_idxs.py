#!/usr/bin/env python3
"""
Compute target row_idxs from sensor influence zones and save to /tmp/target_row_idx.

This script:
1. Fetches sensor locations from OpenAQ API for the given city/bbox
2. Loads grid artifact
3. Computes sensor influence zones (all grid cells within radius R of sensors)
4. Extracts the union of all row_idxs that are influenced by any sensor
5. Writes to /tmp/target_row_idx (parquet with single column: row_idx)

This is a self-contained, one-time computation per city. The output is used by
downstream patliputra fetchers (landsat8, sentinel2, openmeteo) to filter their
outputs to only the grid cells that matter for training.

Usage:
    python patliputra/compute_target_row_idxs.py --city mumbai
    
    # Custom bbox
    python patliputra/compute_target_row_idxs.py \
        --city mumbai \
        --bbox 72.7763,18.8939,72.9797,19.2701
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_ROOT = _SCRIPT_PATH.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import requests

from nalanda.spatial_utils import GridIndex, add_utm_coords_to_sensors


DEFAULT_RADIUS_M = 250.0
TARGET_ROW_IDX_PATH = Path("/tmp/target_row_idx.parquet")

# City-specific bounding boxes (west, south, east, north in EPSG:4326)
CITY_BBOXES = {
    "mumbai": "72.7763,18.8939,72.9797,19.2701",
    # Add more cities as needed
}

# OpenAQ API configuration
OPENAQ_API_BASE = "https://api.openaq.org/v3"
OPENAQ_API_TOKEN_ENV = "OPENAQ_API_KEY"  # Optional: can fetch without token (rate-limited)


OPENAQ_API_BASE = "https://api.openaq.org/v3"
OPENAQ_API_TOKEN_ENV = "OPENAQ_API_KEY"  # Optional: can fetch without token (rate-limited)


def fetch_openaq_sensors(
    bbox: str,
    api_token: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch PM2.5 sensor locations from OpenAQ API.
    
    Args:
        bbox: Bounding box as "west,south,east,north" (EPSG:4326).
        api_token: Optional OpenAQ API token. If None, uses rate-limited public access.
        timeout: Request timeout in seconds.
        
    Returns:
        DataFrame with columns:
            - sensor_id: int, unique sensor identifier
            - location_id: int, OpenAQ location identifier
            - location_name: str, human-readable station name
            - latitude: float, WGS84 latitude
            - longitude: float, WGS84 longitude
    """
    headers = {}
    if api_token:
        headers["X-API-Key"] = api_token
    
    sensor_rows = []
    page = 1
    limit = 100
    
    print(f"Fetching PM2.5 sensors from OpenAQ API (bbox={bbox})...")
    
    while True:
        params = {"bbox": bbox, "limit": limit, "page": page}
        
        try:
            r = requests.get(
                f"{OPENAQ_API_BASE}/locations",
                params=params,
                headers=headers,
                timeout=timeout,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"✗ API request failed: {e}")
            break
        
        results = r.json().get("results", [])
        if not results:
            break
        
        for location in results:
            loc_id = location.get("id")
            if loc_id is None:
                continue
            
            # Fetch detailed location info to check sensors
            detail_url = f"{OPENAQ_API_BASE}/locations/{loc_id}"
            try:
                detail_r = requests.get(detail_url, headers=headers, timeout=timeout)
                detail_r.raise_for_status()
            except requests.RequestException:
                continue
            
            detail_results = detail_r.json().get("results", [])
            if not detail_results:
                continue
            
            loc = detail_results[0]
            sensors = loc.get("sensors", [])
            
            # Extract location name and coordinates
            loc_name = loc.get("name") or loc.get("location") or loc.get("city")
            coords = loc.get("coordinates") or {}
            lat = coords.get("latitude") or coords.get("lat") or loc.get("latitude")
            lon = coords.get("longitude") or coords.get("lon") or loc.get("longitude")
            
            # Filter for PM2.5 sensors
            for sensor in sensors:
                param = sensor.get("parameter")
                if isinstance(param, dict):
                    param_name = param.get("name") or param.get("parameter")
                else:
                    param_name = param
                
                sensor_id = sensor.get("id")
                if sensor_id is None:
                    continue
                
                try:
                    sid = int(sensor_id)
                except (ValueError, TypeError):
                    continue
                
                if isinstance(param_name, str) and param_name.lower() == "pm25":
                    sensor_rows.append({
                        "sensor_id": sid,
                        "location_id": loc_id,
                        "location_name": loc_name,
                        "latitude": lat,
                        "longitude": lon,
                    })
        
        if len(results) < limit:
            break
        page += 1
    
    print(f"  → Fetched {len(sensor_rows)} sensor records")
    
    # Create DataFrame and clean
    df = pd.DataFrame(sensor_rows)
    
    if df.empty:
        print("  ✗ No sensors found")
        return pd.DataFrame(columns=["sensor_id", "location_id", "location_name", "latitude", "longitude"])
    
    # Filter to valid coordinates
    df = df.dropna(subset=["latitude", "longitude"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    
    # Deduplicate by rounded lat/lon (unique sensor locations)
    df["lat_round"] = df["latitude"].round(6)
    df["lon_round"] = df["longitude"].round(6)
    df["latlon_key"] = df["lat_round"].astype(str) + "_" + df["lon_round"].astype(str)
    df = df.sort_values(["latlon_key", "sensor_id"])
    df = df.drop_duplicates(subset=["latlon_key"]).reset_index(drop=True)
    df = df.drop(columns=["lat_round", "lon_round", "latlon_key"])
    
    print(f"  ✓ {len(df)} unique PM2.5 sensor locations")
    
    return df


def compute_target_row_idxs(
    grid_path: Path,
    sensors_df: pd.DataFrame,
    radius_m: float = DEFAULT_RADIUS_M,
) -> pd.DataFrame:
    """Compute target row_idxs from sensor influence zones.
    
    Args:
        grid_path: Path to grid parquet artifact.
        sensors_df: DataFrame with sensor locations (latitude, longitude columns).
        radius_m: Influence radius in meters.
        
    Returns:
        DataFrame with single column: row_idx (int32).
    """
    print(f"Loading grid from {grid_path}...")
    grid_index = GridIndex(grid_path)
    print(f"  ✓ Loaded {len(grid_index.grid):,} grid cells")
    
    # Add UTM coordinates to sensors
    sensors_df = add_utm_coords_to_sensors(sensors_df)
    print(f"  ✓ Converted {len(sensors_df):,} sensors to UTM")
    
    print(f"Computing influence zones (R={radius_m}m)...")
    all_row_idxs = set()
    sensors_in_grid = 0
    
    for _, sensor in sensors_df.iterrows():
        x_utm = sensor['x_utm']
        y_utm = sensor['y_utm']
        
        # Query all cells within radius
        cells = grid_index.query_radius(x_utm, y_utm, radius_m)
        
        if len(cells) == 0:
            # Sensor is outside grid coverage
            continue
        
        sensors_in_grid += 1
        all_row_idxs.update(cells['row_idx'].values)
    
    print(f"  ✓ {sensors_in_grid}/{len(sensors_df)} sensors in grid")
    print(f"  ✓ {len(all_row_idxs):,} unique row_idxs influenced")
    
    # Convert to DataFrame
    target_df = pd.DataFrame({
        'row_idx': sorted(all_row_idxs)
    })
    target_df['row_idx'] = target_df['row_idx'].astype('int32')
    
    return target_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute target row_idxs from sensor influence zones (self-contained)"
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        help="City name (e.g., mumbai)",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Bounding box as 'west,south,east,north' (EPSG:4326). If not provided, uses city default.",
    )
    parser.add_argument(
        "--grid-artifact",
        type=Path,
        default=None,
        help="Path to grid parquet (default: artifacts/v1/grids/{city}/grid.parquet)",
    )
    parser.add_argument(
        "--radius-m",
        type=float,
        default=DEFAULT_RADIUS_M,
        help=f"Influence radius in meters (default: {DEFAULT_RADIUS_M})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TARGET_ROW_IDX_PATH,
        help=f"Output path (default: {TARGET_ROW_IDX_PATH})",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help=f"OpenAQ API token (optional; can also set {OPENAQ_API_TOKEN_ENV} env var)",
    )
    
    args = parser.parse_args()
    
    # Set default grid artifact path if not provided
    if args.grid_artifact is None:
        args.grid_artifact = Path("artifacts/v1/grids") / args.city / "grid.parquet"
    
    # Determine bounding box
    if args.bbox is None:
        if args.city.lower() in CITY_BBOXES:
            args.bbox = CITY_BBOXES[args.city.lower()]
        else:
            print(f"✗ No default bbox for city '{args.city}'. Please provide --bbox.")
            sys.exit(1)
    
    # Get API token from env if not provided
    api_token = args.api_token or os.getenv(OPENAQ_API_TOKEN_ENV)
    
    # Validate grid artifact exists
    if not args.grid_artifact.exists():
        print(f"✗ Grid artifact not found: {args.grid_artifact}")
        sys.exit(1)
    
    # Fetch sensors from OpenAQ API
    print(f"\nFetching PM2.5 sensors for {args.city}")
    print(f"Bounding box: {args.bbox}")
    if api_token:
        print(f"Using API token from {'--api-token' if args.api_token else f'{OPENAQ_API_TOKEN_ENV} env var'}")
    else:
        print("No API token provided; using rate-limited public access")
    print()
    
    sensors_df = fetch_openaq_sensors(bbox=args.bbox, api_token=api_token)
    
    if sensors_df.empty:
        print(f"\n✗ No sensors found in bbox. Cannot compute target row_idxs.")
        sys.exit(1)
    
    # Compute target row_idxs
    print(f"\nComputing target row_idxs for {args.city}")
    print(f"Grid: {args.grid_artifact}")
    print(f"Radius: {args.radius_m}m\n")
    
    target_df = compute_target_row_idxs(
        grid_path=args.grid_artifact,
        sensors_df=sensors_df,
        radius_m=args.radius_m,
    )
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target_df.to_parquet(args.output, index=False)
    
    print(f"\n✓ Wrote {len(target_df):,} target row_idxs to {args.output}")
    print(f"  Size: {args.output.stat().st_size / 1024:.2f} KB")
    
    # Print sample
    print(f"\nSample row_idxs (first 10):")
    print(target_df.head(10).to_string(index=False))
    
    print(f"\nTarget row_idx range: [{target_df['row_idx'].min()}, {target_df['row_idx'].max()}]")


if __name__ == "__main__":
    main()
