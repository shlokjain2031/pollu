#!/usr/bin/env python3
"""
Compute target row_idxs from sensor influence zones and save to /tmp/target_row_idx.

This script:
1. Loads grid artifact
2. Loads sensor locations from cache/openaq_sensor_ids.json
3. Computes sensor influence zones (all grid cells within radius R of sensors)
4. Extracts the union of all row_idxs that are influenced by any sensor
5. Writes to /tmp/target_row_idx (parquet with single column: row_idx)

This is a one-time computation per city. The output is used by downstream
patliputra fetchers (landsat8, sentinel2, openmeteo) to filter their outputs
to only the grid cells that matter for training.

Usage:
    python patliputra/compute_target_row_idxs.py --city mumbai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from nalanda.sensor_locations import load_sensor_locations
from nalanda.spatial_utils import GridIndex, add_utm_coords_to_sensors


DEFAULT_RADIUS_M = 250.0
TARGET_ROW_IDX_PATH = Path("/tmp/target_row_idx")


def compute_target_row_idxs(
    grid_path: Path,
    sensor_json_path: Path,
    radius_m: float = DEFAULT_RADIUS_M,
) -> pd.DataFrame:
    """Compute target row_idxs from sensor influence zones.
    
    Args:
        grid_path: Path to grid parquet artifact.
        sensor_json_path: Path to sensor locations JSON.
        radius_m: Influence radius in meters.
        
    Returns:
        DataFrame with single column: row_idx (int32).
    """
    print(f"Loading grid from {grid_path}...")
    grid_index = GridIndex(grid_path)
    print(f"  ✓ Loaded {len(grid_index.grid):,} grid cells")
    
    print(f"Loading sensors from {sensor_json_path}...")
    sensors_df = load_sensor_locations(sensor_json_path)
    sensors_df = add_utm_coords_to_sensors(sensors_df)
    print(f"  ✓ Loaded {len(sensors_df):,} sensors")
    
    print(f"Computing influence zones (R={radius_m}m)...")
    all_row_idxs = set()
    sensors_in_grid = 0
    
    for _, sensor in sensors_df.iterrows():
        site_id = sensor['site_id']
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
        description="Compute target row_idxs from sensor influence zones"
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        help="City name (e.g., mumbai)",
    )
    parser.add_argument(
        "--grid-artifact",
        type=Path,
        default=None,
        help="Path to grid parquet (default: artifacts/v1/grids/{city}/grid.parquet)",
    )
    parser.add_argument(
        "--sensor-json",
        type=Path,
        default=Path("cache/openaq_sensor_ids.json"),
        help="Path to sensor locations JSON (default: cache/openaq_sensor_ids.json)",
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
    
    args = parser.parse_args()
    
    # Set default grid artifact path if not provided
    if args.grid_artifact is None:
        args.grid_artifact = Path("artifacts/v1/grids") / args.city / "grid.parquet"
    
    # Validate inputs
    if not args.grid_artifact.exists():
        print(f"✗ Grid artifact not found: {args.grid_artifact}")
        sys.exit(1)
    
    if not args.sensor_json.exists():
        print(f"✗ Sensor JSON not found: {args.sensor_json}")
        sys.exit(1)
    
    # Compute target row_idxs
    print(f"\nComputing target row_idxs for {args.city}")
    print(f"Grid: {args.grid_artifact}")
    print(f"Sensors: {args.sensor_json}")
    print(f"Radius: {args.radius_m}m\n")
    
    target_df = compute_target_row_idxs(
        grid_path=args.grid_artifact,
        sensor_json_path=args.sensor_json,
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
