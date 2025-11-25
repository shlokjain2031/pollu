"""Migrate old parquet files to new grid schema.

This script updates parquet files created with the old grid schema (x, y, grid_id)
to the new schema (row_idx, grid_id, x_m, y_m, centroid_lon, centroid_lat, 
col_idx, row_idx_grid, h3_cell).

The migration is done by spatial join on coordinates (x ≈ x_m, y ≈ y_m) with the
canonical new grid. No external data fetching required - pure geometry math.

Usage:
    python scripts/migrate_old_parquets.py --input cache/old_svf.parquet
    python scripts/migrate_old_parquets.py --input cache/old_svf.parquet --output cache/static/svf.parquet
    python scripts/migrate_old_parquets.py --batch cache/*.parquet --output-dir cache/static/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd


# Canonical new grid path
DEFAULT_GRID_PATH = Path("data/mumbai/grid_30m_nd.parquet")
TOLERANCE_METERS = 0.1  # Coordinate matching tolerance


def load_canonical_grid(grid_path: Path) -> pd.DataFrame:
    """Load the canonical new grid as reference."""
    print(f"Loading canonical grid from {grid_path}...")
    grid = pd.read_parquet(grid_path)
    
    required_cols = ["row_idx", "grid_id", "x_m", "y_m"]
    missing = [c for c in required_cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Grid missing required columns: {missing}")
    
    print(f"  Loaded {len(grid):,} points")
    print(f"  Columns: {list(grid.columns)}")
    return grid


def detect_old_schema(df: pd.DataFrame) -> bool:
    """Detect if dataframe uses old schema."""
    # Old schema: x, y, grid_id (int64)
    # New schema: row_idx, grid_id (str), x_m, y_m
    
    has_old_coords = "x" in df.columns and "y" in df.columns
    has_new_coords = "x_m" in df.columns and "y_m" in df.columns
    has_row_idx = "row_idx" in df.columns
    
    if has_row_idx and has_new_coords:
        return False  # Already new schema
    
    if has_old_coords and not has_new_coords:
        return True  # Old schema
    
    # Ambiguous
    print(f"WARNING: Cannot determine schema. Columns: {list(df.columns)}")
    return False


def migrate_parquet(
    input_path: Path,
    canonical_grid: pd.DataFrame,
    output_path: Optional[Path] = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Migrate a single parquet file to new schema.
    
    Parameters
    ----------
    input_path : Path
        Input parquet file (old schema)
    canonical_grid : pd.DataFrame
        Reference grid with new schema
    output_path : Path, optional
        Output path (defaults to input_path with .migrated.parquet suffix)
    dry_run : bool
        If True, don't write output
    
    Returns
    -------
    pd.DataFrame
        Migrated dataframe with new schema
    """
    print(f"\nMigrating {input_path}...")
    
    # Load old parquet
    try:
        old_df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"ERROR: Failed to load: {e}")
        return None
    
    print(f"  Old schema: {len(old_df):,} rows, columns: {list(old_df.columns)}")
    
    # Check if already new schema
    if not detect_old_schema(old_df):
        print(f"  OK: Already uses new schema, skipping")
        return old_df
    
    # Identify coordinate columns
    if "x" in old_df.columns and "y" in old_df.columns:
        coord_cols = ["x", "y"]
        new_coord_cols = ["x_m", "y_m"]
    else:
        print(f"  ERROR: Cannot find coordinate columns")
        return None
    
    # Spatial join on coordinates
    print(f"  Joining with canonical grid on coordinates...")
    
    # Round coordinates for matching (handle floating point precision)
    old_df["_x_round"] = old_df[coord_cols[0]].round(2)
    old_df["_y_round"] = old_df[coord_cols[1]].round(2)
    
    canonical_grid["_x_round"] = canonical_grid["x_m"].round(2)
    canonical_grid["_y_round"] = canonical_grid["y_m"].round(2)
    
    # Join on rounded coordinates
    merged = old_df.merge(
        canonical_grid,
        left_on=["_x_round", "_y_round"],
        right_on=["_x_round", "_y_round"],
        how="left",
        suffixes=("_old", "")
    )
    
    # Check for unmatched points
    unmatched = merged["row_idx"].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched:,} points could not be matched to grid")
        
        # Try fuzzy matching for unmatched points
        if unmatched < 100:  # Only if small number
            print(f"    Attempting fuzzy matching...")
            unmatched_mask = merged["row_idx"].isna()
            unmatched_df = old_df[unmatched_mask].copy()
            
            # For each unmatched point, find nearest grid point
            from scipy.spatial import cKDTree
            
            grid_coords = canonical_grid[["x_m", "y_m"]].values
            tree = cKDTree(grid_coords)
            
            unmatched_coords = unmatched_df[[coord_cols[0], coord_cols[1]]].values
            distances, indices = tree.query(unmatched_coords, k=1)
            
            # Only accept if within tolerance
            close_enough = distances < TOLERANCE_METERS
            n_recovered = close_enough.sum()
            
            if n_recovered > 0:
                print(f"    Recovered {n_recovered} points via nearest neighbor")
                # Update merged dataframe with recovered matches
                recovered_rows = canonical_grid.iloc[indices[close_enough]]
                for i, (idx, recovered_row) in enumerate(zip(merged.index[unmatched_mask][close_enough], recovered_rows.iterrows())):
                    for col in canonical_grid.columns:
                        if col not in ["_x_round", "_y_round"]:
                            merged.at[idx, col] = recovered_row[1][col]
    
    # Drop temporary columns
    merged = merged.drop(columns=["_x_round", "_y_round"], errors="ignore")
    
    # Rename old columns if they conflict
    if "x" in merged.columns and "x_m" in merged.columns:
        merged = merged.drop(columns=["x", "y"])  # Drop old coordinate columns
    
    # Handle grid_id: old was int64, new is string
    if "grid_id_old" in merged.columns:
        merged = merged.drop(columns=["grid_id_old"])
    
    # Reorder columns: new schema first, then data columns
    new_schema_cols = [
        "row_idx", "grid_id", "x_m", "y_m", 
        "centroid_lon", "centroid_lat", 
        "col_idx", "row_idx_grid", "h3_cell"
    ]
    
    data_cols = [c for c in merged.columns if c not in new_schema_cols]
    final_cols = [c for c in new_schema_cols if c in merged.columns] + data_cols
    
    migrated = merged[final_cols].copy()
    
    print(f"  OK: Migrated {len(migrated):,} rows")
    print(f"    New columns: {list(migrated.columns)}")
    
    # Validate
    missing_row_idx = migrated["row_idx"].isna().sum()
    if missing_row_idx > 0:
        print(f"  WARNING: {missing_row_idx} rows missing row_idx")
    
    # Write output
    if not dry_run:
        if output_path is None:
            output_path = input_path  # Replace the original file
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup in tmp/ directory before overwriting
        backup_path = None
        if output_path.exists() and output_path == input_path:
            tmp_dir = Path("tmp")
            tmp_dir.mkdir(exist_ok=True)
            backup_path = tmp_dir / f"{input_path.stem}.backup.parquet"
            print(f"  Creating backup: {backup_path}")
            import shutil
            shutil.copy2(input_path, backup_path)
        
        migrated.to_parquet(output_path, index=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  OK: Replaced {output_path} ({file_size_mb:.2f} MB)")
        
        # Write migration manifest to tmp/ directory
        tmp_dir = Path("tmp")
        tmp_dir.mkdir(exist_ok=True)
        manifest_path = tmp_dir / f"{output_path.stem}_migration.json"
        manifest = {
            "migrated_ts": datetime.utcnow().isoformat() + "Z",
            "source_file": str(input_path),
            "output_file": str(output_path),
            "n_rows": len(migrated),
            "old_schema": list(old_df.columns),
            "new_schema": list(migrated.columns),
            "unmatched_points": int(missing_row_idx),
            "backup_created": str(backup_path) if backup_path else None,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    
    return migrated


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate old parquet files to new grid schema"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input parquet file (old schema)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output parquet file (new schema). Defaults to input.migrated.parquet"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Glob pattern for batch processing (e.g., 'cache/*.parquet')"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch processing"
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        default=DEFAULT_GRID_PATH,
        help="Path to canonical new grid"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test migration without writing files"
    )
    
    args = parser.parse_args(argv)
    
    # Load canonical grid
    try:
        canonical_grid = load_canonical_grid(args.grid_path)
    except Exception as e:
        print(f"ERROR: Failed to load canonical grid: {e}")
        return 1
    
    # Single file or batch?
    if args.input:
        # Single file
        if not args.input.exists():
            print(f"ERROR: Input file not found: {args.input}")
            return 1
        
        migrated = migrate_parquet(
            args.input,
            canonical_grid,
            args.output,
            args.dry_run
        )
        
        if migrated is None:
            return 1
        
        print("\nMigration complete")
        return 0
    
    elif args.batch:
        # Batch processing
        from glob import glob
        
        files = sorted(glob(args.batch))
        if not files:
            print(f"ERROR: No files found matching pattern: {args.batch}")
            return 1
        
        print(f"\nFound {len(files)} files to migrate")
        
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for file_path in files:
            input_path = Path(file_path)
            
            if args.output_dir:
                output_path = args.output_dir / f"{input_path.stem}.parquet"
            else:
                output_path = None
            
            try:
                migrated = migrate_parquet(
                    input_path,
                    canonical_grid,
                    output_path,
                    args.dry_run
                )
                
                if migrated is not None:
                    if "row_idx" in migrated.columns and migrated["row_idx"].notna().all():
                        success_count += 1
                    else:
                        skip_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                fail_count += 1
        
        print(f"\n{'='*60}")
        print(f"BATCH MIGRATION SUMMARY")
        print(f"{'='*60}")
        print(f"Success: {success_count} files")
        print(f"Skipped: {skip_count} files (already new schema)")
        print(f"Failed:  {fail_count} files")
        
        return 0 if fail_count == 0 else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
