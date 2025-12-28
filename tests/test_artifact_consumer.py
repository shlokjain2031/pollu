#!/usr/bin/env python3
"""
Generic artifact consumer test for validating data streams.

This test acts as a downstream consumer that:
1. Queries the manifest for artifacts of a given type
2. Loads parquet files from manifest-registered paths
3. Validates schema, row counts, data quality
4. Reports statistics and any issues

Supports any artifact type: landsat_signals, s5p_no2_signals, sentinel_evi_signals, etc.

Usage:
  # Test landsat artifacts
  python tests/test_artifact_consumer.py --artifact-type landsat_signals --grid-artifact artifacts/v1/grids/mumbai/grid.parquet
  
  # Test specific dates only
  python tests/test_artifact_consumer.py --artifact-type landsat_signals --grid-artifact artifacts/v1/grids/mumbai/grid.parquet --dates 2018-01-01 2018-01-17
  
  # Test S5P NO2 (future)
  python tests/test_artifact_consumer.py --artifact-type s5p_no2_signals --grid-artifact artifacts/v1/grids/mumbai/grid.parquet
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set
import sqlite3

import pandas as pd
import geopandas as gpd

from manifest.db import Manifest


# Schema definitions for each artifact type
ARTIFACT_SCHEMAS = {
    "landsat_signals": {
        "required_columns": ["grid_id", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
        "numeric_columns": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
        "expected_range": (-0.01, 2.0),  # TOA reflectance can exceed 1.0 for bright surfaces
        "description": "Landsat-8 raw band values (B1-B11)",
    },
    "s5p_no2_signals": {
        "required_columns": ["grid_id", "no2_column"],
        "numeric_columns": ["no2_column"],
        "expected_range": (0.0, 0.001),  # mol/m² typical range
        "description": "Sentinel-5P NO2 column density",
    },
    "sentinel_evi_signals": {
        "required_columns": ["grid_id", "evi"],
        "numeric_columns": ["evi"],
        "expected_range": (-1.0, 1.0),  # EVI typical range
        "description": "Sentinel-2 Enhanced Vegetation Index",
    },
}


class ArtifactConsumer:
    """Generic consumer for validating artifact data streams."""
    
    def __init__(self, manifest: Manifest, grid_gdf: gpd.GeoDataFrame):
        self.manifest = manifest
        self.grid_gdf = grid_gdf
        self.grid_ids = set(grid_gdf["grid_id"].values)
        self.n_grid_cells = len(grid_gdf)
    
    def get_artifacts(self, artifact_type: str, dates: Optional[List[str]] = None) -> List[Dict]:
        """Query manifest for artifacts of given type."""
        conn = self.manifest._conn
        cur = conn.cursor()
        
        if dates:
            placeholders = ",".join(["?" for _ in dates])
            query = f"""
                SELECT id, row_idx, timestamp, artifact_type, version, state, artifact_path, error
                FROM artifacts
                WHERE artifact_type = ? AND state = 'ok' AND timestamp IN ({placeholders})
                ORDER BY timestamp
            """
            cur.execute(query, [artifact_type] + dates)
        else:
            query = """
                SELECT id, row_idx, timestamp, artifact_type, version, state, artifact_path, error
                FROM artifacts
                WHERE artifact_type = ? AND state = 'ok'
                ORDER BY timestamp
            """
            cur.execute(query, [artifact_type])
        
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]
    
    def validate_parquet(
        self,
        parquet_path: Path,
        artifact_type: str,
        timestamp: str,
    ) -> Dict:
        """
        Validate a single parquet artifact.
        
        Returns dict with validation results.
        """
        result = {
            "timestamp": timestamp,
            "path": str(parquet_path),
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }
        
        # Check file exists
        if not parquet_path.exists():
            result["valid"] = False
            result["errors"].append(f"File not found: {parquet_path}")
            return result
        
        # Load parquet
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to read parquet: {e}")
            return result
        
        result["stats"]["row_count"] = len(df)
        result["stats"]["columns"] = list(df.columns)
        
        # Get schema for this artifact type
        schema = ARTIFACT_SCHEMAS.get(artifact_type)
        if not schema:
            result["warnings"].append(f"No schema defined for {artifact_type}, skipping schema validation")
            return result
        
        # Validate required columns
        missing_cols = set(schema["required_columns"]) - set(df.columns)
        if missing_cols:
            result["valid"] = False
            result["errors"].append(f"Missing required columns: {missing_cols}")
        
        # Validate row count matches grid
        if len(df) != self.n_grid_cells:
            result["warnings"].append(
                f"Row count mismatch: expected {self.n_grid_cells:,} (grid size), got {len(df):,}"
            )
        
        # Validate grid_id coverage
        if "grid_id" in df.columns:
            artifact_grid_ids = set(df["grid_id"].values)
            missing_in_artifact = self.grid_ids - artifact_grid_ids
            extra_in_artifact = artifact_grid_ids - self.grid_ids
            
            if missing_in_artifact:
                result["warnings"].append(
                    f"Missing {len(missing_in_artifact):,} grid_ids from canonical grid"
                )
            if extra_in_artifact:
                result["warnings"].append(
                    f"Found {len(extra_in_artifact):,} extra grid_ids not in canonical grid"
                )
            
            result["stats"]["grid_id_coverage"] = f"{len(artifact_grid_ids):,}/{self.n_grid_cells:,}"
        
        # Validate numeric columns
        for col in schema["numeric_columns"]:
            if col not in df.columns:
                continue
            
            col_stats = {
                "count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum()),
                "min": float(df[col].min()) if df[col].count() > 0 else None,
                "max": float(df[col].max()) if df[col].count() > 0 else None,
                "mean": float(df[col].mean()) if df[col].count() > 0 else None,
            }
            result["stats"][col] = col_stats
            
            # Check for nulls
            if col_stats["null_count"] > 0:
                pct_null = 100 * col_stats["null_count"] / len(df)
                result["warnings"].append(
                    f"{col}: {col_stats['null_count']:,} nulls ({pct_null:.2f}%)"
                )
            
            # Check value ranges
            if col_stats["min"] is not None and col_stats["max"] is not None:
                expected_min, expected_max = schema["expected_range"]
                if col_stats["min"] < expected_min or col_stats["max"] > expected_max:
                    result["warnings"].append(
                        f"{col}: values outside expected range {schema['expected_range']}: "
                        f"[{col_stats['min']:.6f}, {col_stats['max']:.6f}]"
                    )
        
        # Check row ordering (should match grid order if deterministic)
        if "grid_id" in df.columns:
            # Check if first N grid_ids match
            sample_size = min(100, len(df), len(self.grid_gdf))
            expected_grid_ids = self.grid_gdf["grid_id"].iloc[:sample_size].values
            actual_grid_ids = df["grid_id"].iloc[:sample_size].values
            
            if not all(expected_grid_ids == actual_grid_ids):
                result["warnings"].append(
                    "Row ordering does not match canonical grid order (first 100 rows checked)"
                )
        
        return result
    
    def run_validation(
        self,
        artifact_type: str,
        dates: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run full validation for all artifacts of given type.
        
        Returns summary dict with overall results.
        """
        print(f"{'='*70}")
        print(f"ARTIFACT CONSUMER TEST: {artifact_type}")
        print(f"{'='*70}")
        
        schema = ARTIFACT_SCHEMAS.get(artifact_type)
        if schema:
            print(f"Description: {schema['description']}")
            print(f"Expected schema: {', '.join(schema['required_columns'])}")
        print(f"Grid size: {self.n_grid_cells:,} cells")
        print()
        
        # Query manifest for artifacts
        print(f"Querying manifest for {artifact_type} artifacts...")
        artifacts = self.get_artifacts(artifact_type, dates)
        print(f"  ✓ Found {len(artifacts)} artifacts in manifest (state=ok)")
        
        if not artifacts:
            print("\n⚠ No artifacts found. Nothing to validate.")
            return {
                "artifact_type": artifact_type,
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "results": [],
            }
        
        print()
        
        # Validate each artifact
        results = []
        valid_count = 0
        error_count = 0
        
        for idx, artifact in enumerate(artifacts, start=1):
            timestamp = artifact["timestamp"]
            artifact_path = Path(artifact["artifact_path"])
            
            print(f"[{idx}/{len(artifacts)}] Validating {timestamp}...")
            print(f"  Path: {artifact_path}")
            
            result = self.validate_parquet(artifact_path, artifact_type, timestamp)
            results.append(result)
            
            if result["valid"]:
                valid_count += 1
                print(f"  ✓ VALID")
                print(f"    Rows: {result['stats']['row_count']:,}")
                print(f"    Columns: {len(result['stats']['columns'])}")
                
                if result["warnings"]:
                    print(f"    Warnings: {len(result['warnings'])}")
                    for warning in result["warnings"][:3]:  # Show first 3
                        print(f"      - {warning}")
                    if len(result["warnings"]) > 3:
                        print(f"      ... and {len(result['warnings']) - 3} more")
            else:
                error_count += 1
                print(f"  ✗ INVALID")
                for error in result["errors"]:
                    print(f"    ERROR: {error}")
            
            print()
        
        # Summary
        print(f"{'='*70}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Artifact type:    {artifact_type}")
        print(f"Total artifacts:  {len(results)}")
        print(f"Valid:            {valid_count}")
        print(f"Invalid:          {error_count}")
        print(f"Grid coverage:    {self.n_grid_cells:,} cells expected")
        print()
        
        if valid_count == len(results):
            print("✓ All artifacts passed validation!")
        elif error_count > 0:
            print(f"⚠ {error_count} artifact(s) failed validation")
        
        return {
            "artifact_type": artifact_type,
            "total": len(results),
            "valid": valid_count,
            "invalid": error_count,
            "results": results,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Test artifact consumer - validates data streams from manifest"
    )
    parser.add_argument(
        "--artifact-type",
        type=str,
        required=True,
        choices=["landsat_signals", "s5p_no2_signals", "sentinel_evi_signals"],
        help="Artifact type to validate",
    )
    parser.add_argument(
        "--grid-artifact",
        type=Path,
        required=True,
        help="Path to canonical grid parquet",
    )
    parser.add_argument(
        "--manifest-db",
        type=Path,
        default=Path("manifest/manifest.db"),
        help="Path to manifest database",
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=None,
        help="Optional: specific dates to test (YYYY-MM-DD). If omitted, tests all available.",
    )
    
    args = parser.parse_args()
    
    # Load grid
    if not args.grid_artifact.exists():
        print(f"✗ Grid artifact not found: {args.grid_artifact}")
        sys.exit(1)
    
    print(f"Loading canonical grid from {args.grid_artifact}...")
    grid_gdf = gpd.read_parquet(args.grid_artifact)
    print(f"  ✓ Loaded {len(grid_gdf):,} grid cells")
    print()
    
    # Initialize manifest
    manifest = Manifest(db_path=args.manifest_db)
    print(f"✓ Manifest loaded: {manifest.db_path}")
    print()
    
    # Create consumer and run validation
    consumer = ArtifactConsumer(manifest, grid_gdf)
    summary = consumer.run_validation(args.artifact_type, args.dates)
    
    # Exit with error code if any failures
    if summary["invalid"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
