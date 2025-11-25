"""Assemble static features from precomputed components.

This script combines all Stage 0 static features into a single consolidated file:
- Grid base (row_idx, grid_id, coordinates)
- Sky View Factor (SVF)
- OSM industrial/dust aggregates (30m, 90m, 270m)
- Elevation
- KNN sensor indices and distances

The output is a single parquet file with all static features that never change
over time. This file is joined with dynamic features during daily pipeline runs.

Usage:
    python -m nalanda.assemble_static
    python -m nalanda.assemble_static --output cache/static/static_features_v2.parquet
    python -m nalanda.assemble_static --skip-knn  # If sensor data not ready
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Default paths
DEFAULT_GRID_PATH = Path("data/mumbai/grid_30m_nd.parquet")
DEFAULT_SVF_PATH = Path("cache/static/svf.parquet")
DEFAULT_OSM_INDUSTRIAL_PATH = Path("cache/static/osm_industrial_aggregates.parquet")
DEFAULT_OSM_DUST_PATH = Path("cache/static/osm_dust_aggregates.parquet")
DEFAULT_ELEVATION_PATH = Path("cache/static/elevation.parquet")
DEFAULT_KNN_SENSORS_PATH = Path("cache/static/knn_sensors_k5.parquet")
DEFAULT_OUTPUT_PATH = Path("cache/static/static_features.parquet")


class StaticFeatureAssembler:
    """Assembles static features from multiple sources."""
    
    def __init__(
        self,
        grid_path: Path,
        svf_path: Optional[Path],
        osm_industrial_path: Optional[Path],
        osm_dust_path: Optional[Path],
        elevation_path: Optional[Path],
        knn_sensors_path: Optional[Path],
        output_path: Path,
        verbose: bool = False
    ):
        self.grid_path = grid_path
        self.svf_path = svf_path
        self.osm_industrial_path = osm_industrial_path
        self.osm_dust_path = osm_dust_path
        self.elevation_path = elevation_path
        self.knn_sensors_path = knn_sensors_path
        self.output_path = output_path
        self.verbose = verbose
        
        self.base_df = None
        self.missing_features = []
        self.loaded_features = []
    
    def log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"  {msg}")
    
    def load_grid_base(self) -> pd.DataFrame:
        """Load base grid with row_idx, grid_id, and coordinates."""
        print(f"Loading base grid from {self.grid_path}...")
        
        if not self.grid_path.exists():
            raise FileNotFoundError(f"Grid file not found: {self.grid_path}")
        
        df = pd.read_parquet(self.grid_path)
        
        # Keep only essential columns for static features
        essential_cols = [
            "row_idx", "grid_id", "x_m", "y_m",
            "centroid_lon", "centroid_lat", "h3_cell"
        ]
        available_cols = [c for c in essential_cols if c in df.columns]
        
        df = df[available_cols].copy()
        
        self.log(f"Loaded {len(df):,} points with columns: {list(df.columns)}")
        return df
    
    def load_svf(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Load and merge Sky View Factor."""
        if self.svf_path is None or not self.svf_path.exists():
            print(f"⚠ SVF file not found: {self.svf_path}")
            self.missing_features.append("svf")
            base_df["svf"] = np.nan
            return base_df
        
        print(f"Loading SVF from {self.svf_path}...")
        svf_df = pd.read_parquet(self.svf_path)
        
        # Expected columns: row_idx, svf
        if "row_idx" not in svf_df.columns or "svf" not in svf_df.columns:
            print(f"⚠ SVF file missing required columns (row_idx, svf)")
            self.missing_features.append("svf")
            base_df["svf"] = np.nan
            return base_df
        
        # Merge on row_idx
        merged = base_df.merge(
            svf_df[["row_idx", "svf"]],
            on="row_idx",
            how="left"
        )
        
        missing_count = merged["svf"].isna().sum()
        if missing_count > 0:
            print(f"⚠ {missing_count:,} points missing SVF values")
        
        self.log(f"Merged SVF: range [{merged['svf'].min():.3f}, {merged['svf'].max():.3f}]")
        self.loaded_features.append("svf")
        return merged
    
    def load_osm_industrial(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Load and merge OSM industrial aggregates."""
        if self.osm_industrial_path is None or not self.osm_industrial_path.exists():
            print(f"⚠ OSM industrial file not found: {self.osm_industrial_path}")
            self.missing_features.append("osm_industrial")
            for radius in [30, 90, 270]:
                base_df[f"osm_industrial_{radius}m"] = 0
            return base_df
        
        print(f"Loading OSM industrial from {self.osm_industrial_path}...")
        osm_df = pd.read_parquet(self.osm_industrial_path)
        
        # Expected columns: row_idx, osm_industrial_30m, osm_industrial_90m, osm_industrial_270m
        expected_cols = ["row_idx"] + [f"osm_industrial_{r}m" for r in [30, 90, 270]]
        missing_cols = [c for c in expected_cols if c not in osm_df.columns]
        
        if missing_cols:
            print(f"⚠ OSM industrial missing columns: {missing_cols}")
            self.missing_features.append("osm_industrial")
            for radius in [30, 90, 270]:
                base_df[f"osm_industrial_{radius}m"] = 0
            return base_df
        
        # Merge
        merged = base_df.merge(
            osm_df[expected_cols],
            on="row_idx",
            how="left"
        )
        
        # Fill missing with 0
        for radius in [30, 90, 270]:
            col = f"osm_industrial_{radius}m"
            merged[col] = merged[col].fillna(0).astype(np.int16)
        
        self.log(f"Merged OSM industrial aggregates")
        self.loaded_features.append("osm_industrial")
        return merged
    
    def load_osm_dust(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Load and merge OSM dust source aggregates."""
        if self.osm_dust_path is None or not self.osm_dust_path.exists():
            print(f"⚠ OSM dust file not found: {self.osm_dust_path}")
            self.missing_features.append("osm_dust")
            for radius in [30, 90, 270]:
                base_df[f"osm_dust_{radius}m"] = 0
            return base_df
        
        print(f"Loading OSM dust from {self.osm_dust_path}...")
        osm_df = pd.read_parquet(self.osm_dust_path)
        
        # Expected columns: row_idx, osm_dust_30m, osm_dust_90m, osm_dust_270m
        expected_cols = ["row_idx"] + [f"osm_dust_{r}m" for r in [30, 90, 270]]
        missing_cols = [c for c in expected_cols if c not in osm_df.columns]
        
        if missing_cols:
            print(f"⚠ OSM dust missing columns: {missing_cols}")
            self.missing_features.append("osm_dust")
            for radius in [30, 90, 270]:
                base_df[f"osm_dust_{radius}m"] = 0
            return base_df
        
        # Merge
        merged = base_df.merge(
            osm_df[expected_cols],
            on="row_idx",
            how="left"
        )
        
        # Fill missing with 0
        for radius in [30, 90, 270]:
            col = f"osm_dust_{radius}m"
            merged[col] = merged[col].fillna(0).astype(np.int16)
        
        self.log(f"Merged OSM dust aggregates")
        self.loaded_features.append("osm_dust")
        return merged
    
    def load_elevation(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Load and merge elevation data."""
        if self.elevation_path is None or not self.elevation_path.exists():
            print(f"⚠ Elevation file not found: {self.elevation_path}")
            self.missing_features.append("elevation")
            base_df["elevation_m"] = np.nan
            return base_df
        
        print(f"Loading elevation from {self.elevation_path}...")
        elev_df = pd.read_parquet(self.elevation_path)
        
        # Expected columns: row_idx, elevation_m
        if "row_idx" not in elev_df.columns or "elevation_m" not in elev_df.columns:
            print(f"⚠ Elevation file missing required columns")
            self.missing_features.append("elevation")
            base_df["elevation_m"] = np.nan
            return base_df
        
        # Merge
        merged = base_df.merge(
            elev_df[["row_idx", "elevation_m"]],
            on="row_idx",
            how="left"
        )
        
        missing_count = merged["elevation_m"].isna().sum()
        if missing_count > 0:
            print(f"⚠ {missing_count:,} points missing elevation values")
        
        self.log(f"Merged elevation: range [{merged['elevation_m'].min():.1f}m, {merged['elevation_m'].max():.1f}m]")
        self.loaded_features.append("elevation")
        return merged
    
    def load_knn_sensors(self, base_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """Load and merge KNN sensor indices."""
        if self.knn_sensors_path is None or not self.knn_sensors_path.exists():
            print(f"⚠ KNN sensors file not found: {self.knn_sensors_path}")
            self.missing_features.append("knn_sensors")
            # Create placeholder columns
            for i in range(1, k + 1):
                base_df[f"sensor_id_{i}"] = ""
                base_df[f"sensor_dist_{i}"] = np.inf
            return base_df
        
        print(f"Loading KNN sensors from {self.knn_sensors_path}...")
        knn_df = pd.read_parquet(self.knn_sensors_path)
        
        # Expected columns: row_idx, sensor_id_1..k, sensor_dist_1..k
        expected_cols = ["row_idx"]
        for i in range(1, k + 1):
            expected_cols.extend([f"sensor_id_{i}", f"sensor_dist_{i}"])
        
        missing_cols = [c for c in expected_cols if c not in knn_df.columns]
        if missing_cols:
            print(f"⚠ KNN sensors missing columns: {missing_cols}")
            self.missing_features.append("knn_sensors")
            for i in range(1, k + 1):
                base_df[f"sensor_id_{i}"] = ""
                base_df[f"sensor_dist_{i}"] = np.inf
            return base_df
        
        # Merge
        merged = base_df.merge(
            knn_df[expected_cols],
            on="row_idx",
            how="left"
        )
        
        # Check for missing values
        missing_count = merged[f"sensor_id_1"].isna().sum()
        if missing_count > 0:
            print(f"⚠ {missing_count:,} points missing KNN sensor data")
        
        self.log(f"Merged KNN sensors (k={k})")
        self.loaded_features.append("knn_sensors")
        return merged
    
    def write_output(self, df: pd.DataFrame):
        """Write assembled static features to output path."""
        print(f"\nWriting static features to {self.output_path}...")
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write parquet
        df.to_parquet(self.output_path, index=False, compression="snappy")
        
        file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Written {len(df):,} rows, {len(df.columns)} columns ({file_size_mb:.2f} MB)")
        
        # Write manifest
        manifest_path = self.output_path.parent / f"{self.output_path.stem}_manifest.json"
        manifest = {
            "generated_ts": datetime.utcnow().isoformat() + "Z",
            "n_points": len(df),
            "columns": list(df.columns),
            "loaded_features": self.loaded_features,
            "missing_features": self.missing_features,
            "source_files": {
                "grid": str(self.grid_path),
                "svf": str(self.svf_path) if self.svf_path else None,
                "osm_industrial": str(self.osm_industrial_path) if self.osm_industrial_path else None,
                "osm_dust": str(self.osm_dust_path) if self.osm_dust_path else None,
                "elevation": str(self.elevation_path) if self.elevation_path else None,
                "knn_sensors": str(self.knn_sensors_path) if self.knn_sensors_path else None,
            },
            "output_path": str(self.output_path),
            "schema_version": "1.0"
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✓ Manifest written to {manifest_path}")
    
    def run(self):
        """Run the assembly pipeline."""
        print("\n" + "=" * 60)
        print("STATIC FEATURE ASSEMBLY")
        print("=" * 60)
        
        try:
            # Load base grid
            df = self.load_grid_base()
            
            # Load and merge each feature type
            df = self.load_svf(df)
            df = self.load_osm_industrial(df)
            df = self.load_osm_dust(df)
            df = self.load_elevation(df)
            df = self.load_knn_sensors(df, k=5)
            
            # Write output
            self.write_output(df)
            
            # Summary
            print("\n" + "=" * 60)
            print("ASSEMBLY SUMMARY")
            print("=" * 60)
            print(f"✓ Loaded features: {', '.join(self.loaded_features) if self.loaded_features else 'None'}")
            
            if self.missing_features:
                print(f"⚠ Missing features: {', '.join(self.missing_features)}")
                print("  (These will have placeholder/NaN values)")
            
            print(f"\nFinal schema ({len(df.columns)} columns):")
            for col in df.columns:
                dtype = df[col].dtype
                null_count = df[col].isna().sum()
                null_pct = 100 * null_count / len(df)
                print(f"  {col:30s} {dtype:12s} ({null_pct:5.1f}% null)")
            
            print(f"\n✓ Static features ready at: {self.output_path}")
            
            if self.missing_features:
                print("\n⚠ WARNING: Some features are missing. Run the following:")
                if "svf" in self.missing_features:
                    print("  - SVF: python patliputra/svf_earth_engine.py")
                if "osm_industrial" in self.missing_features:
                    print("  - OSM industrial: python patliputra/aggregate_osm_industrial_30m.py")
                if "osm_dust" in self.missing_features:
                    print("  - OSM dust: python patliputra/aggregate_osm_dust_30m.py")
                if "elevation" in self.missing_features:
                    print("  - Elevation: Extract from DSM/SRTM")
                if "knn_sensors" in self.missing_features:
                    print("  - KNN sensors: python -m nalanda.precompute --mode knn-sensors")
                print("\nThen re-run this script to update static_features.parquet")
                return 1
            
            return 0
        
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Assemble static features from precomputed components"
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        default=DEFAULT_GRID_PATH,
        help="Path to base grid parquet"
    )
    parser.add_argument(
        "--svf-path",
        type=Path,
        default=DEFAULT_SVF_PATH,
        help="Path to SVF parquet"
    )
    parser.add_argument(
        "--osm-industrial-path",
        type=Path,
        default=DEFAULT_OSM_INDUSTRIAL_PATH,
        help="Path to OSM industrial aggregates"
    )
    parser.add_argument(
        "--osm-dust-path",
        type=Path,
        default=DEFAULT_OSM_DUST_PATH,
        help="Path to OSM dust aggregates"
    )
    parser.add_argument(
        "--elevation-path",
        type=Path,
        default=DEFAULT_ELEVATION_PATH,
        help="Path to elevation parquet"
    )
    parser.add_argument(
        "--knn-sensors-path",
        type=Path,
        default=DEFAULT_KNN_SENSORS_PATH,
        help="Path to KNN sensors parquet"
    )
    parser.add_argument(
        "--skip-svf",
        action="store_true",
        help="Skip SVF (use NaN placeholders)"
    )
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help="Skip OSM aggregates (use 0 placeholders)"
    )
    parser.add_argument(
        "--skip-elevation",
        action="store_true",
        help="Skip elevation (use NaN placeholders)"
    )
    parser.add_argument(
        "--skip-knn",
        action="store_true",
        help="Skip KNN sensors (use placeholder values)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for assembled features"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args(argv)
    
    # Handle skip flags
    svf_path = None if args.skip_svf else args.svf_path
    osm_industrial_path = None if args.skip_osm else args.osm_industrial_path
    osm_dust_path = None if args.skip_osm else args.osm_dust_path
    elevation_path = None if args.skip_elevation else args.elevation_path
    knn_sensors_path = None if args.skip_knn else args.knn_sensors_path
    
    assembler = StaticFeatureAssembler(
        grid_path=args.grid_path,
        svf_path=svf_path,
        osm_industrial_path=osm_industrial_path,
        osm_dust_path=osm_dust_path,
        elevation_path=elevation_path,
        knn_sensors_path=knn_sensors_path,
        output_path=args.output,
        verbose=args.verbose
    )
    
    return assembler.run()


if __name__ == "__main__":
    sys.exit(main())
