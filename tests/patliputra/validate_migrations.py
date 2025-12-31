"""Validate parquet files after migration to new grid schema.

This script checks all parquet files in specified directories to ensure:
1. Files can be loaded without corruption
2. Required columns exist (row_idx, grid_id, x_m, y_m)
3. row_idx values are valid (0 to N-1, no duplicates)
4. Coordinate consistency with canonical grid
5. No data loss (row counts match expected)

Usage:
    python scripts/validate_migrations.py
    python scripts/validate_migrations.py --dir cache/landsat_processed
    python scripts/validate_migrations.py --dir cache/landsat_processed --check-coords
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

import numpy as np
import pandas as pd
from glob import glob


DEFAULT_GRID_PATH = Path("data/mumbai/grid_30m_nd.parquet")
EXPECTED_N_POINTS = 509209


class ValidationError(Exception):
    """Validation failure."""
    pass


class MigrationValidator:
    """Validator for migrated parquet files."""
    
    def __init__(
        self,
        grid_path: Path,
        check_coordinates: bool = False,
        verbose: bool = False
    ):
        self.grid_path = grid_path
        self.check_coordinates = check_coordinates
        self.verbose = verbose
        
        # Load canonical grid for reference
        self.canonical_grid = None
        if check_coordinates:
            print(f"Loading canonical grid from {grid_path}...")
            self.canonical_grid = pd.read_parquet(grid_path)
            print(f"  Loaded {len(self.canonical_grid):,} points")
        
        # Results tracking
        self.results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }
    
    def log(self, msg: str):
        """Log if verbose."""
        if self.verbose:
            print(f"  {msg}")
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate a single parquet file.
        
        Returns
        -------
        Tuple[bool, Optional[str]]
            (is_valid, error_message)
        """
        try:
            # Test 1: Can we load it?
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                return False, f"CORRUPT: Cannot load file - {str(e)[:100]}"
            
            # Test 2: Has any data?
            if len(df) == 0:
                return False, "EMPTY: File has 0 rows"
            
            # Test 3: Check for new schema columns
            required_cols = ["row_idx", "grid_id", "x_m", "y_m"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                # Check if it's old schema (not migrated)
                if "x" in df.columns and "y" in df.columns:
                    return False, f"NOT_MIGRATED: Still has old schema (x, y) - missing {missing_cols}"
                else:
                    return False, f"INVALID_SCHEMA: Missing columns {missing_cols}"
            
            # Test 4: row_idx validity
            if df["row_idx"].isna().any():
                null_count = df["row_idx"].isna().sum()
                return False, f"NULL_ROW_IDX: {null_count} rows have null row_idx"
            
            # Test 5: row_idx range
            min_idx = df["row_idx"].min()
            max_idx = df["row_idx"].max()
            
            if min_idx < 0:
                return False, f"INVALID_ROW_IDX: row_idx < 0 (min={min_idx})"
            
            if max_idx >= EXPECTED_N_POINTS:
                return False, f"INVALID_ROW_IDX: row_idx >= {EXPECTED_N_POINTS} (max={max_idx})"
            
            # Test 6: row_idx duplicates
            if df["row_idx"].duplicated().any():
                dup_count = df["row_idx"].duplicated().sum()
                return False, f"DUPLICATE_ROW_IDX: {dup_count} duplicate row_idx values"
            
            # Test 7: Expected row count
            if len(df) != EXPECTED_N_POINTS:
                return False, f"WRONG_COUNT: Expected {EXPECTED_N_POINTS} rows, got {len(df):,}"
            
            # Test 8: Coordinate consistency (optional, expensive)
            if self.check_coordinates and self.canonical_grid is not None:
                # Sample check: verify 100 random points match canonical grid
                sample_size = min(100, len(df))
                sample_indices = np.random.choice(len(df), sample_size, replace=False)
                
                for idx in sample_indices:
                    row = df.iloc[idx]
                    row_idx = int(row["row_idx"])
                    
                    # Get canonical coordinates
                    canonical_row = self.canonical_grid.loc[
                        self.canonical_grid["row_idx"] == row_idx
                    ]
                    
                    if len(canonical_row) == 0:
                        return False, f"COORD_MISMATCH: row_idx {row_idx} not in canonical grid"
                    
                    canonical_row = canonical_row.iloc[0]
                    
                    # Check coordinates match (within 1m tolerance)
                    x_diff = abs(row["x_m"] - canonical_row["x_m"])
                    y_diff = abs(row["y_m"] - canonical_row["y_m"])
                    
                    if x_diff > 1.0 or y_diff > 1.0:
                        return False, f"COORD_MISMATCH: row_idx {row_idx} coords differ by ({x_diff:.2f}m, {y_diff:.2f}m)"
            
            # All tests passed
            return True, None
        
        except Exception as e:
            return False, f"UNEXPECTED_ERROR: {str(e)[:150]}"
    
    def validate_directory(self, directory: Path, pattern: str = "*.parquet") -> None:
        """Validate all parquet files in a directory."""
        print(f"\nValidating files in: {directory}")
        print(f"Pattern: {pattern}")
        print("=" * 80)
        
        # Find all parquet files
        search_path = directory / pattern
        files = sorted(glob(str(search_path)))
        
        if not files:
            print(f"No files found matching {search_path}")
            return
        
        print(f"Found {len(files)} files to validate\n")
        
        # Validate each file
        for file_path_str in files:
            file_path = Path(file_path_str)
            file_name = file_path.name
            
            # Skip backup files
            if ".backup." in file_name or "_migration.json" in file_name:
                self.results["skipped"] += 1
                continue
            
            self.results["total"] += 1
            
            # Validate
            is_valid, error_msg = self.validate_file(file_path)
            
            if is_valid:
                self.results["passed"] += 1
                print(f"OK    {file_name}")
                self.log(f"All checks passed")
            else:
                self.results["failed"] += 1
                print(f"FAIL  {file_name}")
                print(f"      {error_msg}")
                self.results["errors"].append({
                    "file": str(file_path),
                    "error": error_msg
                })
    
    def generate_report(self, output_path: Optional[Path] = None) -> None:
        """Generate validation report."""
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        
        total = self.results["total"]
        passed = self.results["passed"]
        failed = self.results["failed"]
        skipped = self.results["skipped"]
        
        print(f"Total files checked:  {total}")
        print(f"Passed:              {passed} ({100*passed/total:.1f}%)" if total > 0 else "Passed: 0")
        print(f"Failed:              {failed} ({100*failed/total:.1f}%)" if total > 0 else "Failed: 0")
        print(f"Skipped:             {skipped} (backups/manifests)")
        
        if failed > 0:
            print(f"\nFailed files:")
            for error_info in self.results["errors"]:
                file_name = Path(error_info["file"]).name
                error_msg = error_info["error"]
                print(f"  - {file_name}")
                print(f"    {error_msg}")
        
        # Write JSON report
        if output_path:
            report = {
                "generated_ts": datetime.utcnow().isoformat() + "Z",
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                },
                "errors": self.results["errors"],
            }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            
            print(f"\nDetailed report written to: {output_path}")
        
        # Exit code
        if failed > 0:
            print("\nVALIDATION FAILED")
            return 1
        else:
            print("\nALL VALIDATIONS PASSED")
            return 0


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate parquet files after migration"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        action="append",
        help="Directory to validate (can specify multiple times)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.parquet",
        help="File pattern to match (default: *.parquet)"
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        default=DEFAULT_GRID_PATH,
        help="Path to canonical grid"
    )
    parser.add_argument(
        "--check-coords",
        action="store_true",
        help="Perform expensive coordinate validation (slow)"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("tmp/validation_report.json"),
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args(argv)
    
    # Default directories if none specified
    if not args.dir:
        args.dir = [
            Path("cache/landsat_processed"),
            Path("cache/static"),
        ]
    
    # Create validator
    validator = MigrationValidator(
        grid_path=args.grid_path,
        check_coordinates=args.check_coords,
        verbose=args.verbose
    )
    
    # Validate each directory
    for directory in args.dir:
        if not directory.exists():
            print(f"WARNING: Directory not found: {directory}")
            continue
        
        validator.validate_directory(directory, args.pattern)
    
    # Generate report
    exit_code = validator.generate_report(args.report)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
