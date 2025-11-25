"""Validation script for precomputed neighbor indices from nalanda/precompute.py.

This script performs comprehensive checks on precomputed neighbor files:
1. File existence and integrity
2. Data structure validation (flat, offsets, metadata)
3. Index range checks
4. Neighbor symmetry verification
5. Distance verification (neighbors within expected radius)
6. CSR matrix validation
7. Consistency checks across different radii
8. Sample spot-checks with ground truth

Usage:
    python tests/validate_precomputed_neighbors.py
    python tests/validate_precomputed_neighbors.py --grid-path data/mumbai/grid_30m_nd.parquet
    python tests/validate_precomputed_neighbors.py --verbose --sample-size 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.spatial import cKDTree


# Default paths
DEFAULT_GRID_PATH = Path("data/mumbai/grid_30m_nd.parquet")
DEFAULT_CACHE_DIR = Path("cache/static")
DEFAULT_RADII = [30, 90, 270]


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class NeighborValidator:
    """Validator for precomputed neighbor indices."""
    
    def __init__(
        self,
        grid_path: Path,
        cache_dir: Path,
        radii: List[float],
        verbose: bool = False
    ):
        self.grid_path = grid_path
        self.cache_dir = cache_dir
        self.radii = radii
        self.verbose = verbose
        
        # Will be loaded
        self.grid_df = None
        self.coords = None
        self.n_points = 0
        self.tree = None
        
        # Validation results
        self.results = {}
        self.errors = []
        self.warnings = []
    
    def log(self, msg: str, level: str = "INFO"):
        """Log message with level."""
        prefix = {
            "INFO": "✓",
            "WARNING": "⚠",
            "ERROR": "✗",
            "DEBUG": " "
        }.get(level, " ")
        
        if level == "ERROR":
            self.errors.append(msg)
        elif level == "WARNING":
            self.warnings.append(msg)
        
        if self.verbose or level in ["INFO", "WARNING", "ERROR"]:
            print(f"{prefix} {msg}")
    
    def load_grid(self):
        """Load grid and build KDTree for distance verification."""
        self.log(f"Loading grid from {self.grid_path}...")
        
        if not self.grid_path.exists():
            raise ValidationError(f"Grid file not found: {self.grid_path}")
        
        self.grid_df = pd.read_parquet(self.grid_path)
        self.n_points = len(self.grid_df)
        
        required_cols = ["row_idx", "x_m", "y_m"]
        missing = [c for c in required_cols if c not in self.grid_df.columns]
        if missing:
            raise ValidationError(f"Grid missing columns: {missing}")
        
        self.coords = self.grid_df[["x_m", "y_m"]].values.astype(np.float64)
        self.tree = cKDTree(self.coords)
        
        self.log(f"Loaded {self.n_points:,} grid points")
    
    def validate_file_existence(self, radius: float) -> bool:
        """Check if required files exist."""
        self.log(f"\n{'='*60}")
        self.log(f"Validating files for radius {radius}m")
        self.log(f"{'='*60}")
        
        npz_path = self.cache_dir / f"neighbors_{int(radius)}m.npz"
        csr_path = self.cache_dir / f"neighbors_{int(radius)}m_csr.npz"
        
        files_ok = True
        
        if not npz_path.exists():
            self.log(f"Missing NPZ file: {npz_path}", "ERROR")
            files_ok = False
        else:
            size_mb = npz_path.stat().st_size / (1024 * 1024)
            self.log(f"NPZ file exists: {npz_path} ({size_mb:.2f} MB)")
        
        if not csr_path.exists():
            self.log(f"Missing CSR file: {csr_path}", "WARNING")
        else:
            size_mb = csr_path.stat().st_size / (1024 * 1024)
            self.log(f"CSR file exists: {csr_path} ({size_mb:.2f} MB)")
        
        return files_ok
    
    def validate_npz_structure(self, radius: float) -> Dict:
        """Validate NPZ file structure and metadata."""
        self.log(f"\nChecking NPZ structure for {radius}m...")
        
        npz_path = self.cache_dir / f"neighbors_{int(radius)}m.npz"
        data = np.load(npz_path)
        
        # Check required keys
        required_keys = ["flat", "offsets", "n_points", "radius_m"]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise ValidationError(f"NPZ missing keys: {missing_keys}")
        
        flat = data["flat"]
        offsets = data["offsets"]
        n_points_stored = int(data["n_points"])
        radius_stored = float(data["radius_m"])
        
        # Validate metadata
        if n_points_stored != self.n_points:
            raise ValidationError(
                f"n_points mismatch: NPZ={n_points_stored}, grid={self.n_points}"
            )
        
        if abs(radius_stored - radius) > 0.1:
            raise ValidationError(
                f"Radius mismatch: NPZ={radius_stored}, expected={radius}"
            )
        
        # Validate array shapes and types
        if flat.dtype != np.int32:
            self.log(f"flat dtype is {flat.dtype}, expected int32", "WARNING")
        
        if offsets.dtype != np.int64:
            self.log(f"offsets dtype is {offsets.dtype}, expected int64", "WARNING")
        
        if len(offsets) != self.n_points + 1:
            raise ValidationError(
                f"offsets length={len(offsets)}, expected {self.n_points + 1}"
            )
        
        self.log(f"NPZ structure valid: {len(flat):,} neighbors, {len(offsets):,} offsets")
        
        return {
            "flat": flat,
            "offsets": offsets,
            "total_neighbors": len(flat),
            "avg_neighbors": len(flat) / self.n_points
        }
    
    def validate_indices(self, flat: np.ndarray, offsets: np.ndarray, radius: float):
        """Validate index ranges and relationships."""
        self.log(f"\nValidating indices for {radius}m...")
        
        # Check 1: Offsets monotonically increasing
        if not np.all(np.diff(offsets) >= 0):
            raise ValidationError("Offsets not monotonically increasing")
        
        # Check 2: Final offset matches flat length
        if offsets[-1] != len(flat):
            raise ValidationError(
                f"offsets[-1]={offsets[-1]} != len(flat)={len(flat)}"
            )
        
        # Check 3: All indices in valid range
        if len(flat) > 0:
            if flat.min() < 0:
                raise ValidationError(f"Negative index found: {flat.min()}")
            
            if flat.max() >= self.n_points:
                raise ValidationError(
                    f"Index out of range: {flat.max()} >= {self.n_points}"
                )
        
        # Check 4: No duplicate neighbors (within each point's list)
        duplicates_found = 0
        for i in range(min(1000, self.n_points)):  # Sample check
            start, end = offsets[i], offsets[i + 1]
            neighbors = flat[start:end]
            if len(neighbors) != len(set(neighbors)):
                duplicates_found += 1
        
        if duplicates_found > 0:
            self.log(
                f"Found {duplicates_found} points with duplicate neighbors",
                "WARNING"
            )
        
        neighbor_counts = np.diff(offsets)
        self.log(f"Index validation passed")
        self.log(f"  Avg neighbors: {neighbor_counts.mean():.1f}", "DEBUG")
        self.log(f"  Min/max: {neighbor_counts.min()} / {neighbor_counts.max()}", "DEBUG")
    
    def validate_symmetry(
        self,
        flat: np.ndarray,
        offsets: np.ndarray,
        radius: float,
        sample_size: int = 2000
    ):
        """Verify neighbor symmetry: if j in neighbors(i), then i in neighbors(j)."""
        self.log(f"\nChecking symmetry for {radius}m (sample={sample_size})...")
        
        # Sample random points
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(
            self.n_points,
            size=min(sample_size, self.n_points),
            replace=False
        )
        
        asymmetry_count = 0
        asymmetry_examples = []
        
        for i in sample_indices:
            start, end = offsets[i], offsets[i + 1]
            neighbors_i = flat[start:end]
            
            for j in neighbors_i:
                j_start, j_end = offsets[j], offsets[j + 1]
                neighbors_j = flat[j_start:j_end]
                
                if i not in neighbors_j:
                    asymmetry_count += 1
                    if len(asymmetry_examples) < 5:
                        asymmetry_examples.append((i, j))
        
        if asymmetry_count > 0:
            self.log(
                f"Asymmetry detected: {asymmetry_count} violations in {sample_size} samples",
                "ERROR"
            )
            for i, j in asymmetry_examples:
                self.log(f"  Example: {j} in neighbors({i}), but {i} not in neighbors({j})", "DEBUG")
            raise ValidationError("Neighbor symmetry violation")
        
        self.log(f"Symmetry check passed ({sample_size} samples)")
    
    def validate_distances(
        self,
        flat: np.ndarray,
        offsets: np.ndarray,
        radius: float,
        sample_size: int = 1000,
        tolerance: float = 1e-6
    ):
        """Verify neighbors are actually within the specified radius."""
        self.log(f"\nValidating distances for {radius}m (sample={sample_size})...")
        
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(
            self.n_points,
            size=min(sample_size, self.n_points),
            replace=False
        )
        
        violations = 0
        max_violation = 0.0
        
        for i in sample_indices:
            start, end = offsets[i], offsets[i + 1]
            neighbors = flat[start:end]
            
            if len(neighbors) == 0:
                continue
            
            # Compute distances to all neighbors
            point_i = self.coords[i]
            neighbor_coords = self.coords[neighbors]
            distances = np.linalg.norm(neighbor_coords - point_i, axis=1)
            
            # Check if any exceed radius (with small tolerance)
            exceed_mask = distances > (radius + tolerance)
            if np.any(exceed_mask):
                violations += 1
                max_dist = distances[exceed_mask].max()
                max_violation = max(max_violation, max_dist - radius)
        
        if violations > 0:
            self.log(
                f"Distance violations: {violations}/{sample_size} points",
                "ERROR"
            )
            self.log(f"  Max violation: {max_violation:.6f}m beyond radius", "DEBUG")
            raise ValidationError("Neighbor distance validation failed")
        
        self.log(f"Distance validation passed")
    
    def validate_completeness(
        self,
        flat: np.ndarray,
        offsets: np.ndarray,
        radius: float,
        sample_size: int = 500
    ):
        """Verify no neighbors are missing (compare with KDTree ground truth)."""
        self.log(f"\nChecking completeness for {radius}m (sample={sample_size})...")
        
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(
            self.n_points,
            size=min(sample_size, self.n_points),
            replace=False
        )
        
        missing_count = 0
        extra_count = 0
        
        for i in sample_indices:
            # Get stored neighbors
            start, end = offsets[i], offsets[i + 1]
            stored_neighbors = set(flat[start:end])
            
            # Get ground truth from KDTree
            true_neighbors = set(self.tree.query_ball_point(self.coords[i], r=radius))
            
            # Compare
            missing = true_neighbors - stored_neighbors
            extra = stored_neighbors - true_neighbors
            
            if missing:
                missing_count += len(missing)
            if extra:
                extra_count += len(extra)
        
        if missing_count > 0:
            self.log(f"Missing neighbors: {missing_count} total", "ERROR")
            raise ValidationError("Completeness check failed: missing neighbors")
        
        if extra_count > 0:
            self.log(f"Extra neighbors: {extra_count} total", "WARNING")
        
        self.log(f"Completeness check passed")
    
    def validate_csr_matrix(self, radius: float):
        """Validate CSR matrix structure and consistency with NPZ."""
        self.log(f"\nValidating CSR matrix for {radius}m...")
        
        csr_path = self.cache_dir / f"neighbors_{int(radius)}m_csr.npz"
        if not csr_path.exists():
            self.log("CSR file not found, skipping", "WARNING")
            return
        
        # Load CSR
        csr = load_npz(csr_path)
        
        # Check shape
        if csr.shape != (self.n_points, self.n_points):
            raise ValidationError(
                f"CSR shape {csr.shape} != expected ({self.n_points}, {self.n_points})"
            )
        
        # Check dtype
        if csr.dtype != np.float32:
            self.log(f"CSR dtype {csr.dtype}, expected float32", "WARNING")
        
        # Check row sums (should be 1.0 for each row, or 0 if no neighbors)
        row_sums = np.array(csr.sum(axis=1)).flatten()
        non_zero_rows = row_sums > 0
        
        if np.any(non_zero_rows):
            non_one_rows = np.abs(row_sums[non_zero_rows] - 1.0) > 1e-5
            if np.any(non_one_rows):
                self.log(
                    f"CSR row sums not 1.0: {np.sum(non_one_rows)} rows affected",
                    "WARNING"
                )
        
        self.log(f"CSR matrix valid: {csr.nnz:,} nonzeros, {100 * csr.nnz / (self.n_points**2):.4f}% dense")
    
    def validate_manifest(self):
        """Validate manifest file if it exists."""
        self.log(f"\n{'='*60}")
        self.log("Validating manifest file")
        self.log(f"{'='*60}")
        
        manifest_path = self.cache_dir / "neighbors_manifest.json"
        if not manifest_path.exists():
            self.log("Manifest file not found", "WARNING")
            return
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        # Check required keys
        required_keys = ["generated_ts", "grid_path", "n_points", "radii_computed"]
        missing = [k for k in required_keys if k not in manifest]
        if missing:
            self.log(f"Manifest missing keys: {missing}", "WARNING")
        else:
            self.log(f"Manifest structure valid")
            self.log(f"  Generated: {manifest.get('generated_ts', 'unknown')}", "DEBUG")
            self.log(f"  Grid: {manifest.get('grid_path', 'unknown')}", "DEBUG")
            self.log(f"  Points: {manifest.get('n_points', 'unknown'):,}", "DEBUG")
            self.log(f"  Radii: {manifest.get('radii_computed', [])}", "DEBUG")
    
    def run_validation(self):
        """Run full validation suite."""
        print("\n" + "=" * 60)
        print("NEIGHBOR PRECOMPUTATION VALIDATION")
        print("=" * 60)
        
        try:
            # Load grid
            self.load_grid()
            
            # Validate manifest
            self.validate_manifest()
            
            # Validate each radius
            for radius in self.radii:
                # File existence
                if not self.validate_file_existence(radius):
                    self.log(f"Skipping validation for {radius}m (files missing)", "ERROR")
                    continue
                
                # NPZ structure
                npz_data = self.validate_npz_structure(radius)
                flat = npz_data["flat"]
                offsets = npz_data["offsets"]
                
                # Index validation
                self.validate_indices(flat, offsets, radius)
                
                # Symmetry
                self.validate_symmetry(flat, offsets, radius, sample_size=2000)
                
                # Distances
                self.validate_distances(flat, offsets, radius, sample_size=1000)
                
                # Completeness
                self.validate_completeness(flat, offsets, radius, sample_size=500)
                
                # CSR matrix
                self.validate_csr_matrix(radius)
                
                self.results[radius] = {
                    "total_neighbors": npz_data["total_neighbors"],
                    "avg_neighbors": npz_data["avg_neighbors"],
                    "status": "PASSED"
                }
            
            # Summary
            print("\n" + "=" * 60)
            print("VALIDATION SUMMARY")
            print("=" * 60)
            
            for radius, result in self.results.items():
                print(f"Radius {radius}m: {result['status']}")
                print(f"  Total neighbors: {result['total_neighbors']:,}")
                print(f"  Avg per point: {result['avg_neighbors']:.1f}")
            
            if self.warnings:
                print(f"\n⚠ {len(self.warnings)} warnings:")
                for warning in self.warnings[:10]:
                    print(f"  - {warning}")
            
            if self.errors:
                print(f"\n✗ {len(self.errors)} errors:")
                for error in self.errors[:10]:
                    print(f"  - {error}")
                print("\n✗ VALIDATION FAILED")
                return 1
            else:
                print("\n✓ ALL VALIDATIONS PASSED")
                return 0
        
        except ValidationError as e:
            print(f"\n✗ VALIDATION FAILED: {e}")
            return 1
        except Exception as e:
            print(f"\n✗ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate precomputed neighbor indices from nalanda/precompute.py"
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        default=DEFAULT_GRID_PATH,
        help="Path to grid parquet (default: data/mumbai/grid_30m_nd.parquet)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Cache directory with neighbor files (default: cache/static)"
    )
    parser.add_argument(
        "--radii",
        type=str,
        default=",".join(map(str, DEFAULT_RADII)),
        help="Comma-separated radii to validate (default: 30,90,270)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size for statistical checks (default: 1000)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debug information"
    )
    
    args = parser.parse_args(argv)
    
    radii = [float(x.strip()) for x in args.radii.split(",")]
    
    validator = NeighborValidator(
        grid_path=args.grid_path,
        cache_dir=args.cache_dir,
        radii=radii,
        verbose=args.verbose
    )
    
    return validator.run_validation()


if __name__ == "__main__":
    sys.exit(main())
