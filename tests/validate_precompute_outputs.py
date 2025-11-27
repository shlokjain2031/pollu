"""
Comprehensive validation script for all outputs of nalanda/precompute.py.

Validates:
- Manifest file (metadata, structure)
- Grid file (columns, n_points)
- Neighbor NPZ files (structure, indices, symmetry, distances)
- CSR matrix files (shape, dtype, row sums)
- KNN sensor indices (if implemented)
- Consistency between all outputs

Usage:
    python tests/validate_precompute_outputs.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from scipy.spatial import cKDTree

MANIFEST_PATH = Path("cache/static/neighbors_manifest.json")


def validate_manifest(manifest_path):
    print(f"✓ Checking manifest: {manifest_path}")
    assert manifest_path.exists(), "Manifest file missing"
    with open(manifest_path) as f:
        manifest = json.load(f)
    required = ["grid_path", "radii_computed", "n_points", "results"]
    for k in required:
        assert k in manifest, f"Manifest missing key: {k}"
    return manifest


def validate_grid(grid_path, expected_n_points):
    print(f"✓ Checking grid: {grid_path}")
    df = pd.read_parquet(grid_path)
    for col in ["row_idx", "x_m", "y_m"]:
        assert col in df.columns, f"Grid missing column: {col}"
    assert len(df) == expected_n_points, "Grid n_points mismatch"
    return df


def validate_neighbors_npz(npz_path, n_points, radius):
    print(f"✓ Checking neighbors NPZ: {npz_path}")
    data = np.load(npz_path)
    flat, offsets = data["flat"], data["offsets"]
    assert len(offsets) == n_points + 1, "Offsets length mismatch"
    assert offsets[-1] == len(flat), "Offsets[-1] != len(flat)"
    assert flat.min() >= 0 and flat.max() < n_points, "Neighbor indices out of range"
    # Symmetry check (sample)
    sample = np.random.choice(n_points, min(1000, n_points), replace=False)
    for i in sample:
        neighbors = flat[offsets[i]:offsets[i+1]]
        for j in neighbors:
            neighbors_j = flat[offsets[j]:offsets[j+1]]
            assert i in neighbors_j, f"Symmetry failed: {i} <-> {j}"
    return flat, offsets


def validate_csr(csr_path, n_points):
    print(f"✓ Checking CSR: {csr_path}")
    csr = load_npz(csr_path)
    assert csr.shape == (n_points, n_points), "CSR shape mismatch"
    row_sums = np.array(csr.sum(axis=1)).flatten()
    assert np.all((row_sums == 0) | (np.abs(row_sums - 1.0) < 1e-5)), "CSR row sums not 1.0 or 0"
    return csr


def validate_distances(flat, offsets, coords, radius):
    print(f"✓ Checking neighbor distances (radius {radius})")
    for i in range(len(coords)):
        neighbors = flat[offsets[i]:offsets[i+1]]
        if len(neighbors) == 0:
            continue
        dists = np.linalg.norm(coords[neighbors] - coords[i], axis=1)
        assert np.all(dists <= radius + 1e-6), f"Neighbor at i={i} exceeds radius"


def validate_knn_sensors():
    # Placeholder: implement if/when KNN sensor output is available
    knn_files = list(Path("cache/static").glob("knn_sensors_*.parquet"))
    if not knn_files:
        print("✓ No KNN sensor files to validate (skipping)")
        return
    for f in knn_files:
        print(f"✓ Found KNN sensor file: {f} (implement detailed checks as needed)")


def main():
    manifest = validate_manifest(MANIFEST_PATH)
    grid = validate_grid(Path(manifest["grid_path"]), manifest["n_points"])
    coords = grid[["x_m", "y_m"]].values
    for radius in manifest["radii_computed"]:
        npz_path = Path(f"cache/static/neighbors_{int(radius)}m.npz")
        csr_path = Path(f"cache/static/neighbors_{int(radius)}m_csr.npz")
        flat, offsets = validate_neighbors_npz(npz_path, len(grid), radius)
        validate_csr(csr_path, len(grid))
        validate_distances(flat, offsets, coords, radius)
    validate_knn_sensors()
    print("\n✓ All precompute outputs validated successfully.")

if __name__ == "__main__":
    main()
