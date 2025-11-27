"""Stage 0 - Part 2: Precompute spatial neighbor indices for Pollu feature engineering.

This module precomputes spatial relationships once and stores them for fast reuse:
1. Fixed-radius neighbors (30m, 90m, 270m) → NPZ format
2. Sparse CSR matrices for fast matrix-vector aggregation
3. KNN sensor indices (TODO: implement after sensor metadata available)

Moves O(N log N) spatial queries offline, making daily aggregates O(N·k) lookups.

Usage:
    python -m nalanda.precompute --mode neighbors --grid-path data/mumbai/grid_30m_nd.parquet

Output:
    cache/static/neighbors_30m.npz, neighbors_90m.npz, neighbors_270m.npz
    cache/static/neighbors_30m_csr.npz, etc. (sparse matrices)
    cache/static/neighbors_manifest.json (metadata)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, save_npz

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("INFO: tqdm not installed. Install for progress bars: pip install tqdm")


# Default paths
DEFAULT_GRID_PATH = Path("data/mumbai/grid_30m_nd.parquet")
DEFAULT_OUTPUT_DIR = Path("cache/static")
DEFAULT_RADII = [30, 90, 270]  # meters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)



# =============================================================================
# Helper Functions
# =============================================================================

# Global variable for per-process KDTree (for process pool parallelism)
_WORKER_TREE = None

def _init_worker_build_tree(coords_np: np.ndarray, leafsize: int = 16):
    """Build KDTree once inside each worker process.
    This avoids pickling the cKDTree object, which is not supported and would be inefficient for large trees.
    Instead, each process builds its own KDTree from the shared coordinates array.
    """
    global _WORKER_TREE
    _WORKER_TREE = cKDTree(coords_np, leafsize=leafsize)

def _worker_query_chunk(coords_chunk: np.ndarray, radius: float):
    """Query neighbors using global worker KDTree.
    This function is called in each worker process after the KDTree is initialized.
    """
    global _WORKER_TREE
    return _WORKER_TREE.query_ball_point(coords_chunk, r=radius)

def _build_kdtree(coords: np.ndarray, leafsize: int = 16) -> cKDTree:
    """Build KDTree from 2D coordinates with validation.
    
    Args:
        coords: Nx2 array of (x, y) coordinates in meters
        leafsize: KDTree leafsize parameter (default: 16)
    
    Returns:
        scipy.spatial.cKDTree object
    
    Raises:
        ValueError: If coordinates contain NaN or are out of expected range
    """
    if np.any(np.isnan(coords)):
        raise ValueError("Coordinates contain NaN values")
    
    if coords.shape[1] != 2:
        raise ValueError(f"Expected Nx2 coordinates, got shape {coords.shape}")
    
    n_points = len(coords)
    logger.info(f"Building KDTree for {n_points:,} points...")
    start = time.time()
    tree = cKDTree(coords, leafsize=leafsize)
    elapsed = time.time() - start
    logger.info(f"  ✓ KDTree built in {elapsed:.2f}s")
    
    return tree


def _neighbors_to_flat_offsets(
    neighbor_lists: List[List[int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert variable-length neighbor lists to flattened array + offsets.
    
    Args:
        neighbor_lists: List of lists, where neighbor_lists[i] = [j, k, ...] indices
    
    Returns:
        (flat, offsets) where:
            flat: All neighbor indices concatenated (int32)
            offsets: Start index for each point (int64)
    
    Example:
        Input: [[0, 1], [2], [3, 4, 5]]
        Output: (array([0,1,2,3,4,5]), array([0,2,3,6]))
    """
    n_points = len(neighbor_lists)
    offsets = np.zeros(n_points + 1, dtype=np.int64)
    
    # Compute offsets
    for i, neighbors in enumerate(neighbor_lists):
        offsets[i + 1] = offsets[i] + len(neighbors)
    
    # Flatten
    flat = np.empty(offsets[-1], dtype=np.int32)
    for i, neighbors in enumerate(neighbor_lists):
        start, end = offsets[i], offsets[i + 1]
        flat[start:end] = neighbors
    
    return flat, offsets


def _validate_neighbor_indices(
    flat: np.ndarray,
    offsets: np.ndarray,
    n_points: int,
    radius: float
) -> Dict:
    """Extensive validation of neighbor indices with statistics.
    
    Args:
        flat: Flattened neighbor indices
        offsets: Offset array (length n_points + 1)
        n_points: Expected number of grid points
        radius: Radius in meters (for logging)
    
    Returns:
        Dict with validation statistics
    
    Raises:
        ValueError: If validation fails
    """
    logger.info(f"Validating neighbor indices for radius {radius}m...")
    
    # Check 1: Offsets monotonically increasing
    if not np.all(np.diff(offsets) >= 0):
        raise ValueError("Offsets not monotonically increasing")
    
    # Check 2: Final offset matches flat length
    if offsets[-1] != len(flat):
        raise ValueError(
            f"Offsets[-1]={offsets[-1]} != len(flat)={len(flat)}"
        )
    
    # Check 3: All indices in valid range
    if len(flat) > 0:
        if flat.min() < 0 or flat.max() >= n_points:
            raise ValueError(
                f"Neighbor indices out of range [0, {n_points}): "
                f"min={flat.min()}, max={flat.max()}"
            )
    
    # Check 4: Offset length
    if len(offsets) != n_points + 1:
        raise ValueError(
            f"Expected {n_points + 1} offsets, got {len(offsets)}"
        )
    
    # Compute statistics
    neighbor_counts = np.diff(offsets)
    stats = {
        "n_points": n_points,
        "total_neighbors": int(len(flat)),
        "avg_neighbors": float(neighbor_counts.mean()),
        "min_neighbors": int(neighbor_counts.min()),
        "max_neighbors": int(neighbor_counts.max()),
        "median_neighbors": float(np.median(neighbor_counts)),
        "std_neighbors": float(neighbor_counts.std()),
        "points_with_zero_neighbors": int(np.sum(neighbor_counts == 0))
    }
    
    logger.info(f"  ✓ Validation passed for radius {radius}m")
    logger.info(f"    Avg neighbors: {stats['avg_neighbors']:.1f}")
    logger.info(f"    Min/max: {stats['min_neighbors']} / {stats['max_neighbors']}")
    logger.info(f"    Points with 0 neighbors: {stats['points_with_zero_neighbors']:,}")
    
    return stats


def _validate_symmetry(
    flat: np.ndarray,
    offsets: np.ndarray,
    n_points: int,
    sample_size: int = 1000
) -> bool:
    """Check neighbor symmetry: if j in neighbors(i), then i in neighbors(j).
    
    Samples random points to avoid checking all pairs (expensive).
    
    Args:
        flat: Flattened neighbor indices
        offsets: Offset array
        n_points: Number of grid points
        sample_size: Number of random points to check
    
    Returns:
        True if symmetry check passes
    
    Raises:
        ValueError: If asymmetry detected
    """
    logger.info(f"Checking neighbor symmetry (sample size: {sample_size})...")
    
    # Sample random points
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(n_points, size=min(sample_size, n_points), replace=False)
    
    asymmetry_count = 0
    for i in sample_indices:
        # Get neighbors of point i
        start, end = offsets[i], offsets[i + 1]
        neighbors_i = flat[start:end]
        
        # For each neighbor j, check if i is in neighbors(j)
        for j in neighbors_i:
            j_start, j_end = offsets[j], offsets[j + 1]
            neighbors_j = flat[j_start:j_end]
            
            if i not in neighbors_j:
                asymmetry_count += 1
                if asymmetry_count <= 5:  # Log first few
                    logger.warning(
                        f"Asymmetry: {j} in neighbors({i}), but {i} not in neighbors({j})"
                    )
    
    if asymmetry_count > 0:
        raise ValueError(
            f"Neighbor asymmetry detected: {asymmetry_count} violations in sample. "
            "This indicates a bug in neighbor computation."
        )
    
    logger.info("  ✓ Symmetry check passed")
    return True


def _build_csr_matrix(
    flat: np.ndarray,
    offsets: np.ndarray,
    n_points: int
) -> csr_matrix:
    """Build sparse CSR matrix for fast matrix-vector aggregation.
    
    Matrix M where M[i,j] = 1/|neighbors_i| if j in neighbors(i), else 0.
    Then: agg_pm25 = M @ pm25_vec (single sparse matvec operation).
    
    Args:
        flat: Flattened neighbor indices
        offsets: Offset array
        n_points: Number of grid points
    
    Returns:
        scipy.sparse.csr_matrix (n_points × n_points)
    """
    logger.info("Building sparse CSR matrix for fast aggregation...")
    
    row_indices = []
    col_indices = []
    data = []
    
    for i in range(n_points):
        start, end = offsets[i], offsets[i + 1]
        neighbors = flat[start:end]
        n_neighbors = len(neighbors)
        
        if n_neighbors > 0:
            weight = 1.0 / n_neighbors
            for j in neighbors:
                row_indices.append(i)
                col_indices.append(j)
                data.append(weight)
    
    matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_points, n_points),
        dtype=np.float32
    )
    
    nnz = matrix.nnz
    sparsity = 100 * (1 - nnz / (n_points ** 2))
    logger.info(f"  ✓ CSR matrix built: {n_points:,} × {n_points:,}, {nnz:,} nonzeros ({sparsity:.4f}% sparse)")
    
    return matrix


# =============================================================================
# Chunked Processing for Memory Efficiency
# =============================================================================

def _query_neighbors_chunk(
    tree: cKDTree,
    coords_chunk: np.ndarray,
    radius: float,
    chunk_id: int
) -> List[List[int]]:
    """Query neighbors for a chunk of coordinates (for parallel processing).
    
    Args:
        tree: Pre-built KDTree
        coords_chunk: Subset of coordinates to query
        radius: Search radius in meters
        chunk_id: Chunk identifier for logging
    
    Returns:
        List of neighbor lists for this chunk
    """
    return tree.query_ball_point(coords_chunk, r=radius)


def precompute_neighbors_chunked(
    coords: np.ndarray,
    radii: List[float],
    output_dir: Path,
    chunk_size: int = 50000,
    n_workers: int = 4,
    validate: bool = True,
    leafsize: int = 16
) -> Dict[float, Path]:
    """Precompute fixed-radius neighbors using chunked parallel processing.
    
    Splits grid into chunks to reduce memory usage and parallelizes across CPU cores.
    
    Args:
        coords: Nx2 array of (x, y) coordinates in meters
        radii: List of radii in meters (e.g., [30, 90, 270])
        output_dir: Directory to save NPZ files
        chunk_size: Points per chunk (smaller = less memory, more overhead)
        n_workers: Number of parallel workers
        validate: Whether to run extensive validation
    
    Returns:
        Dict mapping radius → output file path
    """
    n_points = len(coords)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build KDTree once in main process (for single-threaded or thread-based fallback)
    tree = _build_kdtree(coords, leafsize=leafsize)
    
    results = {}
    
    for radius in radii:
        logger.info("=" * 60)
        logger.info(f"Processing radius: {radius}m")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Split into chunks
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        logger.info(f"Splitting {n_points:,} points into {n_chunks} chunks (size {chunk_size:,})")
        
        chunks = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_points)
            chunks.append((start_idx, end_idx, coords[start_idx:end_idx]))
        
        # Parallel processing
        all_neighbors = [None] * n_points  # Pre-allocate
        
        if n_workers > 1 and n_chunks > 1:
            logger.info(f"Querying neighbors with {n_workers} workers (process pool, per-process KDTree)...")
            # Each process builds its own KDTree from coords (not pickled!)
            # This avoids the inefficiency and errors of pickling cKDTree objects.
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker_build_tree,
                initargs=(coords, leafsize)
            ) as executor:
                futures = {}
                for chunk_id, (start_idx, end_idx, coords_chunk) in enumerate(chunks):
                    future = executor.submit(_worker_query_chunk, coords_chunk, radius)
                    futures[future] = (chunk_id, start_idx, end_idx)

                iterator = as_completed(futures)
                if HAS_TQDM:
                    iterator = tqdm(iterator, total=len(futures), desc=f"Querying {radius}m")

                for future in iterator:
                    chunk_id, start_idx, end_idx = futures[future]
                    chunk_neighbors = future.result()
                    for i, neighbors in enumerate(chunk_neighbors):
                        all_neighbors[start_idx + i] = neighbors
        else:
            # Single-threaded fallback (or n_workers == 1)
            logger.info("Querying neighbors (single-threaded)...")
            iterator = range(n_chunks)
            if HAS_TQDM:
                iterator = tqdm(iterator, desc=f"Querying {radius}m")

            for chunk_id in iterator:
                start_idx, end_idx, coords_chunk = chunks[chunk_id]
                chunk_neighbors = tree.query_ball_point(coords_chunk, r=radius)
                for i, neighbors in enumerate(chunk_neighbors):
                    all_neighbors[start_idx + i] = neighbors
        
        # Convert to flat + offsets format
        logger.info("Converting to flat + offsets format...")
        flat, offsets = _neighbors_to_flat_offsets(all_neighbors)
        
        # Validation
        if validate:
            stats = _validate_neighbor_indices(flat, offsets, n_points, radius)
            _validate_symmetry(flat, offsets, n_points, sample_size=min(1000, n_points))
        else:
            neighbor_counts = np.diff(offsets)
            stats = {
                "avg_neighbors": float(neighbor_counts.mean()),
                "max_neighbors": int(neighbor_counts.max())
            }
        
        # Save NPZ
        npz_path = output_dir / f"neighbors_{int(radius)}m.npz"
        logger.info(f"Saving to {npz_path}...")
        
        np.savez_compressed(
            npz_path,
            flat=flat,
            offsets=offsets,
            n_points=n_points,
            radius_m=radius,
            avg_neighbors=stats['avg_neighbors'],
            max_neighbors=stats['max_neighbors']
        )
        
        file_size_mb = npz_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Saved ({file_size_mb:.2f} MB)")
        
        # Build and save CSR matrix
        logger.info("Building CSR matrix...")
        csr = _build_csr_matrix(flat, offsets, n_points)
        csr_path = output_dir / f"neighbors_{int(radius)}m_csr.npz"
        save_npz(csr_path, csr)
        csr_size_mb = csr_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Saved CSR matrix to {csr_path} ({csr_size_mb:.2f} MB)")
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Radius {radius}m complete in {elapsed:.1f}s")
        
        results[radius] = {
            "npz_path": str(npz_path),
            "csr_path": str(csr_path),
            "file_size_mb": file_size_mb,
            "csr_size_mb": csr_size_mb,
            "runtime_seconds": elapsed,
            "stats": stats
        }
    
    return results


# =============================================================================
# TODO: KNN Sensor Indices
# =============================================================================

def precompute_knn_sensors(
    grid_path: Path,
    sensor_metadata_path: Path,
    k: int = 5,
    output_path: Path = None
) -> Path:
    """Precompute K-nearest sensors for each grid point with inverse distance weights.
    
    TODO: Implement after sensor metadata is available from OpenAQ processing.
    
    This function will:
    1. Load sensor locations and transform to EPSG:32643
    2. Build KDTree from sensor coordinates
    3. For each grid point, find k nearest sensors
    4. Store sensor_ids and distances for IDW interpolation
    5. Save as parquet: row_idx, sensor_id_1..k, distance_1..k
    
    Args:
        grid_path: Path to grid parquet
        sensor_metadata_path: Path to sensor metadata (sensor_id, lon, lat, active_start, active_end)
        k: Number of nearest sensors
        output_path: Output parquet path (default: cache/static/knn_sensors_k{k}.parquet)
    
    Returns:
        Path to saved parquet file
    
    Raises:
        NotImplementedError: Function not yet implemented
    
    Example usage (future):
        >>> knn_path = precompute_knn_sensors(
        ...     grid_path=Path("data/mumbai/grid_30m_nd.parquet"),
        ...     sensor_metadata_path=Path("cache/sensors/sensor_metadata.parquet"),
        ...     k=5
        ... )
        >>> # Later in pm25_interpolation.py:
        >>> knn_df = pd.read_parquet(knn_path)
        >>> # Use for IDW: weights = 1 / distances
    """
    raise NotImplementedError(
        "KNN sensor precomputation not yet implemented. "
        "Requires sensor metadata from OpenAQ processing (patliputra/openaq_pm25.py). "
        "TODO: Implement after Stage 1 (data ingestion) is complete."
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main(argv=None):
    """CLI entry point for neighbor precomputation."""
    parser = argparse.ArgumentParser(
        description="Precompute spatial neighbor indices for Pollu feature engineering (Stage 0, Part 2)"
    )
    parser.add_argument(
        "--mode",
        choices=["neighbors", "knn_sensors", "all"],
        default="neighbors",
        help="What to precompute (default: neighbors)"
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        default=DEFAULT_GRID_PATH,
        help="Path to grid_30m_nd.parquet (default: data/mumbai/grid_30m_nd.parquet)"
    )
    parser.add_argument(
        "--radii",
        type=str,
        default=",".join(map(str, DEFAULT_RADII)),
        help="Comma-separated radii in meters (default: 30,90,270)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: cache/static/)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Points per chunk for parallel processing (default: 50000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--leafsize",
        type=int,
        default=16,
        help="Leafsize parameter for KDTree (default: 16)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip extensive validation (faster, less safe)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args(argv)
    
    # Parse radii
    radii = [float(x.strip()) for x in args.radii.split(",")]
    
    logger.info("=" * 60)
    logger.info("STAGE 0 - PART 2: Neighbor Precomputation")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Grid path: {args.grid_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Radii: {radii}")
    logger.info(f"Chunk size: {args.chunk_size:,} points")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Validation: {'disabled' if args.no_validate else 'enabled'}")
    logger.info("=" * 60)
    
    # Check grid file exists
    if not args.grid_path.exists():
        logger.error(f"Grid file not found: {args.grid_path}")
        logger.error("Run scripts/create_grid_mumbai.py first to generate grid.")
        return 1
    
    # Load grid
    logger.info(f"Loading grid from {args.grid_path}...")
    grid_df = pd.read_parquet(args.grid_path)
    
    required_cols = ["row_idx", "x_m", "y_m"]
    missing_cols = [col for col in required_cols if col not in grid_df.columns]
    if missing_cols:
        logger.error(f"Grid missing required columns: {missing_cols}")
        return 1
    
    n_points = len(grid_df)
    logger.info(f"  ✓ Loaded {n_points:,} grid points")
    logger.info(f"    Columns: {list(grid_df.columns)}")
    
    # Extract coordinates
    coords = grid_df[["x_m", "y_m"]].values.astype(np.float64)
    
    # Run precomputation
    start_time = time.time()

    if args.mode in ["neighbors", "all"]:
        results = precompute_neighbors_chunked(
            coords=coords,
            radii=radii,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            n_workers=args.workers,
            validate=not args.no_validate,
            leafsize=args.leafsize
        )
        
        # Write manifest
        manifest_path = args.output_dir / "neighbors_manifest.json"
        manifest = {
            "generated_ts": datetime.now(timezone.utc).isoformat(),
            "grid_path": str(args.grid_path),
            "n_points": n_points,
            "radii_computed": radii,
            "chunk_size": args.chunk_size,
            "n_workers": args.workers,
            "validation_enabled": not args.no_validate,
            "results": results,
            "total_runtime_seconds": time.time() - start_time
        }
        
        logger.info(f"Writing manifest to {manifest_path}...")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("  ✓ Manifest saved")
    
    if args.mode in ["knn_sensors", "all"]:
        logger.warning("KNN sensor precomputation not yet implemented (TODO)")
        logger.warning("Requires sensor metadata from OpenAQ processing")
        logger.warning("Skip for now or implement after data ingestion stage")
    
    total_elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"✓ Precomputation complete in {total_elapsed:.1f}s")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Compute static features (SVF, OSM, elevation)")
    logger.info("  2. Continue signal processing (S5P NO2, meteorology)")
    logger.info("  3. Implement feature engineering modules (nalanda/)")
    logger.info("")
    logger.info(f"Output files in: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
