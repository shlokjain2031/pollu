"""Utility helpers for Patliputra ingestion tasks.

This module contains small helpers used by per-signal providers (sampling
rasters at grid centroids, simple caching, and filesystem helpers). Keep
these pure and small so they are reusable across cities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def to_date_pair(start, end):
    """Normalize date-like inputs into (start, end) strings or None.

    This is intentionally permissive: callers can pass strings, datetime,
    or None. Providers should convert to their preferred internal type.
    """
    return start, end


def sample_raster_at_points(
    raster_path: Path, points: Iterable[Tuple[float, float]]
) -> np.ndarray:
    """Sample raster values at given (x, y) coordinates.

    This helper tries to import rasterio. If rasterio is not available the
    function raises a clear ImportError with instructions. The returned
    array shape is (n_points,) for single-band rasters.
    """
    try:
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling
    except Exception as e:
        raise ImportError(
            "rasterio is required for sampling rasters. Install with: pip install rasterio"
        ) from e

    coords = list(points)
    if len(coords) == 0:
        return np.array([])

    with rasterio.open(raster_path) as src:
        # Use WarpedVRT to ensure we can sample in the raster CRS if needed.
        with WarpedVRT(src) as vrt:
            vals = [v[0] for v in vrt.sample(coords)]
    return np.array(vals, dtype=float)
