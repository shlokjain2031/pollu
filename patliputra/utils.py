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


def validate_raster_file(
    path: Path,
    expected_crs: str | None = "EPSG:4326",
    expected_band: int = 1,
    min_nonzero_ratio: float = 1e-6,
    expected_resolution_m: float | None = None,
    bbox: Tuple[float, float, float, float] | None = None,
) -> dict:
    """Validate a raster file produced by preprocess.

    Checks performed:
    - file exists and is readable
    - CRS matches (if expected_crs provided)
    - contains at least `expected_band` bands
    - band statistics (non-NaN count, non-zero count, min/max/mean)
    - pixel size reported and (optionally) compared to expected_resolution_m

    Returns a diagnostics dict with keys:
    - ok: bool
    - notes: list[str]
    - stats: {width,height,crs,band_count,pixel_size,band_stats}
    """
    try:
        import rasterio
        from rasterio import open as rio_open
    except Exception as e:
        return {"ok": False, "notes": ["rasterio not available: %s" % e]}

    p = Path(path)
    notes: list[str] = []
    if not p.exists():
        return {"ok": False, "notes": [f"file not found: {p}"]}

    try:
        with rio_open(p) as src:
            crs = str(src.crs) if src.crs is not None else None
            width = src.width
            height = src.height
            count = src.count
            transform = src.transform

            # pixel sizes (in CRS units)
            px = abs(transform.a)
            py = abs(transform.e)

            # band availability
            if count < expected_band:
                notes.append(f"band_count={count} < expected_band={expected_band}")

            # read the requested band safely (1-based)
            try:
                arr = src.read(expected_band).astype(float)
            except Exception as e:
                return {
                    "ok": False,
                    "notes": [f"failed reading band {expected_band}: {e}"],
                }

            total = arr.size
            n_nans = int(np.isnan(arr).sum())
            non_nan = total - n_nans
            non_zero = int((arr != 0).sum())
            band_min = float(np.nanmin(arr)) if non_nan > 0 else float("nan")
            band_max = float(np.nanmax(arr)) if non_nan > 0 else float("nan")
            band_mean = float(np.nanmean(arr)) if non_nan > 0 else float("nan")

            stats = {
                "crs": crs,
                "width": width,
                "height": height,
                "band_count": count,
                "pixel_size": (px, py),
                "band": expected_band,
                "total_pixels": total,
                "non_nan": non_nan,
                "non_zero": non_zero,
                "min": band_min,
                "max": band_max,
                "mean": band_mean,
            }

            # CRS check
            if (
                expected_crs is not None
                and crs is not None
                and str(expected_crs) not in str(crs)
            ):
                notes.append(f"CRS mismatch: expected {expected_crs}, got {crs}")

            # resolution check (best-effort): if expected_resolution_m provided,
            # compare using meters. If CRS is geographic (degrees), approximate
            # meters per degree at raster centre.
            if expected_resolution_m is not None:
                try:
                    from pyproj import Geod, Transformer

                    if crs is None:
                        notes.append("cannot validate resolution: raster has no CRS")
                    else:
                        crs_upper = str(crs).upper()
                        if (
                            "4326" in crs_upper
                            or "CRS:84" in crs_upper
                            or "GEOGCS" in crs_upper
                        ):
                            # approx meters per degree at centre
                            left, bottom, right, top = src.bounds
                            center_x = (left + right) / 2.0
                            center_y = (bottom + top) / 2.0
                            geod = Geod(ellps="WGS84")
                            _, _, meters_per_deg_lon = geod.inv(
                                center_x, center_y, center_x + 1.0, center_y
                            )
                            meters_per_deg_lon = (
                                abs(meters_per_deg_lon)
                                if meters_per_deg_lon != 0
                                else 111320.0
                            )
                            approx_m_per_pixel = px * meters_per_deg_lon
                        else:
                            # assume projected in meters
                            approx_m_per_pixel = px

                        diff = abs(approx_m_per_pixel - float(expected_resolution_m))
                        rel = (
                            diff / float(expected_resolution_m)
                            if expected_resolution_m > 0
                            else 0.0
                        )
                        stats["approx_m_per_pixel"] = approx_m_per_pixel
                        if rel > 0.25:
                            notes.append(
                                f"pixel size mismatch: approx {approx_m_per_pixel:.2f}m vs expected {expected_resolution_m}m (rel {rel:.2%})"
                            )
                except Exception as _e:
                    notes.append(f"resolution check failed: {_e}")

            # non-zero ratio check
            nonzero_ratio = non_zero / float(non_nan) if non_nan > 0 else 0.0
            stats["nonzero_ratio"] = nonzero_ratio
            if non_nan == 0:
                notes.append("all pixels are NaN")
            elif nonzero_ratio < min_nonzero_ratio:
                notes.append(
                    f"non-zero pixel ratio very low: {nonzero_ratio:.6f} (< {min_nonzero_ratio})"
                )

            ok = len(notes) == 0
            return {"ok": ok, "notes": notes, "stats": stats}

    except Exception as e:
        return {"ok": False, "notes": [f"failed to open/read raster: {e}"]}
