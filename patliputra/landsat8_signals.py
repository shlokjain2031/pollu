"""
landsat8_signals.py

Read a Landsat-8 TOA band GeoTIFF, reproject/resample/clip it to EPSG:32643
at 30 m, sample at a 30 m grid saved as a GeoParquet, and return a pandas
DataFrame ready for GTWR.

Primary function:
    raster_to_grid_df(...)

Helpers:
    _ensure_cache_dir
    _open_reprojected_raster
    _apply_scale_offset
    _sample_points_from_vrt
    _safe_convert_to_float_and_mask_nodata

Requirements:
    rasterio, geopandas, pyproj, numpy, pandas, pyarrow
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict, Any
import json
import tempfile
import math
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pyproj

import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds
import requests
import zipfile
import io
import datetime

from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.merge import merge as rio_merge
from pyproj import Transformer

# Configure simple logger
logger = logging.getLogger("landsat8_signals")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _ensure_cache_dir(cache_dir: Optional[Union[str, Path]]) -> Path:
    """
    Ensure the cache directory exists. If None, create a temp dir and return it.

    Returns Path to cache_dir (created).
    """
    if cache_dir is None:
        tmp = Path(tempfile.mkdtemp(prefix="landsat8_signals_cache_"))
        logger.debug("No cache_dir provided; using temporary: %s", tmp)
        return tmp
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _apply_scale_offset(
    values: np.ndarray, dataset: rasterio.io.DatasetReader, band_index: int = 1
) -> np.ndarray:
    """
    Apply scale/offset if present in the dataset for the given band index (1-based).

    - dataset.scales and dataset.offsets are arrays (or tuples). If None or 1.0/0.0,
      this returns the input values unchanged.
    - This operates elementwise and returns float64 dtype.
    """
    # Ensure float dtype so NaN can be used
    arr = values.astype("float64", copy=True)

    try:
        scales = dataset.scales
        offsets = dataset.offsets
        # rasterio uses 1-based band indexing for metadata arrays
        scale = scales[band_index - 1] if scales is not None else None
        offset = offsets[band_index - 1] if offsets is not None else None
    except Exception:
        scale = None
        offset = None

    if scale is None:
        # try tags
        tags = dataset.tags()
        scale = tags.get(f"scale") or tags.get("SCALE")
        offset = tags.get(f"offset") or tags.get("OFFSET")
        # convert to float if present
        try:
            scale = float(scale) if scale is not None else None
        except Exception:
            scale = None
        try:
            offset = float(offset) if offset is not None else None
        except Exception:
            offset = None

    if scale is not None and not math.isclose(scale, 1.0):
        arr = arr * float(scale)
    if offset is not None and not math.isclose(offset, 0.0):
        arr = arr + float(offset)
    return arr


class _ReprojectedRasterContext:
    """
    Context manager that yields either a WarpedVRT (in-memory reprojection)
    or a rasterio dataset (for cached pre-warped GeoTIFF). Designed to be
    used with "with _open_reprojected_raster(...) as ds:" and ds will be a
    rasterio dataset-like object supporting .read(), .sample(), .nodata, .count, etc.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        target_crs: str,
        target_res: float,
        cache_dir: Path,
        use_vrt: bool = True,
        clip_bounds: Optional[Tuple[float, float, float, float]] = None,
        resampling: Resampling = Resampling.bilinear,
    ):
        self.dataset_path = Path(dataset_path)
        self.target_crs = target_crs
        self.target_res = float(target_res)
        self.cache_dir = cache_dir
        self.use_vrt = use_vrt
        self.clip_bounds = clip_bounds
        self.resampling = resampling
        self._src = None
        self._vrt = None
        self._cached_tif = None

    def __enter__(self):
        # Open source
        self._src = rasterio.open(self.dataset_path)

        if self._src.crs is None:
            # Can't proceed without a CRS
            raise RuntimeError(f"Source raster has no CRS: {self.dataset_path}")

        # Compute transform/width/height for target CRS/resolution and clip bounds
        if self.clip_bounds is not None:
            # clip_bounds are provided in the target CRS; transform them back
            # to the source CRS before calling calculate_default_transform which
            # expects bounds in the source CRS.
            left_t, bottom_t, right_t, top_t = self.clip_bounds
            try:
                from rasterio.warp import transform_bounds

                left, bottom, right, top = transform_bounds(
                    self.target_crs,
                    self._src.crs,
                    left_t,
                    bottom_t,
                    right_t,
                    top_t,
                    densify_pts=21,
                )
            except Exception:
                # Fall back to using the raw target bounds (best-effort) if
                # transform_bounds fails for any reason.
                left, bottom, right, top = left_t, bottom_t, right_t, top_t

            # calculate_default_transform signature: src_crs, dst_crs, src_width, src_height,
            # left, bottom, right, top (in source CRS), resolution=...
            transform, width, height = calculate_default_transform(
                self._src.crs,
                self.target_crs,
                self._src.width,
                self._src.height,
                left,
                bottom,
                right,
                top,
                resolution=self.target_res,
            )
        else:
            transform, width, height = calculate_default_transform(
                self._src.crs,
                self.target_crs,
                self._src.width,
                self._src.height,
                *self._src.bounds,
                resolution=self.target_res,
            )

        # Defensive: ensure width/height are integers > 0
        width = max(1, int(math.ceil(width)))
        height = max(1, int(math.ceil(height)))

        if self.use_vrt:
            # Create WarpedVRT that performs reprojection/resampling on-the-fly
            self._vrt = WarpedVRT(
                self._src,
                crs=self.target_crs,
                transform=transform,
                width=width,
                height=height,
                resampling=self.resampling,
                add_alpha=False,
            )
            return self._vrt
        else:
            # Create a cached GeoTIFF under cache_dir and return that dataset.
            out_name = (
                self.cache_dir
                / f"{self.dataset_path.stem}_reproj_{self.target_res:.0f}m.tif"
            )
            self._cached_tif = out_name
            # If file exists, open and reuse
            if self._cached_tif.exists():
                logger.debug(
                    "Reprojected cached file exists, reusing: %s", self._cached_tif
                )
                return rasterio.open(self._cached_tif)

            # Build metadata for output
            dst_meta = self._src.meta.copy()
            dst_meta.update(
                {
                    "crs": self.target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "driver": "GTiff",
                    "dtype": dst_meta.get("dtype", "float32"),
                    "compress": "deflate",
                }
            )

            # Reproject to file
            with rasterio.open(self._cached_tif, "w", **dst_meta) as dst:
                # Reproject each band
                for i in range(1, self._src.count + 1):
                    reproject(
                        source=rasterio.band(self._src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=self._src.transform,
                        src_crs=self._src.crs,
                        dst_transform=transform,
                        dst_crs=self.target_crs,
                        resampling=self.resampling,
                    )
            # Open and return cached file
            return rasterio.open(self._cached_tif)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close VRT/datasets as needed
        try:
            if self._vrt is not None:
                self._vrt.close()
            if self._src is not None:
                self._src.close()
            # Note: if opening cached tif we returned a new dataset; the caller should allow it to close or use the dataset returned.
        except Exception:
            pass


def _sample_points_from_vrt(
    dataset_path: Union[str, Path],
    coords: List[Tuple[float, float]],
    target_crs: str,
    target_res: float,
    cache_dir: Path,
    use_vrt: bool = True,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Sample a list of (x,y) coordinates (in target_crs) from raster_path,
    returning a 1-D numpy array of sampled values (first band) in the same order.

    This helper opens its own WarpedVRT/context so it can be used inside
    worker processes.
    """
    if len(coords) == 0:
        return np.array([], dtype="float64")

    # The dataset will be opened and a WarpedVRT or cached file returned.
    with _ReprojectedRasterContext(
        dataset_path, target_crs, target_res, cache_dir, use_vrt, None, resampling
    ) as ds:
        # ds is a dataset-like object in target_crs
        # ds.sample expects coords in the dataset CRS (target_crs)
        sampled = []
        # Read generator from ds.sample - yields arrays with shape (bands,)
        for val in ds.sample(coords):
            if val is None:
                # represent as NaN-filled row matching band count
                sampled.append(np.full((ds.count,), np.nan))
            else:
                # val is a 1-D array of length ds.count
                sampled.append(np.asarray(val, dtype="float64"))
        return np.asarray(sampled, dtype="float64")


def _safe_convert_to_float_and_mask_nodata(
    values: np.ndarray, nodata_value: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sampled values to float array and mask nodata:
    Returns (values_float, is_nodata_bool_array)
    """
    vals = values.astype("float64", copy=True)
    # If nodata_value is None, use NaN detection
    if nodata_value is None:
        is_nodata = np.isnan(vals)
    else:
        # nodata may be a sentinel like -9999; treat that as nodata
        is_nodata = np.isclose(vals, nodata_value, atol=0.0)
        # also treat NaN as nodata
        is_nodata = np.logical_or(is_nodata, np.isnan(vals))
    # Convert nodata to np.nan for consistent downstream behavior
    vals[is_nodata] = np.nan
    return vals, is_nodata


def raster_to_grid_df(
    raster_path: Union[str, Path],
    grid_parquet_path: Union[str, Path],
    out_parquet: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    target_crs: str = "EPSG:32643",
    target_res: float = 30.0,
    sample_on: str = "centroid",  # or 'inner' (not implemented for polygon mean; centroid recommended)
    buffer_m: float = 0.0,
    use_vrt: bool = True,
    parallel: bool = True,
    batch_size: int = 10000,
    bands: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Main entrypoint: reproject/resample/clip raster to `target_crs` at `target_res` meters,
    sample at grid centroids loaded from `grid_parquet_path`, and return a DataFrame
    with columns [grid_id, x, y, geometry, toa_b2, is_nodata, raster_meta].

    Parameters
    ----------
    raster_path : str | Path
        Input raster (Landsat TOA GeoTIFF). Function will not overwrite this file.
    grid_parquet_path : str | Path
        GeoParquet file containing the 30 m grid. Must include columns:
        - grid_id (unique identifier)
        - geometry (polygons in some CRS)
    out_parquet : Optional[str | Path]
        If provided, write the resulting table to this path (parquet).
    cache_dir : Optional[str | Path]
        Directory to store temporary cached reprojected rasters. If None, a temp dir is used.
    target_crs : str
        Target CRS (default EPSG:32643).
    target_res : float
        Target resolution in metres per pixel (default 30.0).
    sample_on : str
        'centroid' recommended. ('inner' would sample polygon interior mean - not implemented).
    buffer_m : float
        Buffer in metres to expand grid bbox before clipping raster. Useful to avoid edge artifacts.
    use_vrt : bool
        If True, create an on-the-fly WarpedVRT. If False, write a cached GeoTIFF under cache_dir.
    parallel : bool
        If True, sample in parallel across processes (useful for very large grids). Default False.
    batch_size : int
        Number of points per batch when sampling (controls memory and parallel chunk size).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by grid_id, columns: grid_id, x, y, geometry, toa_b2, is_nodata, raster_meta
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    raster_path = Path(raster_path)
    grid_parquet_path = Path(grid_parquet_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")
    if not grid_parquet_path.exists():
        raise FileNotFoundError(f"Grid parquet not found: {grid_parquet_path}")

    # Load grid parquet
    gdf = gpd.read_parquet(grid_parquet_path)  # relies on pyarrow or fastparquet
    if "grid_id" not in gdf.columns:
        raise KeyError("grid_parquet must contain a 'grid_id' column")
    # Ensure geometry exists
    if gdf.geometry.is_empty.any():
        logger.warning(
            "Some grid geometries are empty; those rows will yield NaN samples"
        )

    # Reproject grid to target_crs
    gdf = gdf.set_geometry("geometry")
    # If crs undefined, assume EPSG:4326 (best effort) — but warn user
    if gdf.crs is None:
        logger.warning("Grid GeoParquet has no CRS; assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf_tgt = gdf.to_crs(target_crs)

    # Compute sampling points
    if sample_on == "centroid":
        # Use centroid — explicit choice to sample cell center for GTWR grid.
        centroids = gdf_tgt.geometry.centroid
        coords = [(pt.x, pt.y) for pt in centroids]
    else:
        raise NotImplementedError(
            "Only 'centroid' sampling is implemented. 'inner' (mean) not implemented."
        )

    # Compute clipping bounds (in target_crs) from grid bounds and buffer
    minx, miny, maxx, maxy = gdf_tgt.total_bounds
    if buffer_m and buffer_m > 0.0:
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m
    clip_bounds = (minx, miny, maxx, maxy)

    # Determine nodata and scale/offset by probing reprojected dataset (VRT or cached)
    # We'll open a small context to inspect metadata
    with _ReprojectedRasterContext(
        raster_path, target_crs, target_res, cache_dir, use_vrt, clip_bounds
    ) as ds:
        nodata = getattr(ds, "nodata", None)
        band_count = ds.count
        # get scales/offsets from underlying dataset if available
        try:
            # If ds is WarpedVRT it may not expose scales/offsets, so try underlying dataset tags
            scales = getattr(ds, "scales", None)
            offsets = getattr(ds, "offsets", None)
        except Exception:
            scales = None
            offsets = None

        # Determine band names: prefer user-provided `bands`, otherwise infer or default to B1..Bn
        if bands is None:
            try:
                desc = [d for d in getattr(ds, "descriptions", [])]
                if desc and any(desc):
                    band_names = [
                        d if d else f"B{i+1}" for i, d in enumerate(desc[:band_count])
                    ]
                else:
                    band_names = [f"B{i+1}" for i in range(band_count)]
            except Exception:
                band_names = [f"B{i+1}" for i in range(band_count)]
        else:
            band_names = list(bands)

        raster_meta = {
            "crs": str(ds.crs),
            "width": int(ds.width),
            "height": int(ds.height),
            "bounds": tuple(ds.bounds),
            "transform": tuple(ds.transform) if hasattr(ds, "transform") else None,
            "count": int(band_count),
            "nodata": nodata,
            "scales": tuple(scales) if scales is not None else None,
            "offsets": tuple(offsets) if offsets is not None else None,
            "band_names": band_names,
        }

    # If a sidecar meta JSON exists next to the raster (created by the mosaic step), read it
    try:
        meta_sidecar = None
        sidecar_path = Path(raster_path).with_suffix(".meta.json")
        if sidecar_path.exists():
            with open(sidecar_path, "r") as sf:
                meta_sidecar = json.load(sf)
        if meta_sidecar:
            # merge into raster_meta under 'angles'
            raster_meta.setdefault("angles", {}).update(meta_sidecar)
    except Exception:
        logger.debug("Failed to read raster sidecar metadata", exc_info=True)

    logger.info(
        "Raster metadata (post-reprojection preview): %s",
        json.dumps(
            {k: raster_meta[k] for k in ("crs", "width", "height", "nodata")},
            default=str,
        ),
    )

    # Sampling: in batches, optionally parallelized
    n_points = len(coords)
    n_bands = len(band_names)
    # results: shape (n_points, n_bands)
    results = np.full((n_points, n_bands), np.nan, dtype="float64")
    is_nodata_mask = np.ones((n_points, n_bands), dtype=bool)

    indices = list(range(0, n_points, batch_size))
    batches = [(i, min(i + batch_size, n_points)) for i in indices]

    if parallel and n_points > 0:
        # Use ProcessPoolExecutor to parallelize per-batch sampling.
        # Each worker will open its own dataset and WarpedVRT (costly but safe).
        logger.info(
            "Sampling in parallel with %d batches (batch_size=%d).",
            len(batches),
            batch_size,
        )
        with ProcessPoolExecutor() as ex:
            futures = {}
            for start, end in batches:
                coords_slice = coords[start:end]
                futures[
                    ex.submit(
                        _sample_points_from_vrt,
                        raster_path,
                        coords_slice,
                        target_crs,
                        target_res,
                        cache_dir,
                        use_vrt,
                        Resampling.bilinear,
                    )
                ] = (start, end)
            for fut in as_completed(futures):
                start, end = futures[fut]
                try:
                    sampled = fut.result()
                except Exception as e:
                    logger.error(
                        "Sampling batch %d:%d failed: %s", start, end, e, exc_info=True
                    )
                    sampled = np.full((end - start, n_bands), np.nan, dtype="float64")

                # If worker returned 1D (single-band), expand to 2D
                if isinstance(sampled, np.ndarray) and sampled.ndim == 1:
                    sampled = sampled.reshape(-1, 1)
                with _ReprojectedRasterContext(
                    raster_path, target_crs, target_res, cache_dir, use_vrt, clip_bounds
                ) as ds:
                    batch_n = sampled.shape[0]
                    # pad if fewer bands returned
                    if sampled.shape[1] < n_bands:
                        pad = np.full((batch_n, n_bands - sampled.shape[1]), np.nan)
                        sampled = np.hstack([sampled, pad])
                    for b in range(n_bands):
                        col = sampled[:, b].astype("float64", copy=True)
                        try:
                            col = _apply_scale_offset(col, ds, band_index=b + 1)
                        except Exception:
                            pass
                        if ds.nodata is None:
                            nodata_mask = np.isnan(col)
                        else:
                            nodata_mask = np.logical_or(
                                np.isnan(col), np.isclose(col, ds.nodata)
                            )
                        results[start:end, b] = col
                        is_nodata_mask[start:end, b] = nodata_mask
    else:
        logger.info(
            "Sampling sequentially in %d batches (batch_size=%d).",
            len(batches),
            batch_size,
        )
        # Open single context and iterate batches
        with _ReprojectedRasterContext(
            raster_path, target_crs, target_res, cache_dir, use_vrt, clip_bounds
        ) as ds:
            for start, end in batches:
                coords_slice = coords[start:end]
                sampled_rows = []
                for val in ds.sample(coords_slice):
                    if val is None:
                        sampled_rows.append([np.nan] * n_bands)
                    else:
                        arr = np.asarray(val, dtype="float64")
                        if arr.shape[0] < n_bands:
                            pad = np.full((n_bands - arr.shape[0],), np.nan)
                            arr = np.concatenate([arr, pad])
                        sampled_rows.append(arr[:n_bands])
                sampled = np.asarray(sampled_rows, dtype="float64")
                batch_n = sampled.shape[0]
                # Apply scale/offset and nodata masking per-band
                for b in range(n_bands):
                    col = sampled[:, b].astype("float64", copy=True)
                    try:
                        col = _apply_scale_offset(col, ds, band_index=b + 1)
                    except Exception:
                        pass
                    if ds.nodata is None:
                        nodata_mask = np.isnan(col)
                    else:
                        nodata_mask = np.logical_or(
                            np.isnan(col), np.isclose(col, ds.nodata)
                        )
                    results[start:end, b] = col
                    is_nodata_mask[start:end, b] = nodata_mask

    # Build DataFrame with one column per band
    data = {
        "grid_id": gdf_tgt["grid_id"].values,
        "x": [c[0] for c in coords],
        "y": [c[1] for c in coords],
        "geometry": [Point(xy) for xy in coords],
    }
    for bi, bname in enumerate(band_names):
        col_name = f"toa_{bname.lower()}"
        data[col_name] = results[:, bi]
    # mark a cell as nodata if any band is nodata (conservative)
    data["is_nodata"] = np.any(is_nodata_mask, axis=1)

    df = pd.DataFrame(data)

    # Attach raster_meta as JSON string (lightweight)
    df["raster_meta"] = json.dumps(raster_meta)

    # If raster_meta contains angles, add them as per-cell columns (repeated)
    try:
        angles = raster_meta.get("angles", {}) if isinstance(raster_meta, dict) else {}
        solar_az = angles.get("solar_azimuth")
        solar_zen = angles.get("solar_zenith")
        sensor_az = angles.get("sensor_azimuth")
        sensor_zen = angles.get("sensor_zenith")
        df["solar_azimuth"] = float(solar_az) if solar_az is not None else np.nan
        df["solar_zenith"] = float(solar_zen) if solar_zen is not None else np.nan
        df["sensor_azimuth"] = float(sensor_az) if sensor_az is not None else np.nan
        df["sensor_zenith"] = float(sensor_zen) if sensor_zen is not None else np.nan
    except Exception:
        # don't fail if angles missing
        df["solar_azimuth"] = np.nan
        df["solar_zenith"] = np.nan
        df["sensor_azimuth"] = np.nan
        df["sensor_zenith"] = np.nan

    # Compute canonical indices per Xu (2008) when bands available:
    # NDVI = (B5 - B4) / (B5 + B4)
    # MNDWI = (B3 - B6) / (B3 + B6)
    # NDBI = (B6 - B5) / (B6 + B5)
    # IBI = (NDBI - (NDVI + MNDWI)/2) / (NDBI + (NDVI + MNDWI)/2)
    try:
        # NDVI
        if "toa_b5" in df.columns and "toa_b4" in df.columns:
            b5 = df["toa_b5"].to_numpy(dtype="float64")
            b4 = df["toa_b4"].to_numpy(dtype="float64")
            denom = b5 + b4
            ndvi = np.where(
                np.isfinite(denom) & (np.abs(denom) > 0), (b5 - b4) / denom, np.nan
            )
            df["ndvi"] = ndvi

        # MNDWI (requires green B3 and SWIR1 B6)
        if "toa_b3" in df.columns and "toa_b6" in df.columns:
            b3 = df["toa_b3"].to_numpy(dtype="float64")
            b6 = df["toa_b6"].to_numpy(dtype="float64")
            denom = b3 + b6
            mndwi = np.where(
                np.isfinite(denom) & (np.abs(denom) > 0), (b3 - b6) / denom, np.nan
            )
            df["mndwi"] = mndwi

        # NDBI using SWIR1 (B6) and NIR (B5)
        if "toa_b6" in df.columns and "toa_b5" in df.columns:
            b6 = df["toa_b6"].to_numpy(dtype="float64")
            b5 = df["toa_b5"].to_numpy(dtype="float64")
            denom = b6 + b5
            ndbi = np.where(
                np.isfinite(denom) & (np.abs(denom) > 0), (b6 - b5) / denom, np.nan
            )
            df["ndbi"] = ndbi

        # Canonical IBI per Xu (2008) if NDVI, MNDWI and NDBI are present
        if "ndbi" in df.columns and "ndvi" in df.columns and "mndwi" in df.columns:
            a = df["ndbi"].to_numpy(dtype="float64")
            b = df["ndvi"].to_numpy(dtype="float64")
            c = df["mndwi"].to_numpy(dtype="float64")
            combo = (b + c) / 2.0
            numer = a - combo
            denom = a + combo
            ibi = np.where(
                np.isfinite(denom) & (np.abs(denom) > 0), numer / denom, np.nan
            )
            df["ibi"] = ibi
    except Exception:
        logger.debug("Index computation failed", exc_info=True)

    # Set index to grid_id for GTWR compatibility
    df = df.set_index("grid_id", drop=False)

    # Optionally write parquet
    if out_parquet is not None:
        out_path = Path(out_parquet)
        # Convert geometry to WKB or use geopandas to write GeoParquet
        gdf_out = gpd.GeoDataFrame(
            df.drop(columns=["geometry"]), geometry=df["geometry"], crs=target_crs
        )
        # Use pyarrow-based parquet writer via geopandas
        logger.info("Writing output to parquet: %s", out_path)
        gdf_out.to_parquet(out_path, engine="pyarrow", index=False)

    return df
