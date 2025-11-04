"""
landsat8_signals.py

Utility helpers for sampling a raster to the 30 m grid. This module mirrors the
primary `patliputra.landsat8_signals` module but exposes a smaller, utility-focused
surface for quick calls. It contains the same core helpers used by the main code.
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

from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure simple logger
logger = logging.getLogger("landsat8_signals")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _ensure_cache_dir(cache_dir: Optional[Union[str, Path]]) -> Path:
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
    arr = values.astype("float64", copy=True)
    try:
        scales = dataset.scales
        offsets = dataset.offsets
        scale = scales[band_index - 1] if scales is not None else None
        offset = offsets[band_index - 1] if offsets is not None else None
    except Exception:
        scale = None
        offset = None
    if scale is None:
        tags = dataset.tags()
        scale = tags.get(f"scale") or tags.get("SCALE")
        offset = tags.get(f"offset") or tags.get("OFFSET")
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
        self._src = rasterio.open(self.dataset_path)
        if self._src.crs is None:
            raise RuntimeError(f"Source raster has no CRS: {self.dataset_path}")
        if self.clip_bounds is not None:
            left, bottom, right, top = self.clip_bounds
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
        width = max(1, int(math.ceil(width)))
        height = max(1, int(math.ceil(height)))
        if self.use_vrt:
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
            out_name = (
                self.cache_dir
                / f"{self.dataset_path.stem}_reproj_{self.target_res:.0f}m.tif"
            )
            self._cached_tif = out_name
            if self._cached_tif.exists():
                logger.debug(
                    "Reprojected cached file exists, reusing: %s", self._cached_tif
                )
                return rasterio.open(self._cached_tif)
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
            with rasterio.open(self._cached_tif, "w", **dst_meta) as dst:
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
            return rasterio.open(self._cached_tif)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._vrt is not None:
                self._vrt.close()
            if self._src is not None:
                self._src.close()
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
    if len(coords) == 0:
        return np.array([], dtype="float64")
    with _ReprojectedRasterContext(
        dataset_path, target_crs, target_res, cache_dir, use_vrt, None, resampling
    ) as ds:
        sampled = []
        for val in ds.sample(coords):
            if val is None:
                sampled.append(np.nan)
            else:
                v = val[0] if val.shape[0] >= 1 else val
                sampled.append(v)
        return np.asarray(sampled, dtype="float64")


def _safe_convert_to_float_and_mask_nodata(
    values: np.ndarray, nodata_value: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    vals = values.astype("float64", copy=True)
    if nodata_value is None:
        is_nodata = np.isnan(vals)
    else:
        is_nodata = np.isclose(vals, nodata_value, atol=0.0)
        is_nodata = np.logical_or(is_nodata, np.isnan(vals))
    vals[is_nodata] = np.nan
    return vals, is_nodata


def raster_to_grid_df(
    raster_path: Union[str, Path],
    grid_parquet_path: Union[str, Path],
    out_parquet: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    target_crs: str = "EPSG:32643",
    target_res: float = 30.0,
    sample_on: str = "centroid",
    buffer_m: float = 0.0,
    use_vrt: bool = True,
    parallel: bool = False,
    batch_size: int = 10000,
) -> pd.DataFrame:
    cache_dir = _ensure_cache_dir(cache_dir)
    raster_path = Path(raster_path)
    grid_parquet_path = Path(grid_parquet_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")
    if not grid_parquet_path.exists():
        raise FileNotFoundError(f"Grid parquet not found: {grid_parquet_path}")
    gdf = gpd.read_parquet(grid_parquet_path)
    if "grid_id" not in gdf.columns:
        raise KeyError("grid_parquet must contain a 'grid_id' column")
    if gdf.crs is None:
        logger.warning("Grid GeoParquet has no CRS; assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf_tgt = gdf.to_crs(target_crs)
    if sample_on == "centroid":
        centroids = gdf_tgt.geometry.centroid
        coords = [(pt.x, pt.y) for pt in centroids]
    else:
        raise NotImplementedError("Only 'centroid' sampling is implemented.")
    minx, miny, maxx, maxy = gdf_tgt.total_bounds
    if buffer_m and buffer_m > 0.0:
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m
    clip_bounds = (minx, miny, maxx, maxy)
    with _ReprojectedRasterContext(
        raster_path, target_crs, target_res, cache_dir, use_vrt, clip_bounds
    ) as ds:
        nodata = getattr(ds, "nodata", None)
        band_count = ds.count
        try:
            scales = getattr(ds, "scales", None)
            offsets = getattr(ds, "offsets", None)
        except Exception:
            scales = None
            offsets = None
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
        }
    logger.info(
        "Raster metadata (post-reprojection preview): %s",
        json.dumps(
            {k: raster_meta[k] for k in ("crs", "width", "height", "nodata")},
            default=str,
        ),
    )
    n_points = len(coords)
    results = np.full((n_points,), np.nan, dtype="float64")
    is_nodata_mask = np.ones((n_points,), dtype=bool)
    indices = list(range(0, n_points, batch_size))
    batches = [(i, min(i + batch_size, n_points)) for i in indices]
    if parallel and n_points > 0:
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
                    sampled = np.full((end - start,), np.nan, dtype="float64")
                with _ReprojectedRasterContext(
                    raster_path, target_crs, target_res, cache_dir, use_vrt, clip_bounds
                ) as ds:
                    vals_float, nodata_mask = _safe_convert_to_float_and_mask_nodata(
                        sampled, ds.nodata
                    )
                    vals_float = _apply_scale_offset(vals_float, ds, band_index=1)
                results[start:end] = vals_float
                is_nodata_mask[start:end] = nodata_mask
    else:
        logger.info(
            "Sampling sequentially in %d batches (batch_size=%d).",
            len(batches),
            batch_size,
        )
        with _ReprojectedRasterContext(
            raster_path, target_crs, target_res, cache_dir, use_vrt, clip_bounds
        ) as ds:
            for start, end in batches:
                coords_slice = coords[start:end]
                sampled = []
                for val in ds.sample(coords_slice):
                    if val is None:
                        sampled.append(np.nan)
                    else:
                        v = val[0] if val.shape[0] >= 1 else val
                        sampled.append(v)
                sampled = np.asarray(sampled, dtype="float64")
                vals_float = _apply_scale_offset(sampled, ds, band_index=1)
                vals_float, nodata_mask = _safe_convert_to_float_and_mask_nodata(
                    vals_float, ds.nodata
                )
                results[start:end] = vals_float
                is_nodata_mask[start:end] = nodata_mask
    df = pd.DataFrame(
        {
            "grid_id": gdf_tgt["grid_id"].values,
            "x": [c[0] for c in coords],
            "y": [c[1] for c in coords],
            "geometry": [Point(xy) for xy in coords],
            "toa_b2": results,
            "is_nodata": is_nodata_mask,
        }
    )
    df["raster_meta"] = json.dumps(raster_meta)
    df = df.set_index("grid_id", drop=False)
    if out_parquet is not None:
        out_path = Path(out_parquet)
        gdf_out = gpd.GeoDataFrame(
            df.drop(columns=["geometry"]), geometry=df["geometry"], crs=target_crs
        )
        logger.info("Writing output to parquet: %s", out_path)
        gdf_out.to_parquet(out_path, engine="pyarrow", index=False)
    return df
