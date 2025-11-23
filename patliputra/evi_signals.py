"""Sentinel-2 EVI sampler that mirrors the Landsat and S5P pipelines.

For every Landsat processing date, fetch COPERNICUS/S2_SR_HARMONIZED data,
apply the QA60 cloud mask, compute the Enhanced Vegetation Index (EVI),
project to EPSG:32643 at 30 m, and update the existing
cache/landsat_YYYY-MM-DD.parquet files with new columns.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from patliputra.utils.earthengine import EarthEngineClient
from patliputra.utils.raster_sampling import raster_to_grid_df


COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
TARGET_CRS = "EPSG:32643"
TARGET_RES = 30.0
DEFAULT_GRID = Path("data/mumbai/grid_30m.parquet")
DEFAULT_DATES = Path("patliputra/landsat8_image_dates.txt")
DEFAULT_S2_MONTHLY_DATES = Path("patliputra/sentinel2_low_cloud_dates.txt")
DEFAULT_CACHE_ROOT = Path("cache")
DEFAULT_BBOX = (72.7763, 18.8939, 72.9797, 19.2701)
DEFAULT_ESTIMATE_WINDOW = 7
TIF_TEMPLATE = "sentinel2_{date}.tif"
LANDSAT_TEMPLATE = "landsat_processed/landsat8_{date}_signals.parquet"
EVI_VALUE_COLUMN = "sentinel2_evi"
EVI_FLAG_COLUMN = "sentinel2_evi_is_nodata"
EVI_META_COLUMN = "sentinel2_evi_meta"

logger = logging.getLogger("evi_signals")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class SampledSeries:
    values: pd.Series
    nodata: pd.Series


def _init_client(project_id: Optional[str]) -> EarthEngineClient:
    logger.debug("Initializing Earth Engine client with project=%s", project_id)
    client = (
        EarthEngineClient(project_id=project_id) if project_id else EarthEngineClient()
    )
    client.initialize()
    return client


def read_dates(path: Path = DEFAULT_DATES) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dates file not found: {path}")
    with open(path, "r") as fh:
        dates = [line.strip() for line in fh.readlines() if line.strip()]
    logger.info("Loaded %d Landsat dates from %s", len(dates), path)
    return dates


def read_monthly_s2_dates(
    path: Path = DEFAULT_S2_MONTHLY_DATES,
) -> Dict[Tuple[int, int], dt.date]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sentinel-2 monthly dates file not found: {path}")
    monthly: Dict[Tuple[int, int], dt.date] = {}
    with open(path, "r") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            date_val = dt.date.fromisoformat(raw)
            key = (date_val.year, date_val.month)
            if key in monthly:
                logger.warning(
                    "Duplicate Sentinel-2 monthly entry for %s-%s; replacing %s with %s",
                    key[0],
                    key[1],
                    monthly[key],
                    date_val,
                )
            monthly[key] = date_val
    logger.info("Loaded %d monthly Sentinel-2 dates from %s", len(monthly), path)
    return monthly


def _mask_s2_clouds(image):
    ee = __import__("ee")
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)


def _evi_image_for_date(
    client: EarthEngineClient, date_str: str, bbox: Tuple[float, float, float, float]
):
    ee = __import__("ee")
    start = ee.Date(date_str)
    end = start.advance(1, "day")
    geom = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])
    logger.debug("Querying Sentinel-2 collection for %s within %s", date_str, bbox)
    collection = (
        ee.ImageCollection(COLLECTION_ID)
        .filterDate(start, end)
        .filterBounds(geom)
        .map(_mask_s2_clouds)
    )
    count = int(collection.size().getInfo())
    if count == 0:
        logger.warning("No Sentinel-2 imagery available on %s", date_str)
        return None
    logger.debug("Found %d Sentinel-2 images for %s", count, date_str)
    image = collection.median()
    evi = (
        image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": image.select("B8"),
                "RED": image.select("B4"),
                "BLUE": image.select("B2"),
            },
        )
        .rename("EVI")
        .clip(geom)
    )
    return evi


def _neighbor_dates(
    client: EarthEngineClient, target_date: str, max_days: int
) -> Tuple[Optional[str], Optional[str]]:
    ee = __import__("ee")
    target = dt.date.fromisoformat(target_date)
    prev_date: Optional[str] = None
    next_date: Optional[str] = None

    def _has_image(start: dt.date, end: dt.date) -> bool:
        collection = ee.ImageCollection(COLLECTION_ID).filterDate(
            start.isoformat(), end.isoformat()
        )
        try:
            return int(collection.size().getInfo()) > 0
        except Exception:
            return False

    logger.debug("Searching neighbors for %s within %d days", target_date, max_days)
    for delta in range(1, max_days + 1):
        start = target - dt.timedelta(days=delta)
        end = target - dt.timedelta(days=delta - 1)
        if _has_image(start, end):
            prev_date = start.isoformat()
            break

    for delta in range(1, max_days + 1):
        start = target + dt.timedelta(days=delta)
        end = target + dt.timedelta(days=delta + 1)
        if _has_image(start, end):
            next_date = start.isoformat()
            break

    logger.info(
        "Neighbor search for %s => prev: %s next: %s",
        target_date,
        prev_date,
        next_date,
    )
    return prev_date, next_date


def _cache_layout(cache_root: Path, date_str: str) -> Tuple[Path, Path]:
    base = Path(cache_root) / f"sentinel2_{date_str}"
    base.mkdir(parents=True, exist_ok=True)
    tif_path = base / TIF_TEMPLATE.format(date=date_str)
    logger.debug("Cache layout for %s => %s", date_str, tif_path)
    return base, tif_path


def _landsat_parquet_path(cache_root: Path, date_str: str) -> Path:
    path = Path(cache_root) / LANDSAT_TEMPLATE.format(date=date_str)
    logger.debug("Landsat parquet path for %s => %s", date_str, path)
    return path


def _has_evi_columns(parquet_path: Path) -> bool:
    if not parquet_path.exists():
        return False
    schema = pq.ParquetFile(parquet_path).schema
    has_cols = EVI_VALUE_COLUMN in schema.names
    logger.debug("%s already has EVI columns: %s", parquet_path, has_cols)
    return has_cols


def _download_image(
    client: EarthEngineClient,
    image,
    out_tif: Path,
    bbox: Tuple[float, float, float, float],
) -> Path:
    params = {"scale": TARGET_RES, "crs": TARGET_CRS}
    if bbox is not None:
        params["region"] = [float(v) for v in bbox]
    url = client.get_download_url(image, params=params)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_tif.with_suffix(out_tif.suffix + ".download")
    logger.info("Downloading Sentinel-2 EVI image to %s", tmp_path)
    with requests.get(url, stream=True, timeout=240) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    logger.debug(
        "Finished download for %s (%d bytes)", tmp_path, tmp_path.stat().st_size
    )
    with open(tmp_path, "rb") as fh:
        magic = fh.read(4)
    if magic.startswith(b"PK"):
        with ZipFile(tmp_path) as zf:
            members = [
                info
                for info in zf.infolist()
                if not info.is_dir()
                and info.filename.lower().endswith((".tif", ".tiff"))
            ]
            if not members:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError("Zip download did not contain a GeoTIFF")
            with zf.open(members[0]) as src, open(out_tif, "wb") as dst:
                shutil.copyfileobj(src, dst)
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.replace(out_tif)
    logger.info("Stored EVI GeoTIFF at %s", out_tif)
    return out_tif


def _series_from_df(df: pd.DataFrame, date_str: str) -> SampledSeries:
    if "toa_b2" not in df.columns:
        raise KeyError("Expected 'toa_b2' column from raster sampling")
    values = df["toa_b2"].astype("float64").copy()
    values.index = df["grid_id"].to_numpy()
    values.name = date_str
    nodata = df["is_nodata"].astype("bool").copy()
    nodata.index = values.index
    nodata.name = f"{date_str}_nodata"
    logger.debug("Sampled %d pixels for %s", len(values), date_str)
    return SampledSeries(values=values, nodata=nodata)


def _empty_sample(grid_parquet: Path, date_str: str) -> SampledSeries:
    gdf = gpd.read_parquet(grid_parquet)
    gdf = gdf.set_index("grid_id", drop=False)
    values = pd.Series(np.nan, index=gdf.index, name=date_str, dtype="float64")
    nodata = pd.Series(True, index=gdf.index, name=f"{date_str}_nodata", dtype="bool")
    logger.warning("Creating empty sample for %s (no data)", date_str)
    return SampledSeries(values=values, nodata=nodata)


def _update_landsat_parquet(
    landsat_parquet: Path, sample: SampledSeries, meta: Optional[dict] = None
) -> Path:
    if not landsat_parquet.exists():
        raise FileNotFoundError(f"Landsat parquet not found: {landsat_parquet}")
    gdf = gpd.read_parquet(landsat_parquet)
    if "grid_id" not in gdf.columns:
        raise KeyError("landsat parquet missing 'grid_id'")
    gdf = gdf.set_index("grid_id", drop=False)
    aligned_values = sample.values.reindex(gdf.index)
    aligned_nodata = sample.nodata.reindex(gdf.index).fillna(True)
    gdf[EVI_VALUE_COLUMN] = aligned_values.to_numpy(dtype="float64")
    gdf[EVI_FLAG_COLUMN] = aligned_nodata.astype("bool").to_numpy(dtype="bool")
    gdf[EVI_META_COLUMN] = json.dumps(meta or {"source": COLLECTION_ID})
    gdf.reset_index(drop=True, inplace=True)
    gdf.to_parquet(landsat_parquet, engine="pyarrow", index=False)
    method = meta.get("method") if meta else "unknown"
    logger.info(
        "Updated %s with %d EVI values (%s)",
        landsat_parquet,
        len(aligned_values),
        method,
    )
    return landsat_parquet


def _sample_series(
    client: EarthEngineClient,
    source_date: str,
    grid_parquet: Path,
    cache_root: Path,
    bbox: Tuple[float, float, float, float],
) -> SampledSeries:
    cache_dir, tif_path = _cache_layout(cache_root, source_date)
    if not tif_path.exists():
        logger.info("Cache miss for %s; generating new raster", source_date)
        image = _evi_image_for_date(client, source_date, bbox)
        if image is None:
            raise RuntimeError(f"No Sentinel-2 data available for {source_date}")
        _download_image(client, image, tif_path, bbox)
    else:
        logger.debug("Reusing cached raster %s", tif_path)
    df = raster_to_grid_df(
        tif_path,
        grid_parquet,
        out_parquet=None,
        cache_dir=cache_dir,
        target_crs=TARGET_CRS,
        target_res=TARGET_RES,
        use_vrt=True,
        parallel=False,
    )
    return _series_from_df(df, source_date)


def _interpolate_series(
    prev_series: Optional[pd.Series],
    next_series: Optional[pd.Series],
    prev_date: Optional[str],
    next_date: Optional[str],
    target_date: str,
) -> pd.Series:
    if prev_series is None and next_series is None:
        raise ValueError("Cannot interpolate without neighbors")
    if prev_series is None:
        logger.info("Using next neighbor only to fill %s", target_date)
        return next_series.copy()
    if next_series is None:
        logger.info("Using previous neighbor only to fill %s", target_date)
        return prev_series.copy()
    prev_dt = dt.date.fromisoformat(prev_date)
    next_dt = dt.date.fromisoformat(next_date)
    target_dt = dt.date.fromisoformat(target_date)
    total = (next_dt - prev_dt).days
    weight = 0.5 if total == 0 else (target_dt - prev_dt).days / float(total)
    logger.info(
        "Interpolating %s between %s and %s (weight=%.3f)",
        target_date,
        prev_date,
        next_date,
        weight,
    )
    aligned_next = next_series.reindex(prev_series.index)
    interpolated = prev_series * (1.0 - weight) + aligned_next * weight
    interpolated.name = target_date
    return interpolated


def process_date(
    date_str: str,
    grid_parquet: Path = DEFAULT_GRID,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    ee_project: Optional[str] = None,
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    estimate_window_days: int = DEFAULT_ESTIMATE_WINDOW,
    force: bool = False,
    monthly_s2_dates: Optional[Mapping[Tuple[int, int], dt.date]] = None,
) -> Path:
    logger.info("Processing Sentinel-2 EVI for %s", date_str)
    landsat_parquet = _landsat_parquet_path(cache_root, date_str)
    if not landsat_parquet.exists():
        raise FileNotFoundError(
            f"Missing Landsat parquet for {date_str}: {landsat_parquet}"
        )
    if not force and _has_evi_columns(landsat_parquet):
        logger.info("EVI already stored for %s", date_str)
        return landsat_parquet

    target_dt = dt.date.fromisoformat(date_str)
    source_date_str = date_str
    if monthly_s2_dates is not None:
        key = (target_dt.year, target_dt.month)
        source_dt = monthly_s2_dates.get(key)

        # Fallback: search backward month-by-month until we find a mapping
        if source_dt is None:
            logger.warning(
                "No Sentinel-2 monthly date mapped for %s (key=%s), searching backward...",
                date_str,
                key,
            )
            current_date = target_dt
            max_lookback_months = 12

            for _ in range(max_lookback_months):
                # Go back one month
                if current_date.month == 1:
                    current_date = current_date.replace(
                        year=current_date.year - 1, month=12
                    )
                else:
                    current_date = current_date.replace(month=current_date.month - 1)

                fallback_key = (current_date.year, current_date.month)
                source_dt = monthly_s2_dates.get(fallback_key)

                if source_dt is not None:
                    logger.info(
                        "Found fallback for %s: using %s-%02d data (%s)",
                        date_str,
                        current_date.year,
                        current_date.month,
                        source_dt.isoformat(),
                    )
                    break

            if source_dt is None:
                logger.warning(
                    "No fallback found for %s after %d months lookback",
                    date_str,
                    max_lookback_months,
                )

        if source_dt is not None:
            source_date_str = source_dt.isoformat()
            if source_date_str != date_str:
                logger.info(
                    "Mapped Landsat %s to Sentinel-2 %s", date_str, source_date_str
                )

    client = _init_client(ee_project)
    cache_dir, tif_path = _cache_layout(cache_root, source_date_str)
    if force and tif_path.exists():
        tif_path.unlink()

    image = _evi_image_for_date(client, source_date_str, bbox)
    if image is not None:
        _download_image(client, image, tif_path, bbox)
        sample = _sample_series(client, source_date_str, grid_parquet, cache_root, bbox)
        if source_date_str != date_str:
            sample = SampledSeries(
                values=sample.values.rename(date_str),
                nodata=sample.nodata.rename(f"{date_str}_nodata"),
            )
        return _update_landsat_parquet(
            landsat_parquet,
            sample,
            {
                "source": COLLECTION_ID,
                "method": "observed",
                "sentinel_date": source_date_str,
            },
        )

    prev_date, next_date = _neighbor_dates(client, date_str, estimate_window_days)
    if prev_date is None and next_date is None:
        sample = _empty_sample(grid_parquet, date_str)
        return _update_landsat_parquet(
            landsat_parquet, sample, {"source": COLLECTION_ID, "method": "blank"}
        )

    prev_sample = (
        _sample_series(client, prev_date, grid_parquet, cache_root, bbox)
        if prev_date
        else None
    )
    next_sample = (
        _sample_series(client, next_date, grid_parquet, cache_root, bbox)
        if next_date
        else None
    )
    interpolated_values = _interpolate_series(
        prev_sample.values if prev_sample else None,
        next_sample.values if next_sample else None,
        prev_date,
        next_date,
        date_str,
    )
    nodata = pd.Series(
        ~np.isfinite(interpolated_values.to_numpy()),
        index=interpolated_values.index,
        dtype="bool",
        name=f"{date_str}_nodata",
    )
    estimate = SampledSeries(values=interpolated_values, nodata=nodata)
    meta = {
        "source": COLLECTION_ID,
        "method": "interpolated",
        "estimated_from": {"prev": prev_date, "next": next_date},
    }
    return _update_landsat_parquet(landsat_parquet, estimate, meta)


def process_dates(
    dates_path: Path = DEFAULT_DATES,
    sentinel_dates_path: Path = DEFAULT_S2_MONTHLY_DATES,
    grid_parquet: Path = DEFAULT_GRID,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    ee_project: Optional[str] = None,
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    estimate_window_days: int = DEFAULT_ESTIMATE_WINDOW,
    force: bool = False,
) -> List[Path]:
    outputs: List[Path] = []
    all_dates = read_dates(dates_path)
    try:
        monthly_s2 = read_monthly_s2_dates(sentinel_dates_path)
    except FileNotFoundError as exc:
        logger.error("Sentinel-2 monthly dates unavailable: %s", exc)
        monthly_s2 = None
    logger.info("Starting EVI processing for %d dates", len(all_dates))
    for date_str in all_dates:
        try:
            logger.info("--> %s", date_str)
            outputs.append(
                process_date(
                    date_str,
                    grid_parquet=grid_parquet,
                    cache_root=cache_root,
                    ee_project=ee_project,
                    bbox=bbox,
                    estimate_window_days=estimate_window_days,
                    force=force,
                    monthly_s2_dates=monthly_s2,
                )
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", date_str, exc, exc_info=True)
    return outputs


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sample Sentinel-2 EVI onto the 30 m grid and update Landsat parquets."
    )
    parser.add_argument(
        "--dates-path",
        type=Path,
        default=DEFAULT_DATES,
        help="Path to landsat8_image_dates.txt",
    )
    parser.add_argument(
        "--grid-parquet",
        type=Path,
        default=DEFAULT_GRID,
        help="30 m grid parquet (must contain grid_id)",
    )
    parser.add_argument(
        "--sentinel-dates-path",
        type=Path,
        default=DEFAULT_S2_MONTHLY_DATES,
        help="Monthly Sentinel-2 low-cloud date list (one ISO date per line)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Cache root containing landsat_YYYY-MM-DD.parquet files",
    )
    parser.add_argument(
        "--ee-project",
        default=None,
        help="Optional Earth Engine project id to initialize",
    )
    parser.add_argument(
        "--bbox",
        default=",".join(str(v) for v in DEFAULT_BBOX),
        help="bbox lon/lat minx,miny,maxx,maxy used for EE region",
    )
    parser.add_argument(
        "--estimate-window",
        type=int,
        default=DEFAULT_ESTIMATE_WINDOW,
        help="Days to look for neighbors when estimating missing days",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if EVI columns already exist",
    )
    args = parser.parse_args(argv)

    bbox = tuple(float(x) for x in args.bbox.split(","))

    logger.info(
        "Running evi_signals with dates=%s grid=%s cache=%s force=%s",
        args.dates_path,
        args.grid_parquet,
        args.cache_root,
        args.force,
    )
    process_dates(
        dates_path=args.dates_path,
        grid_parquet=args.grid_parquet,
        sentinel_dates_path=args.sentinel_dates_path,
        cache_root=args.cache_root,
        ee_project=args.ee_project,
        bbox=bbox,  # type: ignore[arg-type]
        estimate_window_days=args.estimate_window,
        force=args.force,
    )


if __name__ == "__main__":
    main()
