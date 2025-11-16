"""Sentinel-5P NO2 sampler that mirrors `patliputra.landsat8_signals`.

Reads the canonical `landsat8_image_dates.txt`, fetches COPERNICUS/
S5P/OFFL/L3_NO2 for each date, reprojects to EPSG:32643 at 30 m, samples
on the same GeoParquet grid, and writes outputs under cache/copernicus_DATE/.
Missing days fall back to linear interpolation between the nearest
available S5P observations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from patliputra.backends.earthengine import EarthEngineClient
from patliputra.utils.landsat8_signals import raster_to_grid_df, _ensure_cache_dir


COLLECTION_ID = "COPERNICUS/S5P/OFFL/L3_NO2"
BAND_NAME = "tropospheric_NO2_column_number_density"
TARGET_CRS = "EPSG:32643"
TARGET_RES = 30.0
DEFAULT_GRID = Path("data/mumbai/grid_30m.parquet")
DEFAULT_DATES = Path("patliputra/landsat8_image_dates.txt")
DEFAULT_CACHE_ROOT = Path("cache")
DEFAULT_BBOX = (72.7763, 18.8939, 72.9797, 19.2701)
DEFAULT_ESTIMATE_WINDOW = 7
PARQUET_TEMPLATE = "copernicus_{date}.parquet"
TIF_TEMPLATE = "copernicus_{date}.tif"

logger = logging.getLogger("s5p_no2_signals")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _init_client(project_id: Optional[str]) -> EarthEngineClient:
    client = (
        EarthEngineClient(project_id=project_id) if project_id else EarthEngineClient()
    )
    client.initialize()
    return client


def read_dates(path: Path = DEFAULT_DATES) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dates file not found: {path}")
    logger.info("Reading dates from %s", path)
    with open(path, "r") as fh:
        dates = [line.strip() for line in fh.readlines() if line.strip()]
    logger.info("Loaded %d dates", len(dates))
    return dates


def _validate_tif(path: Path) -> None:
    with open(path, "rb") as fh:
        header = fh.read(4)
    if header not in (b"II*\x00", b"MM\x00*"):
        raise ValueError(
            f"Downloaded file at {path} is not a GeoTIFF (header={header!r})"
        )


def _download_image(
    client: EarthEngineClient,
    image,
    out_tif: Path,
    bbox: Tuple[float, float, float, float],
) -> Path:
    params = {"scale": TARGET_RES, "crs": TARGET_CRS, "format": "GEO_TIFF"}
    if bbox is not None:
        params["region"] = [float(v) for v in bbox]
    url = client.get_download_url(image, params=params)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_tif.with_suffix(out_tif.suffix + ".download")
    logger.info("Downloading S5P image to %s", tmp_path)
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)

    with open(tmp_path, "rb") as fh:
        magic = fh.read(4)

    if magic.startswith(b"PK"):
        logger.info("Download is a ZIP archive; extracting GeoTIFF to %s", out_tif)
        with ZipFile(tmp_path) as zf:
            tif_members = [
                info
                for info in zf.infolist()
                if not info.is_dir()
                and info.filename.lower().endswith((".tif", ".tiff"))
            ]
            if not tif_members:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError("Zip download did not contain a GeoTIFF")
            with zf.open(tif_members[0]) as src, open(out_tif, "wb") as dst:
                shutil.copyfileobj(src, dst)
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.replace(out_tif)

    _validate_tif(out_tif)
    logger.info("Saved GeoTIFF: %s", out_tif)
    return out_tif


def _image_for_date(client: EarthEngineClient, date_str: str):
    ee = __import__("ee")
    start = dt.date.fromisoformat(date_str)
    end = start + dt.timedelta(days=1)
    collection = ee.ImageCollection(COLLECTION_ID).filterDate(
        start.isoformat(), end.isoformat()
    )
    if int(collection.size().getInfo()) == 0:
        return None
    return collection.median().select(BAND_NAME)


def _neighbor_dates(
    client: EarthEngineClient, target_date: str, max_days: int
) -> Tuple[Optional[str], Optional[str]]:
    ee = __import__("ee")
    target = dt.date.fromisoformat(target_date)
    prev_date: Optional[str] = None
    next_date: Optional[str] = None

    for delta in range(1, max_days + 1):
        start = (target - dt.timedelta(days=delta)).isoformat()
        end = (target - dt.timedelta(days=delta - 1)).isoformat()
        col = ee.ImageCollection(COLLECTION_ID).filterDate(start, end)
        if int(col.size().getInfo()) > 0:
            prev_date = start
            break

    for delta in range(1, max_days + 1):
        start = (target + dt.timedelta(days=delta)).isoformat()
        end = (target + dt.timedelta(days=delta + 1)).isoformat()
        col = ee.ImageCollection(COLLECTION_ID).filterDate(start, end)
        if int(col.size().getInfo()) > 0:
            next_date = start
            break

    return prev_date, next_date


def _cache_layout(cache_root: Path, date_str: str) -> Tuple[Path, Path, Path]:
    base = Path(cache_root) / f"copernicus_{date_str}"
    base.mkdir(parents=True, exist_ok=True)
    tif_path = base / TIF_TEMPLATE.format(date=date_str)
    parquet_path = base / PARQUET_TEMPLATE.format(date=date_str)
    return base, tif_path, parquet_path


def _grid_template(grid_parquet: Path) -> Tuple[pd.DataFrame, List[Point]]:
    gdf = gpd.read_parquet(grid_parquet)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.to_crs(TARGET_CRS)
    centroids = gdf.geometry.centroid
    base = pd.DataFrame(
        {
            "grid_id": gdf["grid_id"].values,
            "x": centroids.x.values,
            "y": centroids.y.values,
        }
    ).set_index("grid_id", drop=False)
    geom = [Point(pt.x, pt.y) for pt in centroids]
    return base, geom


def _serialize_meta(extra: Optional[dict] = None) -> str:
    meta = {"source": COLLECTION_ID, "band": BAND_NAME}
    if extra:
        meta.update(extra)
    return json.dumps(meta)


def _write_series(
    series: pd.Series,
    grid_parquet: Path,
    out_parquet: Path,
    extra_meta: Optional[dict] = None,
) -> Path:
    base, geom = _grid_template(grid_parquet)
    aligned = series.reindex(base.index).to_numpy(dtype="float64")
    nodata = ~np.isfinite(aligned)
    df = pd.DataFrame(
        {
            "grid_id": base["grid_id"].values,
            "x": base["x"].values,
            "y": base["y"].values,
            "geometry": geom,
            "toa_b2": aligned,
            "is_nodata": nodata,
            "raster_meta": _serialize_meta(extra_meta),
        }
    )
    gdf_out = gpd.GeoDataFrame(
        df.drop(columns=["geometry"]), geometry=df["geometry"], crs=TARGET_CRS
    )
    gdf_out.to_parquet(out_parquet, engine="pyarrow", index=False)
    return out_parquet


def _sample_series(
    client: EarthEngineClient,
    date_str: str,
    grid_parquet: Path,
    cache_root: Path,
    bbox: Tuple[float, float, float, float],
) -> pd.Series:
    tile_dir, tif_path, _ = _cache_layout(cache_root, date_str)
    if not tif_path.exists():
        image = _image_for_date(client, date_str)
        if image is None:
            raise RuntimeError(f"No S5P image available for {date_str}")
        _download_image(client, image, tif_path, bbox)
    df = raster_to_grid_df(
        tif_path,
        grid_parquet,
        out_parquet=None,
        cache_dir=tile_dir,
        target_crs=TARGET_CRS,
        target_res=TARGET_RES,
        use_vrt=True,
        parallel=False,
    )
    series = df.get("toa_b2")
    if series is None and "toa_b1" in df.columns:
        series = df["toa_b1"]
    series = series.astype("float64")
    series.name = date_str
    return series


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
        return next_series.copy()
    if next_series is None:
        return prev_series.copy()
    prev_dt = dt.date.fromisoformat(prev_date)
    next_dt = dt.date.fromisoformat(next_date)
    target_dt = dt.date.fromisoformat(target_date)
    total = (next_dt - prev_dt).days
    weight = 0.5 if total == 0 else (target_dt - prev_dt).days / float(total)
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
) -> Path:
    grid_parquet = Path(grid_parquet)
    cache_root = Path(cache_root)
    bbox = bbox or DEFAULT_BBOX

    cache_dir, tif_path, parquet_path = _cache_layout(cache_root, date_str)
    if parquet_path.exists() and not force:
        logger.info("Parquet already exists for %s", date_str)
        return parquet_path

    client = _init_client(ee_project)
    image = _image_for_date(client, date_str)

    if image is not None:
        _download_image(client, image, tif_path, bbox)
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
        if "toa_b2" not in df.columns and "toa_b1" in df.columns:
            df = df.rename(columns={"toa_b1": "toa_b2"})
        gdf_out = gpd.GeoDataFrame(
            df.drop(columns=["geometry"]), geometry=df["geometry"], crs=TARGET_CRS
        )
        gdf_out.to_parquet(parquet_path, engine="pyarrow", index=False)
        logger.info("Wrote NO2 parquet: %s", parquet_path)
        return parquet_path

    prev_date, next_date = _neighbor_dates(client, date_str, estimate_window_days)
    if prev_date is None and next_date is None:
        template, _ = _grid_template(grid_parquet)
        blank = pd.Series(np.nan, index=template.index, name=date_str, dtype="float64")
        return _write_series(
            blank, grid_parquet, parquet_path, {"estimated_from": None}
        )

    prev_series = (
        _sample_series(client, prev_date, grid_parquet, cache_root, bbox)
        if prev_date
        else None
    )
    next_series = (
        _sample_series(client, next_date, grid_parquet, cache_root, bbox)
        if next_date
        else None
    )
    estimate = _interpolate_series(
        prev_series, next_series, prev_date, next_date, date_str
    )
    meta = {"estimated_from": {"prev": prev_date, "next": next_date}}
    return _write_series(estimate, grid_parquet, parquet_path, meta)


def process_dates(
    dates_path: Path = DEFAULT_DATES,
    grid_parquet: Path = DEFAULT_GRID,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    ee_project: Optional[str] = None,
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    estimate_window_days: int = DEFAULT_ESTIMATE_WINDOW,
    force: bool = False,
) -> List[Path]:
    outputs: List[Path] = []
    for date_str in read_dates(dates_path):
        try:
            out = process_date(
                date_str,
                grid_parquet=grid_parquet,
                cache_root=cache_root,
                ee_project=ee_project,
                bbox=bbox,
                estimate_window_days=estimate_window_days,
                force=force,
            )
            outputs.append(out)
        except Exception as exc:
            logger.error("Failed to process %s: %s", date_str, exc, exc_info=True)
    return outputs


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sample Sentinel-5P NO2 onto the 30 m grid."
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
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Cache root to write copernicus_YYYY-MM-DD folders",
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
        "--force", action="store_true", help="Recompute parquet even if already present"
    )
    args = parser.parse_args(argv)

    bbox = tuple(float(x) for x in args.bbox.split(","))

    logger.info("Queueing processing for dates listed in %s", args.dates_path)
    process_dates(
        dates_path=args.dates_path,
        grid_parquet=args.grid_parquet,
        cache_root=args.cache_root,
        ee_project=args.ee_project,
        bbox=bbox,  # type: ignore[arg-type]
        estimate_window_days=args.estimate_window,
        force=args.force,
    )


if __name__ == "__main__":
    main()
