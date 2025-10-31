import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np


GRID_PATH = Path("data/mumbai/grid_30m.parquet")
WARD_PATH = Path("resources/mumbai_wards.geojson")
EXPECTED_CRS = "EPSG:32643"
RES = 30.0
TOL = 1.0


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def test_grid_file_exists_and_loads():
    assert GRID_PATH.exists(), f"Grid file not found: {GRID_PATH}"
    grid = gpd.read_parquet(GRID_PATH)
    assert len(grid) > 0, "Loaded grid is empty"


def test_grid_has_expected_columns_and_types():
    grid = gpd.read_parquet(GRID_PATH)
    assert "grid_id" in grid.columns, "grid_id column missing"
    assert grid["grid_id"].is_unique, "grid_id values are not unique"
    geom_types = set(grid.geometry.geom_type.unique())
    assert geom_types == {"Point"}, f"Unexpected geometry types: {geom_types}"


def test_crs_and_spacing_and_coverage():
    grid = gpd.read_parquet(GRID_PATH)
    crs = grid.crs
    assert crs is not None, "Grid has no CRS"
    crs_str = crs.to_string() if hasattr(crs, "to_string") else str(crs)
    assert crs_str == EXPECTED_CRS, f"Grid CRS {crs_str} != expected {EXPECTED_CRS}"

    xs = grid["x"].to_numpy(dtype=float)
    ys = grid["y"].to_numpy(dtype=float)
    ux = np.sort(np.unique(xs))
    uy = np.sort(np.unique(ys))
    dx = np.diff(ux)
    dy = np.diff(uy)
    med_dx = float(np.median(dx[dx > 1e-9]))
    med_dy = float(np.median(dy[dy > 1e-9]))
    assert (
        abs(med_dx - RES) <= TOL and abs(med_dy - RES) <= TOL
    ), f"Grid spacing not approx {RES} m"

    # coverage vs wards
    wards = gpd.read_file(WARD_PATH).to_crs(grid.crs)
    union = wards.unary_union
    inside_mask = grid.geometry.within(union)
    inside_count = int(inside_mask.sum())
    total = len(grid)
    pct_inside = 100.0 * inside_count / total
    assert (
        pct_inside >= 90.0
    ), f"Less than 90% points inside wards union ({pct_inside:.2f}%)"


def test_no_duplicates_and_checksum():
    grid = gpd.read_parquet(GRID_PATH)
    geo_wkb = grid.geometry.apply(lambda g: g.wkb).tolist()
    dup_count = len(geo_wkb) - len(set(geo_wkb))
    assert dup_count == 0, f"Duplicate geometries found: {dup_count}"

    # checksum present (non-functional check, records file state)
    ch = sha256(GRID_PATH)
    assert len(ch) == 64, "Checksum length unexpected"
