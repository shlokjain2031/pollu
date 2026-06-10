"""Build the static feature table for the Delhi prediction points.

Real data (from pollu_sop GEE extraction): satellite bands/indices
(B2, B3, B4, B7, EVI, NDVI, IBI) and terrain (elevation, slope, aspect),
keyed by (lat, lon).

Synthetic data (Overpass/OSM API not available): the 110 osm_* features.
They are generated deterministically per point, driven by the real built-up
index (IBI) and vegetation (NDVI) so urban-form features correlate with the
real built-up signal, and calibrated to the value ranges observed in the
station-level OSM extraction (delhi_station_osm_lur_features.csv).

Output: kashi/data/delhi_features.parquet
  lat, lon, 10 real static features, 110 synthetic osm features. Dynamic
  features (sun, meteo, AOD) are filled at serving time.

Run: .venv/bin/python -m kashi.build_features
"""

import zlib

import numpy as np
import pandas as pd

from kashi.config import FEATURES_PARQUET, SOURCE_POINTS_CSV, STATION_OSM_CSV


def _stable(s: str) -> int:
    """Process-independent string hash (builtin hash() is salted per run)."""
    return zlib.crc32(s.encode()) % 1000


STATIC_REAL = ["B2", "B3", "B4", "B7", "EVI", "NDVI", "IBI", "elevation", "slope", "aspect"]

BUFFERS_M = [50, 200, 500, 1000, 2000]
LANDUSE = [
    "cemetery", "commercial", "farmland", "forest", "grass", "industrial",
    "military", "railway", "recreation_ground", "residential", "retail",
]
ROAD_TYPES = ["all", "arterial", "major", "secondary_tertiary"]
POI = ["building", "bus_stop", "restaurant", "temple"]

# Land-use affinity: positive → scales with urbanicity, negative → with 1-urbanicity
LANDUSE_AFFINITY = {
    "residential": 1.0, "commercial": 1.0, "industrial": 0.7, "retail": 1.0,
    "railway": 0.6, "cemetery": 0.3, "recreation_ground": 0.4, "grass": -0.5,
    "forest": -1.0, "farmland": -1.0, "military": 0.0,
}


def _hash01(*arrays):
    """Deterministic pseudo-random floats in [0, 1) from integer/float arrays."""
    acc = np.zeros_like(np.asarray(arrays[0], dtype=np.float64))
    for i, a in enumerate(arrays):
        acc = acc + np.asarray(a, dtype=np.float64) * (12.9898 + 17.17 * i)
    return np.modf(np.abs(np.sin(acc) * 43758.5453123))[0]


def build() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_POINTS_CSV)
    df = df.sort_values(["lat", "lon"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    idx = np.arange(n)

    cols = {"lat": df["lat"].to_numpy(), "lon": df["lon"].to_numpy()}
    for col in STATIC_REAL:
        cols[col] = df[col].to_numpy()

    # Urbanicity in [0,1]: built-up index high + vegetation low → urban.
    ibi_pct = df["IBI"].rank(pct=True).to_numpy()
    ndvi_pct = df["NDVI"].rank(pct=True).to_numpy()
    urban = np.clip(0.6 * ibi_pct + 0.4 * (1.0 - ndvi_pct) + 0.1 * (_hash01(idx, 1) - 0.5), 0.0, 1.0)

    stats = pd.read_csv(STATION_OSM_CSV)

    def cap(col):  # calibrate to observed station-level maxima
        return float(stats[col].max())

    for r in BUFFERS_M:
        # Small buffers are often empty; large buffers almost never are.
        p_zero = {50: 0.55, 200: 0.25, 500: 0.10, 1000: 0.02, 2000: 0.0}[r]
        nonzero = _hash01(idx, r) >= p_zero

        for kind in POI:
            col = f"osm_{kind}_count_{r}m"
            scale = cap(col)
            noise = 0.4 + 1.2 * _hash01(idx, r + _stable(kind))
            counts = np.where(nonzero, np.round(scale * urban ** 2 * noise), 0.0)
            cols[col] = np.clip(counts, 0, 2 * scale if scale > 0 else 0)

        # Building density tracks counts but is bounded by observed maxima.
        dcol = f"osm_building_density_{r}m"
        dnoise = 0.5 + _hash01(idx, r + 7)
        cols[dcol] = np.where(nonzero, np.clip(cap(dcol) * urban * dnoise, 0, cap(dcol)), 0.0)

        # Road network: density for each class, plus total segment count.
        for road in ROAD_TYPES:
            col = f"osm_road_density_{road}_{r}m"
            rnoise = 0.5 + _hash01(idx, r + _stable(road))
            cols[col] = np.clip(cap(col) * (0.15 + 0.85 * urban) * rnoise, 0, cap(col))
        ccol = f"osm_road_count_{r}m"
        cols[ccol] = np.round(np.clip(cap(ccol) * (0.1 + 0.9 * urban) * (0.5 + _hash01(idx, r + 11)), 0, cap(ccol)))

        # Land-use fractions: affinity-weighted, normalized to at most ~0.95 coverage.
        fracs = {}
        for lu in LANDUSE:
            col = f"osm_landuse_{lu}_frac_{r}m"
            mx = cap(col)
            if mx == 0.0:  # never observed at this radius → keep 0 (in-distribution)
                fracs[lu] = np.zeros(n)
                continue
            aff = LANDUSE_AFFINITY[lu]
            base = urban if aff >= 0 else (1.0 - urban)
            w = abs(aff) if aff != 0 else 0.2
            raw = base ** 1.5 * w * (0.3 + _hash01(idx, r + _stable(lu)))
            # Sparse classes appear in only a few points.
            if lu in ("cemetery", "military", "railway", "recreation_ground", "retail"):
                raw = np.where(_hash01(idx, r + 13 + _stable(lu)) < 0.07, raw, 0.0)
            fracs[lu] = raw * mx
        total = sum(fracs.values())
        over = np.maximum(total / 0.95, 1.0)
        for lu in LANDUSE:
            cols[f"osm_landuse_{lu}_frac_{r}m"] = fracs[lu] / over
        cols[f"osm_landuse_known_frac_{r}m"] = total / over

    return pd.DataFrame(cols)


def main():
    out = build()
    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURES_PARQUET, index=False)
    print(f"Wrote {len(out)} points x {len(out.columns)} cols -> {FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
