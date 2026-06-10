"""Feature assembly for arbitrary Delhi coordinates.

We hold the real satellite/terrain features (plus precomputed synthetic OSM
features) at the pollu_sop prediction points. A request for any (lat, lon)
snaps to the nearest point for those static features; the 10 dynamic features
(sun geometry, meteorology, AOD) are filled per request by kashi.synthetic
using the request's own coordinates and hour.
"""

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from kashi import synthetic
from kashi.config import FEATURE_CONFIG_PATH, FEATURES_PARQUET

DYNAMIC = [
    "sun_azimuth", "sun_zenith",
    "WS_ms", "T_celsius", "SLP_kpa", "RH_pct",
    "AOD_550_filled", "AOD_470_filled", "AOD_uncertainty_filled", "AOD_daily_available",
]

# Mean metres per degree of latitude (good enough for a snap-distance check)
_M_PER_DEG = 111_000.0


class FeatureStore:
    """Real static features at the Delhi prediction points, indexed by location."""

    def __init__(self):
        with open(FEATURE_CONFIG_PATH) as f:
            self.feature_names = json.load(f)["model_features"]
        assert set(DYNAMIC) <= set(self.feature_names), "model is missing dynamic features"

        df = pd.read_parquet(FEATURES_PARQUET)
        self.point_count = len(df)
        self.lat = df["lat"].to_numpy()
        self.lon = df["lon"].to_numpy()
        self.bbox = {
            "lat_min": float(self.lat.min()), "lat_max": float(self.lat.max()),
            "lon_min": float(self.lon.min()), "lon_max": float(self.lon.max()),
        }
        self._tree = cKDTree(np.column_stack([self.lat, self.lon]))

        self._static = np.zeros((self.point_count, len(self.feature_names)), dtype=np.float64)
        self._dyn_cols = [(name, col) for col, name in enumerate(self.feature_names) if name in DYNAMIC]
        for col, name in enumerate(self.feature_names):
            if name not in DYNAMIC:
                self._static[:, col] = df[name].to_numpy(dtype=np.float64)

    def snap(self, lats: np.ndarray, lons: np.ndarray):
        """Nearest point index and distance (metres) for each coordinate."""
        dist_deg, idx = self._tree.query(np.column_stack([lats, lons]))
        return np.atleast_1d(idx), np.atleast_1d(dist_deg) * _M_PER_DEG

    def assemble(self, lats: np.ndarray, lons: np.ndarray, ts) -> np.ndarray:
        """Feature matrix [len(coords) x 130] for timestamp ts (tz-aware UTC).

        Static features come from the nearest known point; dynamic features use
        the request's own coordinates so they vary smoothly with location.
        """
        idx, _ = self.snap(lats, lons)
        X = self._static[idx].copy()

        az, zen = synthetic.solar_position(lats, lons, ts)
        ws, t, slp, rh = synthetic.meteo(idx, lats, lons, ts)
        a550, a470, unc, avail = synthetic.aod(idx, lats, ts)
        values = {
            "sun_azimuth": az, "sun_zenith": zen,
            "WS_ms": ws, "T_celsius": t, "SLP_kpa": slp, "RH_pct": rh,
            "AOD_550_filled": a550, "AOD_470_filled": a470,
            "AOD_uncertainty_filled": unc, "AOD_daily_available": avail,
        }
        for name, col in self._dyn_cols:
            X[:, col] = values[name]
        return X
