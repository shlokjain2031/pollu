"""Kashi configuration. Paths and constants for the Delhi inference service."""

import os
from pathlib import Path

KASHI_ROOT = Path(__file__).parent
POLLU_SOP_ROOT = Path(os.getenv("POLLU_SOP_ROOT", "/Users/shlokjain/pollu_sop"))

# Serving artifacts are vendored into the repo so kashi is self-contained at
# runtime. After retraining, re-copy them (see scripts/vendor_model.sh).
_VENDORED_MODEL_DIR = KASHI_ROOT / "data" / "model"
_SOP_MODEL_DIR = POLLU_SOP_ROOT / "models" / "delhi_all_features"
MODEL_DIR = _VENDORED_MODEL_DIR if (_VENDORED_MODEL_DIR / "lightgbm_d3_final.joblib").exists() else _SOP_MODEL_DIR
MODEL_PATH = MODEL_DIR / "lightgbm_d3_final.joblib"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.json"

# Build-time only (kashi.build_features). Real GEE features at the Delhi
# prediction points (the feature lookup source) + station OSM for calibration.
SOURCE_POINTS_CSV = (
    POLLU_SOP_ROOT
    / "data/processed/features/delhi_prediction_points_gee_features_2025_12_08_grid250m.csv"
)
STATION_OSM_CSV = (
    POLLU_SOP_ROOT / "data/processed/features/delhi_station_osm_lur_features.csv"
)

FEATURES_PARQUET = KASHI_ROOT / "data" / "delhi_features.parquet"

MODEL_VERSION = "delhi_all_features_lgbm_d3"
FEATURE_VERSION = "v2_delhi_130"

HOST = os.getenv("KASHI_HOST", "127.0.0.1")
PORT = int(os.getenv("KASHI_PORT", "9999"))

# Largest number of coordinates a single /inference call may request
MAX_BATCH = int(os.getenv("KASHI_MAX_BATCH", "5000"))

# A coordinate further than this from any known feature point is rejected (metres)
MAX_SNAP_DISTANCE_M = 5_000

# Requests outside this window are rejected (model has no skill far from training era)
MIN_YEAR = 2015
MAX_YEAR = 2035
