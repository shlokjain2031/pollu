"""Kashi inference service: serves the Delhi PM2.5 model over HTTP.

Endpoints:
    POST /inference  {coords: [[lat, lon], ...], timestamp_hour}
                     -> {predictions: [pm25, ...]}  (aligned to input order)
    GET  /meta       model/coverage metadata
    GET  /health     liveness

Run: .venv/bin/uvicorn kashi.service:app --host 127.0.0.1 --port 9999
"""

import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kashi.config import (
    FEATURE_VERSION,
    FEATURES_PARQUET,
    MAX_BATCH,
    MAX_SNAP_DISTANCE_M,
    MAX_YEAR,
    MIN_YEAR,
    MODEL_PATH,
    MODEL_VERSION,
)
from kashi.features import DYNAMIC, FeatureStore

log = logging.getLogger("kashi")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [kashi] %(levelname)s %(message)s")

_store: FeatureStore | None = None
_model = None
_predict_lock = threading.Lock()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _store, _model
    if not FEATURES_PARQUET.exists():
        raise RuntimeError(f"{FEATURES_PARQUET} missing - run: .venv/bin/python -m kashi.build_features")
    _store = FeatureStore()
    _model = joblib.load(MODEL_PATH)
    if list(_model.feature_name_) != _store.feature_names:
        raise RuntimeError("model feature names do not match feature_config.json order")
    # Warm LightGBM internals so the first request isn't slow
    _predict(np.array([_store.lat[0]]), np.array([_store.lon[0]]), datetime.now(timezone.utc))
    log.info("loaded model %s + %d feature points", MODEL_VERSION, _store.point_count)
    yield


app = FastAPI(title="Kashi - Delhi PM2.5 inference", version="2.0", lifespan=_lifespan)


def _predict(lats: np.ndarray, lons: np.ndarray, ts: datetime) -> np.ndarray:
    """Assemble features and predict, clipped to physical range."""
    X = pd.DataFrame(_store.assemble(lats, lons, ts), columns=_store.feature_names)
    with _predict_lock:
        pm25 = _model.predict(X)
    return np.clip(pm25, 0.0, None)


class InferenceRequest(BaseModel):
    coords: list[tuple[float, float]] = Field(min_length=1)
    timestamp_hour: str


def _parse_hour(raw: str) -> datetime:
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, detail=f"invalid timestamp_hour: {raw!r}")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if not MIN_YEAR <= ts.year <= MAX_YEAR:
        raise HTTPException(400, detail=f"timestamp_hour outside supported window {MIN_YEAR}-{MAX_YEAR}")
    return ts


@app.post("/inference")
def inference(req: InferenceRequest):
    if len(req.coords) > MAX_BATCH:
        raise HTTPException(400, detail=f"batch too large: {len(req.coords)} > {MAX_BATCH}")
    ts = _parse_hour(req.timestamp_hour)

    lats = np.array([c[0] for c in req.coords], dtype=np.float64)
    lons = np.array([c[1] for c in req.coords], dtype=np.float64)
    if not (np.all(np.abs(lats) <= 90) and np.all(np.abs(lons) <= 180)):
        raise HTTPException(400, detail="lat/lon out of range")

    _, dist_m = _store.snap(lats, lons)
    far = np.argmax(dist_m)
    if dist_m[far] > MAX_SNAP_DISTANCE_M:
        raise HTTPException(404, detail=f"({lats[far]}, {lons[far]}) is {dist_m[far] / 1000:.1f}km outside the Delhi coverage area")

    pm25 = _predict(lats, lons, ts)

    return {
        "predictions": [round(float(v), 2) for v in pm25],
        "timestamp_hour": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
    }


@app.get("/meta")
def meta():
    return {
        "point_count": _store.point_count,
        "bbox": _store.bbox,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_count": len(_store.feature_names),
        "synthetic_dynamic_features": DYNAMIC,
        "max_batch": MAX_BATCH,
    }


@app.get("/health")
def health():
    ok = _store is not None and _model is not None
    return {"status": "healthy" if ok else "loading", "point_count": _store.point_count if _store else 0}
