# Kashi — Delhi PM2.5 inference service

Serves the trained Delhi model (`pollu_sop/models/delhi_all_features`,
LightGBM depth-3) over HTTP. The Go coordinator
(`takshashila/distributed/coordinator`) calls it for on-demand inference.

The API is **lat/lon only** — there is no grid or cell-ID concept. A request for
any coordinate snaps to the nearest known feature point for its real static
features; everything time-varying is generated per request.

## Model

- **Artifact:** `lightgbm_d3_final.joblib` (130 features, station-LOSO R²≈0.78)
- **Features:** 16 core satellite/terrain/meteo + 4 AOD + 110 OSM land-use/road.
  Feature order is pinned to `feature_config.json["model_features"]` and verified
  against the booster's `feature_name_` at startup.

### Real vs synthetic features

| Source | Features | Provenance |
|--------|----------|------------|
| Real (pollu_sop GEE extraction) | B2/B3/B4/B7, EVI, NDVI, IBI, elevation, slope, aspect | nearest point in `data/delhi_features.parquet` |
| Synthetic (OSM/Overpass unavailable) | 110 `osm_*` land-use, road density, POI counts | `build_features.py` — deterministic per point, driven by real IBI/NDVI, calibrated to station-level OSM maxima |
| Synthetic (GEOS-FP / MODIS unavailable) | sun geometry, meteo (T/RH/WS/SLP), AOD | `synthetic.py` — sun is real astronomy at 10:30 IST; meteo/AOD are deterministic functions of (location, hour) calibrated to the 2021–2025 Delhi training climatology |

Synthetic providers are **deterministic**: identical `(lat, lon, hour)` always
yields identical features, so predictions are reproducible and cacheable.

## Coverage

The real static features exist at the pollu_sop prediction points (65,736 points
at 250 m over Delhi). Any coordinate within the bbox snaps to the nearest point;
a coordinate more than 5 km from any point is rejected as out of coverage.

## Endpoints

- `POST /inference` `{coords:[[lat,lon],...], timestamp_hour}` → `{predictions:[pm25,...]}` (aligned to input order)
- `GET /meta` → model/coverage metadata
- `GET /health` → liveness

## Run

```bash
# one-time: build the static feature lookup table
.venv/bin/python -m kashi.build_features

# serve (4 workers parallelize concurrent cold inference; ~15 MB each)
.venv/bin/uvicorn kashi.service:app --host 127.0.0.1 --port 9999 --workers 4
```

The service is read-only after startup, so workers are independent — scale the
worker count with available cores. The coordinator absorbs repeat load via its
Redis + in-memory caches and collapses concurrent identical requests via
deduplication, so kashi only sees one call per distinct (location, hour).

## Test

```bash
.venv/bin/python -m pytest kashi/tests/ -q
```
