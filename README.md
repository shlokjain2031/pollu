Pollu

Pollu is a modular, end-to-end system for pollution-aware routing. It combines environmental sensing, geospatial modeling, and routing optimization to help users navigate cities while minimizing air pollution exposure.

The system is inspired by Valhalla’s modular architecture but built from scratch for fine-grained pollution modeling and walking-oriented routing.

1. Overview

Pollu estimates hyperlocal PM2.5 concentrations across entire cities using a spatio-temporal regression model (GTWR) and microscale geographic predictors derived from remote sensing and open geospatial data.
Predicted pollution maps are used as dynamic cost overlays in a routing engine, enabling “healthier route” recommendations that minimize cumulative exposure.

2. System Architecture

Pollu is composed of five autonomous modules, each representing a functional layer of the system.

Module	Codename	Function	Nature	Language
1	Patliputra	Data ingestion and normalization	I/O-bound	Python
2	Dwarka	ML modeling (GTWR / LUR)	CPU-bound	Python (to migrate to C++)
3	Nalanda	Aggregation and summarization	Compute + I/O	Python
4	Takshashila	Database and persistence	I/O	Python
5	Ujjayini	API and external exposure	Network I/O	Python (FastAPI)
2.1 Patliputra (Ingestion)

Responsible for fetching, cleaning, and aligning upstream datasets into a unified geospatial table.
Sources include:

- Landsat-8 raw bands via Google Earth Engine (derived indices: NDVI, NDBI, MNDWI, BSI, IBI)
- Sentinel-2 raw bands via Google Earth Engine (EVI, AOT, cloud mask)
- Sky View Factor (SVF) from Google Open Buildings + NASA DEM via Earth Engine
- Meteorological variables (temperature, humidity, wind, pressure, precipitation) from Open-Meteo
- OSM features: road type, construction sites, traffic routes
- Ground-truth PM2.5 from CPCB AQI monitoring stations

Output: Per-date parquet artifacts organized by data source (30m resolution).

**Pipeline Orchestration:**

Use the pipeline runner for automated data fetching:

```bash
# Full pipeline for a date range
python -m patliputra.pipeline run \
  --city mumbai \
  --start-date 2018-01-01 \
  --end-date 2018-12-31 \
  --ee-project your-earth-engine-project-id \
  --hourly

# Specific stages only
python -m patliputra.pipeline run \
  --stages meteo,sentinel2 \
  --city mumbai \
  --start-date 2021-06-01 \
  --end-date 2021-06-30

# Check artifact status
python -m patliputra.pipeline status --city mumbai
```

**Note**: All data fetching stages should run **sequentially** due to API rate limits and Earth Engine quotas. See [docs/patliputra_pipeline_guide.md](docs/patliputra_pipeline_guide.md) for details.

**Time Estimates** (7 years of data):
- Grid creation: 5 min (one-time)
- Meteorological data: 16-20 hours (API rate limited)
- Satellite data (Landsat-8 + Sentinel-2): 10-16 hours (Earth Engine quota limited)
- Sky View Factor: 4-8 hours (Earth Engine intensive)
- Total: **31-45 hours**

2.2 Dwarka (Modeling Layer)

Implements Land Use Regression (LUR) enhanced with Geographically and Temporally Weighted Regression (GTWR) to predict micro-scale PM2.5.

Model Equation
y_i = β0(u_i, v_i, t_i) + Σ βk(u_i, v_i, t_i) * x_ik + ε_i

2.3 Nalanda (Training Data Assembly)

Orchestrates feature engineering and training data assembly for the modeling layer:

- **Meteo features** (`features_meteo.py`): OpenMeteo hourly data with temporal lags (1h, 3h, 6h, 12h, 24h)
- **Satellite features** (`features_satellite.py`): Landsat8 indices (NDVI, NDBI, MNDWI, BSI, IBI) + Sentinel2 (EVI, cloud mask, spatial features)
- **Training assembly** (`training.py`): Joins target labels with features, outputs `training/{date}/training.parquet`
- **CLI** (`cli.py`): Batch processing, validation, reporting

**CLI Usage:**
```bash
# Check pipeline status
python -m nalanda.cli status

# Assemble single date
python -m nalanda.cli assemble --date 2021-06-12

# Assemble date range (parallel processing)
python -m nalanda.cli assemble --start-date 2021-06-07 --end-date 2021-06-18 --parallel --workers 4

# Validate all training data
python -m nalanda.cli validate --all -v

# Generate pipeline report
python -m nalanda.cli report --output report.json
```

**Parallel Processing:**
Use `--parallel` (or `-p`) to process multiple dates concurrently. Speeds up batch assembly significantly:
```bash
# Sequential (38.6s for 3 dates)
python -m nalanda.cli assemble --start-date 2021-06-10 --end-date 2021-06-12

# Parallel (47.0s for 3 dates, wall time includes overhead but individual tasks run concurrently)
python -m nalanda.cli assemble --start-date 2021-06-10 --end-date 2021-06-12 --parallel
```


Where:

y_i: PM2.5 concentration

(u_i, v_i): spatial coordinates

t_i: temporal coordinate (season or timestamp)

x_ik: predictor variables

βk(u_i, v_i, t_i): location- and time-specific coefficients

ε_i: residual term

Weighting Kernel
W_i(j) = exp( -d_ij² / h_s²  -  (t_i - t_j)² / h_t² )


with adaptive spatial (h_s) and temporal (h_t) bandwidths.

Implementation

Prototype: Python (mgwr library)

Production: C++ (Eigen + OpenMP)

Input: merged table from Patliputra

Output: fine-grained PM2.5 raster + residual map

2.3 Nalanda (Aggregation)

Aggregates Dwarka outputs into:

City-level PM2.5 rasters (30 m or 90 m)

H3 hex-grid summaries

OSM-edge pollution overlays

Statistical summaries (mean, variance, confidence intervals)

2.4 Takshashila (Storage Layer)

Stores model outputs and metadata for efficient spatial querying.

Typical schema:

Field	Type	Description
city_id	TEXT	City name or code
timestamp	DATETIME	Model run timestamp
grid_id	INT	Grid cell ID
lat, lon	FLOAT	Coordinates
pm25_estimate	FLOAT	Predicted PM2.5
uncertainty	FLOAT	Local model residual
source_model	TEXT	Model name/version
version	TEXT	Model version

Backends: DuckDB for local storage, PostGIS or Parquet for distributed use.

2.5 Ujjayini (API Layer)

A FastAPI-based REST service that exposes model results to applications and routing systems.

Endpoints:

/ping → health check

/city/{name}/map → pollution raster

/city/{name}/stats → city-level summaries

/route → returns pollution-aware routes

Future versions may use shared-memory tiles or gRPC to connect with Pollu’s routing engine.

3. Data Schema — Signals

Each training sample represents one 30 m grid cell with a set of static and dynamic predictors.

Variable	Description	Source	Type
TOA_B2	Blue band reflectance	GEE (Landsat-8)	Static
TOA_B4	Red band reflectance	GEE (Landsat-8)	Static
TOA_B7	SWIR-2 reflectance	GEE (Landsat-8)	Static
NDVI	(B5–B4)/(B5+B4) vegetation index	Derived	Static
IBI	Index-based built-up index	Derived	Static
GVI	Green-view index	Google Street View	Static
SVF	Sky-view factor	Copernicus DEM	Static
Meteorology	Temperature, wind, pressure, PBLH	ERA5	Dynamic
Dust	Road surface, construction proximity	OSM	Static
Traffic Emission	Bus route length, road density	OSM	Static
Target (PM2.5)	Ground truth air quality	AQI_cn	Dynamic
4. Model Flow

Input merged feature table from Patliputra

Preprocessing (normalization, missing data interpolation)

Bandwidth selection via cross-validation

Local regression fitting using GTWR kernel

Continuous pollution surface generation

Validation using mobile sensor and station data

Export to Nalanda and Takshashila

5. Implementation Details

Entire prototype stack is in Python for rapid iteration.

Only Dwarka (the GTWR computation engine) will later migrate to C++ for performance.

All inter-module interfaces remain in Python.

Data interchange via Parquet and DuckDB.

Architecture is modular and import-safe for incremental replacement.

6. Scaling Plan
Short-term

Validate GTWR for 1–2 cities.

Generate PM2.5 maps and uncertainty rasters.

Mid-term

Automate ingestion and aggregation across cities.

Parallelize city-level runs on a multi-core cluster.

Long-term

Rewrite Dwarka core in C++ with Eigen and OpenMP.

Enable hourly national-scale model updates (50+ cities/hour).

Integrate with the routing engine as a live pollution overlay.

7. Summary
Component	Purpose	Language	Notes
Patliputra	Data ingestion	Python	Fetch and preprocess inputs
Dwarka	GTWR modeling	Python → C++	Core ML engine
Nalanda	Aggregation	Python	City-level rollups
Takshashila	Storage	Python	DB and persistence
Ujjayini	API layer	Python	Serves models and routes
8. Mission

Pollu’s mission is to enable air-aware routing: an infrastructure-independent framework that allows any city to generate hyperlocal pollution maps and low-exposure travel routes using open data and minimal sensor coverage.

It aims to make environmental exposure a first-class cost metric in navigation systems, redefining the shortest path as the healthiest path.