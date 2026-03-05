# Pollu

Pollu is a PM2.5 modeling and serving codebase for hyperlocal city-scale pollution intelligence.

You can find end-to-end documentation on the machine learning specific part of the project here: [High Resolution PM2.5 Estimation at Hourly Temporal and 30m Spatial Resolution Using LightGBM
](https://gist.github.com/shlokjain2031/f37a5af9108d0ffd40f6e3d0e25c4b27)

## Scope of this README

This document covers:
- Patliputra (ETL/data ingestion)
- Nalanda (feature engineering)
- Dwarka (model training/evaluation)
- pollu-web (frontend)
- Takshashila (inference/serving) — see [takshashila/README.md](takshashila/README.md) for full details

---

## System shape (high level)

There are two major paths in this repo:

1. **Model-building path (offline):**
  Patliputra -> Nalanda -> Dwarka

2. **Product path (online):**
  pollu-web -> Takshashila

In other words, Patliputra/Nalanda/Dwarka exist to produce and improve the model artifacts that backend serving uses.

## Repository modules

## 1) Patliputra — ETL pipelines

**Purpose**
- Ingest raw environmental data from multiple sources.
- Keep artifacts as close to source truth as possible for downstream flexibility.

**Entrypoint**
- [patliputra/pipeline.py](patliputra/pipeline.py)

### Pipeline stages
1. **Grid creation**
  - [patliputra/create_grid.py](patliputra/create_grid.py)
  - Builds Mumbai 30m x 30m canonical grid.

2. **Meteorology (Open-Meteo)**
  - [patliputra/openmeteo_signals.py](patliputra/openmeteo_signals.py)
  - Coarse-to-fine tile mapping strategy: query coarse reanalysis tiles, broadcast to 30m grid cells.

3. **Satellite (Earth Engine)**
  - Landsat-8: [patliputra/landsat8_signals.py](patliputra/landsat8_signals.py)
  - Sentinel-2: [patliputra/sentinel2_signals.py](patliputra/sentinel2_signals.py)
  - Landsat values sampled at 30m grid points.
  - Sentinel native 10m values aggregated/sampled to 30m grid scale.

4. **CPCB AQI ground truth**
  - [patliputra/cpcb_aqi.py](patliputra/cpcb_aqi.py)
  - Parses historical station CSVs into daily parquet artifacts.

### Design notes
- Pipeline orchestration handles dependency order.
- Stage-level manifests track `producing -> ok/error` lifecycle.
- Output is artifact-first; feature synthesis is deferred to Nalanda.

---

## 2) Nalanda — feature engineering

**Purpose**
- Transform Patliputra artifacts into model-ready datasets.
- Build both target labels and training feature matrices.

**Entrypoints**
- Training assembly CLI: [nalanda/cli.py](nalanda/cli.py)
- Target assembly CLI: [nalanda/target_cli.py](nalanda/target_cli.py)

### Target engineering
- Core logic: [nalanda/target.py](nalanda/target.py)
- Uses sensor influence zones + trust weighting to build `pm25_t`.
- Temporal lag columns are created for autoregressive learning.
- Trust is distance-decayed (Epanechnikov kernel).

### Training feature engineering
- Assembly orchestrator: [nalanda/training.py](nalanda/training.py)
- Meteo features/lags: [nalanda/features_meteo.py](nalanda/features_meteo.py)
- Satellite indices/staleness: [nalanda/features_satellite.py](nalanda/features_satellite.py)
- Temporal encodings: [nalanda/features_temporal.py](nalanda/features_temporal.py)
- Canonical feature registry: [nalanda/features.py](nalanda/features.py)
- Feature freeze contract: [nalanda/feature_freeze.py](nalanda/feature_freeze.py)
- Split materialization/logging: [nalanda/training_splits_logged.py](nalanda/training_splits_logged.py)

### Split strategy note
Two split strategies were explored during iteration:
- temporal split strategy,
- location/grid-aware stratified strategy.

LightGBM was chosen so iteration cycles could be fast while testing these alternatives. In our experiments, location-aware splitting gave better behavior for the intended generalization checks, so both paths exist in code history and tooling.

---

## 3) Dwarka — modeling and evaluation

**Purpose**
- Train and evaluate PM2.5 model artifacts used by backend serving.

**Primary training path**
- [dwarka/trainer.py](dwarka/trainer.py)
- [dwarka/config.py](dwarka/config.py)

### What Dwarka does
- Enforces frozen feature schema.
- Loads split data in chunked mode for scale.
- Trains LightGBM with trust-weighted samples.
- Persists model bundle + metadata + manifest.
- Provides stratified evaluation and reporting:
  - [dwarka/evaluate.py](dwarka/evaluate.py)
  - [dwarka/report.py](dwarka/report.py)
- Provides interpretability workflow:
  - [dwarka/explain.py](dwarka/explain.py)
  - [dwarka/EXPLAIN_SPEC.md](dwarka/EXPLAIN_SPEC.md)

### `grid_id` and leakage concerns
You may notice `grid_id` appears in Nalanda’s canonical feature listing.
In Dwarka training flow, this field is explicitly removed before fitting to reduce location memorization risk during model learning.
### Design decisions: constant features
`local_ndvi_std` (spatial neighborhood NDVI standard deviation) appears in the frozen feature schema but is computed as a constant `0.0` in [takshashila/core/feature_assembler.py](takshashila/core/feature_assembler.py). This is intentional: the feature was frozen with this constant value during training, so inference must provide the same value to ensure model consistency. This is not a bug or incomplete TODO — it's a documented design choice to maintain the train-serving contract.
## Quick navigation map (human-friendly)

If you are new, read in this order:
1. This file.
2. [docs/pipeline_design_invariants.yaml](docs/pipeline_design_invariants.yaml)
3. [patliputra/pipeline.py](patliputra/pipeline.py)
4. [nalanda/cli.py](nalanda/cli.py) and [nalanda/target_cli.py](nalanda/target_cli.py)
5. [dwarka/trainer.py](dwarka/trainer.py)

---

## Practical note

For local runs, prefer running module CLIs (`python -m ...`) and validate one stage at a time. This makes quota/rate-limit failures easier to isolate and diagnose.