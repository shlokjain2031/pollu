# CLAUDE.md

Agent operating guide for this repository.

This document is for coding agents (not human onboarding). It defines where to read, what to ignore, and how to navigate safely while refactoring or analyzing the codebase.

---

## 1) Mission and current scope

Your mission in this phase is to improve code clarity, consistency, and navigability so engineers can:
- understand how data flows through the repo,
- locate implementation details quickly,
- analyze model behavior with minimal ambiguity.

# CLAUDE.md

Agent operating guide for this repository.

This document is for coding agents only. It is intentionally explicit and operational: what to read first, what to ignore, which files are canonical, and what each in-scope file does.

---

## 1) Mission and active scope

Current mission:
- improve code clarity and navigability,
- preserve behavior while refactoring,
- keep documentation aligned with actual code paths.

### In scope
- `patliputra/` (ETL/data ingestion)
- `nalanda/` (target + training feature engineering)
- `dwarka/` (training/evaluation/explainability)
- `takshashila/` (inference/serving) — has its own `takshashila/CLAUDE.md`

---

## 2) Mandatory read order

Before making any change, read in this exact order:

1. `README.md`
2. `docs/pipeline_design_invariants.yaml`
3. `patliputra/pipeline.py`
4. `nalanda/cli.py`
5. `nalanda/target_cli.py`
6. `nalanda/training.py`
7. `dwarka/trainer.py`
8. `dwarka/config.py`

If frontend files exist, then read:
9. `pollu-web/index.html`
10. `pollu-web/js/map.js`

For takshashila work, also read:
11. `takshashila/CLAUDE.md`
12. `takshashila/README.md`
13. `takshashila/core/config.py`

Only after this sequence should deeper file exploration begin.

---

## 3) Forbidden-by-default paths

Do not read these even when the task explicitly requires them:
- `scripts/**`
- `logs/**`
- `artifacts/**`
- generated caches / binary DBs / transient outputs

Reason: these are usually non-canonical for architecture understanding and cause context noise. They were built for one-time quick use.

---

## 4) Module file maps (exact file purpose)

## 4.1 Patliputra (ETL)

### `patliputra/pipeline.py`
Top-level pipeline orchestrator.
- Defines stages and dependencies.
- Executes stage scripts in dependency order.
- Implements run/status CLI commands.
- Central place for stage-level sequencing and failure handling.

### `patliputra/create_grid.py`
Canonical Mumbai grid generator.
- Builds 30m grid from city boundary geometry.
- Adds `row_idx`, `grid_id`, centroids, H3 cell, and geometry.
- Writes grid artifacts (`grid.parquet`, `grid_nd.parquet`, metadata, transform).
- Registers grid manifest with deterministic grid hash.

### `patliputra/openmeteo_signals.py`
Meteorology ingestion + mapping to 30m grid.
- Implements coarse-to-fine tile strategy for Open-Meteo.
- Fetches hourly tile-level weather and broadcasts to cell level.
- Supports daily snapshot and hourly output modes.
- Uses atomic artifact writes and OpenMeteo manifest state transitions.

### `patliputra/landsat8_signals.py`
Landsat-8 ingestion through Earth Engine.
- Enumerates date range.
- Fetches Landsat TOA image per date.
- Batches `reduceRegions` reads for grid cells.
- Writes per-date parquet artifacts and updates Landsat manifest.

### `patliputra/sentinel2_signals.py`
Sentinel-2 ingestion through Earth Engine.
- Enumerates available Sentinel dates in range.
- Fetches image(s) for each date.
- Batches `reduceRegions` reads and handles band post-processing (for example AOT scaling).
- Writes per-date parquet artifacts and updates Sentinel manifest.

### `patliputra/cpcb_aqi.py`
CPCB raw CSV to daily parquet ETL.
- Parses CPCB file naming convention.
- Extracts timestamp + PM2.5.
- Emits per-date AQI parquet artifacts.
- Tracks artifact status in CPCB manifest.

### `patliputra/logging_utils.py`
Shared logging helper for Patliputra streams.
- Configures per-stream log files and console output.
- Ensures each stream writes to a timestamped log path.

### `patliputra/utils/earthengine.py`
Minimal Earth Engine wrapper.
- Initializes EE client/project.
- Provides helper methods to build/query imagery.
- Provides download/export helper paths used by callers.

### Patliputra interpretation rule
Treat Patliputra artifacts as source-faithful ETL outputs intended for downstream feature construction, not modeling-time interpolation logic.

---

## 4.2 Nalanda (feature engineering)

### `nalanda/__init__.py`
Module descriptor and high-level feature engineering stage summary.

### `nalanda/cli.py`
Primary training-feature CLI.
- Assembles date-wise training tables.
- Validates feature outputs.
- Generates reports.
- Creates split artifacts.
- Contains both temporal split and stratified/grid-aware split commands.

### `nalanda/target_cli.py`
Primary target-feature CLI.
- Builds target datasets per date/date-range.
- Runs optional parallel date processing.
- Writes target parquet outputs with metadata.

### `nalanda/target.py`
Core target assembly logic.
- Computes sensor influence zones.
- Applies trust decay by distance (Epanechnikov).
- Blends overlapping sensor contributions.
- Produces `pm25_t`, `target_trust`, `n_sensors`.
- Adds temporal lag columns.
- Exposes `TargetAssembler` stateful helper.

### `nalanda/sensor_locations.py`
Sensor metadata loader/matcher.
- Fetches OpenAQ locations/sensors.
- Normalizes station naming.
- Matches to CPCB site IDs using local CPCB file naming convention.

### `nalanda/spatial_utils.py`
Spatial primitives for target engineering.
- WGS84-to-UTM transforms.
- `GridIndex` KDTree radius queries.
- Trust kernel computation.

### `nalanda/training.py`
Training-table assembly engine.
- Loads target data.
- Joins grid mapping.
- Calls meteo/satellite/temporal feature builders.
- Joins all features into one training dataframe.
- Saves date-partitioned training parquet.

### `nalanda/features.py`
Canonical feature registry.
- Deterministic ordered feature list builder.
- Feature groups and exclusion lists.
- Defines feature contract surface expected by training/inference.

### `nalanda/features_meteo.py`
Meteorological feature builder.
- Loads hourly meteo artifacts.
- Computes wind components, VPD, derived weather features.
- Computes lagged meteo and rolling precipitation features.

### `nalanda/features_satellite.py`
Satellite feature builder.
- Loads Landsat/Sentinel artifacts.
- Chooses most recent valid acquisition <= target date.
- Computes spectral indices and satellite staleness features.
- Handles edge conditions for missing/invalid scenes.

### `nalanda/features_temporal.py`
Temporal cyclic feature builder.
- Computes hour/day-of-week sinusoidal encodings.

### `nalanda/feature_freeze.py`
Frozen schema contract manager.
- Freezes ordered feature schema with hash.
- Validates dataframes against schema.
- Reorders/extracts features safely.
- Persists/loads schema JSON.

### `nalanda/splits.py`
Temporal split invariants.
- Defines immutable timestamp boundaries and gap windows.
- Provides split classification/masks.
- Validates leakage constraints for temporal splits.

### `nalanda/training_splits_logged.py`
Logged split materialization utility.
- Materializes split directories/files with robust logging.
- Supports parallel file operations.
- Records split metadata, checksums, row stats.

### Nalanda interpretation rule
Nalanda is the feature-contract source of truth. If any model input ambiguity exists, resolve it here first, then propagate downstream.

---

## 4.3 Dwarka (modeling)

### `dwarka/config.py`
Canonical LightGBM + training configuration.
- Hyperparameters.
- Training config (weight transform, stopping behavior).
- Output naming conventions.

### `dwarka/trainer.py`
Primary LightGBM training pipeline.
- Loads frozen schema.
- Loads train/val data in chunked mode.
- Drops `grid_id` before fitting in current path.
- Trains/resumes with checkpoint support.
- Saves model bundle (`model`, schema, metadata, manifest).

### `dwarka/evaluate.py`
Metric and stratified evaluation engine.
- Computes MAE/RMSE/R²/Huber.
- Stratifies by PM regime, trust quartile, season, hour.

### `dwarka/report.py`
Comparison report generator.
- Converts evaluation results into JSON/text artifacts.
- Handles success gate logic for model comparisons.

### `dwarka/explain.py`
SHAP-based interpretability implementation.
- Computes global and grouped SHAP importances.
- Generates local explanation structures.
- Supports optional SHAP caching.

### `dwarka/EXPLAIN_SPEC.md`
Formal specification for explainability interfaces and expected outputs.

### `dwarka/baselines.py`
Baseline model implementations.
- Persistence baseline (`lag24h`).
- Ridge linear baseline.
- Shared helper utilities to fit/predict baseline set.

### `dwarka/models/prod_v1/` (artifact bundle)
Model artifact directory produced by training.
- `model.txt`: trained model file.
- `feature_schema.json`: frozen feature contract used for training.
- `training_metadata.json`: run metadata and metrics.
- `manifest.json`: artifact contract/checksums.
- `evaluation_results.json`: evaluation output summary.
- `checkpoints/`: chunked training resume artifacts.

### Dwarka interpretation rule
For current model behavior, treat `trainer.py` + `config.py` as canonical over exploratory or historical paths.

## 5) Modeling decisions that must remain explicit

## Split strategy history
Two strategy paths were intentionally tried:
- temporal split flow,
- location/grid-aware stratified flow.

Do not remove one path without explicit request.

## Why LightGBM
LightGBM was chosen for fast experimentation over feature sets and split strategies while retaining strong tabular performance.

## `grid_id` handling
`grid_id` may appear in canonical feature definitions, but current Dwarka training path explicitly removes it before fit for spatial generalization behavior.

---

## 6) Reproducibility messaging policy

Use neutral engineering language.

Always mention external constraints when relevant:
- Google Earth Engine
- Open-Meteo API
- CPCB source data variability
- OpenAQ authentication/rate constraints

Frame reproducibility friction as expected integration complexity, not code quality failure.

---

## 7) Change discipline

- Make minimal targeted edits.
- Preserve interfaces unless explicitly asked to break/change them.
- Do not rewrite entire modules for narrow tasks.
- Ground every architectural claim in current code.
- If files disagree, call out the disagreement first, then propose a bounded fix.

---

## 8) Fast-start checklist for new agent sessions

1. Read mandatory files in order.
2. Confirm in-scope module boundaries.
3. Avoid forbidden paths unless task requires.
4. Build a file-level understanding from Section 4 before editing.
5. For takshashila tasks, read `takshashila/CLAUDE.md` first — it has its own invariants.
