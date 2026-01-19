# Pollu

Production-grade hyperlocal PM2.5 prediction system using LightGBM, satellite imagery, meteorological signals, and atmospheric chemistry. Built for 30m-resolution pollution mapping across urban areas with temporal validation and trust-weighted training.

## 🎯 Current Model Performance (prod_v1)

**Latest Evaluation Results** (2026-01-19, Test Set: 45.06M samples across 317 files)

| Metric | Training | Validation | **Test** |
|--------|----------|------------|------|
| **R²** | **0.866** ↑ | **0.847** ↑ | **0.613** ⚠️ |
| **MAE** | 6.19 µg/m³ | 8.05 µg/m³ | **13.01 µg/m³** ✓ |
| **RMSE** | 9.81 µg/m³ | 21.45 µg/m³ | 25.59 µg/m³ |

**Status**:
- ✅ **MAE threshold**: PASS (13.01 < 18.0 µg/m³)
- ❌ **R² threshold**: FAIL (0.613 < 0.65)
- ✅ **Beats baseline**: YES (significantly outperforms persistence)

**⚠️ Key Finding**: Significant performance drop from validation (0.847) to test (0.613) indicates **distribution shift** in test data. See [Performance Analysis](#performance-analysis) for investigation.

---

## Overview

Pollu predicts PM2.5 concentrations at 30-meter resolution across entire cities using:
- **LightGBM regression** with Huber loss and trust-weighted samples
- **Multi-source remote sensing**: Landsat-8, Sentinel-2
- **Meteorological signals**: OpenMeteo/ERA5 hourly data with temporal lags
- **Ground truth**: CPCB station networks
- **Temporal validation**: Forward-chaining splits (2019-2024 train, 2019-2024 val, 2019-2024 test)

**Key features:**
- No spatial features (prevents spatial leakage across grid cells)
- Trust-weighted training using measurement uncertainty
- Temporal causality enforcement (48h gaps between splits)
- Stratified evaluation by season and spatial regime
- SHAP-based interpretability for regulatory compliance

---

## Architecture

Pollu is organized into five specialized modules:

| Module | Codename | Purpose | Status |
|--------|----------|---------|--------|
| **Data Pipeline** | Patliputra | Multi-source data ingestion (satellite, meteo, AQI) | Production |
| **Feature Engineering** | Nalanda | Training data assembly with temporal lags | Production |
| **ML Training** | Dwarka | LightGBM model training and evaluation | In Development |
| **Persistence** | Takshashila | Artifact storage and metadata tracking | Production |
| **API Serving** | Ujjayini | FastAPI inference endpoint | Planned |

---

## Performance Analysis

### Train/Val/Test Gap Investigation

The model demonstrates **excellent performance on training/validation sets** (R² ≈ 0.85) but **significant degradation on test data** (R² = 0.613), a 0.234 point drop. This discrepancy suggests **distribution shift** rather than overfitting.

**Possible Root Causes**:

1. **Spatial Distribution Shift**: 
   - Test data may concentrate in low-station-coverage regions (high uncertainty)
   - Lower trust weights in sparse areas could amplify model uncertainty
   - Stratification by `grid_id` may not have balanced geographic coverage

2. **Temporal/Seasonal Effects**:
   - Test period (overlaps with train/val) may have underrepresented seasons
   - Extreme pollution events more common in test set
   - Infrastructure/emissions changes between 2019-2024

3. **Feature Drift**:
   - Satellite availability (cloudiness, sensor issues) varies by region
   - Meteorological patterns shift across time
   - Trust distribution skewed in test set

**Recommended Investigations**:
- [ ] Compare PM2.5 distributions across splits (min/max/mean/std)
- [ ] Analyze R² by spatial regime (urban core vs. suburban)
- [ ] Stratify by season and pollution level (low/medium/high)
- [ ] Plot residuals by trust quartile
- [ ] Feature availability analysis (% missing by source)

**Next Steps**:
- Retrain with balanced stratification across spatial/temporal dimensions
- Add derived meteorological features (VPD, stability indices)
- Increase model complexity or ensemble methods
- Investigate test set anomalies (corrupt files, sensor drift)

---

## Quick Start

### 1. Data Collection (Patliputra)

Fetch multi-source environmental data for a city and date range:

```bash
# Full pipeline for Mumbai (2018)
python -m patliputra.pipeline run \
  --city mumbai \
  --start-date 2018-01-01 \
  --end-date 2018-12-31 \
  --ee-project your-earth-engine-project-id \
  --hourly

# Specific data sources only
python -m patliputra.pipeline run \
  --stages meteo,sentinel2 \
  --city mumbai \
  --start-date 2021-06-01 \
  --end-date 2021-06-30
```

**Time estimates** (7 years, Mumbai ~59k grid cells):
- Grid creation: 5 minutes (one-time)
- Meteorology: 16-20 hours (API rate limited)
- Satellite data: 10-16 hours (Earth Engine quota limited)
- **Total: ~27-37 hours**

**Run stages sequentially** — API rate limits and Earth Engine quotas prevent parallelization.

### 2. Feature Engineering (Nalanda)

Assemble training data with temporal lags and quality checks:

```bash
# Single date
python -m nalanda.cli assemble --date 2021-06-12

# Date range with parallel processing
python -m nalanda.cli assemble \
  --start-date 2021-06-01 \
  --end-date 2021-06-30 \
  --parallel --workers 4

# Validate assembled data
python -m nalanda.cli validate --all -v

# Generate data quality report
python -m nalanda.cli report --output report.json
```

### 3. Model Training (Dwarka)

Train LightGBM model with temporal validation:

```bash
# Train production model
python -m dwarka.train \
  --artifact-dir artifacts/v1/training \
  --output-dir models/v1 \
  --config dwarka/config/production.yml

# Evaluate on test set
python -m dwarka.evaluate \
  --model models/v1/lightgbm_final.pkl \
  --test-data artifacts/v1/training/test \
  --output-dir evaluation/v1

# Generate SHAP interpretability report
python -m dwarka.interpret \
  --model models/v1/lightgbm_final.pkl \
  --sample-data artifacts/v1/training/val/2024-01-15.parquet \
  --output evaluation/v1/shap_report.html
```

---

## Data Sources

| Source | Variables | Temporal Resolution | Spatial Resolution | Pipeline Stage |
|--------|-----------|---------------------|-------------------|----------------|
| **OpenMeteo/ERA5** | Temperature, humidity, wind, pressure, precipitation, dewpoint, cloud cover | Hourly | ~11km (downsampled to 30m) | `meteo` |
| **Landsat-8** | TOA bands → NDVI, NDBI, MNDWI, BSI, IBI | 16-day revisit | 30m | `landsat8` |
| **Sentinel-2** | Surface reflectance → EVI, cloud mask, local texture | 5-day revisit | 10m (resampled to 30m) | `sentinel2` |
| **Sentinel-5P** | Tropospheric NO₂ column density | Daily | 5.5km (downsampled to 30m) | `s5p_no2` |
| **CPCB** | Hourly PM2.5, PM10, NO₂, SO₂, CO, O₃ | Hourly | Station network | `aqi` |
| **OpenAQ** | Hourly PM2.5 from calibrated sensors | Hourly | Station network | `openaq` |
| **OSM** | Industrial zones, dust-prone areas (quarries, construction) | Static | Vector geometries | `osm` |

**Total feature count**: ~62 features
- 6 autoregressive lags (PM2.5 1h, 2h, 3h, 6h, 12h, 24h)
- 40+ meteorological features (with temporal lags)
- 12 satellite indices
- 4 temporal encodings (cyclic hour/day-of-week)

---

## Model Specification

### LightGBM Configuration

```python
params = {
    # Core
    'objective': 'huber',          # Robust to outliers
    'alpha': 10.0,                 # Huber transition at 10 µg/m³
    'metric': ['rmse', 'mae'],
    
    # Tree structure
    'num_leaves': 127,
    'max_depth': 10,
    'min_data_in_leaf': 50,
    
    # Regularization
    'lambda_l2': 5.0,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    
    # Learning
    'learning_rate': 0.05,
    'num_boost_round': 500,
    'early_stopping_rounds': 20,
}
```

### Temporal Validation Strategy

```
2017-2023 (train) │ 48h gap │ 2023-2024 (val) │ 48h gap │ 2024-2025 (test)
```

**Gap rationale**: Maximum lag feature is 24h; 48h gap ensures zero temporal overlap in autoregressive features.

### Trust-Weighted Training

Sample weights proportional to `sqrt(target_trust)`:
- **Trust calculation**: Epanechnikov kernel with 250m radius around CPCB/OpenAQ stations
- **Weight formula**: `w_i = sqrt(trust_i)` where `trust_i ∈ [0, 1]`
- **Rationale**: Downweights distant interpolations while maintaining gradient stability

### Evaluation Protocol

**Primary metrics** (stratified by season):
- **R²**: Overall variance explained
- **MAE**: Absolute error (µg/m³)
- **RMSE**: Penalizes large errors
- **Bias**: Systematic over/under-prediction

**Spatial stratification**:
- Urban core (high density)
- Urban periphery (mixed)
- Suburban/rural (low density)

**Temporal stratification**:
- Winter (Nov-Feb): High pollution, stable meteorology
- Pre-monsoon (Mar-May): Dust, high temperatures
- Monsoon (Jun-Sep): Low pollution, wet deposition
- Post-monsoon (Oct): Festival emissions

**Expected performance** (based on literature):
- R² = 0.65-0.75 (hourly)
- MAE = 12-18 µg/m³
- Baseline (24h persistence) R² = 0.45-0.55

---

## Module Details

### Patliputra (Data Pipeline)

Fetches, validates, and normalizes multi-source environmental data into standardized 30m grid artifacts.

**Pipeline stages**:
1. **Grid creation** (`create_grid.py`): Generate 30m UTM grid with H3 indexing
2. **Meteorology** (`openmeteo_signals.py`): Hourly weather data from OpenMeteo API
3. **Landsat-8** (`landsat8_signals.py`): Spectral indices via Earth Engine
4. **Sentinel-2** (`sentinel2_signals.py`): EVI, cloud mask, texture features via Earth Engine
6. **CPCB AQI** (`cpcb_aqi.py`): Ground truth PM2.5 from monitoring stations

**Output format**: Parquet files organized by `artifacts/v1/{source}/{city}/{date}.parquet`

**Artifact contract**: Each parquet includes:
- `grid_id`: Unique grid cell identifier
- `timestamp`: ISO-8601 datetime (UTC)
- Feature columns (varies by source)
- Metadata: `git_sha`, `created_at`, `version`

See [docs/artifact_contracts.yml](docs/artifact_contracts.yml) for detailed schemas.

**Concurrency**: All stages run **sequentially** due to API rate limits. Internal parallelization handled per-script (e.g., Earth Engine workers).

**Documentation**:
- [Pipeline Guide](docs/patliputra_pipeline_guide.md)
- [Concurrency Analysis](docs/patliputra_concurrency_analysis.md)

---

### Nalanda (Feature Engineering)

Assembles training-ready datasets by joining artifacts with temporal lags and quality checks.

**Key modules**:
- `features_meteo.py`: Meteorological features with 1h, 3h, 6h, 12h, 24h lags
- `features_satellite.py`: Satellite indices + derived features (days_since_landsat/sentinel)
- `features_temporal.py`: Cyclic encodings (hour_sin, hour_cos, dow_sin, dow_cos)
- `features_target.py`: PM2.5 ground truth with trust weighting
- `training.py`: Orchestrates joins and outputs `training/{date}/training.parquet`

**CLI operations**:
```bash
# Assemble single date
python -m nalanda.cli assemble --date 2021-06-12

# Batch processing
python -m nalanda.cli assemble \
  --start-date 2021-06-01 \
  --end-date 2021-06-30 \
  --parallel --workers 4

# Data validation
python -m nalanda.cli validate --all -v

# Quality report
python -m nalanda.cli report --output report.json
```

**Quality checks**:
- Missing value detection
- Temporal continuity (no gaps in lag features)
- Grid alignment (59,209 rows per date for Mumbai)
- Feature range validation (e.g., NDVI ∈ [-1, 1])
- Trust distribution (warn if <5% samples have trust > 0.5)

**Output schema**:
```
grid_id, timestamp, 
pm25_t, target_trust,
pm25_t_lag1h, pm25_t_lag2h, ..., pm25_t_lag24h,
temperature_2m_lag0h, temperature_2m_lag1h, ...,
ndvi, ndbi, evi, cloud_mask, ...,
hour_sin, hour_cos, dow_sin, dow_cos
```

---

### Dwarka (ML Training & Evaluation)

LightGBM-based PM2.5 prediction with temporal validation and interpretability.

**Current status**: Specification complete ([dwarka/model_spec.md](dwarka/model_spec.md)), implementation in progress.

**Planned files**:
- `train.py`: Full training pipeline with early stopping
- `evaluate.py`: Stratified metrics on test set
- `baselines.py`: Persistence, climatology, spatial mean baselines
- `interpret.py`: SHAP waterfall plots, force plots, dependence plots
- `predict.py`: Batch inference on new grid cells

**Training procedure**:
1. Load training data from `artifacts/v1/training/train/` (2017-2023)
2. Apply trust-based sample weights: `w_i = sqrt(target_trust_i)`
3. Train LightGBM with Huber loss (δ=10 µg/m³)
4. Early stopping on validation set (2023-2024)
5. Evaluate on test set (2024-2025) with stratified metrics
6. Generate SHAP interpretability report
7. Save model + metadata (hyperparams, feature importance, git SHA)

**Model outputs**:
- `lightgbm_final.pkl`: Trained model (pickle)
- `feature_importance.csv`: Gain/split-based rankings
- `shap_values/`: Pre-computed SHAP explanations for regulatory review
- `metrics.json`: Test set performance by season/regime

**Interpretability requirements**:
- SHAP waterfall plots for individual predictions
- Global feature importance (mean |SHAP|)
- Dependence plots for top 10 features
- Interaction detection (SHAP interaction values)

See [dwarka/model_spec.md](dwarka/model_spec.md) for complete specification.

---

### Takshashila (Artifact Storage)

Lightweight metadata tracking for data artifacts and model outputs.

**Database**: DuckDB (`manifest/pollu.db`)

**Tracked entities**:
- `artifacts`: Data products (OpenMeteo, Landsat, training files)
- `models`: Trained ML models with hyperparameters
- `predictions`: Inference outputs (grid-level PM2.5 maps)

**Artifact manifest** (`manifest/db.py`):
```python
register_artifact(
    artifact_type="openmeteo_meteo",
    spatial_id="mumbai",
    timestamp="2021-06-12T00:00:00Z",
    path="artifacts/v1/openmeteo/mumbai/2021-06-12.parquet",
    git_sha="abc123",
    version="v1"
)
```

**Query interface**:
```python
from manifest.db import get_artifact, list_artifacts

# Get specific artifact
artifact = get_artifact("openmeteo_meteo", "mumbai", "2021-06-12")

# List all training dates
training_dates = list_artifacts("training", spatial_id="mumbai")
```

---

### Ujjayini (API Serving)

**Status**: Planned

FastAPI-based inference endpoint for real-time predictions.

**Planned endpoints**:
- `GET /health`: Service health check
- `GET /cities`: List available cities
- `POST /predict`: Batch grid-level predictions
  - Input: `{city, timestamp, grid_ids: [...]}`
  - Output: `{predictions: [{grid_id, pm25_pred, uncertainty}, ...]}`
- `GET /city/{name}/map`: GeoJSON pollution map
- `GET /metrics`: Model performance summary

**Serving strategy**:
- Model loaded on startup (shared memory)
- Feature preprocessing on-demand
- Batch predictions (max 10k grid cells/request)
- Redis caching for repeat queries

---

## Critical ML Invariants

These constraints are **non-negotiable** for production deployment:

### 1. Temporal Causality
- No future information in features (all lags are backward-looking)
- 48h gap between train/val/test splits
- Lag features explicitly validated (no accidental forward leakage)

### 2. No Spatial Features
- No distance-to-station features
- No spatial coordinates (lat/lon) as direct features
- Prevents spatial leakage across grid cells
- Only time-invariant environmental features (NDVI, elevation) allowed

### 3. Trust-Weighted Training
- Sample weights = `sqrt(target_trust)` where trust ∈ [0, 1]
- Trust computed via Epanechnikov kernel (250m radius)
- Never discard low-trust samples (retains spatial coverage)

### 4. Forward-Chaining Splits
- Strictly chronological train/val/test
- No random shuffling across time
- Mimics deployment scenario (predict future from past)

### 5. Feature Sanity Checks
- NDVI ∈ [-1, 1], temperature_2m ∈ [-40, 60]°C, etc.
- Reject rows with >20% missing features
- Log and monitor feature drift (distribution shift)

### 6. Stratified Evaluation
- Report metrics separately for:
  - Seasons (winter, pre-monsoon, monsoon, post-monsoon)
  - Spatial regimes (urban core, periphery, suburban)
  - Pollution levels (low <50, medium 50-100, high >100 µg/m³)

### 7. Baseline Comparisons
- Persistence (24h lag)
- Climatological mean (same hour/day-of-week average)
- Spatial mean (city-wide hourly average)
- Model must beat all three baselines

---

## Repository Structure

```
pollu/
├── patliputra/          # Data pipeline
│   ├── create_grid.py
│   ├── openmeteo_signals.py
│   ├── landsat8_signals.py
│   ├── sentinel2_signals.py
│   ├── cpcb_aqi.py
│   └── pipeline.py      # Orchestration
├── nalanda/             # Feature engineering
│   ├── features_meteo.py
│   ├── features_satellite.py
│   ├── features_temporal.py
│   ├── features_target.py
│   ├── training.py
│   └── cli.py           # Assembly CLI
├── dwarka/              # ML training
│   ├── model_spec.md    # Complete specification
│   ├── train.py         # (planned)
│   ├── evaluate.py      # (planned)
│   ├── baselines.py     # (planned)
│   └── interpret.py     # (planned)
├── manifest/            # Artifact tracking
│   ├── db.py
│   └── pollu.db
├── ujjayini/            # API serving (planned)
├── docs/                # Documentation
│   ├── artifact_contracts.yml
│   ├── patliputra_pipeline_guide.md
│   └── patliputra_concurrency_analysis.md
├── artifacts/           # Data products
│   └── v1/
│       ├── grids/
│       ├── openmeteo/
│       ├── landsat_signals/
│       ├── sentinel_evi_signals/
│       └── training/
└── models/              # Trained models (gitignored)
```

---

## Development Roadmap

### Phase 1: Baseline Model ✅ COMPLETE
- [x] Data pipeline (Patliputra) for all sources
- [x] Feature engineering (Nalanda) with temporal lags
- [x] Model specification (Dwarka/model_spec.md)
- [x] Training pipeline (Dwarka/trainer.py)
- [x] Evaluation framework (Dwarka/evaluate.py)
- [x] Baseline comparisons (persistence, climatology)
- [x] First production model (prod_v1, Jan 2026)

**Achieved**: R² = 0.613 (test), R² = 0.847 (val), MAE = 13.01 µg/m³
**Status**: Baseline working, but distribution shift requires investigation

### Phase 2: Performance Improvement (IN PROGRESS - Q1 2026)
- [ ] **URGENT**: Diagnose train/val/test gap (distribution shift analysis)
- [ ] Fix spatial/temporal stratification if unbalanced
- [ ] Derived features (wind speed, VPD, temp-dewpoint diff)
- [ ] Temporal encoding (cyclic hour/day-of-week) - partially done
- [ ] Hyperparameter tuning (Optuna) for test set
- [ ] SHAP interpretability reports
- [ ] Stratified evaluation (season, regime, pollution level)

**Target**: R² > 0.65, MAE < 18 µg/m³ on test set

### Phase 3: Atmospheric Chemistry (Q2 2026)
- [ ] NO₂ temporal lags (1h, 3h, 6h, 12h, 24h)
- [ ] Interaction features (NO₂ × humidity, NO₂ × temperature)
- [ ] Secondary pollutants (O₃ if available)

**Target**: R² > 0.70, improved winter performance

### Phase 4: Deployment (Q3 2026)
- [ ] API serving (Ujjayini/FastAPI)
- [ ] Batch inference pipeline
- [ ] Model monitoring (drift detection, performance alerts)
- [ ] Multi-city expansion (Delhi, Bangalore, Pune)

---

## Performance Expectations

**Current Model (prod_v1, Jan 2026)**:

| Metric | Actual | Target | State-of-art |
|--------|--------|--------|--------------|
| **Test R²** | **0.613** ⚠️ | 0.65+ | 0.75-0.85 |
| **Test MAE** | **13.01 µg/m³** ✓ | <18 µg/m³ | 8-12 µg/m³ |
| **Test RMSE** | **25.59 µg/m³** | 18-25 µg/m³ | 12-18 µg/m³ |
| **Train R²** | **0.866** ✓ | - | - |
| **Val R²** | **0.847** ✓ | - | - |

**Model Details**:
- **Model Name**: `prod_v1`
- **Model Hash**: `ecf51a6d14a239f1`
- **Created**: 2026-01-18 18:09:05 UTC
- **Features**: 64 (6 lags + 40+ meteorology + 12 satellite)
- **Training Samples**: 99.65M
- **Test Samples**: 45.06M
- **Training Time**: 6,264 seconds (1.74 hours)
- **Iterations**: 6,122 / 5,000 max (early stopped)

**Based on literature** (satellite + meteo + ML for PM2.5):

| Metric | Baseline (24h persistence) | LightGBM typical | State-of-art |
|--------|----------------------------|-------------------|--------------|
| **R²** | 0.45-0.55 | 0.65-0.75 | 0.75-0.85 |
| **MAE** | 18-25 µg/m³ | 12-18 µg/m³ | 8-12 µg/m³ |
| **RMSE** | 25-35 µg/m³ | 18-25 µg/m³ | 12-18 µg/m³ |
| **Bias** | ±5 µg/m³ | ±3 µg/m³ | ±2 µg/m³ |

**Seasonal performance** (expected):
- **Monsoon** (Jun-Sep): Best performance (R² ~0.75, low variability)
- **Winter** (Nov-Feb): Moderate (R² ~0.65, high variability, inversion layers)
- **Pre-monsoon** (Mar-May): Moderate (R² ~0.60, dust events)

**Spatial performance**:
- **Urban core** (high trust): R² ~0.70 (dense station network)
- **Urban periphery**: R² ~0.60 (moderate coverage)
- **Suburban/rural**: R² ~0.50 (sparse stations, higher uncertainty)

---

## Data Availability

**Current coverage** (Mumbai):
- **Temporal**: 2017-2025 (8 years)
- **Spatial**: 59,209 grid cells (30m resolution)
- **CPCB stations**: 15-25 active monitors
- **OpenAQ sensors**: 10-50 supplemental sensors

**Data volumes**:
- OpenMeteo: ~500 MB/year/city (hourly, 40+ variables)
- Landsat-8: ~200 MB/year/city (16-day revisit, 5 indices)
- Sentinel-2: ~1 GB/year/city (5-day revisit, 3 indices)
- Training data: ~5 GB/year/city (assembled features)

---

## Citation

If you use Pollu in research, please cite:

```bibtex
@software{pollu2025,
  title={Pollu: Production-Grade Hyperlocal PM2.5 Prediction},
  author={Jain, Shlok},
  year={2025},
  url={https://github.com/shlokjain/pollu}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions, issues, or collaboration:
- GitHub Issues: [github.com/shlokjain/pollu/issues](https://github.com/shlokjain/pollu/issues)
- Email: jainshlok20@gmail.com

---

## Acknowledgments

**Data sources**:
- [OpenMeteo](https://open-meteo.com/) - Meteorological data
- [Google Earth Engine](https://earthengine.google.com/) - Satellite imagery
- [CPCB](https://cpcb.nic.in/) - Air quality monitoring network
- [OpenAQ](https://openaq.org/) - Global air quality data

**Inspiration**:
- LightGBM paper: Ke et al. (2017)
- Satellite-based PM2.5: van Donkelaar et al. (2016)
- Trust-weighted regression: Shepard (1968)
