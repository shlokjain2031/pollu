# LightGBM PM2.5 Model Specification

A complete ML specification for 30m-resolution PM2.5 prediction using LightGBM with temporal validation, trust-weighted training, and interpretability requirements. Covers data splits, feature engineering, hyperparameters, evaluation protocol, and incremental roadmap.

## Problem Formulation

| Aspect | Specification |
|--------|---------------|
| **Task** | Regression: predict PM2.5 (µg/m³) at 30m grid cells |
| **Unit of prediction** | `(grid_id, timestamp)` — hourly |
| **Target** | `pm25_t` — trust-weighted average from CPCB + OpenAQ |
| **Training samples** | ~59k grid cells × 24 hours × ~2500 days ≈ 3.5B potential rows |
| **Effective training** | Chunked by temporal windows; not all in RAM |

---

## Temporal Split Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                        2017-2025 Timeline                           │
├─────────────────────────────────────────────────────────────────────┤
│  TRAIN (2017-01 to 2023-06)  │ GAP │  VAL (2023-07 to 2024-06)     │
│                              │ 48h │                                │
├──────────────────────────────┴─────┴────────────────────────────────┤
│                              │ GAP │  TEST (2024-07 to 2025-06)     │
│                              │ 48h │                                │
└─────────────────────────────────────────────────────────────────────┘
```

**Split parameters:**
- **Train**: 2017-01-01 to 2023-06-28 (~6.5 years)
- **Gap**: 48 hours (2 days) — eliminates `pm25_t_lag24h` leakage
- **Validation**: 2023-07-01 to 2024-06-28 (~1 year)
- **Gap**: 48 hours
- **Test**: 2024-07-01 to 2025-06-30 (~1 year)

**Seasonal stratification for reporting:**
- **Winter** (Nov-Feb): High pollution, stable meteorology
- **Pre-monsoon** (Mar-May): Dust, high temperatures
- **Monsoon** (Jun-Sep): Low pollution, wet deposition
- **Post-monsoon** (Oct): Transition, festival emissions

**Gap day rationale:**
- Maximum lag feature is 24h
- 48h gap ensures no temporal overlap in autoregressive features
- This is a leakage control mechanism, not decoration

---

## Feature Groups & Initial Feature Set

### Group A: Autoregressive (Lags) — 6 features

| Feature | Description | Include in Baseline |
|---------|-------------|---------------------|
| `pm25_t_lag1h` | 1-hour lag | ✓ |
| `pm25_t_lag2h` | 2-hour lag | ✓ |
| `pm25_t_lag3h` | 3-hour lag | ✓ |
| `pm25_t_lag6h` | 6-hour lag | ✓ |
| `pm25_t_lag12h` | 12-hour lag | ✓ |
| `pm25_t_lag24h` | 24-hour lag | ✓ |

**Note**: These are the strongest predictors. Persistence baseline = `pm25_t_lag24h` alone.

### Group B: Meteorological — 40+ features

| Feature | Lags | Include |
|---------|------|---------|
| `temperature_2m` | 0, 1, 3, 6, 12, 24h | ✓ |
| `relative_humidity_2m` | 0, 1, 3, 6, 12, 24h | ✓ |
| `dewpoint_2m` | 0, 1, 3, 6, 12, 24h | ✓ |
| `surface_pressure` | 0, 1, 3, 6, 12, 24h | ✓ |
| `wind_u_10m` | 0, 1, 3, 6, 12, 24h | ✓ |
| `wind_v_10m` | 0, 1, 3, 6, 12, 24h | ✓ |
| `cloud_cover` | 0, 6, 24h | ✓ |
| `vpd` | 0h only | ✓ |
| `temp_dewpoint_diff` | 0h only | ✓ |
| `precip_last_1h` | — | ✓ |
| `precip_last_6h` | — | ✓ |
| `precip_last_24h` | — | ✓ |

### Group C: Satellite Indices — 12 features

| Feature | Source | Include |
|---------|--------|---------|
| `ndvi` | Landsat-8 | ✓ |
| `ndbi` | Landsat-8 | ✓ |
| `mndwi` | Landsat-8 | ✓ |
| `bsi` | Landsat-8 | ✓ |
| `ibi` | Landsat-8 | ✓ |
| `evi` | Sentinel-2 | ✓ |
| `cloud_mask` | Sentinel-2 | ✓ |
| `cloud_fraction_30m` | Sentinel-2 | ✓ |
| `local_ndvi_std` | Sentinel-2 | ✓ |
| `days_since_landsat` | Derived | ✓ (add this) |
| `days_since_sentinel` | Derived | ✓ (add this) |

**Exclude initially:**
- Raw TOA bands (B1-B11) — let indices carry signal first
- `ebbi` — redundant with `ndbi`, `ibi`

### Group D: Temporal Encoding — 4 features

| Feature | Encoding | Include |
|---------|----------|---------|
| `hour_sin` | sin(2π × hour/24) | ✓ |
| `hour_cos` | cos(2π × hour/24) | ✓ |
| `dow_sin` | sin(2π × day_of_week/7) | ✓ |
| `dow_cos` | cos(2π × day_of_week/7) | ✓ |

**Exclude initially:**
- `month` — captured by meteorology
- Raw `hour`, `day_of_week` — use cyclic encoding

### Group E: Grid Metadata — EXCLUDE

| Feature | Reason for Exclusion |
|---------|---------------------|
| `grid_id` | Categorical memorization risk |
| `row_idx` | Same as grid_id |
| `x_m`, `y_m` | Spatial coordinate leakage |
| `centroid_lat`, `centroid_lon` | Same issue |
| `h3_cell` | Categorical with spatial structure |

**Rationale**: These features allow LightGBM to memorize spatial patterns without learning transferable relationships. Satellite indices encode spatial characteristics implicitly.

### Group F: Target Metadata — EXCLUDE from features

| Feature | Use |
|---------|-----|
| `target_trust` | Sample weight, not feature |
| `n_sensors` | Evaluation stratification only |

---

## LightGBM Configuration

```python
lgb_params = {
    # ─────────────────────────────────────────────────────────────
    # OBJECTIVE & LOSS
    # ─────────────────────────────────────────────────────────────
    'objective': 'huber',          # Robust to outliers in high-PM regime
    'huber_delta': 10.0,           # δ=10 µg/m³: quadratic for |e|<10, linear beyond
                                   # Matches sensor disagreement scale (±5-15 µg/m³)
    'metric': ['mae', 'rmse', 'huber'],  # Track all three
    
    # ─────────────────────────────────────────────────────────────
    # TREE STRUCTURE (prevent grid_id memorization)
    # ─────────────────────────────────────────────────────────────
    'num_leaves': 127,             # 2^7 - 1: moderate complexity
                                   # Rule: num_leaves < 2^max_depth to avoid full trees
    'max_depth': 10,               # Hard cap on tree depth
                                   # Deep trees risk memorizing (lat, lon) patterns
    'min_data_in_leaf': 100,       # Require 100+ samples per leaf
                                   # Prevents fitting to individual grid cells
    'min_sum_hessian_in_leaf': 10.0,  # Hessian regularization
                                   # For Huber: Hessian=1 in quadratic region
                                   # So this requires ~10 samples in quadratic regime
    
    # ─────────────────────────────────────────────────────────────
    # LEARNING RATE & ITERATIONS
    # ─────────────────────────────────────────────────────────────
    'learning_rate': 0.05,         # Conservative: allows early stopping to find optimum
                                   # With 5000 max iterations, effective capacity = 250 "full trees"
    'num_iterations': 5000,        # Upper bound; early stopping will terminate earlier
    'early_stopping_rounds': 100,  # Stop if no improvement for 100 rounds
                                   # At lr=0.05, this is ~5 effective trees of patience
    
    # ─────────────────────────────────────────────────────────────
    # SAMPLING (reduce overfitting, speed up training)
    # ─────────────────────────────────────────────────────────────
    'feature_fraction': 0.7,       # Use 70% of features per tree
                                   # Decorrelates trees, reduces feature co-adaptation
    'bagging_fraction': 0.8,       # Use 80% of rows per tree
    'bagging_freq': 1,             # Resample every iteration
                                   # Combined: each tree sees 0.7 × 0.8 = 56% of data
    
    # ─────────────────────────────────────────────────────────────
    # REGULARIZATION
    # ─────────────────────────────────────────────────────────────
    'lambda_l1': 0.0,              # No L1 initially (feature selection via importance)
    'lambda_l2': 5.0,              # L2 on leaf weights: ||w||² penalty
                                   # Shrinks leaf values toward zero
                                   # Mathematically: leaf_value = sum(grad) / (sum(hess) + λ₂)
    'min_gain_to_split': 0.01,     # Minimum gain to create a split
                                   # Prevents tiny splits that overfit noise
    'path_smooth': 1.0,            # Smoothing on leaf values (LightGBM 4.0+)
    
    # ─────────────────────────────────────────────────────────────
    # HISTOGRAM BINNING
    # ─────────────────────────────────────────────────────────────
    'max_bin': 255,                # Default; sufficient for continuous features
    'min_data_in_bin': 10,         # Minimum samples per histogram bin
    
    # ─────────────────────────────────────────────────────────────
    # MONOTONIC CONSTRAINTS (none initially)
    # ─────────────────────────────────────────────────────────────
    # 'monotone_constraints': [],  # Add after SHAP inspection
    # Candidates for later:
    #   precip_last_24h: -1 (more rain → lower PM2.5)
    #   wind_speed (if added): -1 (higher dispersion → lower PM2.5)
    
    # ─────────────────────────────────────────────────────────────
    # PERFORMANCE
    # ─────────────────────────────────────────────────────────────
    'num_threads': -1,             # Use all cores
    'device_type': 'cpu',          # CPU training (GPU not needed for tabular)
    'verbosity': 1,                # Show progress
    'seed': 42,                    # Reproducibility
    'deterministic': True,         # Reproducible splits
    'force_row_wise': True,        # Better for wide datasets with sampling
}
```

### Mathematical Justification for Key Parameters

**Huber Loss (δ=10):**
$$
L_\delta(y, \hat{y}) = 
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

- Quadratic for errors < 10 µg/m³ (accurate where sensors agree)
- Linear for errors > 10 µg/m³ (robust to outliers in high-PM events)
- δ=10 matches sensor disagreement scale

**Effective Model Capacity:**
$$
\text{Capacity} \approx \frac{\text{num\_iterations} \times \text{learning\_rate}}{\text{regularization}} = \frac{5000 \times 0.05}{5.0} \approx 50 \text{ effective trees}
$$

With early stopping at ~1000-2000 iterations, expect 20-40 effective trees.

**Leaf Regularization:**
$$
\text{leaf\_value} = \frac{\sum_{i \in \text{leaf}} g_i}{\sum_{i \in \text{leaf}} h_i + \lambda_2}
$$

Where $g_i$ = gradient, $h_i$ = Hessian. λ₂=5.0 shrinks predictions toward zero, especially in leaves with few samples.

---

## Training Procedure

```python
import lightgbm as lgb
import numpy as np

def train_model(
    train_df,
    val_df,
    feature_cols,
    target_col='pm25_t',
    weight_col='target_trust',
    lgb_params=LGB_PARAMS,
):
    """
    Train LightGBM with trust-weighted samples.
    
    Weight transformation: sqrt(target_trust)
    - Preserves low-trust samples (don't discard)
    - Reduces influence of very-low-trust cells
    - sqrt compresses the range [0, 1] → [0, 1] less aggressively
    """
    # Prepare datasets
    train_weights = np.sqrt(train_df[weight_col].values)
    val_weights = np.sqrt(val_df[weight_col].values)
    
    train_ds = lgb.Dataset(
        train_df[feature_cols],
        label=train_df[target_col],
        weight=train_weights,
        feature_name=feature_cols,
    )
    
    val_ds = lgb.Dataset(
        val_df[feature_cols],
        label=val_df[target_col],
        weight=val_weights,
        reference=train_ds,
    )
    
    # Train with early stopping
    model = lgb.train(
        lgb_params,
        train_ds,
        valid_sets=[train_ds, val_ds],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    
    return model
```

**Chunked Training for Large Data:**

```python
def train_chunked(
    date_range,
    chunk_size_days=90,  # 3-month chunks
    ...
):
    """
    Train on temporal chunks to avoid full RAM materialization.
    
    Approach:
    1. Load 90-day chunk from Parquet
    2. Create LightGBM Dataset (histogram binning)
    3. Continue training (init_model=previous)
    4. Discard raw data, keep histograms
    """
    model = None
    for chunk_start in date_range[::chunk_size_days]:
        chunk_df = load_training_chunk(chunk_start, chunk_size_days)
        
        if model is None:
            model = train_model(chunk_df, val_df, feature_cols)
        else:
            # Continue training from previous model
            model = lgb.train(
                lgb_params,
                train_ds,
                init_model=model,
                ...
            )
    return model
```

---

## Evaluation Protocol

### Baseline Models

```python
def persistence_baseline(df):
    """Baseline: predict pm25_t = pm25_t_lag24h"""
    return df['pm25_t_lag24h']

def linear_baseline(train_df, test_df, feature_cols):
    """Baseline: Ridge regression with same features"""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(train_df[feature_cols], train_df['pm25_t'])
    return model.predict(test_df[feature_cols])
```

### Metric Definitions

```python
def evaluate_model(y_true, y_pred, trust_weights, pm_regime_bins=[0, 40, 100, np.inf]):
    """
    Comprehensive evaluation with stratification.
    
    Returns dict with:
    - overall: MAE, RMSE, R², Huber loss
    - by_regime: metrics for low/moderate/high PM
    - by_trust: metrics for trust quartiles
    """
    results = {}
    
    # Overall metrics
    results['overall'] = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'huber_10': huber_loss(y_true, y_pred, delta=10.0),
    }
    
    # Regime stratification
    regimes = pd.cut(y_true, bins=pm_regime_bins, labels=['low', 'moderate', 'high'])
    for regime in ['low', 'moderate', 'high']:
        mask = (regimes == regime)
        if mask.sum() > 0:
            results[f'regime_{regime}'] = {
                'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                'rmse': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
                'n_samples': mask.sum(),
            }
    
    # Trust stratification
    trust_quartiles = pd.qcut(trust_weights, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = (trust_quartiles == q)
        results[f'trust_{q}'] = {
            'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
            'n_samples': mask.sum(),
        }
    
    return results
```

### Expected Performance Targets

| Metric | Persistence | Linear | LightGBM Target |
|--------|-------------|--------|-----------------|
| **R²** | 0.45-0.60 | 0.50-0.60 | **0.65-0.75** |
| **MAE (overall)** | >20 µg/m³ | 18-22 µg/m³ | **12-18 µg/m³** |
| **MAE (low PM <40)** | — | — | **8-12 µg/m³** |
| **MAE (moderate 40-100)** | — | — | **12-18 µg/m³** |
| **MAE (high PM >100)** | — | — | **18-25 µg/m³** |

**Key Rule**: If LightGBM R² is only marginally better than persistence but MAE in high-PM regime improves significantly, that's a win.

### Seasonal Reporting

| Season | Expected Pattern |
|--------|------------------|
| **Winter (Nov-Feb)** | Highest PM2.5, best predictability |
| **Pre-monsoon (Mar-May)** | Dust events, higher variance |
| **Monsoon (Jun-Sep)** | Lowest PM2.5, wet deposition effects |
| **Post-monsoon (Oct)** | Festival emissions, rapid changes |

---

## Interpretability Framework

### Global Feature Importance

```python
import shap

def compute_global_importance(model, X_sample, feature_groups):
    """
    Compute SHAP values and aggregate by feature groups.
    
    Groups:
    - lags: pm25_t_lag*
    - meteo: temperature, humidity, wind, pressure, precip, cloud
    - satellite: ndvi, ndbi, evi, etc.
    - temporal: hour_sin, hour_cos, dow_sin, dow_cos
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Aggregate by group
    group_importance = {}
    for group_name, feature_list in feature_groups.items():
        group_idx = [i for i, f in enumerate(X_sample.columns) if f in feature_list]
        group_importance[group_name] = np.abs(shap_values[:, group_idx]).sum(axis=1).mean()
    
    return group_importance
```

### Feature Groups for SHAP Aggregation

```python
FEATURE_GROUPS = {
    'autoregressive': [
        'pm25_t_lag1h', 'pm25_t_lag2h', 'pm25_t_lag3h',
        'pm25_t_lag6h', 'pm25_t_lag12h', 'pm25_t_lag24h',
    ],
    'meteorology_temperature': [
        'temperature_2m', 'temperature_2m_lag1h', ..., 'temperature_2m_lag24h',
        'dewpoint_2m', ..., 'temp_dewpoint_diff',
    ],
    'meteorology_wind': [
        'wind_u_10m', 'wind_u_10m_lag1h', ...,
        'wind_v_10m', 'wind_v_10m_lag1h', ...,
    ],
    'meteorology_moisture': [
        'relative_humidity_2m', ..., 'vpd',
        'precip_last_1h', 'precip_last_6h', 'precip_last_24h',
    ],
    'meteorology_other': [
        'surface_pressure', ..., 'cloud_cover', ...,
    ],
    'satellite_vegetation': ['ndvi', 'evi', 'mndwi', 'local_ndvi_std'],
    'satellite_urban': ['ndbi', 'bsi', 'ibi'],
    'satellite_cloud': ['cloud_mask', 'cloud_fraction_30m'],
    'temporal': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'],
}
```

### Local Explanations for Stakeholders

```python
def explain_grid_cell(model, X_row, feature_names, top_k=10):
    """
    Generate human-readable explanation for a single prediction.
    
    Output format:
    "PM2.5 prediction: 85 µg/m³
     Top contributors:
     1. pm25_t_lag1h (82 µg/m³): +45 µg/m³
     2. low wind_u_10m (0.5 m/s): +12 µg/m³
     3. low relative_humidity_2m (35%): +8 µg/m³
     4. high ndbi (0.3): +5 µg/m³
     ..."
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    
    # Sort by absolute contribution
    sorted_idx = np.argsort(-np.abs(shap_values[0]))[:top_k]
    
    explanation = []
    for idx in sorted_idx:
        feature = feature_names[idx]
        value = X_row.iloc[0, idx]
        contribution = shap_values[0, idx]
        explanation.append({
            'feature': feature,
            'value': value,
            'contribution': contribution,
        })
    
    return explanation
```

### Monotonic Constraint Candidates (Post-SHAP Inspection)

After training baseline, inspect SHAP dependence plots for:

| Feature | Expected Relationship | Constraint |
|---------|----------------------|------------|
| `precip_last_24h` | More rain → lower PM2.5 | -1 (if confirmed) |
| `wind_speed` (if derived) | Higher wind → lower PM2.5 | -1 (if confirmed) |
| `relative_humidity_2m` | Non-monotonic (complex) | None |
| `temperature_2m` | Non-monotonic (inversion effects) | None |
| `ndvi` | More vegetation → lower PM2.5 | -1 (if confirmed) |

**Process:**
1. Train baseline without constraints
2. Generate SHAP dependence plots
3. Identify unambiguous monotonic relationships
4. Add constraints only where physically justified
5. Retrain and compare performance

---

## Incremental Roadmap

### Phase 1: Baseline (Week 1)

| Task | Details |
|------|---------|
| Assemble training data | Run `nalanda.training` for 2017-2023 |
| Implement temporal split | Forward-chaining with 48h gap |
| Train persistence baseline | `pm25_t_lag24h` only |
| Train linear baseline | Ridge with all features |
| Train LightGBM v0 | Config above, all Group A-D features |
| Evaluate | Overall + regime + trust stratification |

**Success criteria**: R² > 0.65, MAE < 18 µg/m³

### Phase 2: Diagnostics (Week 2)

| Task | Details |
|------|---------|
| SHAP analysis | Global importance, dependence plots |
| Error analysis | Residuals by regime, season, hour |
| Spatial CV diagnostic | Hold out sensor neighborhoods |
| Feature ablation | Remove feature groups, measure degradation |

**Key questions to answer:**
- Which feature group contributes most?
- Is there spatial leakage (spatial CV degrades significantly)?
- Which PM regime has worst errors?

### Phase 3: Refinement (Week 3)

| Task | Details |
|------|---------|
| Add monotonic constraints | Based on SHAP inspection |
| Tune hyperparameters | Bayesian optimization on validation MAE |
| Feature engineering | Explore additional derived features |
| Model ensembling | Evaluate multi-model strategies |

### Phase 4: Production Hardening (Week 4)

| Task | Details |
|------|---------|
| Chunked training | Implement temporal chunking |
| Model versioning | Save model + config + metrics |
| Inference pipeline | Batch prediction for all grid cells |
| Monitoring | Track prediction drift over time |

---

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `dwarka/lgb_config.py` | LightGBM hyperparameter config |
| `dwarka/train.py` | Training loop with trust weighting |
| `dwarka/evaluate.py` | Stratified evaluation metrics |
| `dwarka/explain.py` | SHAP interpretability utilities |
| `dwarka/baselines.py` | Persistence + linear baselines |
| `nalanda/splits.py` | Temporal split with gap days |
| `nalanda/features_temporal.py` | Cyclic temporal encoding |

---

## Critical Invariants (Do Not Violate)

1. **No spatial features** (`grid_id`, `x_m`, `y_m`, `lat`, `lon`) in feature set
2. **48h minimum gap** between train/val/test splits
3. **No hyperparameter tuning on test set** — only validation
4. **Stratified reporting** — never report only overall metrics
5. **Baseline comparison required** — LightGBM must beat persistence
6. **Trust weighting** — use `sqrt(target_trust)` as sample weights
7. **No fabrication** — missing data flags only, no interpolation in ETL
