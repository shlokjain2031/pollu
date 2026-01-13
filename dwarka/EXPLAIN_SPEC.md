# SHAP Interpretability Module Specification

**Module**: `dwarka/explain.py`  
**Purpose**: Provide SHAP-based model interpretability for PM2.5 predictions  
**Dependencies**: `shap`, `lightgbm`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 1. Overview

The `explain.py` module provides three levels of interpretability:

1. **Global Importance**: Understanding overall feature contributions across the dataset
2. **Feature Group Analysis**: Aggregating SHAP values by conceptual feature groups (lags, meteo, satellite, temporal)
3. **Local Explanations**: Explaining individual predictions for stakeholder communication

All explanations are computed using SHAP (SHapley Additive exPlanations) TreeExplainer, optimized for LightGBM.

---

## 2. API Design

### 2.1 Global Feature Importance

```python
def compute_global_importance(
    model: lgb.Booster,
    X: pd.DataFrame,
    sample_size: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute global SHAP feature importance.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame (full dataset or representative sample)
        sample_size: Number of samples to use for SHAP computation
        seed: Random seed for sampling
        
    Returns:
        DataFrame with columns:
            - feature: str (feature name)
            - mean_abs_shap: float (mean |SHAP value|)
            - mean_shap: float (mean SHAP value, signed)
            - std_shap: float (standard deviation of SHAP values)
        Sorted by mean_abs_shap descending.
        
    Example:
        >>> importance = compute_global_importance(model, X_val)
        >>> print(importance.head(10))
        
                    feature  mean_abs_shap  mean_shap  std_shap
        0      pm25_t_lag1h          12.45      11.23      8.12
        1     pm25_t_lag24h           8.34       7.89      6.45
        2      pm25_t_lag6h           6.78       6.12      5.23
        ...
    """
```

**Implementation notes:**
- Use `shap.TreeExplainer(model)` for fast exact computation
- Sample uniformly if X is too large (>10k rows)
- Return importance ranked by mean absolute SHAP value
- Include both mean (signed) and mean absolute for directionality analysis

---

### 2.2 Feature Group Aggregation

```python
def compute_group_importance(
    model: lgb.Booster,
    X: pd.DataFrame,
    feature_groups: Dict[str, List[str]] = None,
    sample_size: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute SHAP importance aggregated by feature groups.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        feature_groups: Dict mapping group name to list of feature names.
                       If None, uses default groups from nalanda.features
        sample_size: Number of samples for SHAP computation
        seed: Random seed
        
    Returns:
        DataFrame with columns:
            - group: str (feature group name)
            - mean_abs_shap: float (mean |SHAP| for group)
            - contribution_pct: float (% of total SHAP magnitude)
            - n_features: int (number of features in group)
        Sorted by mean_abs_shap descending.
        
    Example:
        >>> group_imp = compute_group_importance(model, X_val)
        >>> print(group_imp)
        
                          group  mean_abs_shap  contribution_pct  n_features
        0         autoregressive          32.45             45.2           6
        1  meteorology_temperature          18.23             25.4          11
        2       meteorology_wind          12.34             17.2          12
        ...
    """
```

**Default feature groups** (from model_spec.md):
```python
DEFAULT_FEATURE_GROUPS = {
    'autoregressive': [
        'pm25_t_minus_1h', 'pm25_t_minus_2h', 'pm25_t_minus_3h',
        'pm25_t_minus_6h', 'pm25_t_minus_12h', 'pm25_t_minus_24h',
    ],
    'meteorology_temperature': [
        'temperature_2m', 'temperature_2m_lag1h', 'temperature_2m_lag3h',
        'temperature_2m_lag12h', 'temperature_2m_lag24h',
        'dewpoint_2m', 'dewpoint_2m_lag1h', 'dewpoint_2m_lag3h',
        'dewpoint_2m_lag6h', 'dewpoint_2m_lag12h', 'dewpoint_2m_lag24h',
        'temp_dewpoint_diff', 'vpd',
    ],
    'meteorology_wind': [
        'wind_u_10m', 'wind_u_10m_lag1h', 'wind_u_10m_lag3h',
        'wind_u_10m_lag6h', 'wind_u_10m_lag12h', 'wind_u_10m_lag24h',
        'wind_v_10m', 'wind_v_10m_lag1h', 'wind_v_10m_lag3h',
        'wind_v_10m_lag6h', 'wind_v_10m_lag12h', 'wind_v_10m_lag24h',
    ],
    'meteorology_moisture': [
        'relative_humidity_2m', 'relative_humidity_2m_lag1h',
        'relative_humidity_2m_lag3h', 'relative_humidity_2m_lag6h',
        'relative_humidity_2m_lag12h', 'relative_humidity_2m_lag24h',
        'precip_last_1h', 'precip_last_6h', 'precip_last_24h',
    ],
    'meteorology_other': [
        'surface_pressure', 'surface_pressure_lag1h', 'surface_pressure_lag3h',
        'surface_pressure_lag6h', 'surface_pressure_lag12h', 'surface_pressure_lag24h',
        'cloud_cover', 'cloud_cover_lag6h', 'cloud_cover_lag24h',
    ],
    'satellite_vegetation': ['ndvi', 'evi', 'mndwi', 'local_ndvi_std'],
    'satellite_urban': ['ndbi', 'bsi', 'ibi'],
    'satellite_cloud': ['cloud_mask', 'cloud_fraction_30m'],
    'satellite_staleness': ['days_since_landsat', 'days_since_sentinel'],
    'temporal': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'],
}
```

**Implementation notes:**
- Sum absolute SHAP values across all features in each group
- Normalize by number of samples to get mean
- Calculate contribution_pct = (group_sum / total_sum) × 100
- Auto-detect groups from `nalanda.features.get_feature_groups()` if not provided

---

### 2.3 Local Explanations (Individual Predictions)

```python
def explain_prediction(
    model: lgb.Booster,
    X_row: pd.DataFrame,
    feature_names: List[str] = None,
    top_k: int = 10,
    base_value: float = None,
) -> Dict[str, Any]:
    """Generate human-readable explanation for a single prediction.
    
    Args:
        model: Trained LightGBM model
        X_row: Single-row DataFrame with features
        feature_names: List of feature names (if None, uses X_row.columns)
        top_k: Number of top contributing features to return
        base_value: Base prediction value (if None, computed from model)
        
    Returns:
        Dict with:
            - prediction: float (predicted PM2.5 value)
            - base_value: float (baseline prediction, mean of training set)
            - top_contributors: List[Dict] with:
                - feature: str
                - value: float (feature value)
                - shap: float (SHAP contribution to prediction)
                - rank: int (1 to top_k)
            - total_shap: float (sum of all SHAP values, should ≈ prediction - base)
            
    Example:
        >>> explanation = explain_prediction(model, X_val.iloc[[0]])
        >>> print(f"Prediction: {explanation['prediction']:.1f} µg/m³")
        >>> print(f"Base value: {explanation['base_value']:.1f} µg/m³")
        >>> for contrib in explanation['top_contributors']:
        >>>     print(f"  {contrib['rank']}. {contrib['feature']}: "
        >>>           f"{contrib['value']:.2f} → {contrib['shap']:+.2f} µg/m³")
        
        Prediction: 85.3 µg/m³
        Base value: 42.1 µg/m³
          1. pm25_t_minus_1h: 82.50 → +38.45 µg/m³
          2. pm25_t_minus_24h: 78.30 → +12.23 µg/m³
          3. wind_u_10m: 0.50 → +8.12 µg/m³
          ...
    """
```

**Implementation notes:**
- Single-sample SHAP computation is fast (<10ms)
- Base value should be the expected model output (usually mean of training targets)
- Sort by absolute SHAP value to get top positive and negative contributors
- Include feature value for context (e.g., "low wind → high PM")

---

### 2.4 SHAP Dependence Plots

```python
def plot_dependence(
    model: lgb.Booster,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = None,
    sample_size: int = 5000,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create SHAP dependence plot for a feature.
    
    Shows how SHAP values vary with feature value, revealing:
    - Monotonic vs non-monotonic relationships
    - Interaction effects (color by another feature)
    - Threshold effects or saturation
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        feature: Feature to plot on x-axis
        interaction_feature: Feature to color points by (auto-selected if None)
        sample_size: Max samples to plot
        ax: Matplotlib axes (creates new if None)
        **kwargs: Additional arguments to shap.dependence_plot
        
    Returns:
        Matplotlib axes with dependence plot
        
    Example:
        >>> fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        >>> for ax, feature in zip(axes.flat, ['pm25_t_minus_24h', 'temperature_2m',
        >>>                                      'wind_u_10m', 'relative_humidity_2m',
        >>>                                      'ndvi', 'precip_last_24h']):
        >>>     plot_dependence(model, X_val, feature, ax=ax)
        >>> plt.tight_layout()
        >>> plt.savefig('shap_dependence.png')
    """
```

**Use cases:**
- **Monotonicity check**: Does `precip_last_24h` consistently reduce PM2.5? → Candidate for monotonic constraint
- **Interaction detection**: Does temperature effect vary with humidity?
- **Threshold identification**: Is there a saturation point for NDVI benefit?

---

### 2.5 Summary Plots

```python
def plot_summary(
    model: lgb.Booster,
    X: pd.DataFrame,
    plot_type: str = 'bar',
    max_display: int = 20,
    sample_size: int = 10000,
    figsize: Tuple[int, int] = (10, 8),
    save_path: str = None,
) -> plt.Figure:
    """Create SHAP summary plot.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        plot_type: 'bar' (mean |SHAP|) or 'dot' (SHAP distribution)
        max_display: Number of features to show
        sample_size: Samples for SHAP computation
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
        
    Example:
        >>> # Bar plot: overall importance
        >>> plot_summary(model, X_val, plot_type='bar', max_display=15)
        
        >>> # Dot plot: show value distribution and SHAP direction
        >>> plot_summary(model, X_val, plot_type='dot', max_display=20)
    """
```

**Plot types:**
- **Bar**: Simple ranking by mean |SHAP|
- **Dot**: Each dot = one sample, color = feature value, x = SHAP value
  - Shows directionality: "high temperature → high SHAP → high PM"
  - Shows spread: wide distribution = high variance in contribution

---

### 2.6 Waterfall Plots for Detailed Local Explanation

```python
def plot_waterfall(
    model: lgb.Booster,
    X_row: pd.DataFrame,
    max_display: int = 15,
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
) -> plt.Figure:
    """Create waterfall plot showing step-by-step prediction buildup.
    
    Visualizes: base_value + SHAP_1 + SHAP_2 + ... + SHAP_n = prediction
    
    Args:
        model: Trained LightGBM model
        X_row: Single-row DataFrame
        max_display: Number of features to show
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
        
    Example:
        >>> # For stakeholder report: explain a specific high-PM event
        >>> high_pm_row = X_val[y_val > 150].iloc[[0]]
        >>> plot_waterfall(model, high_pm_row, save_path='high_pm_explanation.png')
    """
```

**Use case**: Explaining specific events to policymakers
- "On Jan 15, 2024 at 18:00, Andheri grid predicted 165 µg/m³ because:"
  - Base prediction: 42 µg/m³
  - Yesterday's PM (150 µg/m³): +68 µg/m³
  - Low wind (0.3 m/s): +24 µg/m³
  - Low humidity (28%): +15 µg/m³
  - ... = 165 µg/m³

---

### 2.7 Monotonic Constraint Validation

```python
def validate_monotonicity(
    model: lgb.Booster,
    X: pd.DataFrame,
    feature: str,
    expected_direction: int,  # +1 or -1
    sample_size: int = 5000,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Check if feature exhibits monotonic relationship with target.
    
    Used to validate candidates for monotonic constraints before retraining.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        feature: Feature to check
        expected_direction: +1 (increasing), -1 (decreasing)
        sample_size: Samples for analysis
        tolerance: Fraction of samples allowed to violate monotonicity
        
    Returns:
        Dict with:
            - is_monotonic: bool (True if passes tolerance threshold)
            - violation_rate: float (fraction of samples violating monotonicity)
            - recommendation: str (human-readable assessment)
            - correlation: float (Spearman correlation between feature and SHAP)
            
    Example:
        >>> # Check if precipitation consistently reduces PM2.5
        >>> result = validate_monotonicity(model, X_val, 'precip_last_24h', 
        >>>                                 expected_direction=-1)
        >>> print(result)
        
        {
            'is_monotonic': True,
            'violation_rate': 0.03,
            'recommendation': 'SAFE to add monotonic constraint (-1)',
            'correlation': -0.82
        }
    """
```

**Threshold logic:**
- violation_rate < 0.05 → "SAFE to add constraint"
- violation_rate 0.05-0.15 → "BORDERLINE: inspect dependence plot"
- violation_rate > 0.15 → "NOT RECOMMENDED: complex relationship"

---

### 2.8 Feature Group Comparison Across Regimes

```python
def compare_group_importance_by_regime(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    feature_groups: Dict[str, List[str]] = None,
    regimes: List[Tuple[float, float, str]] = None,
    sample_size_per_regime: int = 2000,
) -> pd.DataFrame:
    """Compare feature group importance across PM2.5 regimes.
    
    Answers: "Which features matter most in high-PM events?"
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        y: Target values (PM2.5)
        feature_groups: Dict of feature groups (uses default if None)
        regimes: List of (min_pm, max_pm, label) tuples
                 Default: [(0, 40, 'low'), (40, 100, 'moderate'), (100, inf, 'high')]
        sample_size_per_regime: Samples to draw from each regime
        
    Returns:
        DataFrame with columns:
            - group: str
            - regime: str
            - mean_abs_shap: float
            - contribution_pct: float
        Pivoted for easy comparison
        
    Example:
        >>> regime_comp = compare_group_importance_by_regime(model, X_val, y_val)
        >>> print(regime_comp.pivot(index='group', columns='regime', 
        >>>                          values='contribution_pct'))
        
        regime                     low  moderate  high
        group                                         
        autoregressive            38.2      42.1  51.3
        meteorology_temperature   25.4      22.8  18.2
        meteorology_wind          18.3      20.1  22.5
        satellite_vegetation       8.1       6.2   3.4
        ...
    """
```

**Insight**: In high-PM events (>100 µg/m³), autoregressive features dominate (51%) while satellite features become less important (3%).

---

## 3. Utility Functions

### 3.1 SHAP Value Caching

```python
def compute_shap_values(
    model: lgb.Booster,
    X: pd.DataFrame,
    cache_path: str = None,
) -> np.ndarray:
    """Compute SHAP values with optional caching.
    
    SHAP computation for 10k samples × 64 features takes ~30-60 seconds.
    Cache to avoid recomputation during analysis.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        cache_path: Path to save/load cached SHAP values (.npy file)
        
    Returns:
        SHAP values array, shape (n_samples, n_features)
    """
```

### 3.2 Human-Readable Formatting

```python
def format_explanation_text(
    explanation: Dict[str, Any],
    include_baseline: bool = True,
    max_features: int = 5,
) -> str:
    """Format explanation dict as human-readable text.
    
    Returns:
        Multi-line string suitable for reports or logs
        
    Example output:
        "PM2.5 prediction: 85.3 µg/m³ (base: 42.1 µg/m³)
         
         Top contributors:
         1. pm25_t_minus_1h = 82.5 µg/m³ → +38.5 µg/m³ (previous hour pollution)
         2. pm25_t_minus_24h = 78.3 µg/m³ → +12.2 µg/m³ (yesterday same time)
         3. wind_u_10m = 0.5 m/s → +8.1 µg/m³ (low wind, poor dispersion)
         4. relative_humidity_2m = 28% → +5.3 µg/m³ (dry conditions)
         5. ndvi = 0.15 → +3.2 µg/m³ (low vegetation cover)"
    """
```

---

## 4. Integration with Existing Code

### 4.1 Integration with `dwarka/trainer.py`

After training completes, automatically compute and save global importance:

```python
# In trainer.py, after model.save_model()
from dwarka.explain import compute_global_importance, compute_group_importance

# Compute SHAP on validation set
shap_importance = compute_global_importance(model, X_val, sample_size=5000)
shap_importance.to_csv(output_dir / 'shap_importance.csv', index=False)

group_importance = compute_group_importance(model, X_val)
group_importance.to_csv(output_dir / 'group_importance.csv', index=False)
```

### 4.2 Integration with `dwarka/evaluate.py`

Add SHAP-based diagnostics to evaluation report:

```python
# In evaluate.py, add to evaluation metrics
from dwarka.explain import compare_group_importance_by_regime

regime_comparison = compare_group_importance_by_regime(model, X_val, y_val)
# Add to evaluation report JSON
```

### 4.3 Integration with `dwarka/report.py`

Generate SHAP plots for model card:

```python
# In report.py, add SHAP visualizations
from dwarka.explain import plot_summary, plot_dependence

# Overall importance
plot_summary(model, X_val, plot_type='bar', save_path='shap_bar.png')
plot_summary(model, X_val, plot_type='dot', save_path='shap_dot.png')

# Key dependence plots
for feature in ['pm25_t_minus_24h', 'temperature_2m', 'wind_u_10m', 
                'precip_last_24h', 'ndvi', 'relative_humidity_2m']:
    plot_dependence(model, X_val, feature, 
                    save_path=f'shap_dependence_{feature}.png')
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

```python
# tests/dwarka/test_explain.py

def test_compute_global_importance():
    """Test global importance computation."""
    # Create dummy model and data
    model = create_dummy_lgb_model()
    X = create_dummy_features(n_samples=1000, n_features=10)
    
    importance = compute_global_importance(model, X)
    
    # Assertions
    assert len(importance) == 10  # All features present
    assert 'feature' in importance.columns
    assert 'mean_abs_shap' in importance.columns
    assert importance['mean_abs_shap'].iloc[0] >= importance['mean_abs_shap'].iloc[-1]  # Sorted
    assert importance['mean_abs_shap'].min() >= 0  # All non-negative

def test_explain_prediction():
    """Test single prediction explanation."""
    model = create_dummy_lgb_model()
    X_row = create_dummy_features(n_samples=1, n_features=10)
    
    explanation = explain_prediction(model, X_row, top_k=5)
    
    assert 'prediction' in explanation
    assert 'base_value' in explanation
    assert len(explanation['top_contributors']) == 5
    assert explanation['top_contributors'][0]['rank'] == 1
```

### 5.2 Integration Tests

```python
def test_full_workflow():
    """Test complete SHAP workflow with real model."""
    # Load real model and validation data
    model = lgb.Booster(model_file='artifacts/models/v1/model.txt')
    X_val = pd.read_parquet('artifacts/val/chunk_0.parquet')
    
    # Compute importance
    importance = compute_global_importance(model, X_val.head(1000))
    assert len(importance) > 0
    
    # Group importance
    group_imp = compute_group_importance(model, X_val.head(1000))
    assert 'autoregressive' in group_imp['group'].values
    
    # Local explanation
    explanation = explain_prediction(model, X_val.iloc[[0]])
    assert explanation['prediction'] > 0
```

---

## 6. Performance Considerations

### Computational Cost

| Operation | Sample Size | Features | Time (approx) |
|-----------|-------------|----------|---------------|
| TreeExplainer init | - | 64 | <1s |
| SHAP values (batch) | 1,000 | 64 | ~3s |
| SHAP values (batch) | 10,000 | 64 | ~30s |
| SHAP values (batch) | 100,000 | 64 | ~5min |
| Single prediction | 1 | 64 | <10ms |

**Optimization strategies:**
1. **Sample intelligently**: 5-10k samples usually sufficient for stable importance estimates
2. **Cache SHAP values**: Reuse for multiple analyses (save as .npy)
3. **Parallelize**: Use `n_jobs=-1` in TreeExplainer when available
4. **Stratified sampling**: Ensure all PM regimes represented in sample

### Memory Requirements

- SHAP values array: `n_samples × n_features × 8 bytes`
- 10,000 samples × 64 features × 8 bytes ≈ 5 MB
- 100,000 samples × 64 features × 8 bytes ≈ 50 MB

---

## 7. Output Artifacts

After full analysis, generate:

```
artifacts/models/{model_version}/interpretability/
├── shap_importance.csv              # Global feature importance
├── group_importance.csv             # Feature group aggregation
├── regime_comparison.csv            # Importance by PM regime
├── plots/
│   ├── shap_summary_bar.png        # Bar plot: overall importance
│   ├── shap_summary_dot.png        # Dot plot: value distribution
│   ├── dependence_pm25_lag24h.png  # Dependence plots for key features
│   ├── dependence_temperature_2m.png
│   ├── dependence_wind_u_10m.png
│   ├── dependence_precip_last_24h.png
│   ├── dependence_ndvi.png
│   └── dependence_relative_humidity.png
└── cache/
    └── shap_values_val.npy          # Cached SHAP values for reuse
```

---

## 8. Example Workflow

### Complete SHAP Analysis Pipeline

```python
from dwarka.explain import (
    compute_global_importance,
    compute_group_importance,
    compare_group_importance_by_regime,
    plot_summary,
    plot_dependence,
    validate_monotonicity,
)
import lightgbm as lgb
import pandas as pd

# Load model and validation data
model = lgb.Booster(model_file='artifacts/models/v1/model.txt')
X_val = pd.read_parquet('artifacts/val/chunk_0.parquet')
y_val = X_val['pm25_t']
feature_cols = [c for c in X_val.columns if c not in ['pm25_t', 'target_trust', 'timestamp', 'row_idx', 'grid_id']]
X_val_features = X_val[feature_cols]

# 1. Global importance
print("Computing global importance...")
importance = compute_global_importance(model, X_val_features, sample_size=5000)
importance.to_csv('shap_importance.csv', index=False)
print(importance.head(15))

# 2. Feature group aggregation
print("\nComputing group importance...")
group_imp = compute_group_importance(model, X_val_features)
print(group_imp)

# 3. Regime comparison
print("\nComparing importance by PM regime...")
regime_comp = compare_group_importance_by_regime(model, X_val_features, y_val)
print(regime_comp.pivot(index='group', columns='regime', values='contribution_pct'))

# 4. Visualizations
print("\nGenerating plots...")
plot_summary(model, X_val_features.head(5000), plot_type='bar', save_path='shap_bar.png')
plot_summary(model, X_val_features.head(5000), plot_type='dot', save_path='shap_dot.png')

key_features = ['pm25_t_minus_24h', 'temperature_2m', 'wind_u_10m', 'precip_last_24h', 'ndvi']
for feature in key_features:
    plot_dependence(model, X_val_features.head(3000), feature, 
                    save_path=f'dependence_{feature}.png')

# 5. Validate monotonicity candidates
print("\nValidating monotonic constraint candidates...")
candidates = [
    ('precip_last_24h', -1, "More rain should reduce PM"),
    ('ndvi', -1, "More vegetation should reduce PM"),
]

for feature, direction, reason in candidates:
    result = validate_monotonicity(model, X_val_features, feature, direction)
    print(f"\n{feature}: {reason}")
    print(f"  Violation rate: {result['violation_rate']:.2%}")
    print(f"  Recommendation: {result['recommendation']}")

print("\n✓ SHAP analysis complete!")
```

---

## 9. Success Criteria

The `explain.py` module is complete when:

1. ✅ All functions pass unit tests
2. ✅ Can compute global importance in <60s for 10k samples
3. ✅ Feature group aggregation matches model_spec.md definitions
4. ✅ Local explanations are human-readable and accurate
5. ✅ Dependence plots reveal monotonic relationships for constraint candidates
6. ✅ Integration with trainer.py/evaluate.py/report.py works seamlessly
7. ✅ Generated artifacts (CSVs, plots) are publication-ready
8. ✅ Documentation includes usage examples for all major functions

---

## 10. Future Enhancements (Phase 2+)

1. **SHAP Interaction Values**: Compute pairwise feature interactions
2. **Force Plots**: Alternative to waterfall for single predictions
3. **SHAP-based Feature Selection**: Iteratively remove low-importance features
4. **Temporal SHAP Analysis**: Track how importance changes over time
5. **Spatial SHAP Patterns**: Cluster grid cells by SHAP profile similarity
6. **Adversarial SHAP**: Generate counterfactual explanations ("what feature change would reduce PM by 20 µg/m³?")

