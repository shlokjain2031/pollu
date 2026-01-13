"""SHAP-based model interpretability for PM2.5 predictions.

This module provides three levels of interpretability:
1. Global importance - overall feature contributions
2. Feature group analysis - aggregated by conceptual groups
3. Local explanations - individual prediction breakdown

See dwarka/EXPLAIN_SPEC.md for detailed API documentation.

Example usage:
    from dwarka.explain import compute_global_importance, plot_summary
    import lightgbm as lgb
    import pandas as pd
    
    model = lgb.Booster(model_file='model.txt')
    X = pd.read_parquet('features.parquet')
    
    # Global importance
    importance = compute_global_importance(model, X)
    print(importance.head(10))
    
    # Visualizations
    plot_summary(model, X, save_path='shap_summary.png')
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Optional SHAP import with graceful degradation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn(
        "SHAP not available. Install with: pip install shap\n"
        "Interpretability functions will raise ImportError when called."
    )


# =============================================================================
# Default Feature Groups
# =============================================================================

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


# =============================================================================
# Core SHAP Functions
# =============================================================================

def _check_shap_available():
    """Raise ImportError if SHAP is not available."""
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is required for interpretability functions. "
            "Install with: pip install shap"
        )


def compute_shap_values(
    model: lgb.Booster,
    X: pd.DataFrame,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """Compute SHAP values with optional caching.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        cache_path: Path to save/load cached SHAP values (.npy file)
        
    Returns:
        SHAP values array, shape (n_samples, n_features)
    """
    _check_shap_available()
    
    # Try to load from cache
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached SHAP values from {cache_path}")
        return np.load(cache_path)
    
    # Compute SHAP values
    print(f"Computing SHAP values for {len(X):,} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Save to cache if requested
    if cache_path:
        cache_dir = Path(cache_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, shap_values)
        print(f"Cached SHAP values to {cache_path}")
    
    return shap_values


def compute_global_importance(
    model: lgb.Booster,
    X: pd.DataFrame,
    sample_size: int = 10000,
    seed: int = 42,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Compute global SHAP feature importance.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame (full dataset or representative sample)
        sample_size: Number of samples to use for SHAP computation
        seed: Random seed for sampling
        cache_path: Path to cache SHAP values
        
    Returns:
        DataFrame with columns:
            - feature: str (feature name)
            - mean_abs_shap: float (mean |SHAP value|)
            - mean_shap: float (mean SHAP value, signed)
            - std_shap: float (standard deviation of SHAP values)
        Sorted by mean_abs_shap descending.
    """
    _check_shap_available()
    
    # Sample if needed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=seed)
    else:
        X_sample = X
    
    # Compute SHAP values
    shap_values = compute_shap_values(model, X_sample, cache_path=cache_path)
    
    # Calculate statistics
    importance_data = []
    for i, feature in enumerate(X_sample.columns):
        importance_data.append({
            'feature': feature,
            'mean_abs_shap': np.abs(shap_values[:, i]).mean(),
            'mean_shap': shap_values[:, i].mean(),
            'std_shap': shap_values[:, i].std(),
        })
    
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    
    return importance_df


def compute_group_importance(
    model: lgb.Booster,
    X: pd.DataFrame,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    sample_size: int = 10000,
    seed: int = 42,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Compute SHAP importance aggregated by feature groups.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        feature_groups: Dict mapping group name to list of feature names.
                       If None, uses DEFAULT_FEATURE_GROUPS
        sample_size: Number of samples for SHAP computation
        seed: Random seed
        cache_path: Path to cache SHAP values
        
    Returns:
        DataFrame with columns:
            - group: str (feature group name)
            - mean_abs_shap: float (mean |SHAP| for group)
            - contribution_pct: float (% of total SHAP magnitude)
            - n_features: int (number of features in group)
        Sorted by mean_abs_shap descending.
    """
    _check_shap_available()
    
    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS
    
    # Sample if needed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=seed)
    else:
        X_sample = X
    
    # Compute SHAP values
    shap_values = compute_shap_values(model, X_sample, cache_path=cache_path)
    
    # Total SHAP magnitude for normalization
    total_shap = np.abs(shap_values).sum()
    
    # Aggregate by group
    group_data = []
    for group_name, feature_list in feature_groups.items():
        # Find indices of features in this group
        group_idx = [i for i, f in enumerate(X_sample.columns) if f in feature_list]
        
        if not group_idx:
            continue  # Skip groups with no matching features
        
        # Sum absolute SHAP values for this group
        group_shap = np.abs(shap_values[:, group_idx]).sum(axis=1).mean()
        
        group_data.append({
            'group': group_name,
            'mean_abs_shap': group_shap,
            'contribution_pct': 100 * (group_shap * len(X_sample)) / total_shap if total_shap > 0 else 0,
            'n_features': len(group_idx),
        })
    
    group_df = pd.DataFrame(group_data)
    group_df = group_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    
    return group_df


def explain_prediction(
    model: lgb.Booster,
    X_row: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10,
    base_value: Optional[float] = None,
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
            - base_value: float (baseline prediction)
            - top_contributors: List[Dict] with feature, value, shap, rank
            - total_shap: float (sum of all SHAP values)
    """
    _check_shap_available()
    
    if len(X_row) != 1:
        raise ValueError(f"X_row must have exactly 1 row, got {len(X_row)}")
    
    if feature_names is None:
        feature_names = list(X_row.columns)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    
    # Get prediction
    prediction = model.predict(X_row)[0]
    
    # Get base value (expected value)
    if base_value is None:
        base_value = explainer.expected_value
    
    # Sort by absolute SHAP value
    shap_abs = np.abs(shap_values[0])
    sorted_idx = np.argsort(-shap_abs)[:top_k]
    
    # Build top contributors
    top_contributors = []
    for rank, idx in enumerate(sorted_idx, 1):
        top_contributors.append({
            'rank': rank,
            'feature': feature_names[idx],
            'value': X_row.iloc[0, idx],
            'shap': float(shap_values[0, idx]),
        })
    
    return {
        'prediction': float(prediction),
        'base_value': float(base_value),
        'top_contributors': top_contributors,
        'total_shap': float(shap_values[0].sum()),
    }


def compare_group_importance_by_regime(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    regimes: Optional[List[Tuple[float, float, str]]] = None,
    sample_size_per_regime: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare feature group importance across PM2.5 regimes.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        y: Target values (PM2.5)
        feature_groups: Dict of feature groups (uses default if None)
        regimes: List of (min_pm, max_pm, label) tuples
        sample_size_per_regime: Samples to draw from each regime
        seed: Random seed
        
    Returns:
        DataFrame with columns: group, regime, mean_abs_shap, contribution_pct
    """
    _check_shap_available()
    
    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS
    
    if regimes is None:
        regimes = [
            (0, 40, 'low'),
            (40, 100, 'moderate'),
            (100, np.inf, 'high'),
        ]
    
    # Collect results
    all_results = []
    
    for min_pm, max_pm, label in regimes:
        # Filter to regime
        mask = (y >= min_pm) & (y < max_pm)
        X_regime = X[mask]
        
        if len(X_regime) == 0:
            print(f"Warning: No samples in regime '{label}' [{min_pm}, {max_pm})")
            continue
        
        # Sample if needed
        if len(X_regime) > sample_size_per_regime:
            X_regime = X_regime.sample(n=sample_size_per_regime, random_state=seed)
        
        print(f"Computing SHAP for regime '{label}' ({len(X_regime):,} samples)...")
        
        # Compute group importance for this regime
        group_imp = compute_group_importance(
            model, X_regime, feature_groups=feature_groups,
            sample_size=len(X_regime), seed=seed
        )
        group_imp['regime'] = label
        
        all_results.append(group_imp)
    
    return pd.concat(all_results, ignore_index=True)


def validate_monotonicity(
    model: lgb.Booster,
    X: pd.DataFrame,
    feature: str,
    expected_direction: int,
    sample_size: int = 5000,
    tolerance: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Check if feature exhibits monotonic relationship with target.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        feature: Feature to check
        expected_direction: +1 (increasing), -1 (decreasing)
        sample_size: Samples for analysis
        tolerance: Fraction of samples allowed to violate monotonicity
        seed: Random seed
        
    Returns:
        Dict with: is_monotonic, violation_rate, recommendation, correlation
    """
    _check_shap_available()
    
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in X.columns")
    
    if expected_direction not in [-1, 1]:
        raise ValueError(f"expected_direction must be -1 or 1, got {expected_direction}")
    
    # Sample if needed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=seed)
    else:
        X_sample = X
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Get feature index
    feature_idx = list(X_sample.columns).index(feature)
    
    # Extract feature values and SHAP values
    feature_vals = X_sample[feature].values
    shap_vals = shap_values[:, feature_idx]
    
    # Compute correlation
    correlation = stats.spearmanr(feature_vals, shap_vals)[0]
    
    # Check monotonicity violations
    if expected_direction == 1:
        # Expect positive correlation
        violations = (shap_vals < 0).sum()
    else:
        # Expect negative correlation
        violations = (shap_vals > 0).sum()
    
    violation_rate = violations / len(X_sample)
    is_monotonic = violation_rate <= tolerance
    
    # Generate recommendation
    if violation_rate < 0.05:
        recommendation = f"SAFE to add monotonic constraint ({expected_direction:+d})"
    elif violation_rate < 0.15:
        recommendation = "BORDERLINE: inspect dependence plot before adding constraint"
    else:
        recommendation = "NOT RECOMMENDED: complex non-monotonic relationship"
    
    return {
        'is_monotonic': is_monotonic,
        'violation_rate': float(violation_rate),
        'recommendation': recommendation,
        'correlation': float(correlation),
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_summary(
    model: lgb.Booster,
    X: pd.DataFrame,
    plot_type: str = 'bar',
    max_display: int = 20,
    sample_size: int = 10000,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    seed: int = 42,
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
        seed: Random seed
        
    Returns:
        Matplotlib figure
    """
    _check_shap_available()
    
    # Sample if needed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=seed)
    else:
        X_sample = X
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Create plot
    fig = plt.figure(figsize=figsize)
    
    if plot_type == 'bar':
        shap.summary_plot(
            shap_values, X_sample,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
    elif plot_type == 'dot':
        shap.summary_plot(
            shap_values, X_sample,
            plot_type='dot',
            max_display=max_display,
            show=False
        )
    else:
        raise ValueError(f"plot_type must be 'bar' or 'dot', got '{plot_type}'")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot to {save_path}")
    
    return fig


def plot_dependence(
    model: lgb.Booster,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: Optional[str] = None,
    sample_size: int = 5000,
    ax: Optional[plt.Axes] = None,
    seed: int = 42,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Axes:
    """Create SHAP dependence plot for a feature.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        feature: Feature to plot on x-axis
        interaction_feature: Feature to color points by (auto-selected if None)
        sample_size: Max samples to plot
        ax: Matplotlib axes (creates new if None)
        seed: Random seed
        save_path: Path to save figure
        **kwargs: Additional arguments to shap.dependence_plot
        
    Returns:
        Matplotlib axes with dependence plot
    """
    _check_shap_available()
    
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in X.columns")
    
    # Sample if needed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=seed)
    else:
        X_sample = X
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create dependence plot
    shap.dependence_plot(
        feature,
        shap_values,
        X_sample,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
        **kwargs
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dependence plot to {save_path}")
    
    return ax


def plot_waterfall(
    model: lgb.Booster,
    X_row: pd.DataFrame,
    max_display: int = 15,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create waterfall plot showing step-by-step prediction buildup.
    
    Args:
        model: Trained LightGBM model
        X_row: Single-row DataFrame
        max_display: Number of features to show
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    _check_shap_available()
    
    if len(X_row) != 1:
        raise ValueError(f"X_row must have exactly 1 row, got {len(X_row)}")
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_row)
    
    # Create waterfall plot
    fig = plt.figure(figsize=figsize)
    shap.waterfall_plot(shap_values[0], max_display=max_display, show=False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved waterfall plot to {save_path}")
    
    return fig


# =============================================================================
# Utility Functions
# =============================================================================

def format_explanation_text(
    explanation: Dict[str, Any],
    include_baseline: bool = True,
    max_features: int = 5,
) -> str:
    """Format explanation dict as human-readable text.
    
    Args:
        explanation: Output from explain_prediction()
        include_baseline: Whether to show base value
        max_features: Max features to include
        
    Returns:
        Multi-line string suitable for reports or logs
    """
    lines = []
    
    # Header
    pred = explanation['prediction']
    base = explanation['base_value']
    
    if include_baseline:
        lines.append(f"PM2.5 prediction: {pred:.1f} µg/m³ (base: {base:.1f} µg/m³)")
    else:
        lines.append(f"PM2.5 prediction: {pred:.1f} µg/m³")
    
    lines.append("")
    lines.append("Top contributors:")
    
    # Top contributors
    for contrib in explanation['top_contributors'][:max_features]:
        rank = contrib['rank']
        feature = contrib['feature']
        value = contrib['value']
        shap = contrib['shap']
        
        # Format feature name for readability
        feature_display = feature.replace('_', ' ').title()
        
        lines.append(
            f"  {rank}. {feature_display} = {value:.2f} → {shap:+.1f} µg/m³"
        )
    
    return "\n".join(lines)


def get_feature_groups_from_nalanda() -> Dict[str, List[str]]:
    """Get feature groups from nalanda.features module if available.
    
    Returns:
        Dict mapping group names to feature lists
    """
    try:
        from nalanda.features import get_feature_groups
        return get_feature_groups()
    except ImportError:
        # Fall back to default groups
        return DEFAULT_FEATURE_GROUPS


# =============================================================================
# CLI for Quick Analysis
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SHAP interpretability analysis for LightGBM PM2.5 model'
    )
    parser.add_argument('--model', required=True, help='Path to LightGBM model file')
    parser.add_argument('--data', required=True, help='Path to parquet file with features')
    parser.add_argument('--output-dir', default='shap_analysis', help='Output directory')
    parser.add_argument('--sample-size', type=int, default=5000, help='Samples for SHAP')
    parser.add_argument('--target-col', default='pm25_t', help='Target column name')
    
    args = parser.parse_args()
    
    # Load model and data
    print(f"Loading model from {args.model}")
    model = lgb.Booster(model_file=args.model)
    
    print(f"Loading data from {args.data}")
    df = pd.read_parquet(args.data)
    
    # Separate features and target
    exclude_cols = [args.target_col, 'target_trust', 'timestamp', 'row_idx', 'grid_id', 
                    'landsat_date', 'sentinel_date']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    y = df[args.target_col] if args.target_col in df.columns else None
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing {len(X):,} samples with {len(feature_cols)} features")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # 1. Global importance
    print("\n1. Computing global feature importance...")
    importance = compute_global_importance(model, X, sample_size=args.sample_size)
    importance_path = output_dir / 'feature_importance.csv'
    importance.to_csv(importance_path, index=False)
    print(f"   Saved to {importance_path}")
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))
    
    # 2. Group importance
    print("\n2. Computing feature group importance...")
    group_imp = compute_group_importance(model, X, sample_size=args.sample_size)
    group_path = output_dir / 'group_importance.csv'
    group_imp.to_csv(group_path, index=False)
    print(f"   Saved to {group_path}")
    print("\nFeature groups:")
    print(group_imp.to_string(index=False))
    
    # 3. Visualizations
    print("\n3. Generating visualizations...")
    plot_summary(model, X.head(args.sample_size), plot_type='bar', 
                 save_path=output_dir / 'shap_summary_bar.png')
    plot_summary(model, X.head(args.sample_size), plot_type='dot',
                 save_path=output_dir / 'shap_summary_dot.png')
    
    # 4. Regime comparison (if target available)
    if y is not None:
        print("\n4. Computing importance by PM regime...")
        regime_comp = compare_group_importance_by_regime(model, X, y, 
                                                          sample_size_per_regime=2000)
        regime_path = output_dir / 'regime_comparison.csv'
        regime_comp.to_csv(regime_path, index=False)
        print(f"   Saved to {regime_path}")
        
        # Pivot for easy viewing
        pivot = regime_comp.pivot(index='group', columns='regime', values='contribution_pct')
        print("\nContribution % by regime:")
        print(pivot.to_string())
    
    print("\n" + "="*80)
    print(f"✓ SHAP analysis complete! Results in {output_dir}")
