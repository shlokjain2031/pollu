"""Stratified evaluation metrics for PM2.5 models.

This module provides comprehensive evaluation with multiple stratifications:
- Overall metrics (MAE, RMSE, R², Huber loss)
- By PM regime (low <40, moderate 40-100, high >100 µg/m³)
- By trust quartile (Q1-Q4)
- By season (winter, pre-monsoon, monsoon, post-monsoon)
- By hour of day (0-23)

Usage:
    from dwarka.evaluate import evaluate_model, compare_models
    
    results = evaluate_model(
        y_true=y_true,
        y_pred=y_pred,
        model_name="lightgbm_v0",
        timestamps=df["timestamp"],
        trust_weights=df["target_trust"],
    )
    
    # Compare multiple models
    comparison_df = compare_models([results1, results2, results3])
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StratifiedMetrics:
    """Metrics for a single stratum."""
    mae: float
    rmse: float
    r2: float
    huber_loss: float
    n_samples: int
    sample_pct: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResults:
    """Complete evaluation results with all stratifications."""
    
    model_name: str
    overall: StratifiedMetrics
    
    # Stratifications
    by_pm_regime: Dict[str, StratifiedMetrics]
    by_trust_quartile: Dict[str, StratifiedMetrics]
    by_season: Dict[str, StratifiedMetrics]
    by_hour: Dict[int, StratifiedMetrics]
    
    # Metadata
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "model_name": self.model_name,
            "overall": self.overall.to_dict(),
            "by_pm_regime": {k: v.to_dict() for k, v in self.by_pm_regime.items()},
            "by_trust_quartile": {k: v.to_dict() for k, v in self.by_trust_quartile.items()},
            "by_season": {k: v.to_dict() for k, v in self.by_season.items()},
            "by_hour": {str(k): v.to_dict() for k, v in self.by_hour.items()},
            "created_at": self.created_at,
        }


# =============================================================================
# Metric Computation
# =============================================================================

def _huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 10.0) -> float:
    """Compute Huber loss.
    
    Huber loss is quadratic for |error| <= delta, linear beyond.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        delta: Threshold for quadratic vs linear region
        
    Returns:
        Mean Huber loss
    """
    err = np.abs(y_true - y_pred)
    return np.where(
        err <= delta,
        0.5 * err**2,
        delta * err - 0.5 * delta**2,
    ).mean()


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    huber_delta: float = 10.0,
    total_samples: Optional[int] = None,
) -> StratifiedMetrics:
    """Compute all metrics for a stratum.
    
    Args:
        y_true: True PM2.5 values
        y_pred: Predicted PM2.5 values
        huber_delta: Delta parameter for Huber loss
        total_samples: Total dataset size (for computing sample_pct)
        
    Returns:
        StratifiedMetrics with all computed metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    huber = _huber_loss(y_true, y_pred, huber_delta)
    
    n_samples = len(y_true)
    sample_pct = 100.0 * n_samples / total_samples if total_samples else 100.0
    
    return StratifiedMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        huber_loss=huber,
        n_samples=n_samples,
        sample_pct=sample_pct,
    )


# =============================================================================
# Stratified Evaluation
# =============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    timestamps: Optional[pd.Series] = None,
    trust_weights: Optional[np.ndarray] = None,
    huber_delta: float = 10.0,
) -> EvaluationResults:
    """Comprehensive stratified evaluation.
    
    Args:
        y_true: True PM2.5 values (µg/m³)
        y_pred: Predicted PM2.5 values (µg/m³)
        model_name: Name for results (e.g., "persistence", "linear", "lightgbm_v0")
        timestamps: pd.Series of timestamps (for seasonal stratification)
        trust_weights: Array of trust weights (for quartile stratification)
        huber_delta: Delta parameter for Huber loss (default: 10.0)
        
    Returns:
        EvaluationResults with all stratifications
        
    Raises:
        ValueError: If y_true and y_pred lengths don't match
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} vs {len(y_pred)}"
        )
    
    total_samples = len(y_true)
    
    # 1. Compute overall metrics
    overall = _compute_metrics(y_true, y_pred, huber_delta, total_samples)
    
    # 2. Stratify by PM regime
    by_pm_regime = {}
    pm_regimes = [
        ("low", 0, 40),
        ("moderate", 40, 100),
        ("high", 100, np.inf),
    ]
    
    for regime_name, lo, hi in pm_regimes:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            by_pm_regime[regime_name] = _compute_metrics(
                y_true[mask], y_pred[mask], huber_delta, total_samples
            )
    
    # 3. Stratify by trust quartile
    by_trust_quartile = {}
    if trust_weights is not None and len(trust_weights) == len(y_true):
        try:
            quartiles = pd.qcut(
                trust_weights,
                q=4,
                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                duplicates='drop',
            )
            
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                mask = (quartiles == q)
                if mask.sum() > 0:
                    by_trust_quartile[q] = _compute_metrics(
                        y_true[mask], y_pred[mask], huber_delta, total_samples
                    )
        except ValueError:
            # Handle case where quartiles can't be computed (e.g., all same values)
            pass
    
    # 4. Stratify by season
    by_season = {}
    if timestamps is not None and len(timestamps) == len(y_true):
        timestamps_dt = pd.to_datetime(timestamps)
        
        seasons = [
            ("winter", [11, 12, 1, 2]),
            ("pre_monsoon", [3, 4, 5]),
            ("monsoon", [6, 7, 8, 9]),
            ("post_monsoon", [10]),
        ]
        
        for season_name, months in seasons:
            mask = timestamps_dt.dt.month.isin(months).values
            if mask.sum() > 0:
                by_season[season_name] = _compute_metrics(
                    y_true[mask], y_pred[mask], huber_delta, total_samples
                )
    
    # 5. Stratify by hour
    by_hour = {}
    if timestamps is not None and len(timestamps) == len(y_true):
        timestamps_dt = pd.to_datetime(timestamps)
        
        for hour in range(24):
            mask = (timestamps_dt.dt.hour == hour).values
            if mask.sum() > 0:
                by_hour[hour] = _compute_metrics(
                    y_true[mask], y_pred[mask], huber_delta, total_samples
                )
    
    return EvaluationResults(
        model_name=model_name,
        overall=overall,
        by_pm_regime=by_pm_regime,
        by_trust_quartile=by_trust_quartile,
        by_season=by_season,
        by_hour=by_hour,
        created_at=datetime.utcnow().isoformat() + "Z",
    )


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(
    results_list: List[EvaluationResults],
    metrics: List[str] = ["mae", "rmse", "r2", "huber_loss"],
) -> pd.DataFrame:
    """Create comparison table across models.
    
    Args:
        results_list: List of EvaluationResults
        metrics: Which metrics to include in comparison
        
    Returns:
        DataFrame with models as rows, metrics as columns
    """
    data = []
    
    for results in results_list:
        row = {"model": results.model_name}
        
        for metric in metrics:
            if hasattr(results.overall, metric):
                row[metric] = getattr(results.overall, metric)
        
        row["n_samples"] = results.overall.n_samples
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by R² (descending)
    if "r2" in df.columns:
        df = df.sort_values("r2", ascending=False).reset_index(drop=True)
    
    return df


def get_regime_comparison(
    results_list: List[EvaluationResults],
    regime: str = "low",
    metric: str = "mae",
) -> pd.DataFrame:
    """Compare models on a specific PM regime.
    
    Args:
        results_list: List of EvaluationResults
        regime: Which regime to compare ("low", "moderate", "high")
        metric: Which metric to extract
        
    Returns:
        DataFrame with model names and metric values
    """
    data = []
    
    for results in results_list:
        if regime in results.by_pm_regime:
            regime_metrics = results.by_pm_regime[regime]
            row = {
                "model": results.model_name,
                metric: getattr(regime_metrics, metric),
                "n_samples": regime_metrics.n_samples,
            }
            data.append(row)
    
    return pd.DataFrame(data)


def get_seasonal_comparison(
    results_list: List[EvaluationResults],
    season: str = "winter",
    metric: str = "mae",
) -> pd.DataFrame:
    """Compare models on a specific season.
    
    Args:
        results_list: List of EvaluationResults
        season: Which season to compare
        metric: Which metric to extract
        
    Returns:
        DataFrame with model names and metric values
    """
    data = []
    
    for results in results_list:
        if season in results.by_season:
            season_metrics = results.by_season[season]
            row = {
                "model": results.model_name,
                metric: getattr(season_metrics, metric),
                "n_samples": season_metrics.n_samples,
            }
            data.append(row)
    
    return pd.DataFrame(data)


# =============================================================================
# Success Gate Checking
# =============================================================================

def check_success_gate(
    results: EvaluationResults,
    r2_threshold: float = 0.65,
    mae_threshold: float = 18.0,
    baseline_r2: Optional[float] = None,
) -> Dict[str, Any]:
    """Check if model passes success criteria.
    
    Success gate:
    - R² > 0.65
    - MAE < 18 µg/m³
    - (Optional) Beats persistence baseline R²
    
    Args:
        results: EvaluationResults to check
        r2_threshold: Minimum R² required
        mae_threshold: Maximum MAE allowed
        baseline_r2: Optional persistence baseline R² to beat
        
    Returns:
        Dict with pass/fail status and details
    """
    checks = {
        "r2_pass": results.overall.r2 > r2_threshold,
        "mae_pass": results.overall.mae < mae_threshold,
        "beats_baseline": True,  # Default if no baseline provided
    }
    
    if baseline_r2 is not None:
        checks["beats_baseline"] = results.overall.r2 > baseline_r2
    
    overall_pass = all(checks.values())
    
    return {
        "model_name": results.model_name,
        "passed": overall_pass,
        "checks": checks,
        "r2": results.overall.r2,
        "mae": results.overall.mae,
        "r2_threshold": r2_threshold,
        "mae_threshold": mae_threshold,
        "baseline_r2": baseline_r2,
    }
