"""Baseline models for PM2.5 prediction benchmarking.

This module provides simple baseline models to establish performance floors:
- Persistence baseline: predict pm25_t = pm25_t_lag24h (yesterday's PM2.5)
- Linear baseline: Ridge regression with all frozen features

Usage:
    from dwarka.baselines import PersistenceBaseline, LinearBaseline
    
    # Persistence baseline
    persistence = PersistenceBaseline()
    y_pred = persistence.predict(df)
    
    # Linear baseline
    linear = LinearBaseline(alpha=1.0)
    linear.fit(X_train, y_train, weights=train_weights)
    y_pred = linear.predict(X_val)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


# =============================================================================
# Persistence Baseline
# =============================================================================

class PersistenceBaseline:
    """Baseline: predict pm25_t_lag24h (yesterday's PM2.5 at the same hour).
    
    This is the simplest possible baseline: assume tomorrow's pollution
    will be the same as yesterday's at the same time.
    
    Success criteria:
    - Any ML model must beat this baseline in R²
    - Typical persistence R² = 0.45-0.60
    """
    
    def __init__(self):
        """Initialize persistence baseline."""
        self.name = "persistence_lag24h"
        self.params = {}
        self.lag_col = "pm25_t_lag24h"
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return pm25_t_lag24h column directly.
        
        Args:
            df: DataFrame with pm25_t_lag24h column
            
        Returns:
            Array of predictions (same as lag24h values)
            
        Raises:
            ValueError: If pm25_t_lag24h column not found
        """
        if self.lag_col not in df.columns:
            raise ValueError(
                f"Persistence baseline requires '{self.lag_col}' column. "
                f"Available columns: {list(df.columns)}"
            )
        
        return df[self.lag_col].values
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata.
        
        Returns:
            Dict with model name and params
        """
        return {
            "name": self.name,
            "type": "persistence",
            "params": self.params,
            "description": "Predict pm25_t = pm25_t_lag24h",
        }


# =============================================================================
# Linear Baseline
# =============================================================================

class LinearBaseline:
    """Baseline: Ridge regression with all frozen features.
    
    This tests whether linear relationships capture most of the signal,
    or if tree-based models (LightGBM) provide substantial lift.
    
    Expected performance:
    - R² = 0.50-0.60 (better than persistence, worse than LightGBM)
    - MAE = 18-22 µg/m³
    """
    
    def __init__(self, alpha: float = 1.0):
        """Initialize Ridge regression baseline.
        
        Args:
            alpha: Ridge regularization parameter (L2 penalty)
        """
        self.name = "ridge_regression"
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=42)
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        self.params = {"alpha": alpha}
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LinearBaseline":
        """Fit Ridge model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            weights: Optional sample weights (n_samples,)
                     Will be applied as sqrt(weights) to both X and y
            feature_names: Feature names (for tracking)
            
        Returns:
            Self (for method chaining)
        """
        if weights is not None:
            # Apply square root of weights to both X and y
            # This is equivalent to weighted least squares with weights
            sqrt_weights = np.sqrt(weights)
            X_weighted = X.values * sqrt_weights[:, None]
            y_weighted = y.values * sqrt_weights
            
            self.model.fit(X_weighted, y_weighted)
        else:
            self.model.fit(X, y)
        
        self.feature_names = feature_names if feature_names else list(X.columns)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return Ridge predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
            
        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (absolute coefficients).
        
        Returns:
            DataFrame with feature names and absolute coefficients
            
        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        importance = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_,
            "abs_coefficient": np.abs(self.model.coef_),
        })
        
        return importance.sort_values("abs_coefficient", ascending=False)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata.
        
        Returns:
            Dict with model name, params, and fit status
        """
        metadata = {
            "name": self.name,
            "type": "linear",
            "params": self.params,
            "is_fitted": self.is_fitted,
            "n_features": len(self.feature_names) if self.feature_names else 0,
        }
        
        if self.is_fitted:
            metadata["intercept"] = float(self.model.intercept_)
        
        return metadata


# =============================================================================
# Baseline Loading Helper
# =============================================================================

def load_baselines(
    train_df: pd.DataFrame,
    feature_names: List[str],
    target_col: str = "pm25_t",
    weight_col: str = "target_trust",
    ridge_alpha: float = 1.0,
) -> Dict[str, Any]:
    """Instantiate and fit both baselines.
    
    Args:
        train_df: Training dataframe (must have features, target, and lag24h)
        feature_names: List of feature column names (from frozen schema)
        target_col: Target column name (default: "pm25_t")
        weight_col: Weight column name (default: "target_trust")
        ridge_alpha: Ridge regularization parameter
        
    Returns:
        Dict with:
        {
            "persistence": PersistenceBaseline instance,
            "linear": LinearBaseline instance (fitted),
            "metadata": {
                "persistence": {...},
                "linear": {...},
            }
        }
        
    Raises:
        ValueError: If required columns missing
    """
    # Validate required columns
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train_df")
    
    # 1. Persistence baseline (no fitting required)
    persistence = PersistenceBaseline()
    
    # 2. Linear baseline (requires fitting)
    linear = LinearBaseline(alpha=ridge_alpha)
    
    # Extract features and target
    X_train = train_df[feature_names]
    y_train = train_df[target_col]
    
    # Extract weights if available
    weights = None
    if weight_col in train_df.columns:
        weights = np.sqrt(train_df[weight_col].values)
        # Handle NaN weights
        weights = np.nan_to_num(weights, nan=0.0)
    
    # Fit linear baseline
    linear.fit(X_train, y_train, weights=weights, feature_names=feature_names)
    
    return {
        "persistence": persistence,
        "linear": linear,
        "metadata": {
            "persistence": persistence.get_metadata(),
            "linear": linear.get_metadata(),
        },
    }


# =============================================================================
# Baseline Prediction Helper
# =============================================================================

def predict_all_baselines(
    baselines: Dict[str, Any],
    df: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, np.ndarray]:
    """Get predictions from all baselines.
    
    Args:
        baselines: Dict returned from load_baselines()
        df: Dataframe to predict on (must have features and lag24h)
        feature_names: List of feature column names
        
    Returns:
        Dict mapping model name to predictions:
        {
            "persistence": array(...),
            "linear": array(...),
        }
    """
    predictions = {}
    
    # Persistence predictions
    persistence = baselines["persistence"]
    predictions["persistence"] = persistence.predict(df)
    
    # Linear predictions
    linear = baselines["linear"]
    X = df[feature_names]
    predictions["linear"] = linear.predict(X)
    
    return predictions
