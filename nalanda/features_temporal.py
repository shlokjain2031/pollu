"""Temporal feature engineering module.

This module computes cyclic temporal encodings that allow LightGBM to learn
temporal patterns (diurnal, weekly).

Features:
    - hour_sin, hour_cos: Cyclic encoding of hour of day (0-23)
    - dow_sin, dow_cos: Cyclic encoding of day of week (Mon=0, Sun=6)

These features encode temporal structure in a way trees can use (sinusoid
representation maps cyclic variables to 2D Euclidean space).

Example usage:
    from nalanda.features_temporal import compute_temporal_features
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=24, freq='H')
    })
    df = compute_temporal_features(df)
    print(df[['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']])
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cyclic temporal encodings from timestamp.
    
    Converts cyclic time variables (hour, day-of-week) to sinusoidal encodings
    that preserve circular distance and allow tree-based models to capture
    periodic patterns.
    
    Args:
        df: DataFrame with 'timestamp' column (pandas datetime).
        
    Returns:
        DataFrame with added columns:
            - hour_sin: sin(2π * hour / 24)
            - hour_cos: cos(2π * hour / 24)
            - dow_sin: sin(2π * dow / 7)
            - dow_cos: cos(2π * dow / 7)
            
    Raises:
        KeyError: If 'timestamp' column not present.
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2023-01-01', periods=48, freq='H'),
        ...     'value': np.random.randn(48)
        ... })
        >>> df = compute_temporal_features(df)
        >>> df[['timestamp', 'hour_sin', 'hour_cos']].head()
                           timestamp   hour_sin   hour_cos
        0 2023-01-01 00:00:00  0.000000   1.000000
        1 2023-01-01 01:00:00  0.258819   0.965926
        2 2023-01-01 02:00:00  0.500000   0.866025
        3 2023-01-01 03:00:00  0.707107   0.707107
        4 2023-01-01 04:00:00  0.866025   0.500000
    """
    if 'timestamp' not in df.columns:
        raise KeyError("DataFrame must have 'timestamp' column")
    
    result = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
        result['timestamp'] = pd.to_datetime(result['timestamp'])
    
    # Hour of day: 0-23 -> [0, 2π)
    # Maps to unit circle: (sin, cos) = (sin(2πh/24), cos(2πh/24))
    hour = result['timestamp'].dt.hour.astype(np.float64)
    result['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0)
    result['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0)
    
    # Day of week: 0=Monday, 6=Sunday -> [0, 2π)
    # Maps to unit circle: (sin, cos) = (sin(2πd/7), cos(2πd/7))
    dow = result['timestamp'].dt.dayofweek.astype(np.float64)
    result['dow_sin'] = np.sin(2.0 * np.pi * dow / 7.0)
    result['dow_cos'] = np.cos(2.0 * np.pi * dow / 7.0)
    
    return result


def get_temporal_feature_columns() -> list[str]:
    """Return list of temporal feature column names.
    
    Returns:
        List of feature names: ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    """
    return ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']


if __name__ == '__main__':
    # Demo
    import sys
    
    # Create sample data
    print("Temporal Feature Engineering Demo")
    print("=" * 60)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=48, freq='H'),
        'value': np.random.randn(48),
    })
    
    print("Input DataFrame:")
    print(df.head())
    
    df_with_temporal = compute_temporal_features(df)
    
    print("\nWith temporal features:")
    print(df_with_temporal[['timestamp', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']].head(10))
    
    # Verify cyclic properties
    print("\nCyclic property verification:")
    print(f"  hour_sin²+hour_cos² ≈ 1: {(df_with_temporal['hour_sin']**2 + df_with_temporal['hour_cos']**2).mean():.6f}")
    print(f"  dow_sin²+dow_cos² ≈ 1:  {(df_with_temporal['dow_sin']**2 + df_with_temporal['dow_cos']**2).mean():.6f}")
