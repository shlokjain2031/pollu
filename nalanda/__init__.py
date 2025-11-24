"""Nalanda: Feature engineering module for Pollu air quality modeling.

This module handles all feature engineering stages:
- Stage 0: Precomputation (grid index, neighbors, static features)
- Stage 1: Dynamic features (satellite, meteorology)
- Stage 2: PM2.5 interpolation
- Stage 3: Temporal lags
- Stage 4: Spatial aggregates
- Stage 5: Training set filtering
- Stage 6: Final assembly

Pipeline designed for incremental, parallel processing with comprehensive
monitoring and validation.
"""

__version__ = "1.0.0"
