"""Single source of truth for feature definitions.

This module programmatically defines the complete, ordered feature list
used by LightGBM training. It answers three questions unambiguously:

    1. What information exists?
    2. In what form does it exist?
    3. In what order does it exist?

The frozen feature list is the interface definition between:
    - data
    - model
    - inference
    - evaluation

LightGBM only sees feature_0, feature_1, ..., feature_n.
This module defines what those indices MEAN.

Usage:
    from nalanda.features import get_feature_list, EXCLUDED_COLUMNS
    
    feature_list = get_feature_list()  # deterministic, ordered
    X = df[feature_list]               # slice exactly these columns
"""

from __future__ import annotations

from typing import List, Tuple

# =============================================================================
# LAG DEFINITIONS (programmatic, not handwritten)
# =============================================================================

# PM2.5 autoregressive lags
PM25_LAG_HOURS: Tuple[int, ...] = (1, 2, 3, 6, 12, 24)

# Meteorological lag hours (default for most variables)
METEO_LAG_HOURS: Tuple[int, ...] = (1, 3, 6, 12, 24)

# Temperature-specific lags (6h excluded due to low autocorrelation)
TEMPERATURE_LAG_HOURS: Tuple[int, ...] = (1, 3, 12, 24)

# Cloud cover only needs coarse lags
CLOUD_COVER_LAG_HOURS: Tuple[int, ...] = (6, 24)


# =============================================================================
# FEATURE GROUPS (programmatic definitions)
# =============================================================================

def _pm25_lag_features() -> List[str]:
    """Group A: Autoregressive PM2.5 lags.
    
    These are the strongest predictors. Persistence baseline = pm25_t_minus_24h alone.
    """
    return [f"pm25_t_minus_{h}h" for h in PM25_LAG_HOURS]


def _meteo_base_features() -> List[str]:
    """Group B (base): Current-hour meteorological variables."""
    return [
        "temperature_2m",
        "relative_humidity_2m",
        "dewpoint_2m",
        "surface_pressure",
        "wind_u_10m",
        "wind_v_10m",
        "cloud_cover",
    ]


def _meteo_lag_features() -> List[str]:
    """Group B (lags): Lagged meteorological variables.
    
    Variable-specific lag sets:
    - temperature_2m: excludes 6h (low autocorrelation)
    - cloud_cover: only 6h, 24h (coarse dynamics)
    - others: full set (1, 3, 6, 12, 24h)
    """
    features = []
    
    # Temperature lags (exclude 6h)
    for h in TEMPERATURE_LAG_HOURS:
        features.append(f"temperature_2m_lag{h}h")
    
    # Standard variables with full lag set
    standard_vars = [
        "relative_humidity_2m",
        "dewpoint_2m",
        "surface_pressure",
        "wind_u_10m",
        "wind_v_10m",
    ]
    for var in standard_vars:
        for h in METEO_LAG_HOURS:
            features.append(f"{var}_lag{h}h")
    
    # Cloud cover with coarse lags only
    for h in CLOUD_COVER_LAG_HOURS:
        features.append(f"cloud_cover_lag{h}h")
    
    return features


def _derived_features() -> List[str]:
    """Group B (derived): Computed meteorological features."""
    return [
        "vpd",                  # Vapor pressure deficit
        "temp_dewpoint_diff",   # Temperature - dewpoint
    ]


def _precipitation_features() -> List[str]:
    """Group B (precipitation): Rolling precipitation sums."""
    return [
        "precip_last_1h",
        "precip_last_6h",
        "precip_last_24h",
    ]


def _temporal_encoding_features() -> List[str]:
    """Group D: Cyclic temporal encoding.
    
    LightGBM does not understand time. These features encode
    temporal structure in a way trees can use.
    """
    return [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]


def _landsat_features() -> List[str]:
    """Group C (Landsat): Landsat-8 satellite indices."""
    return [
        "ndvi",
        "ndbi",
        "mndwi",
        "bsi",
        "ibi",
    ]


def _sentinel_features() -> List[str]:
    """Group C (Sentinel): Sentinel-2 satellite indices."""
    return [
        "evi",
        "cloud_mask",
        "cloud_fraction_30m",
        "local_ndvi_std",
    ]


def _satellite_staleness_features() -> List[str]:
    """Group C (staleness): Days since last satellite pass."""
    return [
        "days_since_landsat",
        "days_since_sentinel",
    ]


def _spatial_features() -> List[str]:
    """Group D: Spatial location identifier.
    
    grid_id is a categorical feature representing the 30m grid cell.
    LightGBM handles this natively and learns location-specific baselines.
    This is critical for PM2.5 which has strong spatial patterns.
    """
    return ["grid_id"]


# =============================================================================
# EXCLUDED COLUMNS (explicit, not implicit)
# =============================================================================

# Group E: Spatial identifiers — grid_id is NOW INCLUDED as categorical feature
# Other spatial identifiers are EXPLICITLY EXCLUDED
# Rationale: grid_id captures location-specific baseline pollution levels
#            other spatial coords prevent LightGBM from learning a hash map
SPATIAL_IDENTIFIERS: Tuple[str, ...] = (
    "row_idx",
    "x_m",
    "y_m",
    "centroid_lat",
    "centroid_lon",
    "h3_cell",
)

# Group F: Target and trust — EXPLICITLY EXCLUDED
# Rationale: prevent leakage and circular learning
TARGET_COLUMNS: Tuple[str, ...] = (
    "pm25_t",           # the target itself
    "target_trust",     # used as sample weight, not feature
    "n_sensors",        # evaluation stratification only
)

# Timestamps — EXPLICITLY EXCLUDED
# Rationale: use temporal encoding features instead
TIMESTAMP_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "datetime",
    "date",
    "landsat_date",
    "sentinel_date",
)

# All columns that must never be features
EXCLUDED_COLUMNS: Tuple[str, ...] = (
    SPATIAL_IDENTIFIERS + TARGET_COLUMNS + TIMESTAMP_COLUMNS
)


# =============================================================================
# DETERMINISTIC FEATURE LIST GENERATION
# =============================================================================

def get_feature_list(
    include_landsat: bool = True,
    include_sentinel: bool = True,
) -> List[str]:
    """Generate the deterministic, ordered feature list.
    
    This is THE canonical feature order. LightGBM's feature_i corresponds
    to feature_list[i]. This mapping must be stable across:
        - training runs
        - inference
        - evaluation
        - model versions
    
    Args:
        include_landsat: Include Landsat-8 features (default True).
        include_sentinel: Include Sentinel-2 features (default True).
    
    Returns:
        Ordered list of feature names. Order is deterministic.
    
    Feature order:
        1. PM2.5 lags (strongest predictors first)
        2. Meteorological base (current hour)
        3. Meteorological lags (by variable, then by lag)
        4. Derived features (VPD, temp-dewpoint diff)
        5. Precipitation
        6. Temporal encoding
        7. Landsat indices (if included)
        8. Sentinel indices (if included)
        9. Satellite staleness
    """
    features: List[str] = []
    
    # 1. PM2.5 lags
    features.extend(_pm25_lag_features())
    
    # 2. Meteorological base
    features.extend(_meteo_base_features())
    
    # 3. Meteorological lags
    features.extend(_meteo_lag_features())
    
    # 4. Derived features
    features.extend(_derived_features())
    
    # 5. Precipitation
    features.extend(_precipitation_features())
    
    # 6. Temporal encoding
    features.extend(_temporal_encoding_features())
    
    # 7. Landsat indices
    if include_landsat:
        features.extend(_landsat_features())
    
    # 8. Sentinel indices
    if include_sentinel:
        features.extend(_sentinel_features())
    
    # 9. Satellite staleness
    if include_landsat or include_sentinel:
        features.extend(_satellite_staleness_features())
    
    # 10. Spatial location (categorical)
    features.extend(_spatial_features())
    
    return features


def get_feature_groups() -> dict:
    """Get features organized by conceptual group.
    
    Useful for feature importance analysis and ablation studies.
    
    Returns:
        Dict mapping group name to list of feature names.
    """
    return {
        "pm25_lags": _pm25_lag_features(),
        "meteo_base": _meteo_base_features(),
        "meteo_lags": _meteo_lag_features(),
        "derived": _derived_features(),
        "precipitation": _precipitation_features(),
        "temporal": _temporal_encoding_features(),
        "landsat": _landsat_features(),
        "sentinel": _sentinel_features(),
        "staleness": _satellite_staleness_features(),
        "spatial": _spatial_features(),
    }


def validate_no_excluded_columns(columns: List[str]) -> List[str]:
    """Check that no excluded columns are present.
    
    Args:
        columns: List of column names to check.
    
    Returns:
        List of excluded columns found (empty if clean).
    """
    return [c for c in columns if c in EXCLUDED_COLUMNS]


# =============================================================================
# MODULE-LEVEL CONSTANTS (for import convenience)
# =============================================================================

# The canonical feature list (default configuration)
FEATURE_LIST: List[str] = get_feature_list()

# Number of features
N_FEATURES: int = len(FEATURE_LIST)
