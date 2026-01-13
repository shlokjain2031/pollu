# Feature Analysis: Missing Features Root Cause

## Executive Summary

**Expected Features:** 64 (from `nalanda/features.py`)
**Actual Features in Training Data:** 48-55 (depending on satellite data availability)
**Missing Features:** 10-16 depending on satellite availability

### Root Causes Identified

1. **Temporal Encoding Features (4 features)** - CODE DOES NOT EXIST
   - `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
   - Should be in: `nalanda/features_temporal.py` (missing file)
   - Not integrated in: `nalanda/training.py` pipeline

2. **Satellite Staleness Features (2 features)** - CODE DOES NOT EXIST
   - `days_since_landsat`, `days_since_sentinel`
   - Should be computed in: `nalanda/features_satellite.py` or dedicated module
   - Not integrated in: `nalanda/training.py` pipeline

3. **Sentinel Indices Features (4 features)** - CODE EXISTS BUT NOT INTEGRATED
   - `evi`, `cloud_mask`, `cloud_fraction_30m`, `local_ndvi_std`
   - Code location: [nalanda/features_satellite.py](nalanda/features_satellite.py#L332-L620)
   - Computing functions: `compute_sentinel_indices()` (line 332), `compute_local_ndvi_std()` (line 398), `compute_cloud_fraction()` (line 464)
   - **Currently integrated?** YES - `compute_sentinel_features()` at line 534 calls these functions
   - **Output?** Returns DataFrame with these columns
   - **Problem?** Features appear ONLY when spatial computation succeeds; spatial computation skipped for large grids

## Feature Definition vs Implementation Matrix

### Group A: PM2.5 Lags (6 features) ✅
- **Definition:** [nalanda/features.py](nalanda/features.py#L53-L57) `_pm25_lag_features()`
- **Implementation:** [nalanda/features_meteo.py](nalanda/features_meteo.py) (lags computed with meteorological data)
- **Status:** ✅ IMPLEMENTED - Data uses naming: `pm25_t_minus_1h`, `pm25_t_minus_2h`, etc.
- **Recent fix:** Updated features.py to match actual naming convention (pm25_t_minus_ not pm25_t_lag_)

### Group B: Meteorological Features (37 features) ✅
- **Definitions:**
  - Base (7): [nalanda/features.py](nalanda/features.py#L61-L71) `_meteo_base_features()`
  - Lags (25): [nalanda/features.py](nalanda/features.py#L74-L102) `_meteo_lag_features()`
  - Derived (2): [nalanda/features.py](nalanda/features.py#L105-L110) `_derived_features()`
  - Precipitation (3): [nalanda/features.py](nalanda/features.py#L113-L121) `_precipitation_features()`
- **Implementation:** [nalanda/features_meteo.py](nalanda/features_meteo.py) `compute_meteo_features()`
- **Integration:** [nalanda/training.py](nalanda/training.py#L184-L192) calls `compute_meteo_features()`
- **Status:** ✅ IMPLEMENTED

### Group D: Temporal Encoding (4 features) ❌
- **Definition:** [nalanda/features.py](nalanda/features.py#L124-L133) `_temporal_encoding_features()`
  - `hour_sin` - sine encoding of hour of day (0-23)
  - `hour_cos` - cosine encoding of hour of day
  - `dow_sin` - sine encoding of day of week (0-6)
  - `dow_cos` - cosine encoding of day of week
- **Implementation:** **MISSING** - no `nalanda/features_temporal.py` file
- **Integration:** **NOT CALLED** in [nalanda/training.py](nalanda/training.py#L129-L300) pipeline
- **Status:** ❌ NOT IMPLEMENTED

**Root cause:** Temporal encoding requires computing cyclic features from timestamp, but no module exists and it's not integrated.

### Group C1: Landsat Features (5 features) ✅
- **Definition:** [nalanda/features.py](nalanda/features.py#L136-L145) `_landsat_features()`
  - `ndvi`, `ndbi`, `mndwi`, `bsi`, `ibi`
- **Implementation:** [nalanda/features_satellite.py](nalanda/features_satellite.py#L160-L330) `compute_landsat_features()`
- **Integration:** [nalanda/training.py](nalanda/training.py#L194-L207) calls `compute_landsat_features()`
- **Status:** ✅ IMPLEMENTED - Confirmed present in training data when Landsat data available

### Group C2: Sentinel Features (4 features) ✅ (but conditional)
- **Definition:** [nalanda/features.py](nalanda/features.py#L148-L154) `_sentinel_features()`
  - `evi` - Enhanced Vegetation Index
  - `cloud_mask` - Cloud detection binary flag
  - `cloud_fraction_30m` - Local cloud fraction in 30m window
  - `local_ndvi_std` - Local NDVI heterogeneity
- **Implementation:** [nalanda/features_satellite.py](nalanda/features_satellite.py#L534-L582) `compute_sentinel_features()`
  - Internal functions:
    - [Line 332] `compute_sentinel_indices()` - computes evi, ui_light, si, cloud_mask
    - [Line 398] `compute_local_ndvi_std()` - computes local_ndvi_std
    - [Line 464] `compute_cloud_fraction()` - computes cloud_fraction_30m
- **Integration:** [nalanda/training.py](nalanda/training.py#L209-L222) calls `compute_sentinel_features()`
- **Status:** ⚠️ CONDITIONAL
  - Code exists: YES
  - Called in pipeline: YES
  - **ISSUE:** Spatial features (local_ndvi_std, cloud_fraction_30m) only computed when:
    - `include_spatial=True` (default) AND
    - `grid_ids` is not None AND
    - `len(grid_ids) <= 10000` (line 568)
  - For full grid: spatial computation is skipped, sets `np.nan`
  - For subsets: spatial features should appear

### Group C3: Satellite Staleness (2 features) ❌
- **Definition:** [nalanda/features.py](nalanda/features.py#L157-L161) `_satellite_staleness_features()`
  - `days_since_landsat` - Days between target date and Landsat data source
  - `days_since_sentinel` - Days between target date and Sentinel data source
- **Implementation:** **MISSING** - No code computes this
- **Integration:** **NOT CALLED** in [nalanda/training.py](nalanda/training.py#L129-L300) pipeline
- **Status:** ❌ NOT IMPLEMENTED

**Root cause:** No function exists to compute temporal distance between target date and satellite source date. Should be simple: `(datetime.strptime(target_date) - datetime.strptime(source_date)).days`

## Data Evidence

### Example: 2019-01-01 (No Landsat)
- **Columns present:** 48 (target + meteo only)
- **Missing:** ndvi, ndbi, mndwi, bsi, ibi (Landsat), evi, cloud_mask, cloud_fraction_30m, local_ndvi_std (Sentinel), days_since_landsat, days_since_sentinel (staleness), hour_sin, hour_cos, dow_sin, dow_cos (temporal)
- **Count:** 15 missing

### Example: 2019-01-04 (With Landsat, possible Sentinel)
- **Columns present:** 62 (Landsat indices present)
- **Missing:** hour_sin, hour_cos, dow_sin, dow_cos (temporal), days_since_sentinel (if no Sentinel), possibly evi and sentinel features
- **Count:** ~10+ missing

## Implementation Plan

### Priority 1: Temporal Encoding (4 features)
**File to create:** `nalanda/features_temporal.py`
```python
def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cyclic temporal encodings from timestamp.
    
    Args:
        df: DataFrame with 'timestamp' column
        
    Returns:
        DataFrame with added hour_sin, hour_cos, dow_sin, dow_cos columns
    """
    import numpy as np
    result = df.copy()
    
    # Hour of day: 0-23 -> 0-2π
    hour = result['timestamp'].dt.hour
    result['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    result['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week: 0-6 -> 0-2π
    dow = result['timestamp'].dt.dayofweek
    result['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    result['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    
    return result
```

**Integration in training.py:** Call after loading target, before joins, add output to result.

### Priority 2: Satellite Staleness (2 features)
**Function to add:** In `nalanda/features_satellite.py` or new module
```python
def compute_satellite_staleness(
    target_date: str,
    landsat_date: str = None,
    sentinel_date: str = None,
) -> dict:
    """Compute days since last satellite pass.
    
    Returns:
        Dict with 'days_since_landsat', 'days_since_sentinel'
    """
    from datetime import datetime as dt
    
    target = dt.strptime(target_date, '%Y-%m-%d')
    result = {}
    
    if landsat_date:
        source = dt.strptime(landsat_date, '%Y-%m-%d')
        result['days_since_landsat'] = (target - source).days
    else:
        result['days_since_landsat'] = np.nan
        
    if sentinel_date:
        source = dt.strptime(sentinel_date, '%Y-%m-%d')
        result['days_since_sentinel'] = (target - source).days
    else:
        result['days_since_sentinel'] = np.nan
    
    return result
```

**Integration in training.py:** Add to all rows after Landsat/Sentinel joins, before final reordering.

### Priority 3: Verify Sentinel Features
**Current status:** Code exists, should be integrated
**Action:** Test with `include_sentinel_spatial=True` and small grid_ids to verify features appear

## Code Locations Reference

| Feature Group | Definition File | Definition Function | Implementation File | Implementation Function | Integration Point |
|---|---|---|---|---|---|
| PM2.5 lags | features.py | `_pm25_lag_features()` | features_meteo.py | `compute_meteo_features()` | training.py:184 |
| Meteo base+lags+derived | features.py | `_meteo_*_features()` | features_meteo.py | `compute_meteo_features()` | training.py:184 |
| Temporal encoding | features.py | `_temporal_encoding_features()` | ❌ MISSING | ❌ MISSING | ❌ MISSING |
| Landsat | features.py | `_landsat_features()` | features_satellite.py | `compute_landsat_features()` | training.py:194 |
| Sentinel | features.py | `_sentinel_features()` | features_satellite.py | `compute_sentinel_features()` → `compute_sentinel_indices()`, etc. | training.py:209 |
| Staleness | features.py | `_satellite_staleness_features()` | ❌ MISSING | ❌ MISSING | ❌ MISSING |

## Summary

**Immediate Actions Required:**
1. Create `nalanda/features_temporal.py` with `compute_temporal_features(df)`
2. Add temporal feature computation to training.py pipeline (call after loading target)
3. Add satellite staleness computation to training.py pipeline (call after Landsat/Sentinel joins)
4. Verify Sentinel features appear correctly (should work, may need testing)

**After Fixes Expected:**
- All 64 features should appear in training data when satellite data is available
- Temporal features will appear for all dates
- Staleness features will reflect actual days since last satellite pass
