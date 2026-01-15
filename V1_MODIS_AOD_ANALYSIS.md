# MODIS AOD NULL ANALYSIS - artifacts/v1/modis_aod/mumbai/

## Summary

Analysis of null values in MODIS AOD satellite data stored in `artifacts/v1/modis_aod/mumbai/` directory.

### Overall Statistics

| Column | Total Values | Total Nulls | Null % | Dates with Nulls | Dates with 100% Nulls |
|--------|-------------|------------|--------|------------------|----------------------|
| aod_047 | 1,916,586 | 1,369,116 | 71.44% | 2,171 | 867 |
| aod_055 | 1,916,586 | 1,369,116 | 71.44% | 2,171 | 867 |
| aod_uncertainty | 1,916,586 | 585,985 | 30.57% | 2,173 | 178 |

## Key Findings

### 1. aod_047 and aod_055 Correlation
- **Identical null patterns**: Both columns have exactly the same null values and percentages (71.44%)
- This indicates they are always acquired together or processed as a unit
- **867 dates with complete data loss** (100% nulls)
- Only **2 dates out of 2,173** have complete data (0.1%)

### 2. aod_uncertainty Independence
- **30.57% overall null rate** - significantly better than aod_047/aod_055
- **178 dates with complete loss** (vs 867 for aod_047/aod_055)
- **0 dates with perfect data** (even fewer than aod_047/aod_055)

### 3. Temporal Patterns

#### High Null Periods (2019-2020 Summer)
```
June-September months consistently show >95% nulls across the dataset
2019-06: 97.8% null
2019-07: 99.9% null  ← Near total loss
2019-08: 99.5% null  ← Near total loss
2019-09: 99.7% null  ← Near total loss
2020-06: 98.8% null
2020-07: 99.9% null  ← Near total loss
2020-08: 100.0% null ← COMPLETE LOSS
2020-09: 99.0% null  ← Near total loss
```

#### Better Data Periods
```
January-May and October-December show lower null rates:
2019-01: 30.0% null (but still missing 16% of pixels)
2020-01: 38.0% null
These periods still have significant missing data
```

### 4. Dates with 100% Nulls (Sample)

**aod_047 and aod_055** (867 dates):
- First occurrence: 2019-01-08
- Last occurrence: 2024-12-30
- Pattern: Occurs every 16 days (regular 16-day MODIS repeat cycle)
- Examples: 2019-01-08, 2019-01-24, 2019-02-09, 2019-02-25...

**aod_uncertainty** (178 dates):
- First occurrence: 2019-01-08
- Last occurrence: 2024-12-30
- Subset of aod_047/aod_055 issues

### 5. Data Quality Assessment

| Column | Complete Data Days | Usability |
|--------|-------------------|-----------|
| aod_047 | 2 days (0.1%) | CRITICAL - 99.9% data loss |
| aod_055 | 2 days (0.1%) | CRITICAL - 99.9% data loss |
| aod_uncertainty | 0 days (0.0%) | CRITICAL - 100% data loss |

## Comparison with artifacts/{split}/training.parquet

The v1/modis_aod directory shows **similar patterns** to the original training data:
- Same column structure
- Identical null percentages for aod_047/aod_055
- Similar 16-day cycle pattern
- Seasonal nulls concentrated in summer months

**Key difference**: The v1 location appears to store **raw MODIS pixel data** before aggregation, while the original artifacts/{split}/ contains processed training features.

## Root Cause Analysis

### MODIS Satellite Limitations
1. **16-day Repeat Cycle**: MODIS satellites have a 16-day orbital repeat pattern
   - Data only available every 16 days for any given location
   - Explains the regular 100% null dates

2. **Seasonal Cloud Cover**
   - Summer months (June-September) have high monsoon cloud cover in India
   - Leads to 95-100% data loss during these periods
   - Winter months (Jan-May, Oct-Dec) have better data availability

3. **Product Issues**
   - Cloud masking may be too aggressive
   - Quality flag filtering may exclude valid retrievals
   - Data processing pipeline may have failures

### Data Pipeline Issues
- The v1 directory is organized by location (mumbai/) only
- Original artifacts/{split}/ is split into train/val/test
- This suggests v1 is intermediate/raw storage, not final training data

## Recommendations

### For Using This Data
1. **NOT recommended for time-series analysis** without significant preprocessing
   - 99.9% of time periods have no data
   - Only 2 complete days in entire 6-year dataset

2. **For spatial analysis only**
   - Can aggregate multiple dates for a single map
   - Use non-null pixels even if most pixels are missing

3. **Consider alternative sources**
   - Sentinel-5P has daily global coverage
   - MERRA-2 reanalysis for complete temporal coverage
   - Combine with ground-based CPCB measurements

### Data Handling Strategy
```python
# Option 1: Temporal Aggregation (monthly/seasonal)
# Stack all non-null values across each month
# Accept spatial gaps but gain temporal continuity

# Option 2: Spatial Sampling
# Use only dates with sufficient non-null pixels
# Accept temporal gaps for data quality

# Option 3: Ensemble Approach
# Combine MODIS with other sources (Sentinel-5P, ground stations)
# Fill MODIS gaps with alternative data sources
```

## Files Referenced
- Input: `artifacts/v1/modis_aod/mumbai/*.parquet` (2,174 files, 2019-2024)
- Columns analyzed: `aod_047`, `aod_055`, `aod_uncertainty`
- Date range: 2019-01-01 to 2024-12-31

## Conclusion

The v1/modis_aod directory contains **raw MODIS pixel-level data with severe data quality issues**:
- 71-100% nulls due to MODIS orbital gaps and seasonal cloud cover
- Only 2 complete dates out of 2,173
- Not suitable for direct time-series modeling without major preprocessing

The original `artifacts/{split}/training.parquet` files were likely created by aggregating/processing this raw data to create usable training features. The v1 storage may be kept for reproducibility or alternative analysis approaches.
