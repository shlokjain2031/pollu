# AOD Columns Null Values Analysis

## Summary

Investigation of null values in Aerosol Optical Depth (AOD) columns across training, validation, and test datasets.

### Key Findings

| Column | Total Dates with Nulls | Dates with 100% Nulls | Total Null Values | Null Percentage |
|--------|------------------------|----------------------|-------------------|-----------------|
| `aod_047` | 1,838 | 980 | 96,897,033 | 71.42% |
| `aod_055` | 1,838 | 980 | 96,897,033 | 71.42% |
| `aod_uncertainty` | 918 | 171 | 16,960,632 | 29.28% |

## Detailed Analysis

### aod_047 and aod_055 Columns

These two columns have **identical null patterns**:
- **1,838 dates** have at least some null values
- **980 dates** have 100% null values (all rows are null)
- **71.42%** of all values are null across these dates

**Dates with 100% nulls by split:**

**Train split (369 dates):**
- Early 2019: 2019-01-08, 2019-01-24, 2019-02-04, 2019-02-09, 2019-02-22, 2019-02-25, 2019-03-12, 2019-03-13, 2019-03-29, 2019-03-30...
- Pattern continues throughout 2019-2020

**Val split (367 dates):**
- 2023-01-07, 2023-01-14, 2023-01-21, 2023-01-28, 2023-01-29, 2023-02-08, 2023-03-05, 2023-03-06, 2023-03-07, 2023-03-08...
- Concentrated in late 2023 (Nov-Dec particularly problematic)
- 2023-12-01: 133,888 out of 133,888 rows (100%)
- 2023-12-21: 133,888 out of 133,888 rows (100%)

**Test split (244 dates):**
- 2024-01-03, 2024-01-09, 2024-01-10, 2024-01-15, 2024-01-17, 2024-01-21, 2024-01-23, 2024-01-31, 2024-02-03, 2024-02-13...
- Scattered throughout 2024

**Top 20 dates with highest null counts (aod_047 & aod_055):**
1. 2023-12-01 (val): 133,888 nulls out of 133,888 rows
2. 2023-12-21 (val): 133,888 nulls out of 133,888 rows
3. 2023-11-25 (val): 133,699 nulls out of 133,699 rows
4. 2023-12-20 (val): 132,796 nulls out of 132,796 rows
5. 2023-12-04 (val): 130,369 nulls out of 130,369 rows
6. 2023-12-05 (val): 129,350 nulls out of 129,350 rows
7. 2023-11-19 (val): 128,204 nulls out of 128,204 rows
8. 2023-12-22 (val): 128,190 nulls out of 128,190 rows
9. 2023-12-27 (val): 127,752 nulls out of 127,752 rows
10. 2024-07-04 (test): 126,440 nulls out of 126,440 rows
11. 2024-07-03 (test): 126,005 nulls out of 126,005 rows
12. 2024-04-26 (test): 125,349 nulls out of 125,349 rows
13. 2024-07-05 (test): 124,698 nulls out of 124,698 rows
14. 2024-01-23 (test): 123,640 nulls out of 123,640 rows
15. 2024-07-06 (test): 123,148 nulls out of 123,148 rows
16. 2023-12-18 (val): 122,919 nulls out of 122,919 rows
17. 2024-06-18 (test): 122,578 nulls out of 122,578 rows
18. 2023-12-15 (val): 122,548 nulls out of 122,548 rows
19. 2023-12-07 (val): 122,478 nulls out of 122,478 rows
20. 2024-02-03 (test): 121,634 nulls out of 121,634 rows

### aod_uncertainty Column

This column has a **different null pattern**:
- **918 dates** have at least some null values
- **171 dates** have 100% null values
- **29.28%** of all values are null

**Dates with 100% nulls by split (select examples):**

**Test split (majority):**
- 2024-01-03, 2024-01-15, 2024-01-17, 2024-01-21, 2024-01-31, 2024-02-03, 2024-02-15, 2024-02-22, 2024-03-05, 2024-03-17...

**Train split (examples):**
- 2019-01-08, 2019-01-24, 2019-02-09, 2019-02-25, 2019-03-13, 2019-03-29, 2019-04-14, 2019-04-30, 2019-05-16, 2019-06-01...

**Val split (examples):**
- 2023-01-14, 2023-01-21, 2023-01-28, 2023-03-06, 2023-03-13, 2023-03-20, 2023-03-27, 2023-04-03, 2023-04-10, 2023-04-17...

**Top 20 dates with highest null counts (aod_uncertainty):**
1. 2023-11-19 (val): 128,204 nulls out of 128,204 rows
2. 2023-12-27 (val): 127,752 nulls out of 127,752 rows
3. 2023-12-15 (val): 122,548 nulls out of 122,548 rows
4. 2024-02-03 (test): 121,634 nulls out of 121,634 rows
5. 2024-02-15 (test): 119,239 nulls out of 119,239 rows
6. 2024-10-07 (test): 118,842 nulls out of 118,842 rows
7. 2023-12-08 (val): 118,753 nulls out of 118,753 rows
8. 2024-06-15 (test): 116,312 nulls out of 116,312 rows
9. 2024-03-23 (test): 116,295 nulls out of 116,295 rows
10. 2024-04-10 (test): 115,859 nulls out of 115,859 rows
11. 2024-11-08 (test): 115,804 nulls out of 115,804 rows
12. 2024-01-17 (test): 115,800 nulls out of 115,800 rows
13. 2024-07-25 (test): 115,528 nulls out of 115,528 rows
14. 2024-11-17 (test): 114,701 nulls out of 114,701 rows
15. 2024-01-21 (test): 114,375 nulls out of 114,375 rows
16. 2024-11-27 (test): 114,168 nulls out of 114,168 rows
17. 2024-12-30 (test): 114,053 nulls out of 114,053 rows
18. 2024-10-11 (test): 113,808 nulls out of 113,808 rows
19. 2024-01-31 (test): 113,392 nulls out of 113,392 rows
20. 2024-09-19 (test): 113,307 nulls out of 113,307 rows

## Root Cause Analysis

The high concentration of null values suggests:

1. **MODIS AOD Data Availability Issues**: The AOD columns are derived from MODIS satellite data, which has known limitations:
   - Cloud cover blocking observations
   - Satellite orbital coverage gaps
   - Data retrieval failures for certain dates/regions

2. **Data Processing Pipeline Issues**: 
   - Early 2019 has systematic nulls (possible retroactive data fetch issues)
   - Late 2023 (Nov-Dec) shows widespread nulls in validation set (seasonal or data processing issue)
   - July 2024 shows complete nulls for several consecutive dates in test set

3. **Dataset-Specific Patterns**:
   - Validation set heavily affected (particularly December 2023)
   - Test set has scattered nulls throughout 2024
   - Train set has historical nulls concentrated in early 2019

## Recommendations

1. **Data Investigation**: Examine MODIS data availability for the identified date ranges
2. **Feature Engineering**: Consider temporal interpolation or forward-fill for missing values
3. **Model Considerations**: The model should be robust to missing AOD features, particularly for:
   - Validation set: December 2023 dates
   - Test set: January-July 2024 dates

4. **Alternative Features**: Consider using complementary satellite or ground-based AOD products when MODIS data is unavailable

## Files Analyzed

- Total files processed: **Training files from all three splits** (train, val, test)
- Date range: **2019-01-08 to 2024-12-31**
- Pattern: Each date has a corresponding `{YYYY-MM-DD}_training.parquet` file with actual observation rows
